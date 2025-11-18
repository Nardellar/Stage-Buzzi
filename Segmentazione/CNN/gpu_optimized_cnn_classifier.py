"""
GPU ottimizzata versione di CNN + Classificatore.
Riduce il numero di campioni generati per i classificatori boosting
e abilita l'uso della GPU per XGBoost/LightGBM quando disponibile.
"""

import os
import pickle
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import optuna
import tensorflow as tf
import xgboost as xgb
import lightgbm as lgb

from data_module import (
    load_dataset,
    augment_dataset,
    compute_class_weights_dict,
)
from tuning import tune_classifier

#Usiamo l'API nativa di XGBoost per controllare direttamente i parametri (early stopping, parametri gpu, numero round..)
#Pero' la pipeline richiedere uno stimaotre sklearn per il tuning, quindi creiamo un wrapper che lo faccia comportare come uno sklearn estimator.
class XGBBoosterWrapper:
    """Wrapper per rendere un booster XGBoost simile allo sklearn estimator."""
    # Salva il booster e il numero di classi come attributo d'istanza.
    def __init__(self, booster: xgb.Booster, num_classes: int):
        self.booster = booster
        self.num_classes = num_classes

    # Predice le classi per un batch di dati.
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Converte l'array NumPy con i dati del batch in un DMatrix (formato richiesto da XGBoost).
        dmatrix = xgb.DMatrix(X)
        # Predice le probabilità per ogni classe per ogni campione del batch. (es: [[0.1, 0.3, 0.6], [0.8, 0.1, 0.1], ...])
        preds = self.booster.predict(dmatrix)
        #restituisce l'indice della classe con la probabilità più alta per ogni campione del batch. (es: [2, 0, 1, ...])
        return np.argmax(preds, axis=1)

#dataclass che centralizza tutti i parametri di configurazione della segmentazione
@dataclass
class GPUClassifierConfig:
    #backbone cnn utilizzato. puo' essere cambiato all'avvio del programma (convnext_tiny o resnet50)
    #                                                                     (convex migliore ma pesante, resnet50 leggero e veloce)
    cnn_model: str = "convnext_tiny"
    #classificatore utilizzato. puo' essere cambiato all'avvio del programma (xgboost o lightgbm)
    #                                                                      (XGBOost migliore ma lento, lightgbm piu' veloce)
    classifier: str = "xgboost"  # "xgboost" o "lightgbm"
    #percorso delle cartelle con immagini e maschere
    images_dir: str = "../images/Immagini"
    masks_dir: str = "../images/Maschere"
    #dimensione a cui vengono ridimensionate le immagini e maschere prima del pre processing
    image_size: Tuple[int, int] = (1024, 1024)
    #dimensione quadrata della feature map estratta dalla CNN (se None o 0 mantiene la risoluzione originale)
    feature_map_size: Optional[int] = None
    #numero di immagini processate in parallelo
    batch_size: int = 2
    #abilita data augmentation
    use_augmentation: bool = True
    #converte le immagini in scala di grigi prima del preprocessing
    use_grayscale: bool = False
    #limite di pixel campionati per immagine durante l'estrazione delle features
    max_pixels_per_image: int = 20000 
    #abilita l'uso della GPU per il classificatore
    use_gpu: bool = True
    #numero di classi da classificare
    num_classes: int = 5
    #filtri dei blocchi Convoluzionali del decoder che raffinano la feature-map
    decoder_filters: int = 128


class GPUOptimizedCNNSegmentationClassifier:
    """
    Implementa la pipeline di segmentazione suddivisa in 3 fasi:
    Pipeline CNN (backbone + decoder) -> Feature Map -> Classificatore
    Estrae feature spaziali a risoluzione ridotta per mantenere info pixel-level
    senza esplodere in memoria, e abilita parametri GPU per i classificatori.
    """
    #inizializza la classe con la configurazione data in input o la configurazione predefinita
    def __init__(self, config: Optional[GPUClassifierConfig] = None):
        self.config = config or GPUClassifierConfig()
        self.images_dir = self.config.images_dir
        self.masks_dir = self.config.masks_dir

        # Serve a mappare i numeri di classe ai nomi delle classi
        self.class_names = [
            "Resina",
            "Pori/Imperfezioni",
            "Fase Fusa",
            "Belite",
            "Alite",
        ]
        #modello CNN. verra' definito dopo da "build_feature_extractor()"
        self._feature_extractor = None
        #classificatore. definito in "tune_classifier()"
        self.classifier = None
        #migliori parametri trovati durante il tuning. definiti in "tune_classifier()"
        self.best_params = None
        #pesi delle classi, calcolati in "compute_class_weights_dict()"
        self.class_weights = None
        #numero di round di boosting ideale trovato durante il tuning. definiti in "tune_classifier()"
        self.best_num_boost_round = None
        #features estratte dalla CNN per ogni pixel. definite in "extract_features()"
        self.pixels_features = None
        #labels assegnate per ogni pixel. definite in "extract_features()"
        self.pixels_labels = None
        #pesi del modello CNN. definiti in "extract_features()" (serve per ricostruire il modello se la dimensione dell'immagine cambia senza dover ricaricare i pesi)
        self._feature_extractor_weights = None
        #dimensione corrente dell'immagine. definita in "extract_features()" per sapere a che dimensione creare l'estrattore
        self.current_image_size = tuple(self.config.image_size)

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #
    #Con filename specifichiamo le immagini che vogliamo caricare (quelle del training)
    # se filename e' None, carica tutte le immagini e maschere nella cartella
    def load_train_data(self, filenames: Optional[Iterable[str]] = None):
        """Carica le immagini e maschere specificate dal dataset e le salva internamente. Applica Augmentation e/o scala di grigi se richiesto."""

        images, masks = load_dataset(
            #specifico le cartelle da dove prendere i dati
            images_dir=self.images_dir,
            masks_dir=self.masks_dir,
            #specifico la dimensione a cui ridimensionare le immagini e maschere che carico (nel caso le immagini avessero dimensioni diverse dalla risoluzione richiesta)
            image_size=tuple(self.config.image_size),
            #specifico se voglio caricarle in scala di grigi
            use_grayscale=self.config.use_grayscale,
            #specifico le immagini che vogliamo caricare (quelle del training)
            filenames=filenames,
        )

        #applico l'augmentation se richiesto
        if self.config.use_augmentation:
            images, masks = augment_dataset(images, masks)

        #salvo le immagini e maschere nella classe (rappresentate come un numpy di dimensione (Num_immagini, altezza, larghezza, 3) per le immagini e (Num_immagini, altezza, larghezza) per le maschere)
        #ogni immagine e' normalizzata in [0,1] e ogni pxel dell'immagine è un vettore RGB (es: [0.5, 0.5, 0.5])
        #TODO: da capire meglio come sono rappresentate le immagini in "images", che dati contengono e qualche esempio, stessa cosa per la singola immagine e per il singolo pixel
        self.images = images
        self.masks = masks
        #Scansiona le maschere per contare quanti pixel ci sono di ogni classe e calcola i pesi per controbilanciare.
        #I pesi sono utilizzati poi dal classificatore
        self.class_weights = compute_class_weights_dict(
            self.masks, self.config.num_classes
        )

    # ------------------------------------------------------------------ #
    # Feature extraction
    # ------------------------------------------------------------------ #

    def _compute_features(self, images: np.ndarray, masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Estrae feature spaziali prodotte dalla CNN."""
        #se il modello CNN non e' stato ancora costruito, lo costruisce e salva i pesi 
        # (la funzione viene richiamata piu' volte ecco il perche' del controllo)
        if self._feature_extractor is None:

            #Carica e congella il backbone pre-addestrato della CNN
            self._feature_extractor = self._build_feature_extractor()
            #salviamo i pesi per poterli riprisitnare comodamente in caso si ricrei il modello
            self._feature_extractor_weights = self._feature_extractor.get_weights()
            #tracciamo la dimensione corrente dell'immagine per cui il modello e' valido (se cambiasse va ricostruito)
            self.current_image_size = tuple(self.config.image_size)

        #inizializziamo liste per salvare le features e labels estratte
        features = []
        labels = [] #contiene i numeri di classe per ogni pixel (da 0 a 4)
        #generatore di numeri casuali (usato piu' avanti per il batch downsampling). impostato ad un seed fisso per riproducibilità
        random_generator = np.random.default_rng(seed=42) 
        #per ogni batch del totale di immagini:
        for n_batch in range(0, len(images), self.config.batch_size):
            #prendo n-esimo batch delle immagini e il batch delle relative maschere e lo salvo in lista
            batch_imgs = images[n_batch : n_batch + self.config.batch_size]
            batch_masks = masks[n_batch : n_batch + self.config.batch_size]
            #estraggo le features dell'n-esimo batch di immagine
            feature_maps = self._feature_extractor.predict(batch_imgs, verbose=0)

            #per ogni feature map e maschera dentro l'n-esimo batch:
            for feature_map, mask in zip(feature_maps, batch_masks):
                #salvo altezza e larghezza della feature map
                height_feature_map, width_feature_map = feature_map.shape[:2]
                #ridimensiono la maschera alla stessa dimensione
                mask_resized = cv2.resize(
                    mask,
                    (width_feature_map, height_feature_map),
                    #uso interpolazione "nearest" per mantenere i valori interi in modo che corrispondano sempre ad una classe (0, 1, 2, 3, 4)
                    interpolation=cv2.INTER_NEAREST,
                )
                #creo una maschera booleana che vale True per i pixel che non sono background (0)
                #es: Mask_resized = [0,2,4,1,0] -> valid = [False, True, True, True, False]
                valid = mask_resized != 0
                #se la maschera non ha nessun pixel valido, salto il batch
                if not np.any(valid):
                    continue
                
                #trasformiamo ora entrambi i tensori di maschere e immagini in liste (feature + label) per creare coppie (feature, label) per ogni pixel per darle al training
                #appiattiamo feature map in una matrice.
                #feature_map ha shape (height_feature_map, width_feature_map, lista features del pixel) (es. 128×128×768).
                #la appiattiamo flat_features di shape (height_feature_map * width_feature_map, lista features del pixel):
                #una riga per ogni pixel: (128×128 = 16384 righe),
                flat_features = feature_map.reshape(-1, feature_map.shape[-1])
                #appiattiamo la maschera in un array monodimensionale
                #mask_resized ha shape (height_feature_map, width_feature_map) (es. 128×128).
                #la appiattiamo label_flat di shape (height_feature_map * width_feature_map):
                #una cella per ogni pixel con la label del pixel (128×128 = 16384 righe),
                label_flat = mask_resized.reshape(-1)
                #filtriamo entrambi togliendo tutte le informazioni con pixel di background(cioe' con valid = False)
                filtered_features = flat_features[valid.ravel()]
                label_filtered = label_flat[valid.ravel()] - 1 #diminuiamo di 1 le label per usare l'intervallo da 0 ...4 piuttosto che 1 ... 5 (ora che abbiamo tolto la label 0 di background) per poter essere compatibili con i classificatori (XGBoost/LightGBM)

                #ci assicuriamo che eventuali valori fuori range vengano riportati nei limiti [0, 4]
                label_filtered = np.clip(label_filtered, 0, self.config.num_classes - 1)

                #BLOCCO CAMPIONAMENTO PIXEL
                if (
                    #se numero massimo di pixel per immagine e' impostato ed il numero di label calcolato dalla maschera e' maggiore:
                   
                    self.config.max_pixels_per_image is not None
                    and len(label_filtered) > self.config.max_pixels_per_image
                ):  #allora campioniamo un sottoinsieme di pixel per mantenere il numero di pixel entro il limite impostato
                    
                    #Campiono casualmente alcuni pixel
                    sample_indeces = random_generator.choice(
                        #numero di pixel da cui scegliere
                        len(label_filtered),
                        #numero di pixel da scegliere
                        size=self.config.max_pixels_per_image,
                        #non ripetiamo pixel già campionati
                        replace=False,
                    )
                    #prendo solo le features e le label dei pixel campionati
                    filtered_features = filtered_features[sample_indeces]
                    label_filtered = label_filtered[sample_indeces]

                features.append(filtered_features)
                labels.append(label_filtered)

        if not features:
            raise ValueError("Nessuna feature valida estratta.")
        #compattimo tutte le matrici features (cioe num. pixel dell'immagine x num. features ciascuna) in un unica matrice con shape (num. pixel totali, num. features)
        #compattiamo tutti gli array labels (di dimensione = num. pixel della singola immagine) in un unico array con shape (num. pixel totali)
        return np.vstack(features), np.hstack(labels)

    #lavora sul set di training caricato con load data caricando le immagini gia' salvate nel modello. Salva i tensori in memoria
    def extract_train_features(self):
        """Estrae le features, dalle immagini salvate internamente, dei pixel non appartenenti allo sfondo inferte dalla CNN (fino a un massimo di max_pixels_per_image) 
        e salva nel modello tutte le feature estratte (self.pixels_features) e tutto le label corrispondenti (self.pixels_labels)."""
        #calcoliamo le features e la label di ogni pixel della maschera non appartenente allo sfondo
        dataset_features, dataset_labels = self._compute_features(self.images, self.masks)
        #salviamo le features e le labels nella classe
        self.pixels_features = dataset_features
        self.pixels_labels = dataset_labels
        return dataset_features, dataset_labels

    #versione "stateless". riceve le immagini e maschere da usare e restituisce i risulati senza alterare lo stato interno del modello
    #usato per il val/test set
    def extract_features_stateless(self, images: np.ndarray, masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Estrae feature da un dataset fornito senza alterare lo stato interno.
            Restituisce una tupla contenente features : label per ciascun pixel non background
        """
        return self._compute_features(images, masks)

    #necessaria dato che Keras la dimensione dell'input del modello e' fissata alla sua creazione
    def ensure_feature_extractor_size(self, image_size: Tuple[int, int]) -> None:
        """
        Ricostruisce il feature extractor per poter gestire immagini di dimensioni diverse.
        Mantiene i pesi addestrati, ma aggiorna l'input/output spatial size.
        Utile per permettere tra addestramento ed evaluation immagini di dimensioni diverse
        """
        target_size = (int(image_size[0]), int(image_size[1]))
        #Controlla che il modello sia gia' stato costruito. se no lo costruisce
        if self._feature_extractor is None:
            raise RuntimeError("Feature extractor non inizializzato: carica o addestra il modello prima.")

        #Se la dimensione dell'immagine richiesta e' la stessa del modello corrente, non fa nulla
        if tuple(self.current_image_size) == target_size:
            return

        #Se i pesi del modello non sono stati salvati, li salva, cosi' da poterli utilizzare per ricreare l'estrattore
        if self._feature_extractor_weights is None:
            self._feature_extractor_weights = self._feature_extractor.get_weights()

        #Costruisce l'estrattore CNN con la nuova dimensione
        rebuilt = self._build_feature_extractor(override_image_size=target_size)
        #Imposta i pesi dell'estrattore ricostruito con i pesi del modello originale
        rebuilt.set_weights(self._feature_extractor_weights)
        #Salva l'estrattore ricostruito e lo rende quello effettivo
        self._feature_extractor = rebuilt
        #aggiorna il campo che informa per quale dimensione il modello e' valido
        self.current_image_size = target_size
        #aggiorna i pesi del modello ricostruito con i pesi del modello originale
        self._feature_extractor_weights = self._feature_extractor.get_weights()

    #chiediamo in input opzionale l'image size cosi da poterlo risoctruire in caso di necessita con "ensure_feature_extractor_size()"
    def _build_feature_extractor(self, override_image_size: Optional[Tuple[int, int]] = None) -> tf.keras.Model:
        """Crea e restituisce l'estrattore (backbone + decoder) CNN partendo dai parametri in input."""
        
        #se ci e' fornita dimensione dell'immagine la usiamo, altrimenti usiamo quella del modello corrente
        image_size = override_image_size or self.config.image_size
        #definisce la shape dell'input del modello (Height, Width, canali colori)
        input_shape = (*image_size, 3)
        model_name = self.config.cnn_model.lower()

        #costruisce il modello CNN partendo dai parametri in input
        if model_name == "resnet50":
            cnn_extractor = tf.keras.applications.ResNet50(
                weights="imagenet", include_top=False, input_shape=input_shape
            )
        elif model_name == "convnext_tiny":
            cnn_extractor = tf.keras.applications.ConvNeXtTiny(
                weights="imagenet", include_top=False, input_shape=input_shape
            )
        else:
            raise ValueError(
                f"Modello CNN non supportato: {self.config.cnn_model}. "
                "Usa 'resnet50' o 'convnext_tiny'."
            )

        #congeliamo i pesi del modello CNN
        for layer in cnn_extractor.layers:
            layer.trainable = False

        # Determina la dimensione target della feature map finale.
        target_config = self.config.feature_map_size
        if target_config in (None, 0):
            target_h, target_w = image_size
        elif isinstance(target_config, int):
            target_h = target_w = int(target_config)

        else:
            raise ValueError("feature_map_size deve essere un intero oppure None.")
        
        #memorizziamo quanti filtri usera' il decoder
        decoder_filters = self.config.decoder_filters

        #prendiamo il tensore simbolico generato dall'ultimo layer della CNN, cioe' le feature maps (non contiene ancora dati)
        decoder_cnn = cnn_extractor.output

        #-------UPSAMPLING-------
        #Il backbone CNN non mantiene la stessa dimensione di ingresso durante l'estrazione delle feature maps, ma anzi le diminuisce
        #e' necessario quindi applicare prima del classificatore un upsampling per ripristinare la dimensione originale 
        #salviamo le dimensioni correnti dell'output  dell'estrattore CNN
        current_h = decoder_cnn.shape[1]
        current_w = decoder_cnn.shape[2]

        if current_h is None or current_w is None:
            raise ValueError(
                "Impossibile determinare la dimensione delle feature map: specifica un image_size esplicito."
            )
        current_h = int(current_h)
        current_w = int(current_w)

        #Costruiamo ora il decoder per la CNN
        #applico upsampling + convoluzione fino a quando la dimensione delle feature maps non e' uguale o maggiore a quella target
        while (current_h < target_h) or (current_w < target_w):
            #aggiungo un blocco convoluzionale che affina le faetures prima di applicare upsampling
            #Come numero di filtri usiamo i filtri definiti in config.
            #usaimo filtri 3 x 3, 
            #padding "same" forza il processo di ocnv a mantenere le dimensioni spaziali ricevute in input
            #activation "relu" per introdurre non lienarita' e un attivazione veloce
            decoder_cnn = tf.keras.layers.Conv2D(decoder_filters, kernel_size=3, padding="same", activation="relu")(decoder_cnn)
            
            #applico upsampling
            #size: raddoppiamo sia altezza che larghezza delle feature map
            #interpolation: "bilinear" per "riempire" i nuovi pixel -> le feature dei nuovi pixel sono la media pesata dei 4 pixel vicini.
            # per interpolazione: bilineare 
            decoder_cnn = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(decoder_cnn)
            
            #aggiorno a che dimensioni siamo per mantenenre il ciclo while
            current_h *= 2
            current_w *= 2

        #riapplichiamo un ultima convoluzione standard per ricombinare l'informazione contenstuale persa durante l'ultima interpolazione
        decoder_cnn = tf.keras.layers.Conv2D(decoder_filters, kernel_size=3, padding="same", activation="relu")(decoder_cnn)
        
        #applichiamo una convoluzione finale con meta' dei filtri per comprimere le features in un numero di canali piu gestibile al classificatore
        decoder_cnn = tf.keras.layers.Conv2D(decoder_filters // 2, kernel_size=3, padding="same", activation="relu")(decoder_cnn)

        #applichaimo una rifinitura con un resize finale alla dimensione esplcitiamente richiesta nel caso le feature maps siano leggermente piu grandi delle misure desiderate
        resized = tf.keras.layers.Resizing(target_h,target_w,interpolation="bilinear",name="feature_resizer",)(decoder_cnn)
        #creaimo un nuovo modello keras che prende in input il tensore del backbone originale e restiusice il resize (dopo essere passato per il decoder+ upsampling)
        return tf.keras.Model(inputs=cnn_extractor.input, outputs=resized)
        #IL DECODER E' ADDESTRABILE COME IL CLASSIFICATORE
    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train_classifier_optuna(self, n_trials: int = 20, validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None, timeout: Optional[int] = None,) -> float:
        """
        Esegue ottimizzazione Optuna usando il validation set fornito. Se un timeout e' definito, interrompe la ricerca appena scade il tempo
        """
        #richiama la funzione di tuning (in tuning.py) e otteniamo:
        #best_metric = il risultato migliore ottenuto sulla metrica scelta
        #best classifier = il classificatore con la configurazione migliore
        #best_params = gli iperparametri migliori per il classificatore
        #best_iteration = il numero di boost-Round da utilizzare per raggiungere il risultato migliore
        best_metric, best_classifier, best_params, best_iteration = tune_classifier(
            config=self.config,
            pixels_features=self.pixels_features,
            pixels_labels=self.pixels_labels,
            class_weights=self.class_weights,
            n_trials=n_trials,
            timeout=timeout,
            validation_data=validation_data,
        )

        #memorizza il classificatore addestrato e i relativi metadati
        self.classifier = best_classifier
        self.best_params = best_params
        self.best_num_boost_round = best_iteration

        print(f"Migliori parametri: {self.best_params}")
        print(f"Migliore accuracy (val): {best_metric:.4f}")

        return best_metric


    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path_prefix: str):
        """Salva feature extractor (+) classificatore su disco."""
        #controllo che il modello sia stato addestrato e che l'estrattore sia stato costruito
        if self.classifier is None or self._feature_extractor is None:
            raise RuntimeError("Modello non addestrato, nulla da salvare.")

        #definisco il path e salvo il modello per intero con .save (pesi + struttura)
        feature_path = f"{path_prefix}_feature_extractor.keras"
        self._feature_extractor.save(feature_path)

        #creo un dizionario per salvare i metadati del classificatore
        payload = {
            "classifier": self.classifier,
            "best_params": self.best_params,
            "class_weights": self.class_weights,
            "config": self.config,
            "best_num_boost_round": self.best_num_boost_round,
        }
        #salvo gli iperparametri del classificatore in un file apposito
        with open(f"{path_prefix}_classifier.pkl", "wb") as f:
            pickle.dump(payload, f)

        print(f"Modello salvato (feature extractor + classifier) con prefisso {path_prefix}")

    def load(self, path_prefix: str):
        """Carica modello precedentemente salvato."""
        #controllo che i file del CNN e del classificatore siano presenti
        feature_path = f"{path_prefix}_feature_extractor.keras"
        classifier_path = f"{path_prefix}_classifier.pkl"
        if not os.path.exists(feature_path) or not os.path.exists(classifier_path):
            raise FileNotFoundError("File del modello non trovati.")

        #carico il l'estrattore CNN e lo salvo nella classe
        self._feature_extractor = tf.keras.models.load_model(feature_path)
        #carico i pesi dell'estrattore CNN
        self._feature_extractor_weights = self._feature_extractor.get_weights()

        #carico i metadati relativi al classificatore 
        with open(classifier_path, "rb") as f:
            payload = pickle.load(f)

        required_keys = [
            "classifier",
            "best_params",
            "class_weights",
            "config",
            "best_num_boost_round",
        ]
        #controlla che tutti i campi siano presenti
        missing = [key for key in required_keys if key not in payload]
        if missing:
            raise KeyError(
                f"File del modello incompleto: mancano i campi {missing}. "
                "Rigenera i file salvando nuovamente il modello completo."
            )

        #salvo il classificatore e tutti i metadati nel modello
        self.classifier = payload["classifier"]
        self.best_params = payload["best_params"]
        self.class_weights = payload["class_weights"]
        self.config = payload["config"]
        self.best_num_boost_round = payload["best_num_boost_round"]
        self.current_image_size = tuple(self.config.image_size)
