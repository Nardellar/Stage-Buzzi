"""
Valutazione completa del modello GPU-optimized.
Riproduce le stesse trasformazioni del confronto:
 - caricamento dataset
 - inferenza CNN + classificatore
 - raffinamento DenseCRF
 - metriche (accuracy, confusion matrix)
 - salvataggio anteprima con GT e maschera predetta
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional

from data_module import prepare_dataset_splits, load_dataset_stateless
from gpu_optimized_cnn_classifier import GPUOptimizedCNNSegmentationClassifier

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from sklearn.metrics import accuracy_score, confusion_matrix

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import (
    unary_from_labels,
    create_pairwise_gaussian,
    create_pairwise_bilateral,
)

# Cartella contenente lo script (usata come riferimento per i path relativi).
BASE_DIR = Path(__file__).resolve().parent
# Directory di immagini e maschere da valutare; cambia se il dataset è altrove.
IMAGES_DIR = (BASE_DIR / "../images/Immagini").resolve()
MASKS_DIR = (BASE_DIR / "../images/Maschere").resolve()
# Prefisso dei file salvati dal training ((cnn e decoder).keras + classificatre.pk1).
MODEL_PREFIX = "gpu_optimized_cnn_model"
# Percorso del file PNG con le anteprime (GT vs predizione).
PREVIEW_PATH = BASE_DIR / "gpu_model_preview.png"
# Percorso dell'immagine con la matrice di confusione salvata.
CONFUSION_PATH = BASE_DIR / "gpu_confusion_matrix.png"
#Percorso della cartella contenente gli split del dataset.
SPLIT_DIR = (BASE_DIR / "splits").resolve()
#Nomi delle classi da classificare.
CLASS_NAMES = ["Resina", "Pori/Imperfezioni", "Fase Fusa", "Belite", "Alite"]


def apply_dense_crf(image: np.ndarray, predicted_mask: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Applica il DenseCRF per raffinare le predizioni del modello.
    Args:
        image: Immagine in formato uint8 (0-255) con dimensioni (altezza, larghezza, 3)
         predicted_mask: Maschera predetta dal modello in formato float (probabilità per classe) con dimensioni (altezza, larghezza, num_classi)
        num_classes: Numero di classi da classificare (5)
    Returns:
        Maschera raffinata in formato int32 (0-6)
    """
    # DenseCRF richiede array C-contigui; li forziamo per evitare errori di memoryview
    image = np.ascontiguousarray(image)
    predicted_mask = np.ascontiguousarray(predicted_mask, dtype=float)
    #salvo altezza e larghezza dell'immagine
    height, width = image.shape[:2]
    #creo un oggetto DenseCRF2D con le dimensioni dell'immagine e il numero di classi
    crf = dcrf.DenseCRF2D(width, height, num_classes)

    #pixels_prob_label qui rappresenta le probabilità per pixel nella matrice (altezza, larghezza, num_classi)
    #clipping per limitare le probabilita' ad un range di valori valido (evita valori negativi, zero o > 1)
    #evita problemi con log(0) che darebbe -inf
    pixels_prob_labels = np.clip(predicted_mask, 1e-6, 1.0)

    #convertiamo le probabilita' in energia "unaria"(che e' la metrica con cui ragiona la CRF)
    unary = -np.log(pixels_prob_labels)
    #rendiamo l'array C-contiguo/float32 per compatibilità con pydensecrf
    unary = np.ascontiguousarray(unary, dtype=np.float32)
    #il CRF chiede i dati nel formato: (Classi, Numero_Totale_Pixel).
    #percio' cambiamo ordine dei dati e schiacciamo altezza e lunghezza
    #(altezza, lunghezza, numero classi) -> (numero classi, altezza, lunghezza) -> (numero classi, altezza * lunghezza)
    unary = unary.transpose(2, 0, 1).reshape(num_classes, -1)
    unary = np.ascontiguousarray(unary, dtype=np.float32)
    
    #impostiamo l'energia unaria calcolata nel CRF. 
    crf.setUnaryEnergy(unary)

    #creo features gaussiane per energia pairwise spaziale.      obbiettivo --> rimuovere il rumore dall'immagine
    #per rimuovere il rumore dall'immagine, oltre all'energia unaria calcolata dalle probabilita' dei pixel,
    #calcolo l'energia a coppie (pairwise). l'energia a coppie penalizza (aumenta il costo di energia) il crf se assegna 2 o piu pixel vicini classi diverse
    #il modello e' incentivato quindi a fornire a pixel vicini stessa classificazione

    #sdims (standard deviation): significa che il sistema considera "vicini" e influenti i pixel entro un raggio di circa 3 pixel lungo l'asse X e Y.
    #shape: dimensioni dell'immagine (altezza, larghezza)
    #(non cosnidera il colore, solo la posizione X;Y nello spazio)
    gaussian_energy = create_pairwise_gaussian(sdims=(3, 3), shape=image.shape[:2])
    #imposto l'energia a coppie gaussiane nel CRF.
    # compat (indice di compatibilita'): e' una costante moltiplicativa che imposta l'ifluenza sul costo finale del vincolo di vicinanza spaziale rispetto al vincolo di predizione locale
    crf.addPairwiseEnergy(gaussian_energy, compat=3)

    #calcolo l'energia bilaterale.  obbiettivo --> affinare i contorni
    # funziona come l'energia gaussiana, ma oltre alla posizione tiene conto anche dei colori --> l'energia bilaterale penalizza il crf se assegna a 2 pixel vicini, e con stesso colore classi diverse.
    #il fatto che tenga conto del colore gli permette di smussare emglio i bordi dell'immagine (pixel con colori diversi vicini non vengono penalizzati)
    #args:
    #   sdims (deviazione standard spaziale): Determina il raggio di influenza dei pixel vicini. Un valore di 5 significa che il raggio di smoothing è di circa 5 pixel.
    #   schan (deviazione standard del canale): Determina l'ampiezza del raggio di "somiglianza"" dei colori vicini. Un valore di 10 significa che pixel con tonalita' differenti di massimo 10 unita' saranno cosniderati simili (e quindi penalizzati se di classi diverse).
    #   img: immagine in input
    #   chdim: indica in che asse di dimensione di "img" si trovino i canali di colore (2 = RGB)
    bilateral_energy = create_pairwise_bilateral(
        sdims=(5, 5),
        schan=(10, 10, 10), #ripetuto tre volte perche' sono i tre canali di colore (R,G,B)
        img=image,
        chdim=2,
    )
    #imposto l'energia bilatrale nel CRF. con compat = 5
    crf.addPairwiseEnergy(bilateral_energy, compat=5)

    #esegue l'algoritmo di inferenza CRF (il 5 indica il numero di iterazioni)
    #ad ogni iterazione i pixel "scambiano messaggi" con i loro vicini, influenzandosi 
    # reciprocamente. Ad esempio, se un pixel ha una forte pressione per essere Resina 
    # (dall'Energia Unaria) e i suoi vicini hanno una forte pressione per essere Pori (dall'Energia a Coppie), l'algoritmo bilancia queste forze.
    # output: è la matrice delle probabilità marginali ottimizzate per ciascun pixel e ciascuna classe.
    # shape: (Numero classi, numero pixel totali)
    CRF_probabilties = crf.inference(5)
    #convertiamo l'output CRF nella maschera di segmentazione predetta finale
    # Per ogni pixel seleziono la classe con la probabilità più alta. (l'asse 0 e' quello delle classi)
    #successivamente trasformiamo l'elenco 1D dei pixel totali in una matrice 2D con le stesse dimensioni dell'immagine originale.
    # L'output è un array 2D che contiene l'etichetta di classe (da 0 a 4) per ogni pixel.
    refined_mask = np.argmax(CRF_probabilties, axis=0).reshape((height, width))
    return refined_mask


def evaluate_gpu_model(model, dataset: Dict[str, np.ndarray],) -> Dict:
    """
    Valuta il modello sul set di evaluation (applicando CRF)
    Args:
        model: Modello CNN-based addestrato.
        dataset: Dizionario contenente: 
            - images: array numpy con forma (Num_immagini, altezza, larghezza, 3)
            - masks: array numpy con forma (Num_immagini, altezza, larghezza)
            - image_paths: lista dei percorsi delle immagini
    Returns:
        Dizionario contenente:
            - accuracy: accuracy del modello
            - predictions: array numpy con le predizioni del modello
            - ground_truth: array numpy con i ground truth
            - previews: lista di 3 dizionari contenenti ciascuno [immagine, maschera ground truth, maschera predette, path immagine]
    """
    #inizalizzo le liste
    predictions = []      # accumula le predizioni pixel-level (solo pixel validi)
    ground_truth = []     # accumula le label ground truth (solo pixel validi)
    previews = []         # accumula fino a 3 anteprime da usare poi per la visualizzazione


    #per ogni istanza del dataset:
    for i, (img, mask, img_path) in enumerate(
        zip(dataset["images"], dataset["masks"], dataset["image_paths"]), start=1
    ):
        print(f"[GPU] Processando immagine {i}/{len(dataset['images'])}...")
        #salvo le dimensioni dell'immagine. img = (altezza, larghezza, 3)
        image_native_size = (img.shape[0], img.shape[1])
        #controlliamo che il backbone CNN sia configurato per quella dimensione.
        #se le dimensioni sono diverse, viene ricostruito il modello mantenendo i pesi
        model.ensure_feature_extractor_size(image_native_size)
        #convertiamo l'immagine in float32 come richiesto dai modelli Keras (l'imagine e' anche gia' normalizzata in [0,1] da load_dataset_stateless)
        img_prepared = img.astype(np.float32)
        #aggiungiamo una dimensione all'array numPy img_prepared per poterlo passare al feature extractor. img_prepared = (1, altezza, larghezza, 3)
        #calcoliamo le features dell'immagine e otteniamo in output una feature map delle seguenti dimensioni (1, altezza, larghezza, lista features del pixel)
        feature_map = model._feature_extractor.predict(np.expand_dims(img_prepared, axis=0), verbose=0)
        #estraimo le dimensioni della feature map
        height_feature_map, width_feature_map = feature_map.shape[1:3]

        #appiattiamo la feature map in una matrice 2D per il classificatore
        #da (1, altezza, larghezza, lista features del pixel) a (altezza * larghezza, lista features del pixel)
        feature_map_flat = feature_map.reshape(-1, feature_map.shape[-1])
        #eseguiamo la predizione del classificatore su ogni pixel della feature map
        #restituisce un array con le probabilità per ogni pixel es: [[0.1, 0.3, 0.6], [0.8, 0.1, 0.1], ...]
        preds = model.classifier.predict_proba(feature_map_flat)
        #convertiamo in float e ripristiniamo la forma originale della feature map
        preds_map = np.asarray(preds, dtype=float).reshape(height_feature_map, width_feature_map, len(CLASS_NAMES))
        #eseguo upsampling per ogni canale (classe) separatamente (prendo la feature map 2D dell'intera immagine per ogni classe e calcolo upsampling)
        upsampled_channels = [
            cv2.resize(
                preds_map[:, :, c],  #presa la feature map della classe "c"
                (mask.shape[1], mask.shape[0]), #dimensioni target
                interpolation=cv2.INTER_LINEAR) #interpolazione lineare per "riempire" i nuovi pixel
            for c in range(len(CLASS_NAMES)) #per ogni classe
        ]

        #ricostruiamo la feature map 3D con le probabilità per ogni classe (altezza feature_map, larghezza feature_map, num_classi)
        preds_upsampled = np.stack(upsampled_channels, axis=-1)

        #rinormalizziamo per ottenere probabilità valide dopo il resize (che con l'interpolazione puo' generare probabilita' non valide)
        #rinormalizzo le probabilita' di ogni pixel in modo che la somma delle probabilita' sia 1.
        #Nota: il clipping viene fatto dentro apply_dense_crf per rendere la funzione più robusta
        #risultato = array 3D di dimensioni (altezza immagine, larghezza immagine, numero classi) con tutte le probabilita' per ogni pixel
        preds_upsampled /= preds_upsampled.sum(axis=-1, keepdims=True)

        #convertiamo l'immagine in formato uint8 (0-255) per il DenseCRF
        image_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        #applico il DenseCRF per raffinare le predizioni
        #restituisce numpy array 2D con la classe predetta di ogni pixel
        CRF_pred = apply_dense_crf(image_uint8, preds_upsampled, len(CLASS_NAMES))

        #appiattiamo la maschera in un array 1D e sottraiamo 1 per convertire le classi da [0-4] a [-1-3]
        #mappo eventuali label fuori range su background prima del flatten
        mask_clean = np.where(mask > len(CLASS_NAMES), 0, mask)
        gt_flattened_mask = mask_clean.reshape(-1) - 1
        #appiattiamo le predizioni con CRF in un array 1D
        pred_flat = CRF_pred.reshape(-1)
        #creo una maschera booleana che vale True per i pixel che non sono background (-1)
        valid = gt_flattened_mask >= 0
        #se la maschera non ha nessun pixel valido, salto il batch
        if not np.any(valid):
            continue
        
        #salviamo nelle liste la predizione e il ground truth solo per i pixel validi (non background)
        predictions.extend(pred_flat[valid])
        ground_truth.extend(gt_flattened_mask[valid])

        #salvo le prime tra immagini da usare succssivamente nell'anteprima
        if len(previews) < 3:
            previews.append(
                {
                    "image": img, #immagine originale
                    "mask_gt": mask, #maschera ground truth
                    "predicted_mask": CRF_pred, #mashcera predetta dal modello
                    "path": img_path, #percorso file immagine originale
                }
            )

    #calcolo accuracy modello tra pixel maschera ground truth e predizione del modello (solo sui pixel validi)
    accuracy = accuracy_score(ground_truth, predictions) 
    return {
        "accuracy": accuracy,
        "predictions": np.asarray(predictions, dtype=int),
        "ground_truth": np.asarray(ground_truth, dtype=int),
        "previews": previews,
    }


def save_preview(previews: list[Dict], output_path: Path) -> None:
    """
    Salva l'anteprima di 3 immagini, maschere ground truth e predizioni in un file png
    Args:
        previews: lista di 3 dizionari contenenti ciascuno [immagine, maschera ground truth, maschera predetta, path immagine]
        output_path: percorso del file png dove salvare l'anteprima
    """
    if not previews:
        print("[INFO] Nessuna anteprima da salvare.")
        return

    #creaimo i colori da usare nelle immagini
    color_map = colors.ListedColormap(
        ["#9e9e9e", "#ff6f69", "#ffcc5c", "#88d8b0", "#6b5b95", "#2a9d8f"]
    )
    #impostiamo il range di valori che i pixel possono aver per cui assegnare un certo colore
    norm = colors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5], color_map.N)
    #creiamo una griglia di altezza numero prewiews (3) e larghezza 4
    n_rows = len(previews)
    fig, axes = plt.subplots(n_rows, 4, figsize=(15, 4 * n_rows))
   

    for row, sample in enumerate(previews):
        img = sample["image"]

        mask_gt_raw = sample["mask_gt"]
        #sottraggo uno per mappare le classi da [0-6] a [-1-5]
        mask_gt = mask_gt_raw.astype(float) - 1
        #creo una maschera di display per la maschera ground truth
        #imposto a NaN i pixel di background (cosi' che Matplotlib lasci il pixel bianco)
        mask_gt = np.where(mask_gt < -0.5, np.nan, mask_gt)

        #salvo la maschera predetta
        predicted_mask = sample["predicted_mask"].astype(float)
        #creo una copia della maschera predetta e imposto a NaN i pixel di background (cosi' che Matplotlib lasci il pixel bianco)
        #(non devo decrementare di 1 perche' il CRF gia' restituisce una mask con background -1)
        predicted_mask_filtered = predicted_mask.copy()
        predicted_mask_filtered[mask_gt_raw == 0] = np.nan
        #salvo il nome del file dell'immagine
        name = Path(sample["path"]).name
        

        #alla riga i-esima colonna 0 mostro l'immagine originale
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"Immagine\n{name}")
        axes[row, 0].axis("off")
        # alla seconda colonna mostro la maschera ground truth
        axes[row, 1].imshow(mask_gt, cmap=color_map, norm=norm)
        axes[row, 1].set_title("Maschera Ground Truth")
        axes[row, 1].axis("off")

        #alla terza colonna mostro la maschera predetta ma solo i punti che combaciano con la maschera ground truth (senza backgrounf)
        axes[row, 2].imshow(predicted_mask_filtered, cmap=color_map, norm=norm)
        axes[row, 2].set_title("Predizione (solo GT>0)")
        axes[row, 2].axis("off")

        #mostro la maschera predetta dal modello sull'intera immagine
        axes[row, 3].imshow(predicted_mask, cmap=color_map, norm=norm)
        axes[row, 3].set_title("Maschera Predetta (CRF)")
        axes[row, 3].axis("off")

    #creo la legenda dei colori
    handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=color_map(norm(i - 1)), markersize=12) for i in range(len(CLASS_NAMES) + 1)
    ]
    #aggiugno le etichette alla legenda dei colori
    labels = ["Background"] + CLASS_NAMES
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=10)
    #ottimizzo spacing
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    #salvo l'immagine nel perocrso specificato e chiudo la figura
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Anteprima salvata in: {output_path}")


def save_confusion_matrix(cm: np.ndarray, class_names: list[str], output_path: Path, accuracy: float | None = None) -> None:
    """
        Salva la matrice di confusione in un file png
        Args:
            cm: matrice di confusione numpy.    shape -> (num_classi, num_classi)
            class_names: lista dei nomi delle classi
            output_path: percorso del file png dove salvare la matrice di confusione
            accuracy: accuracy complessiva (opzionale) da mostrare nel titolo
    """
    #creo lo spazio grafico maptolib (fig = intera immagine, ax = assi della matrice)
    fig, ax = plt.subplots(figsize=(6, 5))
    #imposto la matrice come una mappa di colore con colore blu
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    #scriviamo la legenda laterale
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    #imposto i tick (le tacche) dell'asse x e y (6, una per ogni classe)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    #imposto i label (i nomi delle classi) dell'asse x e y (rotati di 45 gradi e centrati)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    #etichettiamo gli assi e le immagini
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    #mostro l'accuracy se fornita
    if accuracy is not None:
        ax.set_title(f"Confusion Matrix (pixel) - acc: {accuracy:.4f}")
    else:
        ax.set_title("Confusion Matrix (pixel)")
    #salviamo il valore piu' alto dentro la matrice (quella col blu piu' scuro)
    max_val = cm.max() if cm.size else 0
    #scorriamo ogni quadrato della matrice
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            #se il valore della casella e' maggiore della meta del massimo (sara blu scura) allora scrivi testo in bianco, altrimenti se e' un valore basso (casella chiara) scrivi in nero
            text_color = "white" if value > max_val * 0.5 else "black"
            #scriviamo il valore della casella nel quadrato.
            ax.text(
                j,
                i,
                f"{value:,}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )
    #ricalcoliamo l'intera immagine per far stare tutto
    fig.tight_layout()
    #salviamo l'immagine nel perocrso specificato
    fig.savefig(output_path, dpi=200)
    #chiudo la finestra
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Valutazione modello GPU-optimized con CRF.")
    #leggo da linea di comando il nome del modello da valutare (senza estensioni).
    parser.add_argument(
        "--model_prefix",
        type=str,
        default=MODEL_PREFIX,
        help="Prefisso dei file salvati (senza _classifier.pkl / _feature_extractor.keras).",
    )
    
    args = parser.parse_args()

    model_path = (BASE_DIR / args.model_prefix).resolve()

    print("=== VALUTAZIONE MODELLO GPU ===")
    print(f"Modello: {model_path}")
    print(f"Immagini: {IMAGES_DIR}")
    print(f"Maschere: {MASKS_DIR}")
    print(f"CRF attivato")
    print("=" * 50)
    #creo un istanza del modello
    model = GPUOptimizedCNNSegmentationClassifier()
    #carico il modello addestrato (sia il file keras (CNN + decoder) che il file pkl (classificatore))
    model.load(str(model_path))
    #genero o carico gli split del dataset per ottenere gli ID delle immagini di evaluation 
    # (ovviamente ache se vengono rigenerati gli split, sono gli stessi del training)
    _, eval_ids = prepare_dataset_splits(
        images_dir=str(IMAGES_DIR),
        masks_dir=str(MASKS_DIR),
        split_dir=str(SPLIT_DIR),
        train_ratio=0.8,
        seed=42,
    )
    #carico le immagini e maschere del set di evaluation e le elaboro in scala di grigi se richiesto
    images, masks, image_paths = load_dataset_stateless(
        images_dir=str(IMAGES_DIR),
        masks_dir=str(MASKS_DIR),
        image_size=tuple(model.config.image_size), #usa come dimensione quella salvata nel modello
        use_grayscale=getattr(model.config, "use_grayscale", False), #attivo la conversione in scala di grigi se è presente nel modello
        image_names=eval_ids, #carico solo le immagini di evaluation (ottenuto prima da prepare_dataset_splits)
        return_paths=True, #restituisce anche i percorsi delle immagini (ci servono poi per la generazione dell'immagine di anteprima)
    )
    dataset = {
        "images": images,
        "masks": masks,
        "image_paths": image_paths,
    }
    print(f"Immagini valutate (holdout): {len(dataset['images'])}")
    #valuto il modello sul set di evaluation
    results = evaluate_gpu_model(model, dataset)

    #prendo e stampo l'accuracy del modello sul test set
    accuracy = results["accuracy"]
    print(f"\nAccuracy pixel (post-CRF): {accuracy:.4f}")

    #calcolo la matrice di confusione
    cm = confusion_matrix(
        y_true=results["ground_truth"], 
        y_pred=results["predictions"],
        labels=list(range(len(CLASS_NAMES))) #per assicurarsi che la matrice rappresenti ogni classe (anche se le mashcere fornite non le avessero)
    )
    #salvo la matrice di confusione in un file png
    save_confusion_matrix(
        cm=cm,
        class_names=CLASS_NAMES,
        output_path=CONFUSION_PATH,
        accuracy=accuracy,
    )
    print(f"Matrice di confusione salvata in: {CONFUSION_PATH}")

    #salvo l'anteprima in un file png
    save_preview(results["previews"], PREVIEW_PATH)


if __name__ == "__main__":
    main()
