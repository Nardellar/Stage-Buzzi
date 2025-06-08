# === IMPORT LIBRERIE ===
import os  # Per operazioni sul filesystem
import sys  # Per terminare il programma in caso di errore
import zipfile  # Per estrarre file ZIP
import gdown  # Per scaricare file da Google Drive
import numpy as np  # Per operazioni numeriche e array
import pandas as pd  # Per leggere file CSV
import tensorflow as tf  # Per pipeline di immagini e modelli TensorFlow
from matplotlib import pyplot as plt  # Per visualizzare immagini
from tensorflow.keras import layers, models
from pathlib import Path
from . import csv_config



# === CONFIGURAZIONE ===
# Percorso base relativo a questo file (cioè la root del progetto)
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "Esperimenti"  # Cartella in cui verranno estratte le immagini
CSV_FILE = BASE_DIR / "esperimenti.csv"  # Nome del file CSV contenente gli attributi
ZIP_NAME = BASE_DIR / "esperimenti.zip"  # Nome del file zip da scaricare da Google Drive
GDRIVE_ID = "1JxuABW728R8n_nz2VONDSOIiWzPFO64a"  # ID pubblico del file su Google Drive


# === FUNZIONE PER SCARICARE ED ESTRARRE IL DATASET ===
def download_and_extract():
    if not DATASET_DIR.exists():  # Se la cartella non esiste, scarica il dataset
        print("⬇️ Scaricamento del dataset da Google Drive...")
        url = f"https://drive.google.com/uc?id={GDRIVE_ID}"  # Costruisce l'URL
        gdown.download(url, str(ZIP_NAME), quiet=False)  # Scarica il file ZIP

        print("📦 Estrazione in corso...")
        with zipfile.ZipFile(ZIP_NAME, "r") as zip_ref:  # Apre il file ZIP
            zip_ref.extractall(path=BASE_DIR)  # Estrae tutto nella cartella base
        os.remove(ZIP_NAME)  # Elimina lo ZIP dopo l'estrazione
        print("✅ Dataset pronto!")



def map_labels_to_attribute(ds, df, attribute_name):
    import numpy as np
    import tensorflow as tf

    attribute_name = attribute_name.strip().lower()  # Normalizza il nome dell'attributo

    images_list = []
    attribute_vals_list = []

    # Crea un dizionario ID -> valore dell'attributo in base al CSV
    attribute_map = df.set_index("ID")[attribute_name].to_dict()

    # "ds.unbatch()" consente di ciclare immagine per immagine
    for image, label in ds.unbatch():
        # Recupera il nome effettivo della classe (usando ds.class_names)
        class_name = ds.class_names[label.numpy()]
        # Cerca il valore dal CSV
        val = attribute_map.get(class_name, None)

        # Se il valore è valido, salva l'immagine e il label corrispondente
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            images_list.append(image.numpy())
            attribute_vals_list.append(val)

    # Se nessuna immagine risulta valida, restituisci None
    if not images_list:
        print("⚠️ Nessuna immagine valida trovata con l'attributo selezionato.")
        return None

    # Crea il dataset TensorFlow con le tuple (immagine, valore_attributo)
    ds_final = tf.data.Dataset.from_tensor_slices(
        (images_list, attribute_vals_list)
    ).batch(32)

    return ds_final




def standardize_dataset(train_dataset, validation_dataset=None):
    images_list = []

    # Itera solo sui batch del training dataset
    for images, _ in train_dataset:
        # Converte i tensori in NumPy per facilitarne l'uso
        images_list.append(images.numpy())

    # Concatena tutti i batch in un unico array
    # Avremo una forma (totale_immagini, altezza, larghezza, canali)
    all_train_images = np.concatenate(images_list, axis=0)

    # Calcoliamo media e std sui pixel del training set
    mean = np.mean(all_train_images, axis=(0, 1, 2))
    std = np.std(all_train_images, axis=(0, 1, 2))

    # Funzione di standardizzazione che possiamo riutilizzare
    def standardize_batch(images, labels):
        return (tf.cast(images, tf.float32) - mean) / std, labels

    # Standardizza il training dataset
    standardized_train = train_dataset.map(standardize_batch)

    # Se è stato passato anche il validation dataset, standardizza quello usando
    # gli stessi valori di media e std calcolati sul training
    if validation_dataset is not None:
        standardized_validation = validation_dataset.map(standardize_batch)
        return standardized_train, standardized_validation

    return standardized_train



def normalize_dataset(train_dataset, validation_dataset=None):
    images_list = []

    # Raggruppa solo le immagini del training set per calcolare min e max
    for images, _ in train_dataset:
        images_list.append(images.numpy())

    all_train_images = np.concatenate(images_list, axis=0)

    # Calcoliamo il min e il max solo sulle immagini di training
    min_val = np.min(all_train_images, axis=(0, 1, 2))
    max_val = np.max(all_train_images, axis=(0, 1, 2))

    # Funzione di normalizzazione che possiamo riutilizzare
    def normalize_batch(images, labels):
        return (tf.cast(images, tf.float32) - min_val) / (max_val - min_val), labels

    # Normalizza il training dataset
    normalized_train = train_dataset.map(normalize_batch)

    # Se è stato passato anche il validation dataset, normalizza quello usando
    # gli stessi valori min e max calcolati sul training
    if validation_dataset is not None:
        normalized_validation = validation_dataset.map(normalize_batch)
        return normalized_train, normalized_validation

    return normalized_train



# === FUNZIONE PER VISUALIZZARE UN BATCH DI IMMAGINI ===
def show_images(ds, max_images=32):
    for images, labels in ds.take(1):  # Prende un solo batch
        num_images = min(
            images.shape[0], max_images
        )  # Mostra al massimo max_images immagini
        plt.figure(figsize=(10, 10))  # Imposta la figura

        for i in range(num_images):  # Crea una griglia 4x8
            ax = plt.subplot(4, 8, i + 1)
            plt.imshow(images[i].numpy())  # Mostra l'immagine
            plt.title(str(labels[i].numpy()))  # Mostra il valore dell'attributo
            plt.axis("off")  # Nasconde gli assi

        plt.tight_layout()  # Ottimizza layout
        plt.show()  # Mostra il plot
        break  # Solo il primo batch





def remap_labels(mapping):
    def map_fn(images, labels):
        def map_label(l):
            result = []
            for v in l:
                if isinstance(v, bytes):
                    key = v.decode("utf-8")
                elif isinstance(v, (str, np.str_)):
                    key = v
                elif isinstance(v, (np.integer, int)):
                    # If the numeric label is present as a key in the mapping
                    # dictionary treat it directly as the key, otherwise fall
                    # back to interpreting it as an index of the mapping keys.
                    if v in mapping:
                        key = int(v)
                    else:
                        class_names = list(mapping.keys())
                        if 0 <= v < len(class_names):
                            key = class_names[v]
                        else:
                            key = None
                else:
                    key = None
                result.append(mapping.get(key, -1))
            return np.array(result, dtype=np.int32)

        labels = tf.numpy_function(map_label, [labels], tf.int32)
        labels.set_shape([None])
        return images, labels
    return map_fn


from collections import defaultdict
import random

def balance_dataset(dataset):
    """
    Bilancia le classi del dataset via oversampling.
    """
    # Raccoglie le immagini per classe
    class_data = defaultdict(list)

    for image, label in dataset.unbatch():
        class_index = int(label.numpy())
        class_data[class_index].append((image, label))

    # Trova la classe con più campioni
    max_count = max(len(samples) for samples in class_data.values())

    balanced_samples = []

    for class_index, samples in class_data.items():
        # Ripeti i sample finché non raggiungi max_count
        repeated = samples.copy()
        while len(repeated) < max_count:
            repeated.extend(random.sample(samples, min(max_count - len(repeated), len(samples))))
        balanced_samples.extend(repeated)

    # Shuffle finale
    random.shuffle(balanced_samples)

    # Ricostruisci un nuovo dataset bilanciato
    images, labels = zip(*balanced_samples)
    images = tf.stack(images)
    labels = tf.convert_to_tensor(labels)
    return tf.data.Dataset.from_tensor_slices((images, labels)).batch(32)




# === BLOCCO PRINCIPALE ===
#Prepara e restituisce il train dataset ed il validation dataset
def get_dataset(attributo):

    download_and_extract()  # Scarica ed estrae le immagini ESPERIMENTI se non già presenti

    # Controlla che il file CSV esista altrimenti lo genera
    if not CSV_FILE.exists():
        print(f"⚠️ File CSV '{CSV_FILE}' non trovato. Creazione in corso...")
        csv_config.create_csv(CSV_FILE)

    #Legge il CSV e salva nel dataframe pandas
    #ora conterrà gli ID degli esperimenti (es:EXP01,EXP02...) e tutti i loro attributi (es.temperatura,...)
    data_frame = pd.read_csv(CSV_FILE)

    image_size = (112, 112)

    # Carica le immagini come dataset TensorFlow
    train_dataset, validation_dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,  # la cartella Esperimenti
        labels="inferred",  # Inferisce le etichette dal nome delle cartelle
        label_mode="int",  # Le etichette sono numeri interi
        image_size=image_size,  # Ridimensiona le immagini e ritaglia i lati per togliere l'etichetta
        batch_size=32,
        seed=42,  # deve essere uguale al precedente
        crop_to_aspect_ratio = True,
        validation_split = 0.2,  # Percentuale di split per la validation
        subset = "both",  # Specifica che questo dataset è la sezione 'training'
    )


    print("\n📁 Classi trovate per il training:", train_dataset.class_names)  # Stampa le classi/ID trovate
    print("📁 Classi trovate per la validazione:", validation_dataset.class_names)  # Stampa le classi/ID trovate


    if attributo == "id":

        train_dataset = balance_dataset(train_dataset)
        train_dataset, validation_dataset = standardize_dataset(train_dataset, validation_dataset)

        return train_dataset, validation_dataset

    #richiesta input utente
    if not attributo:
        attributo = input(
            "🔎 Inserisci l'attributo da ricercare: "
        ).strip()  # Richiede input utente
    if not attributo:
        print("❌ Errore: attributo non inserito.")
        sys.exit()

    # Verifica che l'attributo esista nel CSV
    if attributo.lower() not in data_frame.columns.str.lower():
        print(f"❌ Errore: l'attributo '{attributo}' non esiste nel CSV.")
        sys.exit()

    #per ogni immagine si recupera il suo ID (EXP01,...) da train_dataset stesso e gli si associa il valore dell'attributo scelto da utente (grazie al dataframe creato all'inizio che associa ID con attributi)
    train_dataset = map_labels_to_attribute(
        train_dataset, data_frame, attributo
    )


    if train_dataset is not None:
        print("🔁 Bilanciamento del dataset in corso...")
        train_dataset = balance_dataset(train_dataset)

    # Estrai le classi presenti nel dataset bilanciato
    from collections import Counter

    # Conta il numero di immagini per ciascuna classe nel training set
    label_counter = Counter()
    for _, labels in train_dataset.unbatch():
        label = int(labels.numpy())
        label_counter[label] += 1

    print("\n📊 Numero di immagini per classe nel training set (bilanciato):")
    for label, count in sorted(label_counter.items()):
        print(f"  Classe {label}: {count} immagini")


    validation_dataset = map_labels_to_attribute(
        validation_dataset, data_frame, attributo
    )


    #viene calcolata la media e la deviazione standard delle immagini del training set
    train_dataset, validation_dataset = standardize_dataset(train_dataset, validation_dataset)


    #questo blocco visualizza le immagini
    '''
    if train_dataset is not None:
        print(
            f"\n📊 Immagini trovate: {len(list(train_dataset.unbatch()))}"
        )  # Stampa numero immagini valide
        show_images(train_dataset)  # Mostra le immagini
    else:
        print("❌ Nessuna immagine con valore valido.")

    if validation_dataset is not None:
        print(
            f"\n📊 Immagini trovate: {len(list(validation_dataset.unbatch()))}"
        )  # Stampa numero immagini valide
        show_images(validation_dataset)  # Mostra le immagini
    else:
        print("❌ Nessuna immagine con valore valido.")
    '''



    return train_dataset, validation_dataset
