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




# === CONFIGURAZIONE ===
DATASET_DIR = "Esperimenti"  # Cartella in cui verranno estratte le immagini
CSV_FILE = "esperimenti.csv"  # Nome del file CSV contenente gli attributi
ZIP_NAME = "esperimenti.zip"  # Nome del file zip da scaricare da Google Drive
GDRIVE_ID = "1JxuABW728R8n_nz2VONDSOIiWzPFO64a"  # ID pubblico del file su Google Drive


# === FUNZIONE PER SCARICARE ED ESTRARRE IL DATASET ===
def download_and_extract():
    if not os.path.exists(DATASET_DIR):  # Se la cartella non esiste, scarica il dataset
        print("⬇️ Scaricamento del dataset da Google Drive...")
        url = f"https://drive.google.com/uc?id={GDRIVE_ID}"  # Costruisce l'URL
        gdown.download(url, ZIP_NAME, quiet=False)  # Scarica il file ZIP

        print("📦 Estrazione in corso...")
        with zipfile.ZipFile(ZIP_NAME, "r") as zip_ref:  # Apre il file ZIP
            zip_ref.extractall()  # Estrae tutto
        os.remove(ZIP_NAME)  # Elimina lo ZIP dopo l'estrazione
        print("✅ Dataset pronto!")



def map_labels_to_attribute(ds, df, attribute_name):


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
        labels = tf.numpy_function(
            lambda l: np.array([
                mapping.get(round(v), -1)  # Arrottonda al numero intero più vicino
                for v in l
            ], dtype=np.int32),
            [labels],
            tf.int32  # Tipo corretto
        )
        labels.set_shape([None])  # Imposta la forma attesa
        return images, labels
    return map_fn




# === BLOCCO PRINCIPALE ===
def get_dataset(attributo):

    download_and_extract()  # Scarica ed estrae il dataset se non già presente

    # Controlla che il file CSV esista
    if not os.path.exists(CSV_FILE):
        print(f"❌ Errore: File CSV '{CSV_FILE}' non trovato.")
        sys.exit(1)

    data_frame = pd.read_csv(CSV_FILE)  # Legge il CSV e salva nel dataframe

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

    # Mappa le immagini al valore dell'attributo scelto
    train_dataset = map_labels_to_attribute(
        train_dataset, data_frame, attributo
    )

    validation_dataset = map_labels_to_attribute(
        validation_dataset, data_frame, attributo
    )

    train_dataset, validation_dataset = standardize_dataset(train_dataset, validation_dataset)

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





    print(train_dataset)




    return train_dataset, validation_dataset
