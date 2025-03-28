# === IMPORT LIBRERIE ===
import os  # Per operazioni sul filesystem (esistenza file/cartelle)
import sys  # Per terminare il programma in caso di errore
import zipfile  # Per estrarre file ZIP

import gdown  # Per scaricare file da Google Drive
import keras  # Per creare dataset di immagini
import numpy as np  # Per operazioni numeriche e array
import pandas as pd  # Per gestire e leggere file CSV
import tensorflow as tf  # Per pipeline di immagini e modelli TensorFlow
from matplotlib import pyplot as plt  # Per visualizzare immagini

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


# === FUNZIONE PER MAPPARE IMMAGINI CON UN ATTRIBUTO DAL CSV ===
def map_labels_to_attribute(ds, df, attribute_name):
    attribute_name = attribute_name.strip().lower()  # Normalizza il nome dell'attributo

    images_list = []  # Lista per contenere immagini valide
    attribute_vals_list = []  # Lista per contenere i relativi valori dell'attributo

    # Crea un dizionario ID -> valore attributo
    attribute_map = df.set_index("ID")[attribute_name].to_dict()

    for image, label in ds.unbatch():  # Scorre ogni immagine (dataset non batchato)
        class_name = ds.class_names[
            label.numpy()
        ]  # Recupera il nome della classe dalla label
        val = attribute_map.get(
            class_name, None
        )  # Cerca il valore dell'attributo nel CSV

        # Se è un valore valido (non None o NaN), lo aggiunge alla lista
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            images_list.append(image.numpy())
            attribute_vals_list.append(val)

    # Se nessuna immagine è valida, stampa un messaggio di errore
    if not images_list:
        print("⚠️ Nessuna immagine valida trovata con l'attributo selezionato.")
        return None, None

    # Converte le liste in un dataset TensorFlow batchato
    ds_final = tf.data.Dataset.from_tensor_slices(
        (images_list, attribute_vals_list)
    ).batch(32)

    return ds_final, np.array(attribute_vals_list)  # Ritorna dataset e array valori


# === FUNZIONE PER VISUALIZZARE UN BATCH DI IMMAGINI ===
def show_images(ds, max_images=32):
    for images, labels in ds.take(1):  # Prende un solo batch
        num_images = min(
            images.shape[0], max_images
        )  # Mostra al massimo max_images immagini
        plt.figure(figsize=(10, 10))  # Imposta la figura

        for i in range(num_images):  # Crea una griglia 4x8
            ax = plt.subplot(4, 8, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))  # Mostra l'immagine
            plt.title(str(labels[i].numpy()))  # Mostra il valore dell'attributo
            plt.axis("off")  # Nasconde gli assi

        plt.tight_layout()  # Ottimizza layout
        plt.show()  # Mostra il plot
        break  # Solo il primo batch


# === BLOCCO PRINCIPALE ===
if __name__ == "__main__":
    download_and_extract()  # Scarica ed estrae il dataset se non già presente

    # Controlla che il file CSV esista
    if not os.path.exists(CSV_FILE):
        print(f"❌ Errore: File CSV '{CSV_FILE}' non trovato.")
        sys.exit(1)

    data_frame = pd.read_csv(CSV_FILE)  # Legge il CSV e salva nel dataframe

    # Carica le immagini come dataset TensorFlow
    dataset = keras.utils.image_dataset_from_directory(
        DATASET_DIR,  # la cartella Esperimenti
        labels="inferred",  # Inferisce le etichette dal nome delle cartelle
        label_mode="int",  # Le etichette sono numeri interi
        image_size=(108, 192),  # Ridimensiona le immagini
        batch_size=32,
        verbose=True,  # Stampa informazioni (quanti file ha trovato, ecc.)
    )

    print("\n📁 Classi trovate:", dataset.class_names)  # Stampa le classi/ID trovate

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
    train_dataset, values_array = map_labels_to_attribute(
        dataset, data_frame, attributo
    )

    if train_dataset is not None:
        print(
            f"\n📊 Immagini trovate: {len(list(train_dataset.unbatch()))}"
        )  # Stampa numero immagini valide
        show_images(train_dataset)  # Mostra le immagini
    else:
        print("❌ Nessuna immagine con valore valido.")
