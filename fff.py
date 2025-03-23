import os
import sys
import zipfile

import gdown
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

# === CONFIG ===
DATASET_DIR = "Esperimenti"
CSV_FILE = "esperimenti.csv"
ZIP_NAME = "esperimenti.zip"
GDRIVE_ID = "1JxuABW728R8n_nz2VONDSOIiWzPFO64a"  # ← Cambia qui


# === UTILITY ===
def download_and_extract():
    if not os.path.exists(DATASET_DIR):
        print("⬇️ Scaricamento del dataset da Google Drive...")
        url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
        gdown.download(url, ZIP_NAME, quiet=False)

        print("📦 Estrazione in corso...")
        with zipfile.ZipFile(ZIP_NAME, "r") as zip_ref:
            zip_ref.extractall()
        os.remove(ZIP_NAME)
        print("✅ Dataset pronto!")


def map_labels_to_attribute(ds, df, attribute_name):
    attribute_name = attribute_name.strip().lower()
    images_list = []
    attribute_vals_list = []
    attribute_map = df.set_index("ID")[attribute_name].to_dict()

    for image, label in ds.unbatch():
        class_name = ds.class_names[label.numpy()]
        val = attribute_map.get(class_name, None)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            images_list.append(image.numpy())
            attribute_vals_list.append(val)

    if not images_list:
        print("⚠️ Nessuna immagine valida trovata con l'attributo selezionato.")
        return None, None

    ds_final = tf.data.Dataset.from_tensor_slices(
        (images_list, attribute_vals_list)
    ).batch(32)

    return ds_final, np.array(attribute_vals_list)


def show_images(ds, max_images=32):
    for images, labels in ds.take(1):
        num_images = min(images.shape[0], max_images)
        plt.figure(figsize=(10, 10))
        for i in range(num_images):
            ax = plt.subplot(4, 8, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(str(labels[i].numpy()))
            plt.axis("off")
        plt.tight_layout()
        plt.show()
        break


# === MAIN ===
if __name__ == "__main__":
    download_and_extract()

    if not os.path.exists(CSV_FILE):
        print(f"❌ Errore: File CSV '{CSV_FILE}' non trovato.")
        sys.exit(1)

    data_frame = pd.read_csv(CSV_FILE)

    dataset = keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        labels="inferred",
        label_mode="int",
        image_size=(108, 192),
        batch_size=32,
        verbose=True,
    )

    print("\n📁 Classi trovate:", dataset.class_names)

    attributo = input("🔎 Inserisci l'attributo da ricercare: ").strip()
    if not attributo:
        print("❌ Errore: attributo non inserito.")
        sys.exit()

    if attributo.lower() not in data_frame.columns.str.lower():
        print(f"❌ Errore: l'attributo '{attributo}' non esiste nel CSV.")
        sys.exit()

    train_dataset, values_array = map_labels_to_attribute(
        dataset, data_frame, attributo
    )

    if train_dataset is not None:
        print(f"\n📊 Immagini trovate: {len(list(train_dataset.unbatch()))}")
        show_images(train_dataset)
    else:
        print("❌ Nessuna immagine con valore valido.")
