import tensorflow as tf
import pandas as pd
import os
import keras
from matplotlib import pyplot as plt
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

base_path = '/home/nardellar/Scaricati/stage/'

dataset = keras.utils.image_dataset_from_directory(
    base_path,
    labels="inferred",       # Legge le cartelle come classi
    label_mode="int",
    # class_names=None,
    image_size=(224, 224),
    batch_size=32,
    # shuffle=False,
    verbose=True
    # color_mode = "grayscale"
)

print("📂 Classi trovate:", dataset.class_names)


class_attributes = {
    "Esperimenti": [
        {"ID": "EXP01", "Temperatura": 1400, "Tempo": None, "Rampa": None, "Raffreddamento": 0},
      #  {"ID": "EXP02", "Temperatura": 1400, "Tempo": None, "Rampa": None, "Raffreddamento": 1},
      #  {"ID": "EXP03", "Temperatura": 1400, "Tempo": None, "Rampa": None, "Raffreddamento": None},
      #  {"ID": "EXP04", "Temperatura": 1400, "Tempo": None, "Rampa": None, "Raffreddamento": None},
      #  {"ID": "EXP05", "Temperatura": 1500, "Tempo": 45, "Rampa": 40, "Raffreddamento": 1},
      #  {"ID": "EXP06", "Temperatura": 1300, "Tempo": None, "Rampa": None, "Raffreddamento": None},
      #  {"ID": "EXP07", "Temperatura": None, "Tempo": None, "Rampa": None, "Raffreddamento": 1},
      #  {"ID": "EXP08", "Temperatura": None, "Tempo": None, "Rampa": None, "Raffreddamento": 0},
      #  {"ID": "EXP09", "Temperatura": 1300, "Tempo": None, "Rampa": None, "Raffreddamento": None},
      #  {"ID": "EXP10", "Temperatura": 1500, "Tempo": 45, "Rampa": 10, "Raffreddamento": 0},
      #  {"ID": "EXP11", "Temperatura": 1500, "Tempo": 15, "Rampa": 40, "Raffreddamento": 0},
      #  {"ID": "EXP12", "Temperatura": 1300, "Tempo": None, "Rampa": None, "Raffreddamento": None},
      #  {"ID": "EXP13", "Temperatura": 1500, "Tempo": 15, "Rampa": 10, "Raffreddamento": 1},
      #  {"ID": "EXP14", "Temperatura": 1300, "Tempo": None, "Rampa": None, "Raffreddamento": None}
    ]
}



df = pd.DataFrame(class_attributes["Esperimenti"])
df.to_csv("esperimenti.csv", index=False)
class_names = dataset.class_names


for images, labels in dataset.take(1):  # Prendi il primo batch
    plt.figure(figsize=(16, 9))

    for i in range(30):
        ax = plt.subplot(3, 10, i + 1)
        #DA CAPIRE QUESTA RIGA
        plt.imshow(images[i].numpy().astype("uint8"))

        # Converti l'etichetta numerica nel nome della classe
        #DA CAPIRE QUESTA RIGA
        class_label = class_names[labels[i].numpy()]
        plt.title(f"{class_label}")

        plt.axis("off")

    plt.show()


import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tensorflow.keras.utils import load_img, img_to_array

base_path = '/home/nardellar/Scaricati/stage/'

def find_image_by_attribute(attribute, target_size=(224, 224), batch_size=32):
    df = pd.read_csv('esperimenti.csv')

    # Cerca l'attributo indipendentemente dalle maiuscole
    attribute_lower = attribute.lower()
    col_match = next((col for col in df.columns if col.lower() == attribute_lower), None)

    if col_match is None:
        print(f"Attributo '{attribute}' non trovato. Colonne disponibili: {list(df.columns)}")
        return None

    # Filtra gli esperimenti che hanno quell'attributo valorizzato
    selected_exp = df[df[col_match].notnull()]["ID"].tolist()
    selected_df = df[df["ID"].isin(selected_exp)]

    image_paths = []
    labels = []

    # Scansiona le cartelle ed associa immagini ai dati sperimentali
    for _, row in selected_df.iterrows():
        exp_id = row["ID"]
        exp_folder = os.path.join(base_path, exp_id)

        if not os.path.isdir(exp_folder):
            print(f"⚠️ Cartella {exp_folder} non trovata, saltata.")
            continue

        image_files = [
            os.path.join(exp_folder, img) for img in os.listdir(exp_folder)
            if img.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        for img_path in image_files:
            image_paths.append(img_path)
            labels.append(np.nan_to_num(row.drop("ID").values.astype(np.float32), nan=0.0))  # Gestisce NaN e converte in float32

    # Se non ci sono immagini trovate
    if not image_paths:
        print("❌ Nessuna immagine trovata per questo attributo.")
        return None

    # Funzione per caricare e preprocessare le immagini
    def load_and_preprocess(image_path, label):
        image_path = image_path.numpy().decode("utf-8")  # Converti il tensor in stringa

        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0  # Normalizza tra 0 e 1

        return img_array, label

    def preprocess(tensor):
        tensor = tf.where(tf.math.is_nan(tensor), tf.zeros_like(tensor), tensor)
        return tf.cast(tensor, tf.float32)


    # Creazione dataset TensorFlow
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(lambda image_path, label: (tf.py_function(load_and_preprocess, [image_path, label], [tf.float32, tf.float32])))

    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset





train_dataset = find_image_by_attribute("temperatura")

if train_dataset:
    for images, labels in train_dataset.take(1):  # Visualizza un batch
        print("Batch di immagini:", images.shape)
        print("Batch di etichette:", labels.shape)


for images, labels in train_dataset.take(1):
    plt.imshow(images[0].numpy())  # Mostra la prima immagine
    plt.show()
