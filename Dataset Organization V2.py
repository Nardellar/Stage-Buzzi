import sys

import numpy as np
import tensorflow as tf
import pandas as pd
import keras
from matplotlib import pyplot as plt

#Inizalizzo i path utili!
base_path = '/home/nardellar/Scaricati/stage'
#
df = pd.read_csv("esperimenti.csv")


# Carichiamo un dataset di immagini direttamente dal percorso base, giusto per eventuali addestramenti
dataset = keras.utils.image_dataset_from_directory(
    base_path,
    labels="inferred",  # Legge le cartelle come classi
    label_mode="int",
    image_size=(108, 192),
    batch_size=32,
    verbose=True
)

print("📂 Classi trovate:", dataset.class_names)




def map_labels_to_attribute(ds, df, attribute_name):

    attribute_name = attribute_name.strip().lower()  # Rimuove spazi e uniforma a minuscolo

    images_list = []
    attribute_vals_list = []

    # Creiamo una mappa per collegare ID esperimenti con i valori dell'attributo
    attribute_map = df.set_index("ID")[attribute_name].to_dict()

    for image, label in ds.unbatch():
        # Troviamo il nome dell'esperimento (il dataset assegna interi come classi)
        class_name = ds.class_names[label.numpy()]  # Nome dell'esperimento (es. "EXP01")

        # Otteniamo il valore dell'attributo dal CSV
        val = attribute_map.get(class_name, None)  # None se non esiste nel CSV

        # Se il valore è valido (non None e non NaN), lo includiamo
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            images_list.append(image.numpy())
            attribute_vals_list.append(val)

    if not images_list:  # Se non ci sono dati validi, evitiamo errori
        print("⚠️ Nessuna immagine valida trovata con l'attributo selezionato.")
        return None, None

    # Creiamo un nuovo dataset con (immagine, valore_attributo)
    ds_final = tf.data.Dataset.from_tensor_slices((images_list, attribute_vals_list)).batch(32)

    return ds_final, np.array(attribute_vals_list)






def show_images(ds, max_images=32):

    # Preleviamo il primo batch dal dataset
    for images, labels in ds.take(1):
        # sizes[0] = dimensione batch
        batch_size = images.shape[0]
        # Numero reale di immagini da mostrare
        num_images = min(batch_size, max_images)

        plt.figure(figsize=(10, 10))
        for i in range(num_images):
            ax = plt.subplot(4, 8, i + 1)
            # Converte l'immagine in uint8 (se necessario)
            plt.imshow(images[i].numpy().astype("uint8"))

            # Qui mostriamo la "label" così come appare (potrebbe essere un numero, un array, ecc.)
            lbl = labels[i].numpy()
            plt.title(str(lbl))
            plt.axis("off")

        plt.tight_layout()
        plt.show()

        break





attributo = input("Inserisci l'attributo da ricercare: ").strip()
if not attributo:
    print("❌ Errore: non hai inserito nessun attributo.")
    sys.exit()
if attributo not in df.columns.str.lower():
    print(f"❌ Errore: l'attributo '{attributo}' non esiste nel dataset.")
    sys.exit()
train_dataset, values_array = map_labels_to_attribute(dataset, df, attributo)
print(f"Nel dataset ci sono {len(list(train_dataset.unbatch()))} immagini classificate per {attributo.strip().lower()}")


if train_dataset is not None:
    show_images(train_dataset)
else:
    print("❌ Nessuna immagine con valore valido")

