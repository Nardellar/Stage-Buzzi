import numpy as np
import tensorflow as tf
import pandas as pd
import keras
from keras.src.utils import load_img, img_to_array
from matplotlib import pyplot as plt

# Percorso locale dove risiedono le immagini
base_path = '/Users/matteobarbieri/Documents/Uni/Stage/Allexp10x'

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
    """
    Sostituisce le etichette del dataset con il valore di un attributo dal DataFrame CSV,
    escludendo le immagini con valore NULL o NaN.

    ds: tf.data.Dataset creato da image_dataset_from_directory()
    df: DataFrame contenente gli esperimenti e i loro attributi
    attribute_name: stringa, nome dell'attributo da usare come nuova etichetta

    Ritorna:
    - Nuovo dataset con (immagine, valore_attributo) solo per dati validi
    - Array con i valori validi dell'attributo
    """

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




def show_images(ds, max_images=9):
    """
    Mostra alcune immagini da un tf.data.Dataset.
    Assumiamo che ds fornisca batch di (immagini, label).

    ds         : tf.data.Dataset da cui leggere (immagini, label).
    max_images : numero massimo di immagini da visualizzare.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    # Preleviamo il primo batch dal dataset
    for images, labels in ds.take(1):
        # sizes[0] = dimensione batch
        batch_size = images.shape[0]
        # Numero reale di immagini da mostrare
        num_images = min(batch_size, max_images)

        plt.figure(figsize=(10, 10))
        for i in range(num_images):
            ax = plt.subplot(3, 3, i + 1)
            # Converte l'immagine in uint8 (se necessario)
            plt.imshow(images[i].numpy().astype("uint8"))

            # Label associata all'immagine: dipende dal tipo di labels (scalar, array, ecc.)
            # Qui mostriamo la "label" così come appare (potrebbe essere un numero, un array, ecc.)
            lbl = labels[i].numpy()
            plt.title(str(lbl))
            plt.axis("off")

        plt.tight_layout()
        plt.show()

        # Dopo aver mostrato il primo batch, usciamo dal ciclo
        break





columns_no_id = df.columns.tolist()
train_dataset, values_array = map_labels_to_attribute(dataset, df, "Rampa")

if train_dataset is not None:
    show_images(train_dataset)
else:
    print("❌ Nessuna immagine con valore valido per 'Rampa'.")

