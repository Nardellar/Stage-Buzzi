import sys  # Importa il modulo sys per poter terminare il programma in caso di errore

import numpy as np  # Importa NumPy per operazioni matematiche e gestione di array
import tensorflow as tf  # Importa TensorFlow per la gestione del dataset di immagini
import pandas as pd  # Importa Pandas per la gestione dei dati in formato tabellare (CSV)
import keras  # Importa Keras per la gestione del dataset
from matplotlib import pyplot as plt  # Importa Matplotlib per la visualizzazione delle immagini

# Inizializza il percorso base dove sono memorizzate le immagini
base_path = 'Esperimenti'

# Carica il file CSV che contiene i metadati degli esperimenti in un DataFrame Pandas
df = pd.read_csv("esperimenti.csv")

# Carichiamo un dataset di immagini direttamente dalla cartella specificata
dataset = keras.utils.image_dataset_from_directory(
    base_path,
    labels="inferred",  # Assegna automaticamente le etichette in base alle sottocartelle
    label_mode="int",  # Le etichette saranno interi corrispondenti alle classi
    image_size=(108, 192),  # Dimensione a cui vengono ridimensionate le immagini
    batch_size=32,  # Numero di immagini per batch
    verbose=True  # Stampa informazioni sul caricamento
)

# Stampa le classi trovate nel dataset
print("\U0001F4C2 Classi trovate:", dataset.class_names)


def map_labels_to_attribute(ds, df, attribute_name):
    """
    Associa le immagini del dataset con i valori di un attributo specificato.
    """
    attribute_name = attribute_name.strip().lower()  # Pulisce il nome dell'attributo rimuovendo spazi e trasformandolo in minuscolo

    images_list = []  # Lista per salvare le immagini valide
    attribute_vals_list = []  # Lista per salvare i valori corrispondenti dell'attributo

    # Crea un dizionario per associare gli ID esperimenti ai valori dell'attributo richiesto
    attribute_map = df.set_index("ID")[attribute_name].to_dict()

    # Scorre il dataset immagine per immagine
    for image, label in ds.unbatch():
        class_name = ds.class_names[label.numpy()]  # Ottiene il nome della classe in base all'etichetta numerica
        val = attribute_map.get(class_name, None)  # Recupera il valore dell'attributo dal CSV

        # Se il valore è valido (non None e non NaN), aggiunge immagine e valore alla lista
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            images_list.append(image.numpy())
            attribute_vals_list.append(val)

    # Se non ci sono immagini valide, stampa un avviso e restituisce None
    if not images_list:
        print("⚠️ Nessuna immagine valida trovata con l'attributo selezionato.")
        return None, None

    # Crea un dataset TensorFlow con coppie (immagine, valore attributo)
    ds_final = tf.data.Dataset.from_tensor_slices((images_list, attribute_vals_list)).batch(32)

    return ds_final, np.array(attribute_vals_list)  # Restituisce il dataset e un array di valori


def show_images(ds, max_images=32):
    """
    Mostra un massimo di `max_images` immagini da un dataset.
    """
    for images, labels in ds.take(1):  # Prende il primo batch dal dataset
        batch_size = images.shape[0]  # Ottiene la dimensione del batch
        num_images = min(batch_size, max_images)  # Determina il numero massimo di immagini da mostrare

        plt.figure(figsize=(10, 10))  # Imposta la dimensione della figura
        for i in range(num_images):
            ax = plt.subplot(4, 8, i + 1)  # Crea un subplot (4 righe, 8 colonne)
            plt.imshow(images[i].numpy().astype("uint8"))  # Mostra l'immagine
            plt.title(str(labels[i].numpy()))  # Imposta il titolo con l'etichetta
            plt.axis("off")  # Nasconde gli assi

        plt.tight_layout()  # Ottimizza la disposizione delle immagini
        plt.show()  # Mostra le immagini
        break  # Esce dal ciclo dopo aver mostrato il primo batch

# Chiede all'utente di inserire l'attributo e lo pulisce da spazi iniziali e finali
attributo = input("Inserisci l'attributo da ricercare: ").strip()

# Se l'utente non ha inserito nulla, stampa un errore e termina il programma
if not attributo:
    print("❌ Errore: non hai inserito nessun attributo.")
    sys.exit()

# Controlla se l'attributo esiste nelle colonne del DataFrame (in minuscolo per uniformità)
if attributo not in df.columns.str.lower():
    print(f"❌ Errore: l'attributo '{attributo}' non esiste nel dataset.")
    sys.exit()

# Mappa le immagini con i valori dell'attributo scelto
train_dataset, values_array = map_labels_to_attribute(dataset, df, attributo)

# Stampa il numero di immagini classificate per l'attributo selezionato
print(f"Nel dataset ci sono {len(list(train_dataset.unbatch()))} immagini classificate per {attributo.strip().lower()}")

# Se il dataset non è vuoto, mostra le immagini
if train_dataset is not None:
    show_images(train_dataset)
else:
    print("❌ Nessuna immagine con valore valido")
