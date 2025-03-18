import tensorflow as tf
import pandas as pd
import os
import keras
from matplotlib import pyplot as plt

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
        #DA CAOIRE QUESTA RIGA
        plt.imshow(images[i].numpy().astype("uint8"))

        # Converti l'etichetta numerica nel nome della classe
        #DA CAPIRE QUESTA RIGA
        class_label = class_names[labels[i].numpy()]
        plt.title(f"{class_label}")

        plt.axis("off")

    plt.show()


def find_image_by_attribute(attribute):
    import os
    import pandas as pd

    df = pd.read_csv('esperimenti.csv')

    # Trasformo l'attributo cercato in minuscolo
    attribute_lower = attribute.lower()

    # Trovo, se esiste, la colonna del DataFrame che corrisponde a attribute_lower
    col_match = None
    for col in df.columns:
        if col.lower() == attribute_lower:
            col_match = col
            break

    if col_match is None:
        print(f"Attributo '{attribute}' non trovato (case insensitive). Le colonne disponibili sono: {list(df.columns)}")
        return {}

    # A questo punto col_match contiene il nome della colonna “reale” corrispondente
    # Esempio: Se il DF ha “Temperatura”, e l’utente ha inserito “temperatura” o “TEMPerATura”, col_match sarà “Temperatura”.

    # Seleziono le righe in cui la col_match non è null
    selected_exp = df[df[col_match].notnull()]["ID"].tolist()
    selected_df = df[df["ID"].isin(selected_exp)]

    # Costruisco il dizionario degli esperimenti
    experiment_dict = {
        row["ID"]: row.to_dict() for _, row in selected_df.iterrows()
    }

    # Scansiono le cartelle ed associo immagini → dati sperimentali
    #base_path = '/home/nardellar/Scaricati/stage/'
    image_info_dict = {}

    for exp_id in experiment_dict.keys():
        exp_folder = os.path.join(base_path, exp_id)
        # Verifico che la cartella esista, altrimenti la salto
        if not os.path.isdir(exp_folder):
            print(f"Cartella {exp_folder} non trovata.")
            continue

        image_files = [
            os.path.join(exp_folder, img) for img in os.listdir(exp_folder)
            if img.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        for img_path in image_files:
            image_info_dict[img_path] = experiment_dict[exp_id]

    return image_info_dict





result = find_image_by_attribute("tempERAtura")
import json
print(json.dumps(result, indent = 4))
print(len(result))

