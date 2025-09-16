"""
Allenamento generalizzato di un ViT su un attributo specifico scelto dall'utente,
con bilanciamento del dataset, verifica visiva migliorata, Early Stopping e report finale.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import tensorflow as tf
from datasets import load_dataset, Dataset
from transformers import AutoImageProcessor, TFViTModel
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import random

from common import csv_config


# La nostra classe personalizzata per il modello (invariata)
class ViTForCustomClassification(tf.keras.Model):
    def __init__(self, num_labels, **kwargs):
        super().__init__(**kwargs)
        self.vit = TFViTModel.from_pretrained(
            "google/vit-base-patch16-224",
            from_pt=True,
        )
        self.classifier = tf.keras.layers.Dense(num_labels, name="classifier")

    def call(self, pixel_values, training=False):
        # La keyword 'inputs' è quella standard attesa da Keras
        outputs = self.vit(pixel_values['pixel_values'], training=training)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return {"logits": logits}


def prepare_dataset(attribute: str, batch_size: int):
    root_dir = Path(__file__).resolve().parents[2]
    csv_path = root_dir / "esperimenti.csv"
    if not csv_path.exists():
        print("⚠️ File CSV mancante, creazione in corso...")
        csv_config.create_csv(csv_path)

    df = pd.read_csv(csv_path)
    unique_attributes = sorted(list(df[attribute].unique()))
    label2id = {label: i for i, label in enumerate(unique_attributes)}
    id2label = {i: label for label, i in label2id.items()}
    num_classes = len(unique_attributes)
    attr_map = df.set_index("ID")[attribute].to_dict()

    ds = load_dataset("Nardellar/Esperimenti", split="train")
    # Creiamo una mappa per convertire gli ID numerici degli esperimenti (0-13) nei loro nomi (EXP01-EXP14)
    exp_id2name = {i: name for i, name in enumerate(ds.features["label"].names)}


    def add_attribute(example):
        class_name = ds.features["label"].int2str(example["label"])
        raw_attribute_value = attr_map.get(class_name, -1)
        example["attribute"] = label2id.get(raw_attribute_value, -1)
        # Manteniamo l'etichetta originale per la verifica
        example["original_label_id"] = example["label"]
        return example

    ds = ds.map(add_attribute)
    ds = ds.filter(lambda ex: ex["attribute"] != -1)
    ds = ds.train_test_split(test_size=0.2, seed=42)
    train_ds, val_ds = ds["train"], ds["test"]

    print("\n--- Analisi del bilanciamento (PRIMA dell'oversampling) ---")
    class_counts = Counter(train_ds["attribute"])
    for class_id, count in sorted(class_counts.items()):
        print(f"  - Classe '{id2label[class_id]}': {count} immagini")
    print("---------------------------------------------------------")

    # --- NUOVO: Bilanciamento del Dataset di Addestramento con Oversampling ---
    print("\n--- Bilanciamento del dataset di addestramento in corso... ---")
    max_count = max(class_counts.values())
    balanced_train_ds = train_ds.shuffle(seed=42).flatten_indices() # Mescoliamo prima

    # Raggruppiamo i dati per classe
    grouped_by_class = {}
    for i in range(num_classes):
        grouped_by_class[i] = balanced_train_ds.filter(lambda ex: ex['attribute'] == i)

    # Applichiamo l'oversampling
    final_datasets = []
    for i in range(num_classes):
        class_ds = grouped_by_class[i]
        class_count = len(class_ds)
        if class_count < max_count:
            # Campioniamo con ripetizione per raggiungere max_count
            oversampled_ds = class_ds.select(
                random.choices(range(class_count), k=max_count)
            )
            final_datasets.append(oversampled_ds)
        else:
            final_datasets.append(class_ds)

    # Uniamo i dataset bilanciati
    from datasets import concatenate_datasets
    train_ds = concatenate_datasets(final_datasets).shuffle(seed=42)

    print("\n--- Analisi del bilanciamento (DOPO l'oversampling) ---")
    final_counts = Counter(train_ds["attribute"])
    for class_id, count in sorted(final_counts.items()):
        print(f"  - Classe '{id2label[class_id]}': {count} immagini")
    print("---------------------------------------------------------")


    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

    def transform(batch):
        processed = processor(images=batch["image"], return_tensors="tf")
        batch["pixel_values"] = processed["pixel_values"]
        # L'etichetta per l'addestramento
        batch["labels"] = tf.convert_to_tensor(batch["attribute"], dtype=tf.int32)
        # L'etichetta originale per la verifica
        batch["original_labels"] = tf.convert_to_tensor(batch["original_label_id"], dtype=tf.int32)
        return batch

    train_ds = train_ds.map(transform, batched=True, batch_size=batch_size)
    val_ds = val_ds.map(transform, batched=True, batch_size=batch_size)

    # Passiamo sia i pixel che le etichette originali come input al modello
    train_tf = train_ds.to_tf_dataset(
        columns=["pixel_values", "original_labels"], label_cols=["labels"], batch_size=batch_size, shuffle=True
    )
    val_tf = val_ds.to_tf_dataset(
        columns=["pixel_values", "original_labels"], label_cols=["labels"], batch_size=batch_size, shuffle=False
    )

    return train_tf, val_tf, num_classes, id2label, exp_id2name


def visualize_and_verify_data(dataset, id2label, exp_id2name, attribute_name):
    """Mostra un batch di immagini con etichette dettagliate."""
    print("\n--- Verifica visiva dei dati ---")

    plt.figure(figsize=(12, 12))
    plt.suptitle(f"Verifica Campione - Attributo: {attribute_name.capitalize()}", fontsize=16)

    # Prendiamo un solo batch. x è un dizionario, y è l'etichetta dell'attributo.
    for x, y in dataset.take(1):
        images = x['pixel_values']
        original_labels = x['original_labels']
        attribute_labels = y

        for i in range(min(16, len(images))):
            ax = plt.subplot(4, 4, i + 1)

            img = images[i].numpy()
            if img.shape[0] == 3:
                 img = np.transpose(img, (1, 2, 0))
            img = (img - img.min()) / (img.max() - img.min())

            plt.imshow(img)

            # Recuperiamo e mostriamo entrambe le etichette
            exp_id = original_labels[i].numpy()
            exp_name = exp_id2name.get(exp_id, "N/A")

            attr_id = attribute_labels[i].numpy()
            attr_name = id2label.get(attr_id, "N/A")

            plt.title(f"Exp: {exp_name}\n{attribute_name.capitalize()}: {attr_name}")
            plt.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    while True:
        proceed = input("I dati e le etichette sembrano corretti? (s/n): ").lower().strip()
        if proceed in ["s", "si"]: return True
        if proceed in ["n", "no"]:
            print("❌ Addestramento annullato dall'utente.")
            return False


def augment_image(features, labels):
    """Applica trasformazioni casuali a un'immagine."""
    image = features['pixel_values']  # Estraiamo l'immagine dal dizionario di input

    # Le immagini sono in formato (canali, altezza, larghezza), le convertiamo per TensorFlow
    if image.shape[0] == 3 or image.shape[0] == 1:
        image = tf.transpose(image, [1, 2, 0])

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)  # Delta ridotto per non alterare troppo
    image = tf.image.random_contrast(image, 0.9, 1.1)

    # Riconvertiamo al formato originale (canali, altezza, larghezza)
    if len(image.shape) == 3:
        image = tf.transpose(image, [2, 0, 1])

    features['pixel_values'] = image  # Reinseriamo l'immagine aumentata nel dizionario
    return features, labels

def main() -> None:
    # --- Selezione Interattiva dell'Attributo ---
    # (Questa parte è invariata)
    root_dir = Path(__file__).resolve().parents[2]
    csv_path = root_dir / "esperimenti.csv"
    if not csv_path.exists(): csv_config.create_csv(csv_path)
    df = pd.read_csv(csv_path)
    available_attributes = [col for col in df.columns if col.lower() not in ["id", "esperimenti"]]
    print("Ciao! Scegli su quale attributo vuoi addestrare il modello. 🤖")
    print("Attributi disponibili:", ", ".join(available_attributes))
    attribute = ""
    while attribute not in available_attributes:
        attribute = input("➡️ Inserisci il nome dell'attributo scelto: ").strip()
        if attribute not in available_attributes: print(f"❌ Attributo non valido. Scegli tra: {', '.join(available_attributes)}")
    print(f"\n✅ Ottimo! Il modello verrà addestrato sull'attributo: '{attribute}'")

    results_dir = Path(f"results_{attribute}")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"--- I risultati verranno salvati in: '{results_dir}' ---")

    # --- Preparazione del Dataset ---
    train_ds, val_ds, num_classes, id2label, exp_id2name = prepare_dataset(attribute, batch_size=16)
    if train_ds is None: return

    # --- Verifica Visiva ---
    if not visualize_and_verify_data(train_ds, id2label, exp_id2name, attribute): return

    # --- Creazione del Modello ---
    print("\n--- creando il modello... ---")
    model = ViTForCustomClassification(num_labels=num_classes)

    # --- Compilazione e Addestramento (invariato) ---
    # NUOVO: Applichiamo la data augmentation solo al set di addestramento
    print("--- Applicazione della data augmentation... ---")
    # tf.data.AUTOTUNE ottimizza le performance eseguendo le operazioni in parallelo

    train_ds = train_ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-5), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    print("\n--- Inizio dell'addestramento... ---")
    history = model.fit(train_ds, validation_data=val_ds, epochs=25, callbacks=[early_stopping])

    # --- Salvataggio e Report (invariato) ---
    save_path = results_dir / f"vit_from_hf_{attribute}.keras"
    model.save(save_path)
    print(f"\n✅ Modello salvato nel file: {save_path}")

    print("\n--- Generazione del report delle performance... ---")
    best_epoch = np.argmin(history.history['val_loss'])
    best_val_loss = history.history['val_loss'][best_epoch]
    best_val_accuracy = history.history['val_accuracy'][best_epoch]
    best_train_loss = history.history['loss'][best_epoch]
    best_train_accuracy = history.history['accuracy'][best_epoch]
    total_epochs_run = len(history.history['loss'])

    report_content = (f"📄 Report delle Performance per l'attributo: '{attribute}'\n"
                      f"========================================================\n\n"
                      f"L'addestramento si è fermato dopo {total_epochs_run} epoche (su 25) grazie all'Early Stopping.\n"
                      f"I pesi del modello sono stati ripristinati dall'epoca {best_epoch + 1}, che ha ottenuto i risultati migliori.\n\n"
                      f"📊 Metriche dell'epoca migliore (epoca {best_epoch + 1}):\n"
                      f"----------------------------------------\n"
                      f"  - Accuratezza (Training):   {best_train_accuracy:.4f}\n"
      
                f"  - Loss (Training):          {best_train_loss:.4f}\n"
                      f"  - Accuratezza (Validazione):{best_val_accuracy:.4f}\n"
                      f"  - Loss (Validazione):       {best_val_loss:.4f}\n"
                      f"----------------------------------------\n\n"
                      f"Il modello finale è stato salvato nel file: '{save_path}'\n")

    report_filename = results_dir / f"performance_report_{attribute}.txt"
    with open(report_filename, "w", encoding="utf-8") as f: f.write(report_content)
    print(report_content)
    print(f"✅ Report salvato come '{report_filename}'")

if __name__ == "__main__":
    main()