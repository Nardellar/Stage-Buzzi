"""
Allenamento generalizzato di un ViT, con bilanciamento, verifica, Early Stopping,
report finale e salvataggio automatico delle mappe di attenzione.
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


# Classe del modello (invariata)
class ViTForCustomClassification(tf.keras.Model):
    def __init__(self, num_labels, **kwargs):
        super().__init__(**kwargs)
        self.vit = TFViTModel.from_pretrained(
            "google/vit-base-patch16-224",
            from_pt=True,
        )
        self.classifier = tf.keras.layers.Dense(num_labels, name="classifier")

    def call(self, inputs, training=False, output_attentions=False):
        pixel_values = inputs['pixel_values']
        outputs = self.vit(
            pixel_values,
            training=training,
            output_attentions=output_attentions
        )
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        if output_attentions:
            return {"logits": logits, "attentions": outputs.attentions}
        return {"logits": logits}


# Funzione di preparazione dataset (invariata)
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
    exp_id2name = {i: name for i, name in enumerate(ds.features["label"].names)}

    def add_attribute(example):
        class_name = ds.features["label"].int2str(example["label"])
        raw_attribute_value = attr_map.get(class_name, -1)
        example["attribute"] = label2id.get(raw_attribute_value, -1)
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

    print("\n--- Bilanciamento del dataset di addestramento in corso... ---")
    max_count = max(class_counts.values())
    balanced_train_ds = train_ds.shuffle(seed=42).flatten_indices()
    grouped_by_class = {i: balanced_train_ds.filter(lambda ex: ex['attribute'] == i) for i in range(num_classes)}
    final_datasets = []
    for i in range(num_classes):
        class_ds = grouped_by_class[i]
        if len(class_ds) < max_count:
            oversampled_ds = class_ds.select(random.choices(range(len(class_ds)), k=max_count))
            final_datasets.append(oversampled_ds)
        else:
            final_datasets.append(class_ds)
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
        batch["labels"] = tf.convert_to_tensor(batch["attribute"], dtype=tf.int32)
        batch["original_labels"] = tf.convert_to_tensor(batch["original_label_id"], dtype=tf.int32)
        return batch

    train_ds = train_ds.map(transform, batched=True, batch_size=batch_size)
    val_ds = val_ds.map(transform, batched=True, batch_size=batch_size)
    train_tf = train_ds.to_tf_dataset(columns=["pixel_values", "original_labels"], label_cols=["labels"], batch_size=batch_size, shuffle=True)
    val_tf = val_ds.to_tf_dataset(columns=["pixel_values", "original_labels"], label_cols=["labels"], batch_size=batch_size, shuffle=False)

    return train_tf, val_tf, num_classes, id2label, exp_id2name

# Funzione di augmentation (invariata)
def augment_image(features, labels):
    image = features['pixel_values']
    if image.shape[0] == 3 or image.shape[0] == 1:
        image = tf.transpose(image, [1, 2, 0])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    if len(image.shape) == 3:
        image = tf.transpose(image, [2, 0, 1])
    features['pixel_values'] = image
    return features, labels

# Funzione di verifica visiva (invariata)
def visualize_and_verify_data(dataset, id2label, exp_id2name, attribute_name):
    print("\n--- Verifica visiva dei dati ---")
    plt.figure(figsize=(12, 12))
    plt.suptitle(f"Verifica Campione - Attributo: {attribute_name.capitalize()}", fontsize=16)
    for x, y in dataset.take(1):
        images, original_labels, attribute_labels = x['pixel_values'], x['original_labels'], y
        for i in range(min(16, len(images))):
            ax = plt.subplot(4, 4, i + 1)
            img = images[i].numpy()
            if img.shape[0] == 3: img = np.transpose(img, (1, 2, 0))
            img = (img - img.min()) / (img.max() - img.min())
            plt.imshow(img)
            exp_name = exp_id2name.get(original_labels[i].numpy(), "N/A")
            attr_name = id2label.get(attribute_labels[i].numpy(), "N/A")
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

# --- MODIFICA 1: La funzione ora salva il file invece di mostrarlo ---
def save_attention_maps(model, dataset, id2label, results_dir, attribute, num_images=8):
    """
    Esegue il modello su un campione, genera le mappe di attenzione e le salva su file.
    """
    print("\n--- Generazione e salvataggio delle Mappe di Attenzione ---")

    for inputs, labels in dataset.take(1):
        outputs = model(inputs, training=False, output_attentions=True)
        last_layer_attentions = outputs["attentions"][-1]
        avg_attentions = tf.reduce_mean(last_layer_attentions, axis=1)
        cls_token_attention = avg_attentions[:, 0, 1:]
        num_patches_side = int(np.sqrt(cls_token_attention.shape[-1]))
        attention_maps = tf.reshape(cls_token_attention, (-1, num_patches_side, num_patches_side))
        predictions = tf.argmax(outputs["logits"], axis=-1)

        plt.figure(figsize=(20, 10))
        plt.suptitle("Mappe di Attenzione del Modello ViT", fontsize=20)

        for i in range(min(num_images, len(inputs['pixel_values']))):
            img = inputs['pixel_values'][i].numpy()
            if img.shape[0] == 3: img = np.transpose(img, (1, 2, 0))
            img = (img - img.min()) / (img.max() - img.min())

            heatmap = tf.image.resize(tf.expand_dims(attention_maps[i], axis=-1), [img.shape[0], img.shape[1]])

            ax = plt.subplot(2, num_images // 2, i + 1)
            plt.imshow(img)
            plt.imshow(heatmap, cmap='jet', alpha=0.5)

            true_label = id2label.get(labels[i].numpy(), "N/A")
            pred_label = id2label.get(predictions[i].numpy(), "N/A")

            plt.title(f"Vero: {true_label}\nPredetto: {pred_label}",
                      color=("green" if true_label == pred_label else "red"))
            plt.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Salviamo la figura su file
    save_path = results_dir / f"attention_map_{attribute}.png"
    plt.savefig(save_path)
    plt.close() # Chiudiamo la figura per liberare memoria
    print(f"✅ Mappa di attenzione salvata in: {save_path}")


def main() -> None:
    # (Tutta la parte iniziale è invariata)
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
        if attribute not in available_attributes: print(
            f"❌ Attributo non valido. Scegli tra: {', '.join(available_attributes)}")
    print(f"\n✅ Ottimo! Il modello verrà addestrato sull'attributo: '{attribute}'")

    results_dir = Path(f"results_{attribute}")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"--- I risultati verranno salvati in: '{results_dir}' ---")

    train_ds, val_ds, num_classes, id2label, exp_id2name = prepare_dataset(attribute, batch_size=16)
    if train_ds is None: return
    if not visualize_and_verify_data(train_ds, id2label, exp_id2name, attribute): return

    print("\n--- creando il modello... ---")
    model = ViTForCustomClassification(num_labels=num_classes)

    print("--- Applicazione della data augmentation... ---")
    train_ds = train_ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-5),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    print("\n--- Inizio dell'addestramento (con oversampling e augmentation)... ---")
    history = model.fit(train_ds, validation_data=val_ds, epochs=25, callbacks=[early_stopping])

    save_path = results_dir / f"vit_from_hf_{attribute}.keras"
    model.save(save_path)
    print(f"\n✅ Modello salvato nel file: {save_path}")

    # --- ECCO LA CORREZIONE ---
    # Ho ripristinato il contenuto corretto della variabile 'report_content'
    print("\n--- Generazione del report delle performance... ---")
    best_epoch = np.argmin(history.history['val_loss'])
    best_val_loss = history.history['val_loss'][best_epoch]
    best_val_accuracy = history.history['val_accuracy'][best_epoch]
    best_train_loss = history.history['loss'][best_epoch]
    best_train_accuracy = history.history['accuracy'][best_epoch]
    total_epochs_run = len(history.history['loss'])

    report_content = (
        f"📄 Report delle Performance per l'attributo: '{attribute}'\n"
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
        f"Il modello finale è stato salvato nel file: '{save_path}'\n"
    )

    report_filename = results_dir / f"performance_report_{attribute}.txt"
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(report_content)
    print(f"✅ Report salvato come '{report_filename}'")

    save_attention_maps(model, val_ds, id2label, results_dir, attribute)

    print("\n🎉 Processo completato!")
if __name__ == "__main__":
    main()