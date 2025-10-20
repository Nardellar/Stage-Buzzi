"""
Script per l'addestramento del modello ViT Migliorato.
- Divide il dataset in Train (80%) e Validation (20%).
- Salva il validation set su file per essere usato come test set separato.
- Esegue il training e salva i pesi del modello migliore e gli artefatti.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter
from datetime import datetime
import os

import pandas as pd
import numpy as np
import tensorflow as tf
from datasets import load_dataset, ClassLabel
from transformers import AutoImageProcessor, TFViTModel
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras import layers
from tensorflow.keras.optimizers import AdamW

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from common import csv_config

# --- CLASSE DEL MODELLO ---
class ViTForCustomClassificationImproved(tf.keras.Model):
    def __init__(self, num_labels, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.vit = TFViTModel.from_pretrained("google/vit-base-patch16-224", from_pt=True)
        self.dropout = layers.Dropout(dropout_rate)
        self.batch_norm = layers.BatchNormalization()
        self.classifier = layers.Dense(
            num_labels,
            name="classifier",
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            bias_regularizer=tf.keras.regularizers.l2(0.001)
        )

    def call(self, inputs, training=False, output_attentions=False):
        pixel_values = inputs['pixel_values']
        outputs = self.vit(pixel_values, training=training, output_attentions=output_attentions)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output, training=training)
        pooled_output = self.batch_norm(pooled_output, training=training)
        logits = self.classifier(pooled_output)
        if output_attentions:
            return {"logits": logits, "attentions": outputs.attentions}
        return {"logits": logits}

    def get_config(self):
        config = super().get_config()
        config.update({"num_labels": self.classifier.units, "dropout_rate": self.dropout.rate})
        return config

# --- FUNZIONI DI PREPARAZIONE E TRAINING ---

def prepare_and_split_dataset(attribute: str, batch_size: int):
    root_dir = Path(__file__).resolve().parents[2]
    csv_path = root_dir / "esperimenti.csv"
    if not csv_path.exists(): csv_config.create_csv(csv_path)
    df = pd.read_csv(csv_path)

    unique_attributes = sorted([str(attr) for attr in df[attribute].unique()])

    label2id = {label: i for i, label in enumerate(unique_attributes)}
    id2label = {i: label for label, i in label2id.items()}
    num_classes = len(unique_attributes)

    attr_map = {k: str(v) for k, v in df.set_index("ID")[attribute].to_dict().items()}

    ds = load_dataset("Nardellar/Esperimenti", split="train")

    def add_attribute(example):
        class_name = ds.features["label"].int2str(example["label"])
        example["attribute"] = label2id.get(attr_map.get(class_name, -1), -1)
        return example

    ds = ds.map(add_attribute).filter(lambda ex: ex["attribute"] != -1)

    ds = ds.cast_column('attribute', ClassLabel(names=unique_attributes))

    print("\n--- Divisione del dataset (80% Train, 20% Validation/Test) ---")
    ds_split = ds.train_test_split(test_size=0.2, seed=42, stratify_by_column="attribute")
    train_ds, val_ds = ds_split["train"], ds_split["test"]

    print(f"✅ Train set:      {len(train_ds)} immagini")
    print(f"✅ Validation set:   {len(val_ds)} immagini")

    val_ds.to_json("validation_test_set.json")
    print("\n✅ Set di validazione salvato in 'validation_test_set.json' per la valutazione separata.")

    class_counts = Counter(train_ds["attribute"])
    total_samples = sum(class_counts.values())
    class_weights = {
        class_id: total_samples / (num_classes * count) for class_id, count in class_counts.items()
    }

    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    def transform(batch):
        processed = processor(images=batch["image"], return_tensors="tf")
        batch["pixel_values"] = processed["pixel_values"]
        batch["labels"] = tf.convert_to_tensor(batch["attribute"], dtype=tf.int32)
        return batch

    train_ds = train_ds.map(transform, batched=True, batch_size=batch_size)
    val_ds = val_ds.map(transform, batched=True, batch_size=batch_size)

    # Con la nuova versione di datasets, il formato sarà {features}, {labels}
    train_tf = train_ds.to_tf_dataset(columns=["pixel_values"], label_cols=["labels"], batch_size=batch_size, shuffle=True)
    val_tf = val_ds.to_tf_dataset(columns=["pixel_values"], label_cols=["labels"], batch_size=batch_size, shuffle=False)

    return train_tf, val_tf, num_classes, id2label, class_weights

# CORREZIONE: La funzione ora accetta (features, labels) come previsto dalla nuova versione di 'datasets'
def augment_image(pixel_values, labels):
    """Data augmentation per le immagini in formato dizionario."""
    image = pixel_values
    # Transpose da (batch, C, H, W) a (batch, H, W, C) per le funzioni di tf.image
    if len(image.shape) == 4: image = tf.transpose(image, [0, 2, 3, 1])
    # Applica le augmentation
    if image.shape[-1] == 3:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)

    # Transpose di nuovo al formato originale (batch, C, H, W)
    if len(image.shape) == 4: image = tf.transpose(image, [0, 3, 1, 2])

    # Restituisce i dati nel formato a dizionario che il modello si aspetta
    return {'pixel_values': image}, labels

def compile_model(num_classes):
    model = ViTForCustomClassificationImproved(num_labels=num_classes)
    optimizer = AdamW(learning_rate=5e-5, weight_decay=1e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

# --- FUNZIONE PRINCIPALE DI ADDESTRAMENTO ---
def main_train():
    print("🚀 SCRIPT DI ADDESTRAMENTO ViT 🚀")
    attribute = input("➡️ Su quale attributo vuoi addestrare? (es. temperatura): ").strip() or "temperatura"

    results_dir = Path(f"training_results_{attribute}")
    results_dir.mkdir(parents=True, exist_ok=True)

    train_tf, val_tf, num_classes, id2label, class_weights = prepare_and_split_dataset(attribute, batch_size=16)

    model = compile_model(num_classes)

    train_tf_augmented = train_tf.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = results_dir / f"best_model_{attribute}_{timestamp}.weights.h5"

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1),
        ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1),
        TensorBoard(log_dir=results_dir / f"logs_{attribute}_{timestamp}")
    ]

    print("\n--- Inizio addestramento ---")
    model.fit(train_tf_augmented, validation_data=val_tf, epochs=100, callbacks=callbacks, class_weight=class_weights)

    artifacts = {
        'attribute': attribute,
        'id2label': {int(k): v for k, v in id2label.items()},
        'model_weights_path': str(model_path),
        'num_classes': num_classes
    }
    with open(results_dir / "artifacts.json", "w") as f:
        json.dump(artifacts, f, indent=4)

    print("\n🎉 Addestramento completato! 🎉")
    print(f"✅ Pesi del modello migliore salvati in: {model_path}")
    print(f"✅ Artefatti per la valutazione salvati in: {results_dir / 'artifacts.json'}")
    print("\n➡️ Ora puoi eseguire lo script 'evaluate_model.py' per testare le performance.")

if __name__ == "__main__":
    main_train()