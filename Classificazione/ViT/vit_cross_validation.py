"""
Implementazione di Cross-Validation per il modello ViT
"""
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from datasets import load_dataset
from transformers import AutoImageProcessor, TFViTModel
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
from pathlib import Path
from collections import Counter
import random
import sys

# Aggiungi il percorso del progetto per importare moduli comuni
sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import csv_config

# --- Definizione del Modello (integrata da vit_from_hf_attribute_improved.py) ---
class ViTForCustomClassificationCV(tf.keras.Model):
    def __init__(self, num_labels, dropout_rate=0.3, **kwargs):
        super().__init__(name="ViTForCustomClassification", **kwargs)
        self.vit = TFViTModel.from_pretrained(
            "google/vit-base-patch16-224",
            from_pt=True,
        )
        self.dropout = layers.Dropout(dropout_rate)
        self.batch_norm = layers.BatchNormalization()
        self.classifier = layers.Dense(
            num_labels,
            name="classifier",
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        )

    def call(self, inputs, training=False):
        # La chiave 'pixel_values' è gestita da to_tf_dataset
        pixel_values = inputs['pixel_values']
        outputs = self.vit(pixel_values, training=training)

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output, training=training)
        pooled_output = self.batch_norm(pooled_output, training=training)

        logits = self.classifier(pooled_output)
        return {"logits": logits}

# --- Funzioni di supporto ---

def prepare_dataset_for_cv(attribute: str):
    """Prepara il dataset per la cross-validation."""
    root_dir = Path(__file__).resolve().parents[1]
    csv_path = root_dir / "../esperimenti.csv"
    if not csv_path.exists():
        print(f"File CSV non trovato in {csv_path}. Creazione in corso...")
        # Assumendo che esista una funzione per creare il CSV
        # import common.csv_config
        # common.csv_config.create_csv(csv_path)

    df = pd.read_csv(csv_path)

    ds = load_dataset("Nardellar/Esperimenti", split="train")
    exp_id2name = {i: name for i, name in enumerate(ds.features["label"].names)}

    unique_attributes = sorted(list(df[attribute].unique()))
    label2id = {label: i for i, label in enumerate(unique_attributes)}
    attr_map = df.set_index("ID")[attribute].to_dict()

    images = []
    labels = []

    for example in ds:
        exp_name = exp_id2name[example["label"]]
        attr_value = attr_map.get(exp_name)
        if attr_value is not None and attr_value in label2id:
            images.append(example["image"])
            labels.append(label2id[attr_value])

    return np.array(images), np.array(labels), unique_attributes

def create_tf_datasets(train_images, train_labels, val_images, val_labels, batch_size=16):
    """Crea dataset TensorFlow per un fold, con la struttura corretta."""
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

    def preprocess(images):
        # 'images' è una lista di PIL.Image, il processore le gestisce
        return processor(images=images, return_tensors="tf")["pixel_values"]

    # Converte le immagini PIL in tensori
    train_pixel_values = preprocess(list(train_images))
    val_pixel_values = preprocess(list(val_images))

    # Crea i dataset di TensorFlow
    train_ds = tf.data.Dataset.from_tensor_slices(
        ({"pixel_values": train_pixel_values}, train_labels)
    ).shuffle(len(train_labels)).batch(batch_size)

    val_ds = tf.data.Dataset.from_tensor_slices(
        ({"pixel_values": val_pixel_values}, val_labels)
    ).batch(batch_size)

    return train_ds, val_ds

def get_callbacks():
    """Restituisce i callback per l'addestramento."""
    return [
        EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1)
    ]

def cross_validate_vit(attribute: str, n_splits=5, epochs=10, batch_size=16):
    """Esegue cross-validation per il modello ViT."""

    print(f"🔄 INIZIO CROSS-VALIDATION per attributo: {attribute}")
    print(f"📊 Configurazione: {n_splits} folds, {epochs} epoche per fold")

    images, labels, unique_attributes = prepare_dataset_for_cv(attribute)
    num_classes = len(unique_attributes)

    print(f"📈 Dataset preparato:")
    print(f"  - Immagini totali: {len(images)}")
    print(f"  - Classi ({num_classes}): {unique_attributes}")
    print(f"  - Distribuzione: {Counter(labels)}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_losses = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
        print(f"\n🔄 FOLD {fold + 1}/{n_splits}")

        train_images, val_images = images[train_idx], images[val_idx]
        train_labels, val_labels = labels[train_idx], labels[val_idx]

        train_ds, val_ds = create_tf_datasets(train_images, train_labels, val_images, val_labels, batch_size)

        model = ViTForCustomClassificationCV(num_labels=num_classes)

        optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

        print(f"  🏋️ Addestramento fold {fold + 1}...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=get_callbacks(),
            verbose=2 # Meno output per pulizia
        )

        val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)

        fold_accuracies.append(val_accuracy)
        fold_losses.append(val_loss)

        print(f"  ✅ Fold {fold + 1} completato: Val Accuracy: {val_accuracy:.4f}, Val Loss: {val_loss:.4f}")

    # Risultati finali
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    mean_loss = np.mean(fold_losses)
    std_loss = np.std(fold_losses)

    print("\n" + "="*60)
    print(f"📊 RISULTATI FINALI CROSS-VALIDATION ({attribute.upper()})")
    print("="*60)
    print(f"🎯 Accuratezza Media: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"📉 Loss Media: {mean_loss:.4f} ± {std_loss:.4f}")
    print("-" * 60)
    print(f"Accuratezze per fold: {[f'{acc:.4f}' for acc in fold_accuracies]}")
    print("-" * 60)

if __name__ == "__main__":
    # Esempio di esecuzione per l'attributo 'temperatura'
    cross_validate_vit(attribute="temperatura", n_splits=5, epochs=10)