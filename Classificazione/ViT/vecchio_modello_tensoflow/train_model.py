"""
Script per l'addestramento del modello ViT Migliorato (il vecchio vit_from_hf_attribute_improved.py).
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
from PIL import Image

import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from datasets import load_dataset, ClassLabel
from transformers import AutoImageProcessor, TFViTModel
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras import layers
from keras.optimizers import AdamW

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from common import csv_config

# --- CLASSE DEL MODELLO ---
@keras.saving.register_keras_serializable()
class ViTForCustomClassificationImproved(keras.Model):
    def __init__(self, num_labels, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        # Carica il ViT e convertilo da PyTorch (il ViT sarà congelato e non salvato)
        self.vit = TFViTModel.from_pretrained("google/vit-base-patch16-224", from_pt=True)
        self.vit.trainable = False  # ✅ CORRETTO: Congela il ViT per evitare overfitting con dataset piccolo
        self.dropout = layers.Dropout(dropout_rate)
        self.batch_norm = layers.BatchNormalization()
        self.classifier = layers.Dense(
            num_labels,
            name="classifier",
            kernel_regularizer=keras.regularizers.l2(0.001),
            bias_regularizer=keras.regularizers.l2(0.001)
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
        """Necessario per il salvataggio del modello"""
        config = super().get_config()
        config.update({
            "num_labels": self.classifier.units,
            "dropout_rate": self.dropout.rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Necessario per il caricamento del modello"""
        # Gestisce modelli salvati prima dell'aggiunta di get_config()
        num_labels = config.pop("num_labels", 3)  # Default 3 classi
        dropout_rate = config.pop("dropout_rate", 0.1)
        return cls(num_labels=num_labels, dropout_rate=dropout_rate, **config)

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
        raw_attribute_value = attr_map.get(class_name, -1)
        example["attribute"] = label2id.get(raw_attribute_value, -1)
        example["original_label_id"] = example["label"]  # ✅ CORREZIONE: Mantieni l'ID originale
        return example

    ds = ds.map(add_attribute).filter(lambda ex: ex["attribute"] != -1)

    ds = ds.cast_column('attribute', ClassLabel(names=unique_attributes))

    print("\n--- Divisione del dataset (80% Train, 20% Validation/Test) ---")
    ds_split = ds.train_test_split(test_size=0.2, seed=42, stratify_by_column="attribute")
    train_ds, val_ds = ds_split["train"], ds_split["test"]

    print(f"✅ Train set:      {len(train_ds)} immagini")
    print(f"✅ Validation set:   {len(val_ds)} immagini")

    # ✅ CORREZIONE: Salva in formato HuggingFace invece di JSON per coerenza
    val_ds.save_to_disk("validation_test_set")
    print("\n✅ Set di validazione salvato in 'validation_test_set/' per la valutazione separata.")

    class_counts = Counter(train_ds["attribute"])
    total_samples = sum(class_counts.values())
    class_weights = {
        class_id: total_samples / (num_classes * count) for class_id, count in class_counts.items()
    }

    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    def transform(batch):
        # ✅ CORREZIONE: Usa SEMPRE Image.open() per coerenza totale
        # Converti oggetti PIL in percorsi temporanei, poi ricarica con Image.open()
        images_loaded = []
        for img in batch["image"]:
            if hasattr(img, 'filename') and img.filename:
                # Se l'immagine ha un filename, usa quello
                images_loaded.append(Image.open(img.filename))
            else:
                # Se non ha filename, salva temporaneamente e ricarica
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    img.save(tmp.name)
                    images_loaded.append(Image.open(tmp.name))
        
        processed = processor(images=images_loaded, return_tensors="tf")
        batch["pixel_values"] = tf.convert_to_tensor(processed["pixel_values"])
        batch["labels"] = tf.convert_to_tensor(batch["attribute"], dtype=tf.int32)
        batch["original_labels"] = tf.convert_to_tensor(batch["original_label_id"], dtype=tf.int32)
        return batch

    train_ds = train_ds.map(transform, batched=True, batch_size=batch_size)
    val_ds = val_ds.map(transform, batched=True, batch_size=batch_size)

    # Con la nuova versione di datasets, il formato sarà {features}, {labels}
    train_tf = train_ds.to_tf_dataset(columns=["pixel_values", "original_labels"], label_cols=["labels"], batch_size=batch_size, shuffle=True)
    val_tf = val_ds.to_tf_dataset(columns=["pixel_values", "original_labels"], label_cols=["labels"], batch_size=batch_size, shuffle=False)

    return train_tf, val_tf, num_classes, id2label, class_weights

augmentation_layers = keras.Sequential([
    keras.layers.RandomFlip("horizontal_and_vertical"),
    keras.layers.RandomRotation(factor=0.1),
    keras.layers.GaussianNoise(stddev=0.1)
    # Add other Keras augmentation layers here if needed
])

# CORREZIONE: La funzione ora accetta (features, labels) come previsto dalla nuova versione di 'datasets'
def augment_image(features, labels):
    """Data augmentation più aggressiva per le immagini."""
    # ✅ CORREZIONE: Gestisci il nuovo formato con dizionario
    if isinstance(features, dict):
        image = features['pixel_values']
    else:
        image = features

    # Transpose da (batch, C, H, W) a (batch, H, W, C) per le funzioni di tf.image
    if len(image.shape) == 4:
        image = tf.transpose(image, [0, 2, 3, 1])

    # Applica le augmentation DEFINITE FUORI
    if image.shape[-1] == 3:
        # --- INIZIO CORREZIONE ---
        # Call the pre-defined layers, ensure training=True for noise etc.
        image = augmentation_layers(image, training=True)
        # Apply brightness separately as it's not a layer
        image = tf.image.random_brightness(image, max_delta=0.1)
        # --- FINE CORREZIONE ---

    # Transpose di nuovo al formato originale (batch, C, H, W)
    if len(image.shape) == 4:
        image = tf.transpose(image, [0, 3, 1, 2])

    # Return the dictionary format expected by the model
    return {'pixel_values': image}, labels



def compile_model(num_classes):
    model = ViTForCustomClassificationImproved(num_labels=num_classes)
    optimizer = AdamW(learning_rate=5e-5, weight_decay=1e-5)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
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

    def format_for_model(features, labels):
        # ✅ CORREZIONE: Gestisci il nuovo formato con dizionario
        if isinstance(features, dict):
            # Se è già un dizionario, restituiscilo così com'è
            return features, labels
        elif isinstance(features, tuple):
            # Se è una tupla, estrai pixel_values
            pixel_values, original_labels = features
            return {'pixel_values': pixel_values}, labels
        else:
            # Se è un tensore diretto
            return {'pixel_values': features}, labels

    train_tf_final = train_tf_augmented.map(format_for_model, num_parallel_calls=tf.data.AUTOTUNE)
    val_tf_final = val_tf.map(format_for_model, num_parallel_calls=tf.data.AUTOTUNE)  # Applica anche a val_tf


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = results_dir / f"best_model_{attribute}_{timestamp}"

    # Usa un callback custom per salvare SOLO i layer addestrabili (non il ViT congelato)
    class CustomModelCheckpoint(tf.keras.callbacks.Callback):
        def __init__(self, filepath, monitor='val_loss'):
            super().__init__()
            self.filepath = filepath
            self.monitor = monitor
            self.best = float('inf')
            
        def on_epoch_end(self, epoch, logs=None):
            current = logs.get(self.monitor)
            if current is None:
                return
                
            if current < self.best:
                self.best = current
                print(f"\nEpoch {epoch + 1}: {self.monitor} improved to {current:.5f}, saving model...")
                # Salva SOLO i pesi dei layer addestrabili (dropout, batch_norm, classifier)
                # Non salviamo il ViT perché è congelato e verrà ricaricato da HuggingFace
                import numpy as np
                
                # Converti i pesi in array NumPy per compatibilità
                dropout_weights = [w.numpy() if hasattr(w, 'numpy') else w for w in self.model.dropout.get_weights()]
                batch_norm_weights = [w.numpy() if hasattr(w, 'numpy') else w for w in self.model.batch_norm.get_weights()]
                classifier_weights = [w.numpy() if hasattr(w, 'numpy') else w for w in self.model.classifier.get_weights()]
                
                # Salva in formato NumPy .npz (più affidabile di pickle)
                np.savez(str(self.filepath) + '.npz',
                    dropout_0=dropout_weights[0] if len(dropout_weights) > 0 else np.array([]),
                    batch_norm_0=batch_norm_weights[0] if len(batch_norm_weights) > 0 else np.array([]),
                    batch_norm_1=batch_norm_weights[1] if len(batch_norm_weights) > 1 else np.array([]),
                    batch_norm_2=batch_norm_weights[2] if len(batch_norm_weights) > 2 else np.array([]),
                    batch_norm_3=batch_norm_weights[3] if len(batch_norm_weights) > 3 else np.array([]),
                    classifier_0=classifier_weights[0] if len(classifier_weights) > 0 else np.array([]),
                    classifier_1=classifier_weights[1] if len(classifier_weights) > 1 else np.array([])
                )
                print(f"✅ Pesi addestrabili salvati in: {self.filepath}.npz")
    
    model_checkpoint = CustomModelCheckpoint(filepath=str(model_path), monitor='val_loss')
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1),
        model_checkpoint,
        TensorBoard(log_dir=results_dir / f"logs_{attribute}_{timestamp}")
    ]

    print("\n--- Inizio addestramento ---")
    model.fit(train_tf_final, validation_data=val_tf_final, epochs=100, callbacks=callbacks, class_weight=class_weights)

    # Salva gli artifacts
    model_weights_final = str(model_path) + '.npz'
    
    artifacts = {
        'attribute': attribute,
        'id2label': {int(k): v for k, v in id2label.items()},
        'model_weights_path': model_weights_final,
        'num_classes': num_classes,
        'weights_format': 'npz'  # Indica che i pesi sono in formato NumPy
    }
    with open(results_dir / "artifacts.json", "w") as f:
        json.dump(artifacts, f, indent=4)

    print("\n🎉 Addestramento completato! 🎉")
    print(f"✅ Pesi addestrabili salvati in: {model_weights_final}")
    print(f"✅ Artefatti per la valutazione salvati in: {results_dir / 'artifacts.json'}")
    print("\n➡️ Ora puoi eseguire lo script 'evaluate_model.py' per testare le performance.")

if __name__ == "__main__":
    main_train()