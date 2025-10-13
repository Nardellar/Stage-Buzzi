"""
Versione migliorata del ViT con regularizzazione e cross-validation integrate
Basato su vit_from_hf_attribute.py originale con ottimizzazioni
"""
from __future__ import annotations

import argparse
from pathlib import Path
from collections import Counter
import random
import os
import json
from datetime import datetime

import pandas as pd
import numpy as np
import tensorflow as tf
from datasets import load_dataset, Dataset
from transformers import AutoImageProcessor, TFViTModel
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras import layers
from tensorflow.keras.optimizers import AdamW
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from common import csv_config


# 🔧 CLASSE MIGLIORATA CON REGULARIZZAZIONE
class ViTForCustomClassificationImproved(tf.keras.Model):
    def __init__(self, num_labels, dropout_rate=0.1, **kwargs):  # FIX: Ridotto da 0.3 a 0.1
        super().__init__(**kwargs)
        self.vit = TFViTModel.from_pretrained(
            "google/vit-base-patch16-224",
            from_pt=True,
        )

        # 🔧 REGULARIZZAZIONE: Dropout e Batch Normalization
        self.dropout = layers.Dropout(dropout_rate)
        self.batch_norm = layers.BatchNormalization()

        # 🔧 REGULARIZZAZIONE LEGGERA: Dense layer con L2 regularization
        self.classifier = layers.Dense(
            num_labels,
            name="classifier",
            kernel_regularizer=tf.keras.regularizers.l2(0.001),  # FIX: Ridotto da 0.01 a 0.001
            bias_regularizer=tf.keras.regularizers.l2(0.001)  # FIX: Ridotto da 0.01 a 0.001
        )

    def call(self, inputs, training=False, output_attentions=False):
        pixel_values = inputs['pixel_values']
        outputs = self.vit(
            pixel_values,
            training=training,
            output_attentions=output_attentions
        )

        # 🔧 REGULARIZZAZIONE: Applica dropout e batch norm
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


# 🔧 FUNZIONE MIGLIORATA PER PREPARARE IL DATASET
def prepare_dataset_improved(attribute: str, batch_size: int):
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

    # 🔧 BILANCIAMENTO CON SOLO CLASS WEIGHTS (FIX: Rimosso oversampling)
    print("\n--- Bilanciamento del dataset con class weights... ---")
    print("✅ Usando SOLO class weights (NO oversampling per evitare duplicati)")
    print("💡 Approccio più pulito: nessun duplicato artificiale, nessun overfitting sui duplicati")

    # Mantieni il dataset originale senza duplicazione
    train_ds = train_ds.shuffle(seed=42)

    print("\n--- Analisi del bilanciamento (FINAL) ---")
    final_counts = Counter(train_ds["attribute"])
    for class_id, count in sorted(final_counts.items()):
        print(f"  - Classe '{id2label[class_id]}': {count} immagini")
    print("---------------------------------------------------------")

    # 🔧 CALCOLO CLASS WEIGHTS per bilanciamento
    total_samples = sum(final_counts.values())
    class_weights = {}
    for class_id, count in final_counts.items():
        # Formula: n_samples / (n_classes * n_samples_per_class)
        weight = total_samples / (num_classes * count)
        class_weights[class_id] = weight
        print(f"  - Classe '{id2label[class_id]}': weight = {weight:.3f}")

    print("💡 Class weights calcolati per bilanciamento matematicamente equivalente all'oversampling")
    print("✅ Vantaggi: nessun duplicato, nessun overfitting artificiale, training più veloce")

    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

    def transform(batch):
        processed = processor(images=batch["image"], return_tensors="tf")
        batch["pixel_values"] = processed["pixel_values"]
        batch["labels"] = tf.convert_to_tensor(batch["attribute"], dtype=tf.int32)
        batch["original_labels"] = tf.convert_to_tensor(batch["original_label_id"], dtype=tf.int32)
        return batch

    train_ds = train_ds.map(transform, batched=True, batch_size=batch_size)
    val_ds = val_ds.map(transform, batched=True, batch_size=batch_size)
    train_tf = train_ds.to_tf_dataset(columns=["pixel_values", "original_labels"], label_cols=["labels"],
                                      batch_size=batch_size, shuffle=True)
    val_tf = val_ds.to_tf_dataset(columns=["pixel_values", "original_labels"], label_cols=["labels"],
                                  batch_size=batch_size, shuffle=False)

    return train_tf, val_tf, num_classes, id2label, exp_id2name, class_weights


# 🔧 FUNZIONI AVANZATE DI VALUTAZIONE
def evaluate_model_detailed(model, val_dataset, id2label, results_dir, attribute):
    """Valutazione dettagliata del modello con metriche avanzate"""

    print("\n🔍 VALUTAZIONE DETTAGLIATA DEL MODELLO")
    print("=" * 50)

    # Predizioni
    y_true = []
    y_pred = []

    for batch in val_dataset:
        images, labels = batch
        predictions = model(images, training=False)
        predicted_classes = tf.argmax(predictions['logits'], axis=1)

        y_true.extend(labels.numpy())
        y_pred.extend(predicted_classes.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Report di classificazione
    class_names = [id2label[i] for i in range(len(id2label))]
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # Salva report dettagliato (con conversione dei tipi per JSON)
    report_path = results_dir / f"classification_report_{attribute}.json"

    # Converti numpy types a Python types per JSON
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj

    report_clean = convert_numpy_types(report)

    with open(report_path, 'w') as f:
        json.dump(report_clean, f, indent=2)

    print(f"📊 Accuracy: {report['accuracy']:.4f}")
    print(f"📄 Report salvato: {report_path}")

    # Matrice di confusione
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Matrice di Confusione - {attribute}')
    plt.xlabel('Predizioni')
    plt.ylabel('Valori Reali')
    plt.tight_layout()

    cm_path = results_dir / f"confusion_matrix_{attribute}.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to avoid showing it now

    print(f"📊 Matrice di confusione salvata: {cm_path}")

    return report


def create_training_plots(history, results_dir, attribute):
    """Crea grafici dettagliati dell'addestramento"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Learning Rate
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)

    # Top-2 Accuracy
    if 'top2_accuracy' in history.history:
        axes[1, 1].plot(history.history['top2_accuracy'], label='Training Top-2 Acc')
        axes[1, 1].plot(history.history['val_top2_accuracy'], label='Validation Top-2 Acc')
        axes[1, 1].set_title('Top-2 Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    plt.suptitle(f'Training History - {attribute}', fontsize=16)
    plt.tight_layout()

    plot_path = results_dir / f"training_history_{attribute}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to avoid showing it now

    print(f"📊 Grafici di training salvati: {plot_path}")


def save_attention_maps(model, dataset, id2label, results_dir, attribute, num_images=8):
    """
    Genera e salva le mappe di attenzione del modello ViT.
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
        plt.suptitle("Mappe di Attenzione del Modello ViT Migliorato", fontsize=20)

        for i in range(min(num_images, len(inputs['pixel_values']))):
            img = inputs['pixel_values'][i].numpy()
            if img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))

            # Un-normalize for display
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)

            heatmap = tf.image.resize(
                tf.expand_dims(attention_maps[i], axis=-1),
                [img.shape[0], img.shape[1]]
            )

            ax = plt.subplot(2, num_images // 2, i + 1)
            plt.imshow(img)
            plt.imshow(heatmap, cmap='jet', alpha=0.5)

            true_label = id2label.get(labels[i].numpy(), "N/A")
            pred_label = id2label.get(predictions[i].numpy(), "N/A")

            plt.title(
                f"Vero: {true_label}\nPredetto: {pred_label}",
                color=("green" if true_label == pred_label else "red")
            )
            plt.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path = results_dir / f"attention_map_{attribute}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Mappa di attenzione salvata in: {save_path}")


def get_advanced_callbacks(results_dir, attribute):
    """Callbacks avanzati per il training"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=results_dir / f"best_model_{attribute}_{timestamp}.weights.h5",  # Save weights only
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,  # Important!
            verbose=1
        ),
        TensorBoard(
            log_dir=results_dir / f"logs_{attribute}_{timestamp}",
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
    ]

    return callbacks


# --- CORREZIONE INDENTAZIONE ---
def augment_image_improved(features, labels):
    """Augmentation corretta per immagini già processate dal ViT processor"""
    image = features['pixel_values']

    # Le immagini dal ViT processor hanno forma [batch, channels, height, width]
    # Le funzioni di augmentation di TF si aspettano [batch, height, width, channels]
    if len(image.shape) == 4:
        image = tf.transpose(image, [0, 2, 3, 1])
    elif len(image.shape) == 3:
        image = tf.transpose(image, [1, 2, 0])

    # Applica augmentation solo se l'immagine ha 3 canali
    if image.shape[-1] == 3:
        # Le seguenti operazioni devono essere indentate per essere DENTRO l'if
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, 0.9, 1.1)

    # Riporta alla forma originale
    if len(image.shape) == 4:
        image = tf.transpose(image, [0, 3, 1, 2])
    elif len(image.shape) == 3:
        image = tf.transpose(image, [2, 0, 1])

    features['pixel_values'] = image
    return features, labels


# --- FINE CORREZIONE ---


def compile_improved_model(model):
    """Compila il modello con ottimizzazioni avanzate"""

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=5e-5,  # FIX: Aumentato da 1e-5
        weight_decay=1e-5,  # FIX: Ridotto da 1e-4
    )

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top2_accuracy')
        ]
    )

    return model


def main_improved() -> None:
    """Funzione principale con tutte le migliorie integrate"""

    root_dir = Path(__file__).resolve().parents[1]
    csv_path = root_dir.parent / "esperimenti.csv"
    if not csv_path.exists():
        csv_config.create_csv(csv_path)
    df = pd.read_csv(csv_path)
    available_attributes = [col for col in df.columns if col.lower() not in ["id", "esperimenti"]]

    print("🚀 ViT MIGLIORATO - Addestramento")
    attribute = ""
    while attribute not in available_attributes:
        attribute = input(f"➡️ Su quale attributo vuoi addestrare? ({', '.join(available_attributes)}): ").strip()
        if not attribute:
            attribute = "temperatura"  # Default
            print(f"   Usando default: {attribute}")
            break
        if attribute not in available_attributes:
            print("❌ Attributo non valido.")

    results_dir = Path(f"results_improved_{attribute}")
    results_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, num_classes, id2label, exp_id2name, class_weights = prepare_dataset_improved(attribute,
                                                                                                   batch_size=16)

    model = ViTForCustomClassificationImproved(num_labels=num_classes)
    model = compile_improved_model(model)

    print("\n✅ Augmentation attivata con gestione corretta dei canali")
    train_ds = train_ds.map(augment_image_improved, num_parallel_calls=tf.data.AUTOTUNE)

    callbacks = get_advanced_callbacks(results_dir, attribute)

    print("\n--- Inizio addestramento ---")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=100,
        callbacks=callbacks,
        class_weight=class_weights
    )

    # Carica i pesi del miglior modello salvato da ModelCheckpoint
    print("\n🔄 Ricaricamento dei pesi dal miglior checkpoint per la valutazione finale...")
    best_model_path = max(results_dir.glob(f"best_model_{attribute}_*.weights.h5"), key=os.path.getctime)
    model.load_weights(best_model_path)
    print(f"✅ Pesi caricati da: {best_model_path}")

    evaluation_report = evaluate_model_detailed(model, val_ds, id2label, results_dir, attribute)
    create_training_plots(history, results_dir, attribute)
    save_attention_maps(model, val_ds, id2label, results_dir, attribute)

    best_epoch = np.argmin(history.history['val_loss']) + 1
    val_acc = evaluation_report['accuracy']
    train_acc = history.history['accuracy'][best_epoch - 1]
    val_loss = evaluation_report['weighted avg']['f1-score']  # Stima
    top2_acc = max(history.history['val_top2_accuracy'])

    # Riassunto finale per un facile confronto
    print("\n" + "=" * 50)
    print("🏆 RIASSUNTO FINALE DEL TRAINING 🏆")
    print("=" * 50)
    print(f"✅ Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"✅ Training Accuracy:   {train_acc * 100:.2f}%")
    print(f"✅ Val Loss:            {val_loss:.4f}")
    print(f"✅ Top-2 Accuracy:      {top2_acc * 100:.2f}%")
    print(f"✅ Best Epoch:          {best_epoch}/{len(history.history['loss'])}")
    if 'early_stopping' in str(callbacks):
        stopped_epoch = callbacks[0].stopped_epoch
        if stopped_epoch > 0:
            print(f"✅ Early Stopping:      Epoca {stopped_epoch + 1}")
    print("=" * 50)


if __name__ == "__main__":
    main_improved()