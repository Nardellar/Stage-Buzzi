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
            bias_regularizer=tf.keras.regularizers.l2(0.001)     # FIX: Ridotto da 0.01 a 0.001
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
    train_tf = train_ds.to_tf_dataset(columns=["pixel_values", "original_labels"], label_cols=["labels"], batch_size=batch_size, shuffle=True)
    val_tf = val_ds.to_tf_dataset(columns=["pixel_values", "original_labels"], label_cols=["labels"], batch_size=batch_size, shuffle=False)

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
    plt.show()
    
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
    axes[0, 1].plot(history.history['sparse_categorical_accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history.history['val_sparse_categorical_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning Rate (se disponibile)
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
    
    # Metriche aggiuntive
    metrics = [key for key in history.history.keys() if key.startswith('val_') and key != 'val_loss']
    if metrics:
        for metric in metrics[:1]:  # Mostra solo la prima metrica aggiuntiva
            metric_name = metric.replace('val_', '')
            if metric_name in history.history:
                axes[1, 1].plot(history.history[metric_name], label=f'Training {metric_name}')
                axes[1, 1].plot(history.history[metric], label=f'Validation {metric_name}')
                axes[1, 1].set_title(metric_name.title())
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
    
    plt.suptitle(f'Training History - {attribute}', fontsize=16)
    plt.tight_layout()
    
    plot_path = results_dir / f"training_history_{attribute}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Grafici di training salvati: {plot_path}")


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
            filepath=results_dir / f"best_model_{attribute}_{timestamp}.keras",
            monitor='val_loss',
            save_best_only=True,
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




# 🔧 AUGMENTATION MIGLIORATA E CORRETTA
def augment_image_improved(features, labels):
    """Augmentation corretta per immagini già processate dal ViT processor"""
    image = features['pixel_values']
    
    # Le immagini dal ViT processor hanno forma [batch, channels, height, width]
    # Le funzioni di augmentation di TF si aspettano [batch, height, width, channels]
    # Quindi dobbiamo fare il transpose
    
    # Controlla la forma dell'immagine
    if len(image.shape) == 4:  # [batch, channels, height, width]
        # Trasponi a [batch, height, width, channels]
        image = tf.transpose(image, [0, 2, 3, 1])
    elif len(image.shape) == 3:  # [channels, height, width] (singola immagine)
        # Trasponi a [height, width, channels]
        image = tf.transpose(image, [1, 2, 0])
    
    # 🔧 AUGMENTATION MODERATA (per evitare problemi)
    # Applica augmentation solo se l'immagine ha 3 canali
    if image.shape[-1] == 3:
        # Random flip orizzontale (50% probabilità)
        image = tf.image.random_flip_left_right(image)
        
        # Random flip verticale (50% probabilità)  
        image = tf.image.random_flip_up_down(image)
        
        # Random brightness (moderato)
        image = tf.image.random_brightness(image, max_delta=0.1)
        
        # Random contrast (moderato)
        image = tf.image.random_contrast(image, 0.9, 1.1)
    
    # Riporta alla forma originale
    if len(image.shape) == 4:  # [batch, height, width, channels]
        # Trasponi di nuovo a [batch, channels, height, width]
        image = tf.transpose(image, [0, 3, 1, 2])
    elif len(image.shape) == 3:  # [height, width, channels]
        # Trasponi di nuovo a [channels, height, width]
        image = tf.transpose(image, [2, 0, 1])
    
    features['pixel_values'] = image
    return features, labels


# 🔧 CALLBACKS MIGLIORATI
def get_improved_callbacks():
    """Restituisce i callback migliorati per l'addestramento"""
    
    # 🔧 EARLY STOPPING PIÙ AGGRESSIVO
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,  # Era 5
        verbose=1,
        restore_best_weights=True,
        min_delta=0.001
    )
    
    # 🔧 LEARNING RATE SCHEDULING
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
    
    return [early_stopping, reduce_lr]


# 🔧 COMPILAZIONE MIGLIORATA
def compile_improved_model(model):
    """Compila il modello con ottimizzazioni avanzate"""
    
    # 🔧 OPTIMIZER OTTIMIZZATO (FIX UNDERFITTING)
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=5e-5,  # FIX: Aumentato da 1e-5 (era troppo basso!)
        weight_decay=1e-5,   # FIX: Ridotto da 1e-4 (era troppo aggressivo!)
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8
    )
    
    # 🔧 LOSS CON LABEL SMOOTHING (con fallback)
    try:
        # Prova con label smoothing
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            label_smoothing=0.1  # Label smoothing per migliorare generalizzazione
        )
        print("✅ Label smoothing abilitato (0.1)")
    except TypeError:
        # Fallback senza label smoothing
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )
        print("⚠️  Label smoothing non supportato, usando loss standard")
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top2_accuracy')
        ]
    )
    
    return model


# 🔧 CROSS-VALIDATION INTEGRATA
def cross_validate_improved(attribute: str, n_splits=5, epochs=10):
    """Esegue cross-validation con il modello migliorato"""
    
    print(f"🔄 CROSS-VALIDATION MIGLIORATA per attributo: {attribute}")
    print(f"📊 Configurazione: {n_splits} fold, {epochs} epoche per fold")
    
    # Prepara dataset per CV
    root_dir = Path(__file__).resolve().parents[2]
    csv_path = root_dir / "esperimenti.csv"
    df = pd.read_csv(csv_path)
    
    ds = load_dataset("Nardellar/Esperimenti", split="train")
    exp_id2name = {i: name for i, name in enumerate(ds.features["label"].names)}
    
    unique_attributes = sorted(list(df[attribute].unique()))
    label2id = {label: i for i, label in enumerate(unique_attributes)}
    attr_map = df.set_index("ID")[attribute].to_dict()
    
    # Prepara dati per CV
    images = []
    labels = []
    
    for example in ds:
        exp_name = exp_id2name[example["label"]]
        attr_value = attr_map.get(exp_name, -1)
        if attr_value != -1:
            # Controlla che l'immagine non sia None
            if example["image"] is not None:
                images.append(example["image"])
                labels.append(label2id[attr_value])
    
    print(f"📊 Dati filtrati: {len(images)} immagini valide, {len(labels)} etichette")
    
    # Controlla che non ci siano valori None
    none_images = sum(1 for img in images if img is None)
    none_labels = sum(1 for lbl in labels if lbl is None)
    
    if none_images > 0 or none_labels > 0:
        print(f"⚠️  Trovati {none_images} immagini None e {none_labels} etichette None")
        # Rimuovi i valori None
        images = [img for img in images if img is not None]
        labels = [lbl for lbl in labels if lbl is not None]
        print(f"📊 Dati puliti: {len(images)} immagini, {len(labels)} etichette")
    
    print(f"📈 Dataset preparato: {len(images)} immagini, {len(unique_attributes)} classi")
    
    # 🔄 STRATIFIED K-FOLD
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    fold_accuracies = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
        print(f"\n🔄 FOLD {fold + 1}/{n_splits}")
        
        # Prepara dati per questo fold
        train_images = [images[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_images = [images[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        
        # Crea dataset TensorFlow
        train_ds, val_ds = create_tf_datasets_improved(
            train_images, train_labels, val_images, val_labels
        )
        
        # Controlla se i dataset sono stati creati correttamente
        if train_ds is None or val_ds is None:
            print(f"❌ Errore nella creazione dei dataset per fold {fold + 1}")
            continue
        
        # Crea e addestra modello
        model = ViTForCustomClassificationImproved(num_labels=len(unique_attributes))
        model = compile_improved_model(model)
        
        callbacks = get_improved_callbacks()
        
        # Addestra
        print(f"  🏋️ Addestramento fold {fold + 1}...")
        history = model.fit(
            train_ds, 
            validation_data=val_ds, 
            epochs=epochs, 
            callbacks=callbacks,
            verbose=0
        )
        
        # Valuta
        val_results = model.evaluate(val_ds, verbose=0)
        val_loss, val_accuracy = val_results[0], val_results[1]
        
        fold_results.append({
            'fold': fold + 1,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'epochs_trained': len(history.history['loss'])
        })
        
        fold_accuracies.append(val_accuracy)
        print(f"  ✅ Fold {fold + 1}: Acc={val_accuracy:.4f}, Loss={val_loss:.4f}")
    
    # 📊 RISULTATI FINALI
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    
    print(f"\n📊 RISULTATI CROSS-VALIDATION MIGLIORATA")
    print("=" * 60)
    print(f"🎯 ACCURACY: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"📉 LOSS: {np.mean([r['val_loss'] for r in fold_results]):.4f}")
    
    if std_accuracy < 0.05:
        print("✅ Risultati STABILI (deviazione standard < 5%)")
    else:
        print("⚠️ Risultati VARIABILI (deviazione standard ≥ 5%)")
    
    return {
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'fold_results': fold_results
    }


def create_tf_datasets_improved(train_images, train_labels, val_images, val_labels):
    """Crea dataset TensorFlow migliorati"""
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    
    # Controlli di sicurezza
    print(f"🔍 Debug: train_images={len(train_images)}, train_labels={len(train_labels)}")
    print(f"🔍 Debug: val_images={len(val_images)}, val_labels={len(val_labels)}")
    
    # Controlla valori None
    train_none = sum(1 for img in train_images if img is None)
    val_none = sum(1 for img in val_images if img is None)
    
    if train_none > 0 or val_none > 0:
        print(f"⚠️  Trovati {train_none} immagini None in train, {val_none} in val")
        return None, None
    
    def transform_batch(images, labels):
        try:
            processed = processor(images=images, return_tensors="tf")
            return {
                "pixel_values": processed["pixel_values"],
                "labels": tf.convert_to_tensor(labels, dtype=tf.int32)
            }
        except Exception as e:
            print(f"❌ Errore nel processamento: {e}")
            return None
    
    train_processed = transform_batch(train_images, train_labels)
    val_processed = transform_batch(val_images, val_labels)
    
    if train_processed is None or val_processed is None:
        print("❌ Errore nel processamento dei dati")
        return None, None
    
    train_ds = tf.data.Dataset.from_tensor_slices({
        "pixel_values": train_processed["pixel_values"],
        "labels": train_processed["labels"]
    }).batch(16)
    
    val_ds = tf.data.Dataset.from_tensor_slices({
        "pixel_values": val_processed["pixel_values"],
        "labels": val_processed["labels"]
    }).batch(16)
    
    return train_ds, val_ds


# 🔧 FUNZIONE PRINCIPALE MIGLIORATA
def main_improved() -> None:
    """Funzione principale con tutte le migliorie integrate"""
    
    root_dir = Path(__file__).resolve().parents[2]
    csv_path = root_dir / "esperimenti.csv"
    if not csv_path.exists(): 
        csv_config.create_csv(csv_path)
    df = pd.read_csv(csv_path)
    available_attributes = [col for col in df.columns if col.lower() not in ["id", "esperimenti"]]
    
    print("🚀 ViT MIGLIORATO - Scegli il tipo di addestramento:")
    print("1. Addestramento normale (come prima)")
    print("2. Cross-validation migliorata")
    print("3. Confronto tra modello originale e migliorato")
    
    choice = input("➡️ Scegli (1/2/3): ").strip()
    
    if choice == "1":
        # Addestramento normale migliorato
        print("🔧 Addestramento normale con migliorie...")
        attribute = input("➡️ Attributo: ").strip()
        
        results_dir = Path(f"results_improved_{attribute}")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        train_ds, val_ds, num_classes, id2label, exp_id2name, class_weights = prepare_dataset_improved(attribute, batch_size=16)
        
        # Crea modello migliorato
        model = ViTForCustomClassificationImproved(num_labels=num_classes)
        model = compile_improved_model(model)
        
        # 🔧 AUGMENTATION ATTIVATA
        print("✅ Augmentation attivata con gestione corretta dei canali")
        train_ds = train_ds.map(augment_image_improved, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Callback avanzati
        callbacks = get_advanced_callbacks(results_dir, attribute)
        
        print("\n--- Addestramento con migliorie avanzate... ---")
        print(f"📊 Class weights: {class_weights}")
        history = model.fit(
            train_ds, 
            validation_data=val_ds, 
            epochs=75,  # FIX: Aumentato da 25 a 75 per superare underfitting
            callbacks=callbacks,
            class_weight=class_weights
        )
        
        # Valutazione dettagliata
        evaluation_report = evaluate_model_detailed(model, val_ds, id2label, results_dir, attribute)
        
        # Grafici di training
        create_training_plots(history, results_dir, attribute)
        
        # Salva modello finale
        save_path = results_dir / f"vit_improved_{attribute}_final.keras"
        model.save(save_path)
        print(f"\n✅ Modello migliorato salvato: {save_path}")
        
        # Salva risultati riassuntivi
        summary = {
            'attribute': attribute,
            'final_accuracy': evaluation_report['accuracy'],
            'training_epochs': len(history.history['loss']),
            'best_val_loss': min(history.history['val_loss']),
            'best_val_accuracy': max(history.history['val_sparse_categorical_accuracy']),
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = results_dir / f"training_summary_{attribute}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📄 Riassunto training salvato: {summary_path}")
        
    elif choice == "2":
        # Cross-validation
        attribute = input("➡️ Attributo per CV: ").strip()
        results = cross_validate_improved(attribute, n_splits=5, epochs=10)
        
    elif choice == "3":
        # Confronto
        print("🔄 Confronto tra modello originale e migliorato...")
        attribute = input("➡️ Attributo: ").strip()
        
        print("📊 Questo richiederebbe implementazione del confronto diretto...")
        print("💡 Suggerimento: Esegui prima l'opzione 1, poi confronta i risultati!")
    
    else:
        print("❌ Scelta non valida")


if __name__ == "__main__":
    main_improved()
