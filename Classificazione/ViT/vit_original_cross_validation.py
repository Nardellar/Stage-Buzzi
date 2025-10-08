"""
Cross-Validation per il modello ViT ORIGINALE
Per verificare se le performance del 91% sono stabili
"""
from __future__ import annotations

from pathlib import Path
from collections import Counter
import random

import pandas as pd
import numpy as np
import tensorflow as tf
from datasets import load_dataset, Dataset
from transformers import AutoImageProcessor, TFViTModel
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from common import csv_config


# Classe del modello ORIGINALE (senza regularizzazione)
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


def cross_validate_original(attribute: str, n_splits=5, epochs=25):
    """
    Esegue cross-validation con il modello ORIGINALE
    """
    
    print("🔄 CROSS-VALIDATION MODELLO ORIGINALE")
    print(f"📊 Attributo: {attribute}")
    print(f"📊 Configurazione: {n_splits} fold, {epochs} epoche per fold")
    print("=" * 70)
    
    # Prepara dataset
    root_dir = Path(__file__).resolve().parents[2]
    csv_path = root_dir / "esperimenti.csv"
    
    if not csv_path.exists():
        csv_config.create_csv(csv_path)
    
    df = pd.read_csv(csv_path)
    
    # Carica dataset
    ds = load_dataset("Nardellar/Esperimenti", split="train")
    exp_id2name = {i: name for i, name in enumerate(ds.features["label"].names)}
    
    # Mappa attributi
    unique_attributes = sorted(list(df[attribute].unique()))
    label2id = {label: i for i, label in enumerate(unique_attributes)}
    id2label = {i: label for label, i in label2id.items()}
    attr_map = df.set_index("ID")[attribute].to_dict()
    
    # Prepara dati per CV
    images = []
    labels = []
    
    for example in ds:
        exp_name = exp_id2name[example["label"]]
        attr_value = attr_map.get(exp_name, -1)
        if attr_value != -1:
            images.append(example["image"])
            labels.append(label2id[attr_value])
    
    print(f"📈 Dataset preparato: {len(images)} immagini, {len(unique_attributes)} classi")
    print(f"📊 Distribuzione classi: {Counter(labels)}")
    
    # Processor
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = []
    fold_accuracies = []
    fold_losses = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
        print(f"\n{'='*70}")
        print(f"🔄 FOLD {fold + 1}/{n_splits}")
        print(f"{'='*70}")
        print(f"  📊 Training: {len(train_idx)} campioni")
        print(f"  📊 Validation: {len(val_idx)} campioni")
        
        # Prepara dati per questo fold
        train_images = [images[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_images = [images[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        
        print(f"  📊 Distribuzione training: {Counter(train_labels)}")
        print(f"  📊 Distribuzione validation: {Counter(val_labels)}")
        
        # Processa immagini
        print(f"  🔄 Processamento immagini...")
        train_processed = processor(images=train_images, return_tensors="tf")
        val_processed = processor(images=val_images, return_tensors="tf")
        
        # Crea dataset TensorFlow con formato corretto
        # Il modello si aspetta un dizionario come input e le labels separate
        def create_dataset_with_dict(pixel_values, labels, batch_size=16, shuffle=False):
            """Crea dataset nel formato corretto per il modello"""
            dataset = tf.data.Dataset.from_tensor_slices((
                {"pixel_values": pixel_values},
                labels
            ))
            if shuffle:
                dataset = dataset.shuffle(1000, seed=42)
            return dataset.batch(batch_size)
        
        train_ds = create_dataset_with_dict(
            train_processed["pixel_values"],
            tf.convert_to_tensor(train_labels, dtype=tf.int32),
            batch_size=16,
            shuffle=True
        )
        
        val_ds = create_dataset_with_dict(
            val_processed["pixel_values"],
            tf.convert_to_tensor(val_labels, dtype=tf.int32),
            batch_size=16,
            shuffle=False
        )
        
        # Crea modello ORIGINALE
        print(f"  🏗️  Creazione modello...")
        model = ViTForCustomClassification(num_labels=len(unique_attributes))
        
        # Compila con parametri ORIGINALI
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),  # Learning rate originale
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )
        
        # Callback originali
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Addestra
        print(f"  🏋️  Addestramento fold {fold + 1}...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Valuta
        val_results = model.evaluate(val_ds, verbose=0)
        val_loss, val_accuracy = val_results[0], val_results[1]
        
        fold_results.append({
            'fold': fold + 1,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'epochs_trained': len(history.history['loss']),
            'best_epoch': np.argmin(history.history['val_loss']) + 1
        })
        
        fold_accuracies.append(val_accuracy)
        fold_losses.append(val_loss)
        
        print(f"\n  ✅ FOLD {fold + 1} COMPLETATO:")
        print(f"     Accuracy: {val_accuracy:.4f}")
        print(f"     Loss: {val_loss:.4f}")
        print(f"     Epoche addestrate: {len(history.history['loss'])}")
        print(f"     Epoca migliore: {np.argmin(history.history['val_loss']) + 1}")
    
    # Risultati finali
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    mean_loss = np.mean(fold_losses)
    std_loss = np.std(fold_losses)
    
    print("\n" + "="*70)
    print("📊 RISULTATI FINALI CROSS-VALIDATION (MODELLO ORIGINALE)")
    print("="*70)
    print(f"🎯 Accuratezza Media: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"📉 Loss Media: {mean_loss:.4f} ± {std_loss:.4f}")
    print("-"*70)
    print(f"Accuratezze per fold: {[f'{acc:.4f}' for acc in fold_accuracies]}")
    print(f"Loss per fold: {[f'{loss:.4f}' for loss in fold_losses]}")
    print("-"*70)
    
    # Interpretazione
    print("\n🔍 INTERPRETAZIONE:")
    if std_accuracy < 0.05:
        print("  ✅ Risultati STABILI (deviazione standard < 5%)")
    elif std_accuracy < 0.10:
        print("  ⚠️  Risultati MODERATAMENTE STABILI (deviazione standard 5-10%)")
    else:
        print("  ❌ Risultati INSTABILI (deviazione standard > 10%)")
    
    if mean_accuracy > 0.85:
        print("  🎉 Performance ECCELLENTI (>85%)")
    elif mean_accuracy > 0.75:
        print("  ✅ Performance BUONE (75-85%)")
    elif mean_accuracy > 0.60:
        print("  ⚠️  Performance ACCETTABILI (60-75%)")
    else:
        print("  ❌ Performance SCARSE (<60%)")
    
    # Dettagli per fold
    print("\n📋 DETTAGLI PER FOLD:")
    for result in fold_results:
        print(f"  Fold {result['fold']}: "
              f"Acc={result['val_accuracy']:.4f}, "
              f"Loss={result['val_loss']:.4f}, "
              f"Epochs={result['epochs_trained']}, "
              f"Best={result['best_epoch']}")
    
    # Salva risultati
    results_dir = Path(f"results_cv_original_{attribute}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "cv_results.txt", "w") as f:
        f.write("CROSS-VALIDATION MODELLO ORIGINALE\n")
        f.write("="*70 + "\n\n")
        f.write(f"Attributo: {attribute}\n")
        f.write(f"N. Fold: {n_splits}\n")
        f.write(f"Epoche per fold: {epochs}\n\n")
        f.write(f"Accuratezza Media: {mean_accuracy:.4f} ± {std_accuracy:.4f}\n")
        f.write(f"Loss Media: {mean_loss:.4f} ± {std_loss:.4f}\n\n")
        f.write("Dettagli per fold:\n")
        for result in fold_results:
            f.write(f"  Fold {result['fold']}: Acc={result['val_accuracy']:.4f}, Loss={result['val_loss']:.4f}\n")
    
    print(f"\n📄 Risultati salvati in: {results_dir / 'cv_results.txt'}")
    
    return {
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'mean_loss': mean_loss,
        'std_loss': std_loss,
        'fold_results': fold_results
    }


if __name__ == "__main__":
    print("🚀 CROSS-VALIDATION MODELLO ORIGINALE")
    print("="*70)
    
    attribute = input("➡️ Attributo da testare (es. temperatura): ").strip()
    n_splits = int(input("➡️ Numero di fold (default 5): ").strip() or "5")
    epochs = int(input("➡️ Epoche per fold (default 25): ").strip() or "25")
    
    results = cross_validate_original(attribute, n_splits=n_splits, epochs=epochs)
    
    print("\n🎯 RISULTATO FINALE:")
    print(f"Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    print(f"Loss: {results['mean_loss']:.4f} ± {results['std_loss']:.4f}")
