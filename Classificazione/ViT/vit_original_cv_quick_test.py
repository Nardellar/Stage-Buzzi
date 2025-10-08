"""
Test RAPIDO della Cross-Validation per il modello ORIGINALE
Usa solo 3 fold e 5 epoche per avere risultati velocemente
"""
from __future__ import annotations

from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from transformers import AutoImageProcessor, TFViTModel
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from common import csv_config


# Classe del modello ORIGINALE
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


def quick_cv_test(attribute: str = "temperatura", n_splits=3, epochs=5):
    """
    Test RAPIDO di cross-validation con il modello ORIGINALE
    """
    
    print("⚡ QUICK TEST CROSS-VALIDATION MODELLO ORIGINALE")
    print(f"📊 Attributo: {attribute}")
    print(f"📊 Configurazione: {n_splits} fold, {epochs} epoche per fold")
    print("⚠️  Questo è un test rapido, non una valutazione completa!")
    print("=" * 70)
    
    # Prepara dataset
    root_dir = Path(__file__).resolve().parents[2]
    csv_path = root_dir / "esperimenti.csv"
    
    if not csv_path.exists():
        csv_config.create_csv(csv_path)
    
    df = pd.read_csv(csv_path)
    
    # Carica dataset
    print("📥 Caricamento dataset...")
    ds = load_dataset("Nardellar/Esperimenti", split="train")
    exp_id2name = {i: name for i, name in enumerate(ds.features["label"].names)}
    
    # Mappa attributi
    unique_attributes = sorted(list(df[attribute].unique()))
    label2id = {label: i for i, label in enumerate(unique_attributes)}
    id2label = {i: label for label, i in label2id.items()}
    attr_map = df.set_index("ID")[attribute].to_dict()
    
    # Prepara dati
    images = []
    labels = []
    
    for example in ds:
        exp_name = exp_id2name[example["label"]]
        attr_value = attr_map.get(exp_name, -1)
        if attr_value != -1:
            images.append(example["image"])
            labels.append(label2id[attr_value])
    
    print(f"✅ Dataset: {len(images)} immagini, {len(unique_attributes)} classi")
    
    # Processor
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_accuracies = []
    fold_losses = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
        print(f"\n{'='*70}")
        print(f"🔄 FOLD {fold + 1}/{n_splits}")
        print(f"{'='*70}")
        
        # Prepara dati
        train_images = [images[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_images = [images[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        
        # Processa
        train_processed = processor(images=train_images, return_tensors="tf")
        val_processed = processor(images=val_images, return_tensors="tf")
        
        # Crea dataset
        train_ds = tf.data.Dataset.from_tensor_slices((
            {"pixel_values": train_processed["pixel_values"]},
            tf.convert_to_tensor(train_labels, dtype=tf.int32)
        )).batch(16).shuffle(1000, seed=42)
        
        val_ds = tf.data.Dataset.from_tensor_slices((
            {"pixel_values": val_processed["pixel_values"]},
            tf.convert_to_tensor(val_labels, dtype=tf.int32)
        )).batch(16)
        
        # Crea modello
        model = ViTForCustomClassification(num_labels=len(unique_attributes))
        
        # Compila con parametri ORIGINALI
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )
        
        # Callback
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=0
            )
        ]
        
        # Addestra
        print(f"  🏋️  Addestramento...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=2
        )
        
        # Valuta
        val_results = model.evaluate(val_ds, verbose=0)
        val_loss, val_accuracy = val_results[0], val_results[1]
        
        fold_accuracies.append(val_accuracy)
        fold_losses.append(val_loss)
        
        print(f"\n  ✅ Fold {fold + 1}: Acc={val_accuracy:.4f}, Loss={val_loss:.4f}")
    
    # Risultati
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    mean_loss = np.mean(fold_losses)
    std_loss = np.std(fold_losses)
    
    print("\n" + "="*70)
    print("⚡ RISULTATI QUICK TEST (MODELLO ORIGINALE)")
    print("="*70)
    print(f"🎯 Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"📉 Loss: {mean_loss:.4f} ± {std_loss:.4f}")
    print(f"📊 Fold accuracies: {[f'{acc:.4f}' for acc in fold_accuracies]}")
    print("-"*70)
    
    if std_acc < 0.05:
        print("✅ STABILE: Deviazione standard < 5%")
    elif std_acc < 0.10:
        print("⚠️  MODERATAMENTE STABILE: Deviazione standard 5-10%")
    else:
        print("❌ INSTABILE: Deviazione standard > 10%")
    
    if mean_acc > 0.85:
        print("🎉 ECCELLENTE: Accuracy > 85%")
    elif mean_acc > 0.75:
        print("✅ BUONO: Accuracy 75-85%")
    else:
        print("⚠️  DA MIGLIORARE: Accuracy < 75%")
    
    print("\n💡 NOTA: Questo è un test rapido. Per risultati definitivi,")
    print("   esegui la CV completa con 5 fold e 25 epoche.")
    
    return {
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'fold_accuracies': fold_accuracies
    }


if __name__ == "__main__":
    print("⚡ QUICK TEST - Cross-Validation Veloce")
    print("="*70)
    
    results = quick_cv_test(attribute="temperatura", n_splits=3, epochs=5)
    
    print(f"\n🎯 RISULTATO FINALE:")
    print(f"   Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
