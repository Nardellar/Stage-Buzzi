"""
Script per l'addestramento del modello ViT con PyTorch.
- Divide il dataset in Train (80%) e Validation (20%).
- Salva il validation set su file per essere usato come test set separato.
- Esegue il training e salva il modello migliore e gli artefatti.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter
from datetime import datetime
import sys

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset, ClassLabel
from transformers import ViTForImageClassification, AutoImageProcessor, default_data_collator
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))
from common import csv_config


# --- CLASSE DEL MODELLO ---
class ViTForCustomClassification(nn.Module):
    """
    Modello ViT personalizzato per classificazione.
    Usa ViT pretrained di HuggingFace con head personalizzato.
    """
    def __init__(self, num_labels, dropout_rate=0.3):
        super().__init__()
        # Carica il ViT pretrained (nativo PyTorch)
        self.vit = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        
        # Congela il ViT base (solo il classificatore sarà addestrato)
        for param in self.vit.vit.parameters():
            param.requires_grad = False
        
        # Sostituisci il classificatore con uno personalizzato
        hidden_size = self.vit.config.hidden_size
        self.vit.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_size // 2, num_labels)
        )
        
    def forward(self, pixel_values, labels=None):
        return self.vit(pixel_values=pixel_values, labels=labels)


# --- FUNZIONI DI PREPARAZIONE DEL DATASET ---
def prepare_and_split_dataset(attribute: str, batch_size: int):
    """Prepara e divide il dataset in train e validation."""
    root_dir = Path(__file__).resolve().parents[2]
    csv_path = root_dir / "esperimenti.csv"
    if not csv_path.exists():
        csv_config.create_csv(csv_path)
    df = pd.read_csv(csv_path)

    # Crea mappatura label
    unique_attributes = sorted([str(attr) for attr in df[attribute].unique()])
    label2id = {label: i for i, label in enumerate(unique_attributes)}
    id2label = {i: label for label, i in label2id.items()}
    num_classes = len(unique_attributes)

    attr_map = {k: str(v) for k, v in df.set_index("ID")[attribute].to_dict().items()}

    # Carica dataset
    ds = load_dataset("Nardellar/Esperimenti", split="train")

    def add_attribute(example):
        class_name = ds.features["label"].int2str(example["label"])
        raw_attribute_value = attr_map.get(class_name, -1)
        example["attribute"] = label2id.get(raw_attribute_value, -1)
        return example

    ds = ds.map(add_attribute).filter(lambda ex: ex["attribute"] != -1)
    ds = ds.cast_column('attribute', ClassLabel(names=unique_attributes))

    print("\n--- Divisione del dataset (80% Train, 20% Validation) ---")
    ds_split = ds.train_test_split(test_size=0.2, seed=42, stratify_by_column="attribute")
    train_ds, val_ds = ds_split["train"], ds_split["test"]

    print(f"Train set: {len(train_ds)} immagini")
    print(f"Validation set: {len(val_ds)} immagini")

    # Salva il validation set
    val_ds.save_to_disk("validation_test_set")
    print("Validation set salvato in 'validation_test_set/'")

    # Calcola class weights
    class_counts = Counter(train_ds["attribute"])
    total_samples = sum(class_counts.values())
    class_weights = {
        class_id: total_samples / (num_classes * count) 
        for class_id, count in class_counts.items()
    }
    class_weights_tensor = torch.tensor([class_weights[i] for i in range(num_classes)])

    # Preprocessing - usa il metodo HuggingFace standard con set_transform
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

    def transform(examples):
        """
        Trasformazione applicata on-the-fly dal dataset HuggingFace.
        Best practice: converte esplicitamente in RGB e usa il processor.
        """
        # Converti tutte le immagini in RGB (gestisce anche immagini in scala di grigi)
        images = [img.convert("RGB") for img in examples["image"]]
        
        # Preprocessa con l'AutoImageProcessor (fa resize, normalizzazione, ecc.)
        inputs = processor(images, return_tensors="pt")
        
        # Aggiungi le labels
        inputs["labels"] = examples["attribute"]
        return inputs

    # Set transform viene applicato on-the-fly quando il dataset viene iterato
    # Questo è il metodo HuggingFace standard per preprocessing lazy
    train_ds.set_transform(transform)
    val_ds.set_transform(transform)

    return train_ds, val_ds, num_classes, id2label, class_weights_tensor


# --- FUNZIONE DI TRAINING ---
def train_model(model, train_ds, val_ds, class_weights, num_epochs, results_dir, attribute):
    """Addestra il modello."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo: {device}")
    model = model.to(device)
    class_weights = class_weights.to(device)

    # Optimizer con weight decay (L2 regularization)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=5e-5, 
        weight_decay=1e-4  # L2 regularization equivalente a 0.0001
    )
    
    # Loss con label smoothing per ridurre overconfidence e migliorare generalizzazione
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,  # Class balancing
        label_smoothing=0.1    # Label smoothing (10%)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7
    )

    # DataLoaders con collator HuggingFace per gestire i batch correttamente
    train_loader = DataLoader(
        train_ds, 
        batch_size=16, 
        shuffle=True, 
        collate_fn=default_data_collator,
        num_workers=0  # 0 per compatibilità Windows
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=16, 
        shuffle=False, 
        collate_fn=default_data_collator,
        num_workers=0
    )

    # Training loop
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = results_dir / f"best_model_{attribute}_{timestamp}.pth"

    print("\n--- Inizio addestramento ---")
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            
            # Gradient clipping per stabilità
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            train_loss += loss.item()
            predictions = outputs.logits.argmax(dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

                val_loss += loss.item()
                predictions = outputs.logits.argmax(dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        val_loss /= len(val_loader)

        # Logging
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping e salvataggio
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, model_path)
            print(f"  Modello salvato! (val_loss improved to {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    return model_path


# --- MAIN ---
def main():
    print("ADDESTRAMENTO ViT con PyTorch")
    import sys
    # Usa l'attributo da argomenti della linea di comando o default a temperatura
    attribute = sys.argv[1] if len(sys.argv) > 1 else "temperatura"
    print(f"Attributo selezionato: {attribute}")

    results_dir = Path(f"training_results_{attribute}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Prepara dataset
    train_ds, val_ds, num_classes, id2label, class_weights = prepare_and_split_dataset(attribute, batch_size=16)

    # Crea modello
    model = ViTForCustomClassification(num_labels=num_classes)

    # Training
    model_path = train_model(model, train_ds, val_ds, class_weights, num_epochs=100, 
                             results_dir=results_dir, attribute=attribute)

    # Salva artifacts
    artifacts = {
        'attribute': attribute,
        'id2label': {int(k): v for k, v in id2label.items()},
        'model_path': str(model_path),
        'num_classes': num_classes
    }
    with open(results_dir / "artifacts.json", "w") as f:
        json.dump(artifacts, f, indent=4)

    print("\nAddestramento completato!")
    print(f"Modello salvato in: {model_path}")
    print(f"Artifacts salvati in: {results_dir / 'artifacts.json'}")


if __name__ == "__main__":
    main()

