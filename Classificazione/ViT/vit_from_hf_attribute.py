"""Allenamento di un ViT su un attributo specifico utilizzando il dataset su Hugging Face.

Passaggi principali:
1. Download del dataset ``Nardellar/Esperimenti``.
2. Creazione di ``esperimenti.csv`` tramite :func:`csv_config.create_csv` se assente.
3. Associazione di ogni immagine al valore dell'attributo scelto.
4. Preparazione dei ``tf.data.Dataset`` per il fine‑tuning del modello ViT.

Esempio:
    python -m Classificazione.ViT.vit_from_hf_attribute --attribute temperatura
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import tensorflow as tf
from datasets import load_dataset
from transformers import AutoImageProcessor, TFViTForImageClassification

from common import csv_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine‑tuning ViT su un attributo")
    parser.add_argument("--attribute", required=True, help="Nome dell'attributo nel CSV")
    parser.add_argument("--epochs", type=int, default=3, help="Numero di epoche di training")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    return parser.parse_args()


def prepare_dataset(attribute: str, batch_size: int):
    root_dir = Path(__file__).resolve().parents[2]
    csv_path = root_dir / "esperimenti.csv"
    if not csv_path.exists():
        print("⚠️ File CSV mancante, creazione in corso...")
        csv_config.create_csv(csv_path)

    df = pd.read_csv(csv_path)

    # --- INIZIO MODIFICA ---
    # 1. Ottieni i valori unici dell'attributo e crea una mappa verso gli indici (0, 1, 2...)
    unique_attributes = sorted(list(df[attribute].unique()))
    label2id = {label: i for i, label in enumerate(unique_attributes)}
    num_classes = len(unique_attributes)
    # --- FINE MODIFICA ---

    attr_map = df.set_index("ID")[attribute].to_dict()

    ds = load_dataset("Nardellar/Esperimenti", split="train")

    def add_attribute(example):
        class_name = ds.features["label"].int2str(example["label"])
        # Ottieni il valore grezzo (es. 1400)
        raw_attribute_value = attr_map.get(class_name, -1)
        # Mappa il valore grezzo al suo indice (es. 1) usando la mappa label2id
        example["attribute"] = label2id.get(raw_attribute_value, -1)
        return example

    ds = ds.map(add_attribute)
    ds = ds.filter(lambda ex: ex["attribute"] != -1)

    ds = ds.train_test_split(test_size=0.2, seed=42)
    train_ds, val_ds = ds["train"], ds["test"]

    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

    def transform(batch):
        processed = processor(images=batch["image"], return_tensors="tf")
        batch["pixel_values"] = processed["pixel_values"]
        batch["labels"] = tf.convert_to_tensor(batch["attribute"], dtype=tf.int32)
        return batch

    train_ds = train_ds.map(transform, batched=True, batch_size=batch_size)
    val_ds = val_ds.map(transform, batched=True, batch_size=batch_size)

    train_tf = train_ds.to_tf_dataset(
        columns=["pixel_values"], label_cols=["labels"], batch_size=batch_size, shuffle=True
    )
    val_tf = val_ds.to_tf_dataset(
        columns=["pixel_values"], label_cols=["labels"], batch_size=batch_size, shuffle=False
    )

    return train_tf, val_tf, num_classes

def main() -> None:
    # --- Inizio Modifiche ---

    # 1. Definisci i parametri direttamente qui
    attribute = "temperatura"
    epochs = 3  # Puoi cambiare anche questo valore se vuoi
    batch_size = 16 # E anche questo

    # 2. La riga seguente non è più necessaria
    # args = parse_args()

    # 3. Passa le variabili definite sopra alle funzioni
    train_ds, val_ds, num_classes = prepare_dataset(attribute, batch_size)

    model = TFViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    model.save_pretrained(f"vit_from_hf{attribute}")

    # --- Fine Modifiche ---


if __name__ == "__main__":
    main()