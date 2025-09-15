"""Allenamento di un ViT su un attributo specifico utilizzando il dataset su Hugging Face.

Passaggi principali:
1. Download del dataset ``Nardellar/Esperimenti``.
2. Creazione di ``esperimenti.csv`` tramite :func:`csv_config.create_csv` se assente.
3. Associazione di ogni immagine al valore dell'attributo scelto.
4. Preparazione dei ``tf.data.Dataset`` per il fine‑tuning del modello ViT.

Esempio:
    python -m Classificazione.ViT.vit_from_hf_attribute --attribute temperatura

oppure senza argomenti per usare l'attributo predefinito ``temperatura``
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import tensorflow as tf
from datasets import load_dataset
# MODIFICA 1: Aggiungiamo l'import di TFViTModel
from transformers import AutoImageProcessor, TFViTModel

from common import csv_config


# MODIFICA 2: Aggiungiamo una classe personalizzata per il modello
class ViTForCustomClassification(tf.keras.Model):
    """
    Questa classe aggira il bug della libreria caricando il corpo pre-allenato
    di ViT e aggiungendo un nuovo classificatore della dimensione corretta.
    """
    def __init__(self, num_labels, **kwargs):
        super().__init__(**kwargs)
        # Carichiamo solo il "corpo" del modello ViT, senza il classificatore originale
        self.vit = TFViTModel.from_pretrained(
            "google/vit-base-patch16-224",
            from_pt=True,
        )
        # Creiamo un nuovo classificatore con il numero corretto di etichette
        self.classifier = tf.keras.layers.Dense(num_labels, name="classifier")

    def call(self, pixel_values):
        # Definiamo come i dati fluiscono attraverso i livelli
        outputs = self.vit(pixel_values)
        # Usiamo l'output aggregato che rappresenta l'intera immagine
        pooled_output = outputs.pooler_output
        # Lo passiamo al nostro nuovo classificatore per ottenere i risultati (logits)
        logits = self.classifier(pooled_output)
        return {"logits": logits} # Restituiamo un dizionario come fa il modello originale

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine‑tuning ViT su un attributo")
    parser.add_argument(
        "--attribute",
        default="temperatura",
        help="Nome dell'attributo nel CSV (default: temperatura)",
    )
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
    unique_attributes = sorted(list(df[attribute].unique()))
    label2id = {label: i for i, label in enumerate(unique_attributes)}
    num_classes = len(unique_attributes)
    attr_map = df.set_index("ID")[attribute].to_dict()

    ds = load_dataset("Nardellar/Esperimenti", split="train")

    def add_attribute(example):
        class_name = ds.features["label"].int2str(example["label"])
        raw_attribute_value = attr_map.get(class_name, -1)
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

    # Non abbiamo più bisogno di restituire unique_attributes
    return train_tf, val_tf, num_classes

def main() -> None:
    args = parse_args()

    train_ds, val_ds, num_classes = prepare_dataset(args.attribute, args.batch_size)

    # MODIFICA 3: Usiamo la nostra nuova classe invece di quella difettosa
    model = ViTForCustomClassification(num_labels=num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    # Il salvataggio con `save_pretrained` non è compatibile con modelli Keras personalizzati.
    # Usiamo il metodo di salvataggio standard di Keras.
    save_path = f"vit_from_hf_{args.attribute}_keras"
    model.save(save_path)
    print(f"\n✅ Modello salvato in formato Keras nella cartella: {save_path}")


if __name__ == "__main__":
    main()