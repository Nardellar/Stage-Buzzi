"""
Script per rigenerare SOLO il validation_test_set senza training.
Usa la stessa logica del training ma si ferma dopo aver salvato il dataset.
"""
import argparse
from pathlib import Path
import pandas as pd
from datasets import load_dataset, ClassLabel

from ..common import csv_config

def regenerate_validation_set_only(attribute: str):
    """Rigenera solo il validation set, senza training."""
    print("=" * 60)
    print("RIGENERAZIONE VALIDATION SET (senza training)")
    print("=" * 60)
    
    print(f"\nAttributo: {attribute}")
    
    # Stessa logica del training
    root_dir = Path(__file__).resolve().parents[2]
    csv_path = root_dir / "esperimenti.csv"
    if not csv_path.exists():
        csv_config.create_csv(csv_path)
    df = pd.read_csv(csv_path)
    
    if attribute not in df.columns:
        raise ValueError(f"L'attributo '{attribute}' non esiste nel CSV. Colonne disponibili: {list(df.columns)}")

    unique_attributes = sorted([str(attr) for attr in df[attribute].dropna().unique()])
    label2id = {label: i for i, label in enumerate(unique_attributes)}
    attr_map = {k: str(v) for k, v in df.set_index("ID")[attribute].to_dict().items()}
    
    print("Caricamento dataset da HuggingFace...")
    ds = load_dataset("Nardellar/Esperimenti", split="train")
    
    def add_attribute(example):
        class_name = ds.features["label"].int2str(example["label"])
        raw_attribute_value = attr_map.get(class_name, -1)
        example["attribute"] = label2id.get(raw_attribute_value, -1)
        return example
    
    print("Filtraggio dataset...")
    ds = ds.map(add_attribute).filter(lambda ex: ex["attribute"] != -1)
    ds = ds.cast_column('attribute', ClassLabel(names=unique_attributes))
    
    print("\nDivisione dataset (80% Train, 20% Validation)...")
    ds_split = ds.train_test_split(test_size=0.2, seed=42, stratify_by_column="attribute")
    train_ds, val_ds = ds_split["train"], ds_split["test"]
    
    print(f"Train set: {len(train_ds)} immagini")
    print(f"Validation set: {len(val_ds)} immagini")
    
    # Salva SOLO il validation set
    print("\nSalvataggio validation set...")
    val_ds.save_to_disk("validation_test_set")
    
    print("\n" + "=" * 60)
    print("[OK] VALIDATION SET RIGENERATO!")
    print("=" * 60)
    print(f"Salvato in: validation_test_set/")
    print(f"Campioni: {len(val_ds)}")
    print(f"\nIl tuo modello addestrato e' ancora intatto.")
    print("\nOra puoi eseguire:")
    print("  python evaluate_model_pytorch.py")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rigenera il validation set per un attributo specifico.")
    parser.add_argument(
        "attribute",
        nargs="?",
        help="Nome dell'attributo da usare (es. 'temperatura'). Se omesso viene richiesto da input.",
    )
    args = parser.parse_args()

    attribute = args.attribute
    if not attribute:
        attribute = input("Inserisci l'attributo da utilizzare (default: temperatura): ").strip() or "temperatura"

    regenerate_validation_set_only(attribute)

