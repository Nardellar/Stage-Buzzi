from pathlib import Path

import pandas as pd
import tensorflow as tf
from datasets import load_dataset
from transformers import AutoImageProcessor, TFViTForImageClassification

# Analisi dei risultati
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


# ==============================================================================
# Funzione per preparare il dataset (identica alla precedente)
# ==============================================================================
def prepare_dataset(attribute: str, batch_size: int):
    root_dir = Path(__file__).resolve().parents[2]
    csv_path = root_dir / "esperimenti.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"File CSV non trovato in {csv_path}. Assicurati che esista.")

    df = pd.read_csv(csv_path)

    unique_attributes = sorted(list(df[attribute].unique()))
    label2id = {label: i for i, label in enumerate(unique_attributes)}

    attr_map = df.set_index("ID")[attribute].to_dict()

    ds = load_dataset("Nardellar/Esperimenti", split="train")

    def add_attribute(example):
        class_name = ds.features["label"].int2str(example["label"])
        raw_attribute_value = attr_map.get(class_name, -1)
        example["attribute"] = label2id.get(raw_attribute_value, -1)
        return example

    ds = ds.map(add_attribute)
    ds = ds.filter(lambda ex: ex["attribute"] != -1)

    # Usa lo stesso seed per ottenere la stessa suddivisione train/test
    ds = ds.train_test_split(test_size=0.2, seed=42)
    val_ds = ds["test"]  # Ci serve solo il set di validazione

    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

    def transform(batch):
        processed = processor(images=batch["image"], return_tensors="tf")
        batch["pixel_values"] = processed["pixel_values"]
        batch["labels"] = tf.convert_to_tensor(batch["attribute"], dtype=tf.int32)
        return batch

    val_ds = val_ds.map(transform, batched=True, batch_size=batch_size)

    val_tf = val_ds.to_tf_dataset(
        columns=["pixel_values"], label_cols=["labels"], batch_size=batch_size, shuffle=False
    )

    return val_tf, unique_attributes


def main() -> None:
    # --- INIZIO MODIFICA ---
    # 1. Chiedi il nome della cartella del modello
    model_folder_name = input("Inserisci il nome della cartella del modello (es. vit_di_EXP): ")

    # 2. Chiedi il nome della colonna nel file CSV
    attribute_column_name = input("Inserisci il nome della colonna nel file .csv (es. temperatura): ")
    # --- FINE MODIFICA ---

    # Carica il modello usando il nome della cartella
    print(f"--- Caricamento del modello da: {model_folder_name} ---")
    try:
        model = TFViTForImageClassification.from_pretrained(model_folder_name)
    except OSError:
        print(f"ERRORE: La cartella del modello '{model_folder_name}' non è stata trovata.")
        print("Assicurati di aver inserito il nome corretto e che il modello sia stato salvato.")
        return

    # Prepara il dataset usando il nome della colonna
    print("--- Preparazione del set di dati di validazione... ---")
    try:
        val_ds, class_names = prepare_dataset(attribute_column_name, batch_size=16)
    except KeyError:
        print(f"ERRORE: La colonna '{attribute_column_name}' non è stata trovata nel file esperimenti.csv.")
        return

    # Il resto dello script rimane invariato...
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    print("--- Avvio valutazione... ---")
    results = model.evaluate(val_ds)
    print("\n--- Risultati della Valutazione ---")
    print(f"Loss: {results[0]:.4f}")
    print(f"Accuracy: {results[1]:.4f}")

    print("\n--- Generazione analisi dettagliata ---")
    y_pred_logits = model.predict(val_ds).logits
    y_pred = np.argmax(y_pred_logits, axis=1)
    y_true = np.concatenate([y for x, y in val_ds], axis=0)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predetto')
    plt.ylabel('Reale')
    plt.title(f'Matrice di Confusione - {attribute_column_name}')
    plt.savefig(f'evaluation_confusion_matrix_{model_folder_name}.png')
    print(f"Matrice di confusione salvata come 'evaluation_confusion_matrix_{model_folder_name}.png'")

    print("\nReport di Classificazione:")
    print(classification_report(y_true, y_pred, target_names=[str(x) for x in class_names]))
    plt.show()


if __name__ == "__main__":
    main()