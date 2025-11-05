"""
Valuta un modello ViT addestrato con HuggingFace Trainer.
- Carica modello e image processor dalla cartella di training
- Carica il validation set salvato in ``validation_test_set/``
- Calcola metriche e genera confusion matrix e mappe di attenzione
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import Dataset, load_dataset, ClassLabel
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import (
    AutoImageProcessor,
    default_data_collator,
)
from safetensors.torch import load_file

from dataset_utils import (
    add_class_id_column,
    build_vit_batch_preprocessor,
    filter_missing_class,
    load_attribute_metadata,
)
from train_model_pytorch import build_model  


def regenerate_validation_split(attribute: str):
    """Rigenera lo stesso validation set stratificato usato durante il training."""
    classes, class_to_id, _, experiment_to_class = load_attribute_metadata(attribute)
    dataset = load_dataset("Nardellar/Esperimenti", split="train")
    # Per ogni immagine associa l'ID classe.
    dataset = add_class_id_column(dataset, class_to_id, experiment_to_class)
    dataset = filter_missing_class(dataset, class_field="class_id")
    # Trasforma la colonna class_id in un oggetto ClassLabel per gestire correttamente la stratificazione.
    dataset = dataset.cast_column("class_id", ClassLabel(names=classes))

    #divido il dataset in train 80% e validation 20%, stratificando per class_id esattamente come il training
    split = dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column="class_id")
    #salvo il validation set
    split["test"].save_to_disk("validation_test_set")
    print("Validation split rigenerato e salvato in 'validation_test_set/'.")


def load_validation_set(batch_size: int, processor: AutoImageProcessor, use_grayscale: bool) -> DataLoader:
    """Carica il validation set salvato in ``validation_test_set/`` e applica le trasformazioni."""
    dataset = Dataset.load_from_disk("validation_test_set")

    transform = build_vit_batch_preprocessor(processor, use_grayscale, label_field="class_id")
    #applica le trasformazioni al dataset
    dataset.set_transform(transform)
    #creiamo il dataloader per usarlo poi nella fase di valutazione
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
        num_workers=0,
    )


def load_trained_model(model_dir: Path, num_labels: int, device: torch.device) -> torch.nn.Module:
    """
    Ricrea il modello con la stessa architettura usata in training e carica i pesi salvati.
    """
    #ricreo il modello (con classificatore non addestrato)
    model = build_model(num_labels=num_labels)

    #definisco i possibili percorsi dei pesi del modello
    weight_paths = [
        model_dir / "model.safetensors",
        model_dir / "pytorch_model.bin",
    ]

    #cerco il tipo di peso del modello e lo carico in model_weights 
    model_weights = None
    for path in weight_paths:
        if path.exists():
            if path.suffix == ".safetensors":
                model_weights = load_file(str(path))
            else:
                model_weights = torch.load(path, map_location="cpu")
            break

    if model_weights is None:
        raise FileNotFoundError(
            f"Nessun file di pesi trovato in {model_dir}. Attesi 'model.safetensors' o 'pytorch_model.bin'."
        )

    #carico i pesi sul modello
    model.load_state_dict(model_weights)
    #sposto il modello sul device specificato (serve per spostarlo in GPU se disponibile)
    model.to(device)
    #restituisco il modello pronto per l'inferenza
    return model


def evaluate(model, dataloader, device, id_to_class: Dict[int, str]):
    """Valuta il modello su un dataset e calcola le metriche di valutazione."""
    #imposto il modello in modalita' di valutazione (disattivo dropout e batch normalization)
    model.eval()
    predictions, true_class_ids = [], []
    # non servono gradienti durante l'inferenza
    with torch.no_grad():
        #tqdm = mostra barra di progressione su terminale
        for batch in tqdm(dataloader, desc="Evaluating"):
            #sposto i pixel values e i class_id veri sul device (GPU se disponibile)
            pixel_values = batch["pixel_values"].to(device)
            true_class_ids_batch = batch["labels"].to(device)
            #calcolo predizioni
            outputs = model(pixel_values=pixel_values)
            # prendo la classe con probabilita' piu' alta per ogni predizione
            preds = outputs.logits.argmax(dim=-1)
            #aggiugniamo le predizioni e i class_id veri (ground truth) alla lista
            #Scikit-learn richiede array numpy (supportati solo su cpu) percio' spostiamo predizioni e class_id e le convertiamo in numpy
            predictions.extend(preds.cpu().numpy())
            true_class_ids.extend(true_class_ids_batch.cpu().numpy())

    class_values = [id_to_class[i] for i in range(len(id_to_class))]
    #genero il report json
    report = classification_report(true_class_ids, predictions, target_names=class_values, output_dict=True)

    #genero la matrice di confusione
    cm = confusion_matrix(true_class_ids, predictions)
    #restituisco l'aaray numpy con le predizioni, array numpy con i class_id veri, il report e la matrice di confusione
    return np.asarray(predictions), np.asarray(true_class_ids), report, cm


def save_confusion_matrix(cm: np.ndarray, class_names: list[str], output_path: Path):
    """Salva la matrice di confusione in un file png."""
    #inizializzo lo spazio grafico
    plt.figure(figsize=(8, 6))
    #crea una mappa di calore colorata della matrice (blu piu' scuro = valore piu' alto)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    #salvo la matrice di confusione in un file png
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_classification_report(report: Dict, output_path: Path):
    """Scrive in un file json il report"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Valuta un modello ViT salvato con Trainer.")
    #la cartella in cui cercare il modello
    parser.add_argument(
        "model_dir",
        nargs="?",
        default="training_results_temperatura",
        help="Cartella del modello salvato (default: training_results_temperatura).",
    )
    #di che dimensione avere il batch
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per l'inferenza (default: 16).")
    args = parser.parse_args()

    model_path = Path(args.model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Cartella modello non trovata: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    print(f"Modello: {model_path}")

    artifacts_path = model_path / "artifacts.json"

    # Verifica che artifacts.json esista
    if not artifacts_path.exists():
        raise FileNotFoundError(
            f"artifacts.json non trovato in {model_path}. "
            f"Questo file è necessario per l'evaluation. "
            f"Assicurati di aver addestrato il modello con train_model_pytorch.py."
        )

    # Carica artifacts.json
    with open(artifacts_path, encoding="utf-8") as f:
        training_artifacts = json.load(f)

    # Valida che tutti i campi necessari siano presenti
    missing_fields = []
    
   
    if "use_grayscale" not in training_artifacts:
        missing_fields.append("use_grayscale")
        use_grayscale = None
    else:
        use_grayscale = training_artifacts["use_grayscale"]

    
    if "attribute" not in training_artifacts:
        missing_fields.append("attribute")
        attribute = None
    else:
        attribute = training_artifacts["attribute"]

   # Ottengo un dizionario {str : str} ID -> classe
    raw_id_to_class = training_artifacts.get("id_to_class")
    if not raw_id_to_class:
        missing_fields.append("id_to_class")
        id_to_class = None
    else:
        # Converto le chiavi del dizionario in int per poter essere indicizzate dopo
        id_to_class = {int(k): v for k, v in raw_id_to_class.items()}

    # Stesso di prima ma con dizionario invertito classe -> ID
    raw_class_to_id = training_artifacts.get("class_to_id")
    if not raw_class_to_id:
        missing_fields.append("class_to_id")
        class_to_id = None
    else:
        class_to_id = {str(k): int(v) for k, v in raw_class_to_id.items()}

    # Se mancano campi, genera errore
    if missing_fields:
        raise ValueError(
            f"artifacts.json incompleto. Campi mancanti: {', '.join(missing_fields)}. "
            f"Non è possibile valutare il modello senza questi dati. "
            f"Rigenera il modello con train_model_pytorch.py."
        )

    print(f"Caricato da artifacts.json:")
    print(f"  - attribute: {attribute}")
    print(f"  - use_grayscale: {use_grayscale}")
    print(f"  - num_classes: {len(id_to_class)}")

    #carico lo stesso preprocessor del training
    processor = AutoImageProcessor.from_pretrained(model_path)
    #carico il modello con la stessa architettura usata in training e carico i pesi salvati
    model = load_trained_model(model_path, num_labels=len(id_to_class), device=device)
    #imposto i dizionari id2label e label2id del modello con i dizionari ottenuti da artifacts.json
    model.config.id2label = id_to_class
    model.config.label2id = class_to_id

    # 3) Rigeneriamo lo stesso validation set stratificato usato in training.
    regenerate_validation_split(attribute)
    # 4) Applichiamo le trasformazioni e creiamo il DataLoader per l'inferenza.
    dataloader = load_validation_set(
        batch_size=args.batch_size, processor=processor, use_grayscale=use_grayscale
    )

    # 5) Valutazione finale con metriche classiche e salvataggio degli artefatti.
    predictions, true_class_ids, report, cm = evaluate(model, dataloader, device, id_to_class)
    accuracy = report["accuracy"]
    f1_macro = report["macro avg"]["f1-score"]
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")

    output_dir = model_path / "evaluation"
    output_dir.mkdir(exist_ok=True)

    save_classification_report(report, output_dir / "classification_report.json")
    save_confusion_matrix(cm, [id_to_class[i] for i in range(len(id_to_class))], output_dir / "confusion_matrix.png")

    print(f"Report salvato in: {output_dir / 'classification_report.json'}")
    print(f"Confusion matrix salvata in: {output_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
