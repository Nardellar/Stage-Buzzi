"""
Script per la valutazione del modello ViT con PyTorch.
- Carica il modello addestrato
- Carica il validation/test set
- Calcola metriche e genera visualizzazioni
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, ClassLabel
from transformers import AutoImageProcessor, default_data_collator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[2]))
from common import csv_config  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent))
from manual_train_model_pytorch import ViTForCustomClassification


def regenerate_validation_split(attribute: str):
    root_dir = Path(__file__).resolve().parents[2]
    csv_path = root_dir / "esperimenti.csv"
    if not csv_path.exists():
        csv_config.create_csv(csv_path)
    df = pd.read_csv(csv_path)

    if attribute not in df.columns:
        raise ValueError(f"Attributo '{attribute}' non presente nel CSV.")

    unique_attributes = sorted(df[attribute].dropna().astype(str).unique())
    label2id = {label: idx for idx, label in enumerate(unique_attributes)}

    dataset = load_dataset("Nardellar/Esperimenti", split="train")

    attr_map = df.set_index("ID")[attribute].dropna().astype(str).to_dict()

    def add_attribute(example):
        class_name = dataset.features["label"].int2str(example["label"])
        example["attribute"] = label2id.get(attr_map.get(class_name, None), -1)
        return example

    dataset = dataset.map(add_attribute).filter(lambda ex: ex["attribute"] != -1)
    dataset = dataset.cast_column("attribute", ClassLabel(names=unique_attributes))

    split = dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column="attribute")
    val_ds = split["test"]
    val_ds.save_to_disk("validation_test_set")
    print("Validation split rigenerato e salvato in 'validation_test_set/'.")


def load_test_data(batch_size: int, use_grayscale: bool):
    """Carica il test dataset."""
    try:
        # Prova prima con load_from_disk (formato HuggingFace standard)
        from datasets import Dataset
        test_ds = Dataset.load_from_disk("validation_test_set")
        print(f"Dataset caricato: {len(test_ds)} campioni")
    except Exception as e:
        print(f"Errore nel caricamento con load_from_disk: {e}")
        try:
            # Fallback: prova con load_dataset arrow
            test_ds = load_dataset("arrow", data_dir="validation_test_set", split="train")
            print(f"Dataset caricato (arrow): {len(test_ds)} campioni")
        except Exception as e2:
            print(f"Errore anche con arrow: {e2}")
            print("\nIl validation_test_set potrebbe essere corrotto.")
            print("Soluzione: Riesegui il training per rigenerarlo.")
            return None

    # Preprocessing - IDENTICO al training per consistenza
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

    def transform(examples):
        def crop(image):
            width, height = image.size
            crop_height = min(950, height)
            if crop_height == height:
                return image
            return image.crop((0, 0, width, crop_height))

        def maybe_grayscale(image):
            if not use_grayscale:
                return image.convert("RGB")
            gray = image.convert("L")
            return Image.merge("RGB", (gray, gray, gray))

        """
        Trasformazione applicata on-the-fly dal dataset HuggingFace.
        DEVE essere IDENTICA a quella del training per risultati consistenti.
        """
        images = [maybe_grayscale(crop(img)) for img in examples["image"]]

        inputs = processor(images, return_tensors="pt")
        inputs["labels"] = examples["attribute"]
        return inputs

    # Set transform viene applicato on-the-fly quando il dataset viene iterato
    test_ds.set_transform(transform)
    
    # DataLoader con collator HuggingFace per gestire correttamente i batch
    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=default_data_collator,  # Collator HuggingFace per batching automatico
        num_workers=0  # 0 per compatibilità Windows
    )

    return test_loader


def evaluate_model(model, test_loader, id2label, results_dir, attribute, device):
    """Valuta il modello sul test set."""
    print("\nValutazione del modello sul Test Set...")
    model.eval()
    
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values)
            predictions = outputs.logits.argmax(dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification report
    class_names = [id2label[str(i)] for i in range(len(id2label))]
    report = classification_report(all_labels, all_predictions, 
                                   target_names=class_names, output_dict=True)
    
    report_path = results_dir / f"classification_report_{attribute}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"Accuracy sul Test Set: {report['accuracy']:.4f}")
    print(f"Report salvato in: {report_path}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Matrice di Confusione - {attribute}')
    plt.xlabel('Predizioni')
    plt.ylabel('Valori Reali')
    
    cm_path = results_dir / f"confusion_matrix_{attribute}.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix salvata in: {cm_path}")

    return report['accuracy']


def visualize_attention_maps(model, test_loader, id2label, results_dir, attribute, device, num_images=8):
    """Genera mappe di attenzione."""
    print("\nGenerazione Mappe di Attenzione...")
    model.eval()
    
    # Forza l'implementazione "eager" per supportare output_attentions
    original_attn_implementation = None
    if hasattr(model.vit.vit.config, '_attn_implementation'):
        original_attn_implementation = model.vit.vit.config._attn_implementation
        model.vit.vit.config._attn_implementation = "eager"
    
    # Prendi un batch
    batch = next(iter(test_loader))
    pixel_values = batch["pixel_values"][:num_images].to(device)
    labels = batch["labels"][:num_images]

    try:
        with torch.no_grad():
            outputs = model.vit.vit(pixel_values, output_attentions=True)
            
            # Verifica che le attention siano state restituite
            if outputs.attentions is None or len(outputs.attentions) == 0:
                print("Attenzione: il modello non ha restituito attention maps.")
                print("Questo puo' accadere con alcune implementazioni ottimizzate.")
                return
                
            attentions = outputs.attentions[-1]  # Ultimo layer
            
            # Media delle attention heads
            avg_attentions = attentions.mean(dim=1)  # [batch, seq_len, seq_len]
            cls_attentions = avg_attentions[:, 0, 1:]  # CLS token attention ai patch

        # Predizioni
        with torch.no_grad():
            pred_outputs = model(pixel_values=pixel_values)
            predictions = pred_outputs.logits.argmax(dim=1)

        # Visualizzazione
        plt.figure(figsize=(20, 10))
        for idx in range(min(num_images, len(pixel_values))):
            # Immagine originale
            img = pixel_values[idx].cpu().permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min())

            # Attention map
            attention = cls_attentions[idx].cpu().numpy()
            num_patches = int(np.sqrt(len(attention)))
            attention_map = attention.reshape(num_patches, num_patches)

            # Resize attention map alla dimensione dell'immagine
            from scipy.ndimage import zoom
            scale = img.shape[0] / attention_map.shape[0]
            attention_resized = zoom(attention_map, scale, order=1)

            # Plot
            plt.subplot(2, num_images // 2, idx + 1)
            plt.imshow(img)
            plt.imshow(attention_resized, cmap='jet', alpha=0.5)
            
            true_label = id2label[str(labels[idx].item())]
            pred_label = id2label[str(predictions[idx].item())]
            color = "green" if true_label == pred_label else "red"
            plt.title(f"Vero: {true_label}\nPredetto: {pred_label}", color=color)
            plt.axis("off")

        map_path = results_dir / f"attention_maps_{attribute}.png"
        plt.tight_layout()
        plt.savefig(map_path)
        plt.close()
        print(f"Attention maps salvate in: {map_path}")
        
    except Exception as e:
        print(f"Errore durante la generazione delle attention maps: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Ripristina sempre l'implementazione originale
        if original_attn_implementation is not None:
            model.vit.vit.config._attn_implementation = original_attn_implementation


def main():
    print("VALUTAZIONE MODELLO ViT PyTorch")

    # Trova le sessioni di training
    training_dirs = [d for d in Path(".").iterdir() 
                    if d.is_dir() and d.name.startswith("training_results_")]
    
    if not training_dirs:
        print("Errore: Nessuna cartella 'training_results_*' trovata.")
        return

    print("Seleziona la sessione di training da valutare:")
    for i, d in enumerate(training_dirs):
        print(f"  [{i}] - {d.name}")

    choice = int(input("Inserisci il numero: ").strip())
    training_dir = Path("training_results_temperatura/manuale_senza_grigi") #training_dirs[choice]
    
    # Carica artifacts
    artifacts_path = training_dir / "artifacts.json"
    with open(artifacts_path) as f:
        artifacts = json.load(f)

    attribute = artifacts['attribute']
    id2label = artifacts['id2label']
    model_path = artifacts['model_path']
    num_classes = artifacts['num_classes']

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo: {device}")

    # Carica modello
    print(f"\nCaricamento modello da: {model_path}")
    model = ViTForCustomClassification(num_labels=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print("Modello caricato con successo!")

    # Carica test data
    use_grayscale = artifacts.get('use_grayscale', False)
    print(f"Uso scala di grigi (dati di valutazione): {use_grayscale}")
    regenerate_validation_split(attribute)
    test_loader = load_test_data(batch_size=16, use_grayscale=use_grayscale)
    if test_loader is None:
        return

    # Crea cartella risultati
    eval_results_dir = Path(f"evaluation_results_{attribute}")
    eval_results_dir.mkdir(parents=True, exist_ok=True)

    # Valutazione
    accuracy = evaluate_model(model, test_loader, id2label, eval_results_dir, attribute, device)
    
    # Attention maps
    visualize_attention_maps(model, test_loader, id2label, eval_results_dir, attribute, device)

    print("\nValutazione completata!")
    print(f"Risultati salvati in: '{eval_results_dir}'")
    print(f"\nAccuracy finale: {accuracy:.4f}")


if __name__ == "__main__":
    main()
