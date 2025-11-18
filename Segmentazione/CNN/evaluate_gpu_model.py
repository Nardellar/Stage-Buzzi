"""
Valutazione completa del modello GPU-optimized.
Riproduce le stesse trasformazioni del confronto:
 - caricamento dataset
 - inferenza CNN + classificatore
 - raffinamento DenseCRF
 - metriche (accuracy, confusion matrix)
 - salvataggio anteprima con GT e maschera predetta
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional

from data_module import prepare_dataset_splits, load_dataset_stateless
from gpu_optimized_cnn_classifier import GPUOptimizedCNNSegmentationClassifier

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import (
    unary_from_labels,
    create_pairwise_gaussian,
    create_pairwise_bilateral,
)

# Cartella contenente lo script (usata come riferimento per i path relativi).
BASE_DIR = Path(__file__).resolve().parent
# Directory di immagini e maschere da valutare; cambia se il dataset è altrove.
IMAGES_DIR = (BASE_DIR / "../images/Immagini").resolve()
MASKS_DIR = (BASE_DIR / "../images/Maschere").resolve()
# Prefisso dei file salvati dal training ((cnn e decoder).keras + classificatre.pk1).
MODEL_PREFIX = "gpu_optimized_cnn_model"
# Percorso del file PNG con le anteprime (GT vs predizione).
PREVIEW_PATH = BASE_DIR / "gpu_model_preview.png"
# Percorso dell'immagine con la matrice di confusione salvata.
CONFUSION_PATH = BASE_DIR / "gpu_confusion_matrix.png"
#Percorso della cartella contenente gli split del dataset.
SPLIT_DIR = (BASE_DIR / "splits").resolve()
#Nomi delle classi da classificare.
CLASS_NAMES = ["Resina", "Pori/Imperfezioni", "Fase Fusa", "Belite", "Alite"]


def apply_dense_crf(image: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Applica il DenseCRF per raffinare le predizioni del modello.
    Args:
        image: Immagine in formato uint8 (0-255) con dimensioni (altezza, larghezza, 3)
        labels: Maschera predetta dal modello in formato int32 (prima del CRF) con dimensioni (altezza, larghezza) e valori (0-4)
        num_classes: Numero di classi da classificare (5)
    Returns:
        Maschera raffinata in formato int32 (0-4)
    """
    #salvo altezza e larghezza dell'immagine
    height, width = image.shape[:2]
    #creo un oggetto DenseCRF2D con le dimensioni dell'immagine e il numero di classi
    crf = dcrf.DenseCRF2D(width, height, num_classes)

    unary = unary_from_labels(labels.astype(np.int32), num_classes, gt_prob=0.7, zero_unsure=False)
    crf.setUnaryEnergy(unary)

    feats_gaussian = create_pairwise_gaussian(sdims=(3, 3), shape=image.shape[:2])
    crf.addPairwiseEnergy(feats_gaussian, compat=3)

    feats_bilateral = create_pairwise_bilateral(
        sdims=(5, 5),
        schan=(10, 10, 10),
        img=image,
        chdim=2,
    )
    crf.addPairwiseEnergy(feats_bilateral, compat=5)

    Q = crf.inference(5)
    refined = np.argmax(Q, axis=0).reshape((height, width))
    return refined


def evaluate_gpu_model(model, dataset: Dict[str, np.ndarray],) -> Dict:
    """
    Valuta il modello sul set di evaluation.
    Args:
        model: Modello CNN-based addestrato.
        dataset: Dizionario contenente: 
            - images: array numpy con forma (Num_immagini, altezza, larghezza, 3)
            - masks: array numpy con forma (Num_immagini, altezza, larghezza)
            - image_paths: lista dei percorsi delle immagini
    Returns:
        Dizionario contenente:
            - accuracy: accuracy del modello
            - predictions: array numpy con le predizioni del modello
            - ground_truth: array numpy con i ground truth
            - previews: lista di 3 dizionari contenenti ciascuno [immagine, maschera ground truth, maschera predette, path immagine]
    """
    #inizalizzo le liste
    predictions = []      # accumula le predizioni pixel-level (solo pixel validi)
    ground_truth = []     # accumula le label ground truth (solo pixel validi)
    previews = []         # accumula fino a 3 anteprime da usare poi per la visualizzazione


    #per ogni istanza del dataset:
    for i, (img, mask, img_path) in enumerate(
        zip(dataset["images"], dataset["masks"], dataset["image_paths"]), start=1
    ):
        print(f"[GPU] Processando immagine {i}/{len(dataset['images'])}...")
        #salvo le dimensioni dell'immagine. img = (altezza, larghezza, 3)
        image_native_size = (img.shape[0], img.shape[1])
        #controlliamo che il backbone CNN sia configurato per quella dimensione.
        #se le dimensioni sono diverse, viene ricostruito il modello mantenendo i pesi
        model.ensure_feature_extractor_size(image_native_size)
        #convertiamo l'immagine in float32 come richiesto dai modelli Keras (l'imagine e' anche gia' normalizzata in [0,1] da load_dataset_stateless)
        img_prepared = img.astype(np.float32)
        #aggiungiamo una dimensione all'array numPy img_prepared per poterlo passare al feature extractor. img_prepared = (1, altezza, larghezza, 3)
        #calcoliamo le features dell'immagine e otteniamo in output una feature map delle seguenti dimensioni (1, altezza, larghezza, lista features del pixel)
        feature_map = model._feature_extractor.predict(np.expand_dims(img_prepared, axis=0), verbose=0)
        #estraimo le dimensioni della feature map
        height_feature_map, width_feature_map = feature_map.shape[1:3]

        #appiattiamo la feature map in una matrice 2D per il classificatore
        #da (1, altezza, larghezza, lista features del pixel) a (altezza * larghezza, lista features del pixel)
        feature_map_flat = feature_map.reshape(-1, feature_map.shape[-1])
        #eseguiamo la predizione del classificatore su ogni pixel della feature map
        #restituisce un array 1D con la classe predetta di ogni pixel es: [1,3,4,1]
        preds = model.classifier.predict(feature_map_flat)
        #convertiamo in interi e ripristiniamo la forma originale della feature map
        preds = np.asarray(preds, dtype=int).reshape(height_feature_map, width_feature_map)
        #ridimensioniamo la predizione alla dimensione originale dell'immagine
        preds_upsampled = cv2.resize(
            preds,
            (mask.shape[1], mask.shape[0]), #dimensioni target
            interpolation=cv2.INTER_NEAREST, #interpolazione "nearest" per mantenere i valori interi in modo che corrispondano sempre ad una classe (0, 1, 2, 3, 4)
        )

        #convertiamo l'immagine in formato uint8 (0-255) per il DenseCRF
        image_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        #applico il DenseCRF per raffinare le predizioni
        #restituisce numpy array 2D con la classe predetta di ogni pixel
        CRF_pred = apply_dense_crf(image_uint8, preds_upsampled, len(CLASS_NAMES))

        #appiattiamo la maschera in un array 1D e sottraiamo 1 per convertire le classi da [0-5] a [-1-4] 
        gt_flattened_mask = mask.reshape(-1) - 1
        #appiattiamo le predizioni con CRF in un array 1D
        pred_flat = CRF_pred.reshape(-1)
        #creo una maschera booleana che vale True per i pixel che non sono background (-1)
        valid = gt_flattened_mask >= 0
        #se la maschera non ha nessun pixel valido, salto il batch
        if not np.any(valid):
            continue
        
        #salviamo nelle liste la predizione e il ground truth solo per i pixel validi (non background)
        predictions.extend(pred_flat[valid])
        ground_truth.extend(gt_flattened_mask[valid])

        #salvo le prime tra immagini da usare succssivamente nell'anteprima
        if len(previews) < 3:
            previews.append(
                {
                    "image": img, #immagine originale
                    "mask_gt": mask, #maschera ground truth
                    "predicted_mask": CRF_pred, #mashcera predetta dal modello
                    "path": img_path, #percorso file immagine originale
                }
            )

    #calcolo accuracy modello tra pixel maschera ground truth e predizione del modello (solo sui pixel validi)
    accuracy = accuracy_score(ground_truth, predictions) 
    return {
        "accuracy": accuracy,
        "predictions": np.asarray(predictions, dtype=int),
        "ground_truth": np.asarray(ground_truth, dtype=int),
        "previews": previews,
    }


def save_preview(previews: list[Dict], output_path: Path) -> None:
    if not previews:
        print("[INFO] Nessuna anteprima da salvare.")
        return

    cmap = colors.ListedColormap(
        ["#9e9e9e", "#ff6f69", "#ffcc5c", "#88d8b0", "#6b5b95", "#2a9d8f"]
    )
    norm = colors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    n_rows = len(previews)
    fig, axes = plt.subplots(n_rows, 4, figsize=(15, 4 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, sample in enumerate(previews):
        img = sample["image"]
        mask_gt_raw = sample["mask_gt"]
        mask_gt = mask_gt_raw.astype(float) - 1
        predicted_mask = sample["mask_pred"].astype(float)
        predicted_mask_filtered = predicted_mask.copy()
        predicted_mask_filtered[mask_gt_raw == 0] = np.nan
        name = Path(sample["path"]).name

        gt_display = np.where(mask_gt < -0.5, np.nan, mask_gt)

        axes[row_idx, 0].imshow(img)
        axes[row_idx, 0].set_title(f"Immagine\n{name}")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(gt_display, cmap=cmap, norm=norm)
        axes[row_idx, 1].set_title("Maschera Ground Truth")
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(predicted_mask, cmap=cmap, norm=norm)
        axes[row_idx, 2].set_title("Maschera Predetta (CRF)")
        axes[row_idx, 2].axis("off")

        axes[row_idx, 3].imshow(predicted_mask_filtered, cmap=cmap, norm=norm)
        axes[row_idx, 3].set_title("Predizione (solo GT>0)")
        axes[row_idx, 3].axis("off")

    handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=cmap(norm(i - 1)), markersize=12)
        for i in range(len(CLASS_NAMES) + 1)
    ]
    labels = ["Background"] + CLASS_NAMES
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Anteprima salvata in: {output_path}")


def save_confusion_matrix(cm: np.ndarray, class_names: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix (pixel)")

    max_val = cm.max() if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            text_color = "white" if value > max_val * 0.5 else "black"
            ax.text(
                j,
                i,
                f"{value:,}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Valutazione modello GPU-optimized con CRF.")
    #leggo da linea di comando il nome del modello da valutare (senza estensioni).
    parser.add_argument(
        "--model_prefix",
        type=str,
        default=MODEL_PREFIX,
        help="Prefisso dei file salvati (senza _classifier.pkl / _feature_extractor.keras).",
    )
    
    args = parser.parse_args()

    model_path = (BASE_DIR / args.model_prefix).resolve()

    print("=== VALUTAZIONE MODELLO GPU ===")
    print(f"Modello: {model_path}")
    print(f"Immagini: {IMAGES_DIR}")
    print(f"Maschere: {MASKS_DIR}")
    print(f"CRF attivato")
    print("=" * 50)
    #creo un istanza del modello
    model = GPUOptimizedCNNSegmentationClassifier()
    #carico il modello addestrato (sia il file keras (CNN + decoder) che il file pkl (classificatore))
    model.load(str(model_path))
    #genero o carico gli split del dataset per ottenere gli ID delle immagini di evaluation 
    # (ovviamente ache se vengono rigenerati gli split, sono gli stessi del training)
    _, eval_ids = prepare_dataset_splits(
        images_dir=str(IMAGES_DIR),
        masks_dir=str(MASKS_DIR),
        split_dir=str(SPLIT_DIR),
        train_ratio=0.8,
        seed=42,
    )
    #carico le immagini e maschere del set di evaluation e le elaboro in scala di grigi se richiesto
    images, masks, image_paths = load_dataset_stateless(
        images_dir=str(IMAGES_DIR),
        masks_dir=str(MASKS_DIR),
        image_size=tuple(model.config.image_size), #usa come dimensione quella salvata nel modello
        use_grayscale=getattr(model.config, "use_grayscale", False), #attivo la conversione in scala di grigi se è presente nel modello
        image_names=eval_ids, #carico solo le immagini di evaluation (ottenuto prima da prepare_dataset_splits)
        return_paths=True, #restituisce anche i percorsi delle immagini (ci servono poi per la generazione dell'immagine di anteprima)
    )
    dataset = {
        "images": images,
        "masks": masks,
        "image_paths": image_paths,
    }
    print(f"Immagini valutate (holdout): {len(dataset['images'])}")
    #valuto il modello sul set di evaluation
    results = evaluate_gpu_model(model, dataset)

    accuracy = results["accuracy"]
    print(f"\nAccuracy pixel (post-CRF): {accuracy:.4f}")

    cm = confusion_matrix(results["ground_truth"], results["predictions"], labels=list(range(len(CLASS_NAMES))))
    save_confusion_matrix(cm, CLASS_NAMES, CONFUSION_PATH)
    print(f"Matrice di confusione salvata in: {CONFUSION_PATH}")

    preview_path = PREVIEW_PATH if PREVIEW_PATH.is_absolute() else (BASE_DIR / PREVIEW_PATH)
    save_preview(results["previews"], preview_path)


if __name__ == "__main__":
    main()
