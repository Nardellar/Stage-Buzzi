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
import glob
from pathlib import Path
from typing import Dict, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import (
        unary_from_labels,
        create_pairwise_gaussian,
        create_pairwise_bilateral,
    )

    CRF_AVAILABLE = True
except ImportError as exc:
    CRF_AVAILABLE = False
    CRF_IMPORT_ERROR = exc

# Cartella contenente lo script (usata come riferimento per i path relativi).
BASE_DIR = Path(__file__).resolve().parent
# Directory di immagini e maschere da valutare; cambia se il dataset è altrove.
IMAGES_DIR = (BASE_DIR / "../images/Immagini").resolve()
MASKS_DIR = (BASE_DIR / "../images/Maschere").resolve()
# Prefisso dei file salvati dal training (senza estensioni).
MODEL_PREFIX = "gpu_optimized_cnn_model"
# Percorso del file PNG con le anteprime (GT vs predizione).
PREVIEW_PATH = BASE_DIR / "gpu_model_preview.png"
# Numero massimo di immagini da valutare (None = tutte). Riduci per test rapidi.
MAX_IMAGES = None
# Imposta una dimensione fissa (H, W) se vuoi forzare il resize durante la valutazione;
# lascia None per usare la risoluzione originale delle immagini nel dataset.
IMAGE_SIZE_OVERRIDE: Optional[tuple[int, int]] = None
# Se True ricostruisce il feature extractor per ogni immagine per lavorare alla risoluzione originale.
USE_NATIVE_RESOLUTION = True
CLASS_NAMES = ["Resina", "Pori/Imperfezioni", "Fase Fusa", "Belite", "Alite"]


def load_dataset(images_dir: Path, masks_dir: Path, limit: Optional[int]) -> Dict[str, np.ndarray]:
    image_paths = sorted(glob.glob(str(images_dir / "*.png")))
    mask_paths = sorted(glob.glob(str(masks_dir / "*.tif")))

    if not image_paths or not mask_paths:
        raise FileNotFoundError(
            f"Nessuna immagine trovata in {images_dir} o nessuna maschera in {masks_dir}"
        )

    if limit is not None:
        image_paths = image_paths[:limit]
        mask_paths = mask_paths[:limit]

    images, masks = [], []
    for img_path, mask_path in zip(image_paths, mask_paths):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img.astype(np.float32) / 255.0)

        mask = Image.open(mask_path)
        mask = np.array(mask)
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        masks.append(mask.astype(np.int32))

    if not images:
        raise RuntimeError("Dataset vuoto dopo il caricamento.")

    return {
        "images": np.asarray(images),
        "masks": np.asarray(masks),
        "image_paths": image_paths,
    }


def apply_dense_crf(image: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    if not CRF_AVAILABLE:
        raise RuntimeError(
            "pydensecrf non installato. Esegui `pip install pydensecrf` per abilitare il CRF."
        ) from CRF_IMPORT_ERROR

    h, w = image.shape[:2]
    d = dcrf.DenseCRF2D(w, h, num_classes)

    unary = unary_from_labels(labels.astype(np.int32), num_classes, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(unary)

    feats_gaussian = create_pairwise_gaussian(sdims=(3, 3), shape=image.shape[:2])
    d.addPairwiseEnergy(feats_gaussian, compat=3)

    feats_bilateral = create_pairwise_bilateral(
        sdims=(5, 5),
        schan=(10, 10, 10),
        img=image,
        chdim=2,
    )
    d.addPairwiseEnergy(feats_bilateral, compat=5)

    Q = d.inference(5)
    refined = np.argmax(Q, axis=0).reshape((h, w))
    return refined


def evaluate_gpu_model(
    model_path: Path,
    dataset: Dict[str, np.ndarray],
    image_size: Optional[tuple[int, int]] = None,
    use_native_resolution: bool = False,
) -> Dict:
    from gpu_optimized_cnn_classifier import GPUOptimizedCNNSegmentationClassifier

    model = GPUOptimizedCNNSegmentationClassifier()
    model.load(str(model_path))

    expected_size = None if use_native_resolution else (image_size or tuple(model.config.image_size))
    if expected_size is not None:
        width_expected, height_expected = expected_size[1], expected_size[0]

    predictions = []
    ground_truth = []
    previews = []

    for idx, (img, mask, img_path) in enumerate(
        zip(dataset["images"], dataset["masks"], dataset["image_paths"]), start=1
    ):
        print(f"[GPU] Processando immagine {idx}/{len(dataset['images'])}...")
        if expected_size is not None:
            img_prepared = cv2.resize(
                img, (width_expected, height_expected), interpolation=cv2.INTER_LINEAR
            ).astype(np.float32)
        else:
            native_size = (img.shape[0], img.shape[1])
            model.ensure_feature_extractor_size(native_size)
            img_prepared = img.astype(np.float32)

        features = model._feature_extractor.predict(np.expand_dims(img_prepared, axis=0), verbose=0)
        h_f, w_f = features.shape[1:3]

        feat_flat = features.reshape(-1, features.shape[-1])
        preds = model.classifier.predict(feat_flat)
        preds = np.asarray(preds, dtype=int).reshape(h_f, w_f)
        preds_up = cv2.resize(
            preds,
            (mask.shape[1], mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        original_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        refined_pred = apply_dense_crf(original_uint8, preds_up, len(CLASS_NAMES))

        gt_flat = mask.reshape(-1) - 1
        pred_flat = refined_pred.reshape(-1)
        valid = gt_flat >= 0
        if not np.any(valid):
            continue

        predictions.extend(pred_flat[valid])
        ground_truth.extend(gt_flat[valid])

        if len(previews) < 3:
            previews.append(
                {
                    "image": img,
                    "mask_gt": mask,
                    "mask_pred": refined_pred,
                    "path": img_path,
                }
            )

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
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, sample in enumerate(previews):
        img = sample["image"]
        mask_gt = sample["mask_gt"].astype(float) - 1
        mask_pred = sample["mask_pred"].astype(float)
        name = Path(sample["path"]).name

        gt_display = np.where(mask_gt < -0.5, np.nan, mask_gt)

        axes[row_idx, 0].imshow(img)
        axes[row_idx, 0].set_title(f"Immagine\n{name}")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(gt_display, cmap=cmap, norm=norm)
        axes[row_idx, 1].set_title("Maschera Ground Truth")
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(mask_pred, cmap=cmap, norm=norm)
        axes[row_idx, 2].set_title("Maschera Predetta (CRF)")
        axes[row_idx, 2].axis("off")

    handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=cmap(norm(i - 1)), markersize=12)
        for i in range(len(CLASS_NAMES) + 1)
    ]
    labels = ["Background"] + CLASS_NAMES
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Anteprima salvata in: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Valutazione modello GPU-optimized con CRF.")
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
    print(f"CRF disponibile: {CRF_AVAILABLE}")
    print(f"Max immagini valutate: {MAX_IMAGES}")
    print("=" * 50)

    dataset = load_dataset(IMAGES_DIR, MASKS_DIR, MAX_IMAGES)
    results = evaluate_gpu_model(
        model_path,
        dataset,
        IMAGE_SIZE_OVERRIDE,
        use_native_resolution=USE_NATIVE_RESOLUTION,
    )

    accuracy = results["accuracy"]
    print(f"\nAccuracy pixel (post-CRF): {accuracy:.4f}")

    cm = confusion_matrix(results["ground_truth"], results["predictions"], labels=list(range(len(CLASS_NAMES))))
    short_names = [name[:4] for name in CLASS_NAMES]
    print("\nConfusion matrix (classi 0-4):")
    header = "     " + " ".join(f"{name:>6}" for name in short_names)
    print(header)
    for idx, row in enumerate(cm):
        print(f"{short_names[idx]:>4} " + " ".join(f"{val:>6}" for val in row))

    preview_path = PREVIEW_PATH if PREVIEW_PATH.is_absolute() else (BASE_DIR / PREVIEW_PATH)
    save_preview(results["previews"], preview_path)


if __name__ == "__main__":
    main()
