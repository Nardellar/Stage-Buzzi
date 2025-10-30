"""
Training script per GPUOptimizedCNNSegmentationClassifier.
Mantiene pipeline CNN -> feature map -> classificatore boosting con Optuna.
"""

import argparse
from pathlib import Path

from gpu_optimized_cnn_classifier import (
    GPUClassifierConfig,
    GPUOptimizedCNNSegmentationClassifier,
)

# -----------------------------------------------------------------------------
# Costanti di configurazione (modificabili da qui)
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
# Percorsi delle cartelle con immagini e maschere da usare per training.
# Modificali solo se il dataset è stato spostato.
IMAGES_DIR = (BASE_DIR / "../images/Immagini").resolve()
MASKS_DIR = (BASE_DIR / "../images/Maschere").resolve()
# Prefisso completo (senza estensione) dei file del modello addestrato.
# Salva sempre all'interno della cartella dello script per allinearsi allo script di evaluation.
MODEL_OUTPUT_PREFIX = (BASE_DIR / "gpu_optimized_cnn_model").resolve()

# Numero di immagini elaborate in parallelo durante l'estrazione feature.
# Aumenta se hai più RAM/GPU per ridurre il tempo di estrazione; diminuisci se vedi OOM.
BATCH_SIZE = 4
# Risoluzione spaziale finale delle feature estratte (es. 64 -> griglia 64x64).
# Valori più alti aumentano il dettaglio ma anche costi computazionali e memoria.
FEATURE_MAP_SIZE = 64
# Numero massimo di pixel campionati per immagine (prevenzione dataset enorme).
# Aumenta per usare più pixel (migliore stima) ma attenzione alla memoria; None = tutti.
MAX_PIXELS_PER_IMAGE = 16384
# Filtri dei blocchi Conv2D del decoder che raffinano la feature-map.
# Incrementa per decoder più espressivo (ma più lento / pesante).
DECODER_FILTERS = 128
# Tipo di classificatore finale (xgboost o lightgbm).
CLASSIFIER_TYPE = "xgboost"
# Se True usa versioni GPU dei booster (quando disponibili); metti False su CPU pure.
USE_GPU = True
# Numero di trial Optuna per la ricerca iperparametri.
# Aumenta per tuning migliore (più tempo); riduci per prove rapide.
TRIALS = 20
# Timeout massimo in secondi per l'ottimizzazione (None = nessun limite).
# Imposta un valore se vuoi interrompere Optuna dopo N secondi.
TIMEOUT = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training CNN + Classificatore (GPU-Optimized)"
    )
    parser.add_argument(
        "--cnn_model",
        type=str,
        default="mobilenet_v2",
        choices=["mobilenet_v2", "efficientnet_b0", "resnet50", "convnext_tiny"],
        help="Backbone CNN da usare (mobilenet_v2 per velocità, convnext_tiny per qualità).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("GPU-OPTIMIZED CNN + CLASSIFICATORE".center(70))
    print("=" * 70)
    print(f"Backbone CNN            : {args.cnn_model}")
    print(f"Classificatore boosting  : {CLASSIFIER_TYPE}")
    print(f"Batch size feature       : {BATCH_SIZE}")
    print(f"Feature map size         : {FEATURE_MAP_SIZE}x{FEATURE_MAP_SIZE}")
    print(f"Max pixel per immagine   : {MAX_PIXELS_PER_IMAGE}")
    print(f"Decoder filters          : {DECODER_FILTERS}")
    print(f"Numero di trial Optuna   : {TRIALS}")
    print(f"Timeout Optuna (s)       : {TIMEOUT}")
    print(f"Uso GPU                  : {USE_GPU}")
    print(f"Directory immagini       : {IMAGES_DIR}")
    print(f"Directory maschere       : {MASKS_DIR}")
    print("=" * 70)

    max_pixels = (
        MAX_PIXELS_PER_IMAGE
        if MAX_PIXELS_PER_IMAGE and MAX_PIXELS_PER_IMAGE > 0
        else None
    )

    config = GPUClassifierConfig(
        cnn_model=args.cnn_model,
        classifier=CLASSIFIER_TYPE,
        batch_size=BATCH_SIZE,
        feature_map_size=FEATURE_MAP_SIZE,
        max_pixels_per_image=max_pixels,
        use_gpu=USE_GPU,
        images_dir=str(IMAGES_DIR),
        masks_dir=str(MASKS_DIR),
        decoder_filters=DECODER_FILTERS,
    )

    model = GPUOptimizedCNNSegmentationClassifier(config=config)

    try:
        print("\n1. Caricamento dati...")
        model.load_data()

        print("\n2. Estrazione feature map...")
        model.extract_features()
        print(f"Feature totali estratte : {model.X_features.shape}")

        print("\n3. Training classificatore con Optuna...")
        best_acc = model.train_classifier_optuna(
            n_trials=TRIALS,
            timeout=TIMEOUT,
        )
        print(f"Accuracy best trial     : {best_acc:.4f}")

        print("\n4. Salvataggio modello...")
        model.save(str(MODEL_OUTPUT_PREFIX))

        print("\nTraining completato con successo!")
    except Exception as exc:
        print("\nErrore durante il training:")
        print(str(exc))
        raise


if __name__ == "__main__":
    main()
