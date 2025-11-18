"""
Training script per GPUOptimizedCNNSegmentationClassifier.
Segue una pipeline CNN -> feature map -> classificatore boosting con Optuna.
"""

import argparse
from pathlib import Path

from data_module import load_dataset_stateless, prepare_dataset_splits
from gpu_optimized_cnn_classifier import (
    GPUClassifierConfig,
    GPUOptimizedCNNSegmentationClassifier,
)

# -----------------------------------------------------------------------------
# Costanti di configurazione (modificabili da qui)
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent 
# Percorsi delle cartelle con immagini e maschere da usare per training + evaluation/testing.
IMAGES_DIR = (BASE_DIR / "../images/Immagini").resolve()
MASKS_DIR = (BASE_DIR / "../images/Maschere").resolve()
# Prefisso dei file del modello addestrato.
# Salva sempre all'interno della cartella dello script per allinearsi allo script di evaluation.
MODEL_OUTPUT_PREFIX = (BASE_DIR / "gpu_optimized_cnn_model").resolve()
# Percorso della cartella contenente gli split del dataset.
SPLIT_DIR = (BASE_DIR / "splits").resolve()

# Numero di immagini elaborate in parallelo durante l'estrazione feature.
# Aumenta se hai più RAM/GPU per ridurre il tempo di estrazione; diminuisci se vedi OOM.
BATCH_SIZE = 2
# Risoluzione spaziale finale delle feature estratte. (es: 128, 256,...)
# None -> mantiene la risoluzione originale dell'immagine.
FEATURE_MAP_SIZE = None
# Numero massimo di pixel campionati per immagine (prevenzione dataset enorme).
# Aumenta per usare più pixel (migliore stima) ma attenzione alla memoria; 
# None -> campiona tutti i pixel dell'immagine.
MAX_PIXELS_PER_IMAGE = 20000
# Filtri dei blocchi Conv2D del decoder che raffinano la feature-map.
# Incrementa per decoder più espressivo (ma più lento / pesante).
DECODER_FILTERS = 128
# Tipo di classificatore finale (xgboost o lightgbm).
CLASSIFIER_TYPE = "xgboost"
# Se True usa versioni GPU dei booster (quando disponibili);
USE_GPU = False
# Numero di trial Optuna per la ricerca iperparametri.
# Aumenta per tuning migliore (più tempo);
TRIALS = 75
# Timeout massimo in secondi per l'ottimizzazione (None = nessun limite).
# Imposta un valore se vuoi interrompere Optuna dopo N secondi.
TIMEOUT = None

#funzione che parsa gli argomenti da linea di comando
def parse_args():
    parser = argparse.ArgumentParser(
        description="Training CNN + Classificatore (GPU-Optimized)"
    )
    #argomento per selezionare il backbone CNN (tra convexnet_tiny e resnet50)
    parser.add_argument(
        "--cnn_model",
        type=str,
        default="convnext_tiny",
        choices=["resnet50", "convnext_tiny"],
        help="Backbone CNN da usare (resnet50: classico, convnext_tiny: migliore accuratezza).",
    )
    #argomento per selezionare se convertire le immagini in scala di grigi
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Se specificato converte le immagini in scala di grigi replicata su tre canali.",
    )
    return parser.parse_args()


def main():
    #recupero gli argomenti da linea di comando
    args = parse_args()

    #stampo a schermo tutti i parametri di configurazione
    print("=" * 70)
    print("GPU-OPTIMIZED CNN + CLASSIFICATORE".center(70))
    print("=" * 70)
    print(f"Backbone CNN            : {args.cnn_model}")
    print(f"Classificatore boosting  : {CLASSIFIER_TYPE}")
    print(f"Batch size feature       : {BATCH_SIZE}")
    feature_map_desc = (
        "uguale all'immagine"
        if FEATURE_MAP_SIZE in (None)
        else f"{FEATURE_MAP_SIZE}x{FEATURE_MAP_SIZE}"
    )
    print(f"Feature map size         : {feature_map_desc}")
    max_pixels_desc = (
        "tutti quelli dell'immagine"
        if MAX_PIXELS_PER_IMAGE in (None)
        else MAX_PIXELS_PER_IMAGE
    )
    print(f"Max pixel per immagine   : {max_pixels_desc}")
    print(f"Uso scala di grigi       : {args.grayscale}")
    print(f"Decoder filters          : {DECODER_FILTERS}")
    print(f"Numero di trial Optuna   : {TRIALS}")
    print(f"Timeout Optuna (s)       : {TIMEOUT}")
    print(f"Uso GPU                  : {USE_GPU}")
    print(f"Directory immagini       : {IMAGES_DIR}")
    print(f"Directory maschere       : {MASKS_DIR}")
    print("=" * 70)

    #creo un oggetto configurazione
    config = GPUClassifierConfig(
        cnn_model=args.cnn_model,
        classifier=CLASSIFIER_TYPE,
        batch_size=BATCH_SIZE,
        feature_map_size=FEATURE_MAP_SIZE,
        max_pixels_per_image=MAX_PIXELS_PER_IMAGE,
        use_gpu=USE_GPU,
        images_dir=str(IMAGES_DIR),
        masks_dir=str(MASKS_DIR),
        decoder_filters=DECODER_FILTERS,
        use_grayscale=args.grayscale,
    )

    #creo instanzio il modello con la configurazione appena creata
    model = GPUOptimizedCNNSegmentationClassifier(config=config)

    try:
        #divido il dataset in training e evaluation/testing
        train_ids, eval_ids = prepare_dataset_splits(
            images_dir=str(IMAGES_DIR),
            masks_dir=str(MASKS_DIR),
            split_dir=str(SPLIT_DIR),
            train_ratio=0.8, #80% per training, 20% per evaluation/testing
            seed=42,
        )
        print(f"\nSplit dataset -> Train: {len(train_ids)} immagini, Eval/Test: {len(eval_ids)} immagini")

        print("\n1. Caricamento dati (train)...")
        #carico sol il train set (e applico augmentation e scala di grigi se rischiesto)
        model.load_train_data(filenames=train_ids)

        print("\n2. Estrazione feature map (train)...")
        #calcoliamo features e label per ogni pixel valido (non appartennti allo sfondo) di ogni immagine del train set
        model.extract_train_features()
        print(f"Feature train estratte   : {model.pixels_features.shape}")

        #carico il set di validazione e applico scala di grigi se richiesto (senza augmentantion)
        val_images, val_masks = load_dataset_stateless(
        images_dir=str(IMAGES_DIR),
        masks_dir=str(MASKS_DIR),
        image_size=tuple(config.image_size),
        use_grayscale=config.use_grayscale,
        filenames=eval_ids,
        )
        #calcoliamo features e label per ogni pixel valido (non appartennti allo sfondo) di ogni immagine del validation set
        validation_data = model.extract_features_stateless(
            val_images,
            val_masks,
        )
        features_val = validation_data[0]
        print(f"Feature validation       : {features_val.shape}")

        print("\n3. Training classificatore con Optuna...")
        #avviamo il training del classificatore e gli passiamo l'evaluation set per valutare il migliore, otteniamo la miglior accuracy
        best_accuracy = model.train_classifier_optuna(
            n_trials=TRIALS,
            timeout=TIMEOUT,
            validation_data=validation_data,
        )
        print(f"Accuracy best trial     : {best_accuracy:.4f}")

        print("\n4. Salvataggio modello...")
        model.save(str(MODEL_OUTPUT_PREFIX))

        print("\nTraining completato con successo!")
    except Exception as exc:
        print("\nErrore durante il training:")
        print(str(exc))
        raise


if __name__ == "__main__":
    main()

