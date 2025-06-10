import argparse
import os
import shutil
import random
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, models, callbacks, optimizers, mixed_precision
from pathlib import Path

# --- Utility Functions ---

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_gpus():
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def clear_tfhub_cache():
    """
    Rimuove la cache locale di TensorFlow Hub per evitare header corrotti.
    """
    cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'tensorflowhub')
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)

# --- Model Components ---

def build_fc_layers(
    inputs,
    units_list,
    dropout_rates,
    activation,
    use_batch_norm
):
    """
    Costruisce la testa Fully-Connected del modello.
    """
    x = inputs
    for units, dr in zip(units_list, dropout_rates):
        x = layers.Dense(units, activation=None)(x)
        if use_batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        if dr > 0:
            x = layers.Dropout(dr)(x)
    return x


def create_vit_model(
    input_shape,
    num_classes,
    transformer_url,
    base_trainable,
    fc_units,
    fc_dropout,
    fc_activation,
    use_batch_norm
):
    """
    Crea un classificatore basato su Vision Transformer da TF-Hub.
    """
    inputs = layers.Input(shape=input_shape)

    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")
    x = data_augmentation(inputs)

    # Preprocessing
    x = layers.Resizing(input_shape[0], input_shape[1], name="resize")(x)
    x = layers.Rescaling(1./255, name="rescale")(x)

    # Load Transformer encoder
    try:
        transformer_layer = hub.KerasLayer(
            transformer_url,
            trainable=base_trainable,
            name="vit_encoder"
        )
        x = transformer_layer(x)
    except Exception as e:
        raise RuntimeError(
            f"Errore nel caricamento del modello da {transformer_url}: {e}\n"
            "Verifica l'URL (ad es. '/vit-base-patch16-224/feature_vector/1' o '/vit-base-patch16-224/finetuning/1') "
            "e pulisci la cache TF-Hub con clear_tfhub_cache()."
        )

    # Custom FC head
    x = build_fc_layers(
        inputs=x,
        units_list=fc_units,
        dropout_rates=fc_dropout,
        activation=fc_activation,
        use_batch_norm=use_batch_norm
    )

    # Output
    if num_classes == 2:
        outputs = layers.Dense(1, activation='sigmoid', name="output")(x)
    else:
        outputs = layers.Dense(num_classes, activation='softmax', name="output")(x)

    return models.Model(inputs=inputs, outputs=outputs, name="vit_classifier")


def prepare_dataset(ds, batch_size, shuffle, augment, cache=True):
    """
    Applica shuffle, cache, batch e prefetch.
    """
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    if cache:
        ds = ds.cache()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# --- CLI Argument Parsing ---

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Vision Transformer classifier.")
    parser.add_argument('--data_key', type=str, default='temperatura',
                        help='Chiave per dataset_organization.get_dataset')
    parser.add_argument('--transformer_url', type=str,
                        default="https://tfhub.dev/google/vit-base-patch16-224/feature_vector/1",
                        help='URL del modello ViT su TF-Hub')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Dimensione input immagini')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Dimensione del batch')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Numero epoche')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Tasso di apprendimento')
    parser.add_argument('--fc_units', nargs='+', type=int, default=[512, 256],
                        help='Unità livelli FC')
    parser.add_argument('--fc_dropout', nargs='+', type=float, default=[0.5, 0.3],
                        help='Dropout livelli FC')
    parser.add_argument('--no_batch_norm', action='store_true',
                        help='Disabilita BatchNormalization')
    parser.add_argument('--trainable_base', action='store_true',
                        help='Rende addestrabile backbone ViT')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Abilita mixed precision')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Cartella output modelli e log')
    return parser.parse_args()

# --- Main Training Script ---

def main():
    args = parse_args()

    # Clear cache, seeds, GPUs, precision
    clear_tfhub_cache()
    set_seeds()
    configure_gpus()
    if args.mixed_precision:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)

    # Output directories
    out_dir = Path(args.output_dir)
    model_dir = out_dir / 'models'
    log_dir = out_dir / 'logs'
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    from common import dataset_organization
    train_ds, val_ds = dataset_organization.get_dataset(args.data_key)

    # Prepare tf.data pipelines
    train_ds = prepare_dataset(train_ds, args.batch_size, shuffle=True, augment=True)
    val_ds = prepare_dataset(val_ds, args.batch_size, shuffle=False, augment=False)

    # Build model
    num_classes = (train_ds.element_spec[1].shape[-1]
                   if hasattr(train_ds.element_spec[1], 'shape') else 3)
    model = create_vit_model(
        input_shape=(args.image_size, args.image_size, 3),
        num_classes=num_classes,
        transformer_url=args.transformer_url,
        base_trainable=args.trainable_base,
        fc_units=args.fc_units,
        fc_dropout=args.fc_dropout,
        fc_activation='relu',
        use_batch_norm=not args.no_batch_norm
    )

    # Compile
    loss = 'binary_crossentropy' if model.output_shape[-1] == 1 else 'sparse_categorical_crossentropy'
    opt = optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    model.summary()

    # Callbacks
    cb_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
        callbacks.ModelCheckpoint(filepath=str(model_dir / 'best_model.h5'), save_best_only=True),
        callbacks.TensorBoard(log_dir=str(log_dir)),
        callbacks.CSVLogger(str(log_dir / 'training_log.csv'))
    ]

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=cb_list,
        verbose=1
    )

    # Save final model
    model.save(str(model_dir / 'final_model.h5'))

if __name__ == "__main__":
    main()
