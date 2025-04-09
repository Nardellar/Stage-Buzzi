import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers.legacy import Adam


def _build_fc_layers(
        inputs,
        units_list=(512, 256),
        dropout_rates=(0.5, 0.3),
        activation='relu',
        use_batch_norm=True
):
    """
    Costruisce blocchi fully connected personalizzati.

    Args:
        inputs: Tensor di input
        units_list: Lista con numero di unità per ogni livello FC
        dropout_rates: Lista con tassi di dropout per ogni livello
        activation: Funzione di attivazione da usare
        use_batch_norm: Se True, applica la normalizzazione batch

    Returns:
        Tensor di output dell'ultimo livello
    """
    x = inputs

    # Verifica che le liste di unità e dropout abbiano la stessa lunghezza
    assert len(units_list) == len(dropout_rates), "units_list e dropout_rates devono avere la stessa lunghezza"

    for units, dr in zip(units_list, dropout_rates):
        x = layers.Dense(units, activation=activation)(x)

        if use_batch_norm:
            x = layers.BatchNormalization()(x)

        if dr > 0:
            x = layers.Dropout(dr)(x)

    return x


def create_classification_model(
        input_shape=(112, 112, 3),
        num_classes=3,
        base_trainable=False,
        optimizer=None,
        fc_units_list=(512, 256),
        fc_dropout_rates=(0.5, 0.3),
        fc_activation='relu',
        use_batch_norm=True,
        learning_rate=0.001
):
    """
    Crea un modello di classificazione basato su VGG16 con transfer learning.

    Args:
        input_shape: Dimensione immagini di input (altezza, larghezza, canali)
        num_classes: Numero di classi per la classificazione
        base_trainable: Se True, sblocca i pesi del modello base
        optimizer: Ottimizzatore personalizzato (se None, usa Adam)
        fc_units_list: Lista con numero di unità per ogni livello FC
        fc_dropout_rates: Lista con tassi di dropout per ogni livello FC
        fc_activation: Funzione di attivazione per i livelli FC
        use_batch_norm: Se True, applica la normalizzazione batch nei livelli FC
        learning_rate: Learning rate iniziale dell'ottimizzatore

    Returns:
        Modello Keras compilato
    """
    # Verifica dimensioni di input valide per VGG16
    min_dim = 32
    if input_shape[0] < min_dim or input_shape[1] < min_dim:
        raise ValueError(f"Le dimensioni di input devono essere almeno {min_dim}x{min_dim}")

    # Carica VGG16 pre-addestrato come feature extractor
    base_model = VGG16(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    # Configura il congelamento/scongelamento dei pesi del modello base
    base_model.trainable = base_trainable

    # Costruzione modello
    inputs = base_model.input
    x = base_model.output

    # Global Average Pooling per ridurre le feature map 2D
    x = layers.GlobalAveragePooling2D()(x)

    # Aggiunge blocco fully connected personalizzato
    x = _build_fc_layers(
        inputs=x,
        units_list=fc_units_list,
        dropout_rates=fc_dropout_rates,
        activation=fc_activation,
        use_batch_norm=use_batch_norm
    )

    # Livello di output in base al numero di classi
    if num_classes == 2:
        outputs = layers.Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        loss = 'sparse_categorical_crossentropy'

    # Crea il modello Keras completo
    model = models.Model(inputs=inputs, outputs=outputs)

    # Usa l'ottimizzatore fornito o crea Adam con LR specificato
    if optimizer is None:
        optimizer = Adam(learning_rate=learning_rate)

    # Compila il modello con la loss appropriata
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )

    return model


def train_model(
        model,
        train_dataset,
        validation_dataset,
        epochs=50,
        checkpoint_filepath='best_model.h5',
        early_stopping_patience=7,
        lr_reduction_patience=3,
        lr_reduction_factor=0.2,
        min_lr=1e-6,
        additional_callbacks=None
):
    """
    Addestra il modello con best practices come early stopping e learning rate scheduling.

    Args:
        model: Modello Keras compilato
        train_dataset: Dataset di addestramento
        validation_dataset: Dataset di validazione
        epochs: Numero massimo di epoche
        checkpoint_filepath: Percorso per salvare i migliori pesi del modello
        early_stopping_patience: Numero di epoche di attesa prima di interrompere
        lr_reduction_patience: Numero di epoche di attesa prima di ridurre LR
        lr_reduction_factor: Fattore di riduzione del learning rate
        min_lr: Learning rate minimo
        additional_callbacks: Lista di callback aggiuntivi

    Returns:
        Storia dell'addestramento
    """
    # Lista base di callback
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=lr_reduction_factor,
            patience=lr_reduction_patience,
            min_lr=min_lr,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        )
    ]

    # Aggiungi eventuali callback personalizzati
    if additional_callbacks:
        callbacks.extend(additional_callbacks)

    # Avvia addestramento
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks
    )

    return history


def load_and_evaluate_model(
        model_path,
        test_dataset,
        custom_objects=None
):
    """
    Carica un modello salvato e lo valuta sul test set.

    Args:
        model_path: Percorso del modello salvato
        test_dataset: Dataset di test
        custom_objects: Dizionario di oggetti personalizzati

    Returns:
        Risultati della valutazione (dict)
    """
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    results = model.evaluate(test_dataset, verbose=1)

    metrics = {}
    for metric_name, value in zip(model.metrics_names, results):
        metrics[metric_name] = value

    return metrics