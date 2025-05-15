import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import dataset_organization


# 224 224 con 10 epoche mi da 0.047, 0.98, 0.17, 0.95
# loss: 0.0439 - accuracy: 0.9857 - val_loss: 0.1811 - val_accuracy: 0.9429 - lr: 0.0010
# loss: 0.0612 - accuracy: 0.9750 - val_loss: 0.1366 - val_accuracy: 0.9607 - lr: 0.0010

# 112 112 con 10 epoche mi da 0.10, 0.96, 0.37, 0.86
# 112 122 con 10 epoche e con ultimo dropout 0.2:  loss: 0.0110 - accuracy: 0.9973 - val_loss: 0.3503 - val_accuracy: 0.9107 - lr: 2.0000e-04
# loss: 0.0361 - accuracy: 0.9875 - val_loss: 0.3978 - val_accuracy: 0.9000 - lr: 4.0000e-05
# loss: 0.0488 - accuracy: 0.9830 - val_loss: 0.3294 - val_accuracy: 0.9179 - lr: 4.0000e-05


def build_fc_layers(
        inputs,
        units_list=(512, 256, 128),
        dropout_rates=(0.5, 0.3, 0.2),
        activation='relu',
        use_batch_norm=True
):
    """
    Costruisce i livelli fully connected del modello.

    Args:
        inputs: Tensor di input
        units_list: Lista con il numero di unità per ogni livello dense
        dropout_rates: Lista con i tassi di dropout per ogni livello
        activation: Funzione di attivazione da utilizzare
        use_batch_norm: Se utilizzare la normalizzazione batch

    Returns:
        Tensor di output dopo i livelli fully connected
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
        input_shape,
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
    Crea un modello di classificazione basato su VGG16.

    Args:
        input_shape: Forma dell'input (altezza, larghezza, canali)
        num_classes: Numero di classi da classificare
        base_trainable: Se rendere addestrabile il modello base
        optimizer: Ottimizzatore personalizzato (opzionale)
        fc_units_list: Lista con il numero di unità per ogni livello fully connected
        fc_dropout_rates: Lista con i tassi di dropout per ogni livello FC
        fc_activation: Funzione di attivazione per i livelli FC
        use_batch_norm: Se utilizzare la normalizzazione batch
        learning_rate: Tasso di apprendimento per l'ottimizzatore Adam

    Returns:
        Modello TensorFlow compilato
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

    # Imposta se il modello base è addestrabile o meno
    base_model.trainable = base_trainable

    # Costruzione modello
    inputs = base_model.input
    x = base_model.output

    # Global Average Pooling per ridurre le feature map 2D
    x = layers.GlobalAveragePooling2D()(x)

    # Aggiunge blocco fully connected personalizzato
    x = build_fc_layers(
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
        loss = 'sparse_categorical_crossentropy'  # Cambiato da sparse_categorical_crossentropy

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
        batch_size=None,  # Aggiunto parametro batch_size
        early_stopping_patience=7,
        lr_reduction_patience=3,
        lr_reduction_factor=0.2,
        min_lr=1e-6,
        additional_callbacks=None
):
    """
    Addestra il modello con early stopping e riduzione del learning rate.

    Args:
        model: Modello TensorFlow da addestrare
        train_dataset: Dataset di addestramento
        validation_dataset: Dataset di validazione
        epochs: Numero massimo di epoche di addestramento
        batch_size: Dimensione del batch (opzionale)
        early_stopping_patience: Numero di epoche prima di interrompere l'addestramento
        lr_reduction_patience: Numero di epoche prima di ridurre il learning rate
        lr_reduction_factor: Fattore di riduzione del learning rate
        min_lr: Learning rate minimo
        additional_callbacks: Callback aggiuntivi

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
        batch_size=batch_size,  # Aggiunto batch_size
        callbacks=callbacks,
        verbose=1  # Aggiunto verbose per mostrare la barra di progresso
    )

    return history


def load_and_evaluate_model(
        model_path,
        test_dataset,
        custom_objects=None
):
    """
    Carica un modello salvato e lo valuta su un dataset di test.

    Args:
        model_path: Percorso del modello salvato
        test_dataset: Dataset di test
        custom_objects: Oggetti personalizzati per il caricamento del modello

    Returns:
        Dizionario con le metriche di valutazione
    """
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    results = model.evaluate(test_dataset, verbose=1)

    metrics = {}
    for metric_name, value in zip(model.metrics_names, results):
        metrics[metric_name] = value

    return metrics, model  # Restituisce anche il modello caricato


def predict_with_model(model, data, batch_size=32):
    """
    Esegue previsioni utilizzando il modello.

    Args:
        model: Modello TensorFlow
        data: Dati su cui eseguire le previsioni
        batch_size: Dimensione del batch per le previsioni

    Returns:
        Previsioni del modello
    """
    predictions = model.predict(data, batch_size=batch_size, verbose=1)
    return predictions

