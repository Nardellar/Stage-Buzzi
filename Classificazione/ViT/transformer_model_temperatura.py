import tensorflow_hub as hub
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from common import dataset_organization

#costuisce ulteriori strati
def build_fc_layers(
        inputs,
        units_list=(512, 256, 128), #numero di neuroni
        dropout_rates=(0.5, 0.3, 0.2), #tassi dropout per ogni livello
        activation='relu',
        use_batch_norm=True #applica BatchNormalization dopo ogni livello
):
    """
    Costruisce i livelli fully connected del modello.
    """
    x = inputs
    assert len(units_list) == len(dropout_rates), "units_list e dropout_rates devono avere la stessa lunghezza"


    for units, dr in zip(units_list, dropout_rates):
        x = layers.Dense(units, activation=activation)(x)

        if use_batch_norm:
            x = layers.BatchNormalization()(x)

        if dr > 0:
            x = layers.Dropout(dr)(x)

    return x

def create_transformer_classification_model(
        input_shape,
        num_classes=3,
        base_trainable=False,
        optimizer=None,
        fc_units_list=(512, 256),
        fc_dropout_rates=(0.5, 0.3),
        fc_activation='relu',
        use_batch_norm=True,
        learning_rate=0.001,
        transformer_url="https://tfhub.dev/google/vit_base_patch16_224/classification/1"): #il transformer scelto
    """
    Crea un modello di classificazione basato su Vision Transformer (SAM).

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
        transformer_url: URL del modello SAM da TensorFlow Hub

    Returns:
        Modello TensorFlow compilato
    """

    # Carica modello Transformer da TensorFlow Hub
    transformer_layer = hub.KerasLayer(transformer_url, trainable=base_trainable)

    inputs = layers.Input(shape=input_shape)
    
    # I Transformer di solito vogliono input normalizzati [0,1] o [-1,1]
    x = layers.Rescaling(scale=1./255)(inputs)

    x = transformer_layer(x)

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
        loss = 'sparse_categorical_crossentropy'

    model = models.Model(inputs=inputs, outputs=outputs)

    if optimizer is None:
        optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )

    return model



def  train_model(
        model,
        train_dataset,
        validation_dataset,
        checkpoint_filepath,
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
        checkpoint_filepath: Percorso dove salvare il modello migliore
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
        batch_size=batch_size,  # Aggiunto batch_size
        callbacks=callbacks,
        verbose=1  # Aggiunto verbose per mostrare la barra di progresso
    )

    return history



def main():

    train_ds, val_ds = dataset_organization.get_dataset("temperatura")

    mapping_dict = {
        1300: 0,
        1400: 1,
        1500: 2
    }

    train_ds = train_ds.map(dataset_organization.remap_labels(mapping_dict))
    val_ds = val_ds.map(dataset_organization.remap_labels(mapping_dict))

    model = create_transformer_classification_model(
    input_shape=(224, 224, 3),   # Vision Transformer richiede di solito 224x224
    num_classes=3,
    base_trainable=False
)

    history = train_model(
        model=model,
        train_dataset=train_ds,
        validation_dataset=val_ds,
        epochs=30,
        checkpoint_filepath='CNN/Models/Temperatura/best_modello_temperatura.h5'
    )

    # Per esempio, salva il modello finale
    model.save("Models/Temperatura/modello_temperatura.h5")


if __name__ == "__main__":
    main()
