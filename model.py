import dataset_Organization
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16



def crea_modello_vgg16(input_shape, num_classi):

    # Carica VGG16 pre-addestrato su ImageNet, escludendo i livelli fully connected finali
    base_model = VGG16(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet", #pesi degli archi
        pooling='avg'
    )

    base_model.trainable = False

    # Aggiungi livelli personalizzati
    x = layers.Flatten()(base_model.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(num_classi, activation='softmax')(x)

    # Crea il nuovo modello
    model = models.Model(inputs=base_model.input, outputs=output)

    # Compila il modello
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model



if __name__ == "__main__":

    train_dataset, values_array = dataset_Organization.getdataset()

    model = crea_modello_vgg16(
        (108, 140, 3),
        len(values_array))


    model.summary()

