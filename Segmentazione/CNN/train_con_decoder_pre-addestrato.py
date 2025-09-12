# Segmentazione/CNN/train.py

import tensorflow as tf
import os
import cv2 # Importa OpenCV
from sklearn.model_selection import train_test_split

# --- METRICA PERSONALIZZATA (INVARIATA) ---
class ArgmaxMeanIoU(tf.keras.metrics.MeanIoU):
    """MeanIoU metric that argmaxes model predictions."""
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.cast(y_true, tf.int32)
        if len(y_true.shape) == 4:
            y_true = tf.squeeze(y_true, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)


# --- NUOVE FUNZIONI DI CARICAMENTO CON OPENCV ---

def read_and_decode_image(path):
    """Legge un file immagine dal disco usando OpenCV."""
    # Decodifica il path da tensore a stringa python
    path = path.numpy().decode('utf-8')
    # Leggi l'immagine in scala di grigi
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # Aggiungi una dimensione per il canale per coerenza
    img = img[..., tf.newaxis]
    return img.astype('float32') # Ritorna come float32

def read_and_decode_mask(path):
    """Legge un file maschera (anche TIFF) dal disco."""
    path = path.numpy().decode('utf-8')
    # IMREAD_UNCHANGED è importante per leggere i dati grezzi dei TIFF
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # Aggiungi una dimensione per il canale se non è presente
    if len(mask.shape) == 2:
        mask = mask[..., tf.newaxis]
    return mask.astype('uint8') # Le maschere sono interi

def load_and_preprocess(img_path, mask_path, size, num_classes):
    """Funzione wrapper che usa tf.py_function per integrare OpenCV."""
    # Carica immagine
    [image,] = tf.py_function(read_and_decode_image, [img_path], [tf.float32])
    image.set_shape([None, None, 1])
    image = tf.image.resize(image, size)
    image = tf.image.grayscale_to_rgb(image) # Converte a 3 canali per il backbone
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    # Carica maschera
    [mask,] = tf.py_function(read_and_decode_mask, [mask_path], [tf.uint8])
    mask.set_shape([None, None, 1])
    mask = tf.image.resize(mask, size, method='nearest') # 'nearest' per le maschere

    # Controllo sui valori della maschera
    tf.debugging.assert_less(
        mask, tf.cast(num_classes, dtype=mask.dtype),
        message="I valori della maschera superano il numero di classi."
    )
    return image, mask

def create_dataset_from_files(img_paths, mask_paths, size, batch_size, num_classes):
    """Crea un dataset TensorFlow dai percorsi dei file."""
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    dataset = dataset.map(lambda img, msk: load_and_preprocess(img, msk, size, num_classes),
                          num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

# La funzione build_unet_with_backbone rimane identica a prima
def build_unet_with_backbone(input_shape, num_classes):
    """Costruisce un modello U-Net usando MobileNetV2 come encoder pre-addestrato."""
    print("Costruzione del modello U-Net con backbone MobileNetV2...")

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    layer_names = [
        'block_1_expand_relu',
        'block_3_expand_relu',
        'block_6_expand_relu',
        'block_13_expand_relu',
        'out_relu'
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    base_model.trainable = False
    encoder = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
    encoder.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    all_skips = encoder(inputs)
    x = all_skips[-1]
    skips = reversed(all_skips[:-1])

    up_stack = [
        tf.keras.layers.Conv2DTranspose(256, 2, strides=2, padding='same'),
        tf.keras.layers.Conv2DTranspose(128, 2, strides=2, padding='same'),
        tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding='same'),
        tf.keras.layers.Conv2DTranspose(32, 2, strides=2, padding='same'),
    ]

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    last = tf.keras.layers.Conv2DTranspose(
        filters=num_classes,
        kernel_size=3,
        strides=2,
        padding='same',
        activation='softmax'
    )
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


def main():
    """Funzione principale."""
    # --- IPERPARAMETRI E CONFIGURAZIONE ---
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    IMG_CHANNELS = 3
    NUM_CLASSES = 7    #TODO: TROPPE CLASSI, è DA RISOLVERE

    INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    BATCH_SIZE = 8
    EPOCHS = 50 # Ho rimesso 50, sentiti libero di abbassarlo per i test
    LEARNING_RATE = 1e-3
    VALIDATION_SPLIT = 0.2

    # --- PATH DEI DATI ---
    IMAGE_DIR = "../images/Immagini/"
    MASK_DIR = "../images/Maschere/"
    MODEL_SAVE_PATH = "unet_mobilenetv2_model_preaddestrato.keras"

    print("Caricamento e suddivisione dei dati...")
    image_files = sorted([os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    # Questa riga ora funziona perché la funzione di caricamento usa OpenCV
    mask_files = sorted([os.path.join(MASK_DIR, f) for f in os.listdir(MASK_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])

    assert len(image_files) == len(mask_files), "Il numero di immagini e maschere non corrisponde."

    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
        image_files, mask_files, test_size=VALIDATION_SPLIT, random_state=42
    )
    print(f"Trovati {len(image_files)} campioni totali.")
    print(f"Suddivisi in {len(train_img_paths)} di training e {len(val_img_paths)} di validazione.")

    print("Creazione dei dataset TensorFlow...")
    train_dataset = create_dataset_from_files(
        train_img_paths, train_mask_paths, (IMG_HEIGHT, IMG_WIDTH), BATCH_SIZE, NUM_CLASSES
    )
    val_dataset = create_dataset_from_files(
        val_img_paths, val_mask_paths, (IMG_HEIGHT, IMG_WIDTH), BATCH_SIZE, NUM_CLASSES
    )
    print("Dataset creati.")

    model = build_unet_with_backbone(INPUT_SHAPE, NUM_CLASSES)
    model.summary()

    print("Compilazione del modello...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=[ArgmaxMeanIoU(num_classes=NUM_CLASSES)],
    )
    print("Modello compilato.")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor="val_loss", verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir="logs"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_lr=1e-6)
    ]

    print("Inizio dell'addestramento...")
    model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks
    )
    print("Addestramento completato.")


if __name__ == '__main__':
    main()