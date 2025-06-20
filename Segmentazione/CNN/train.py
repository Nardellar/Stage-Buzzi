# Segmentazione/CNN/train.py

import tensorflow as tf
from model_Narde import build_unet
from Dataset import create_dataset_from_files
import os
from sklearn.model_selection import train_test_split


class ArgmaxMeanIoU(tf.keras.metrics.MeanIoU):
    """MeanIoU metric that argmaxes model predictions."""

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.clip_by_value(tf.cast(y_true, tf.int32), 0, self.num_classes - 1)
        return super().update_state(y_true, y_pred, sample_weight)

def main():
    """
    Funzione principale che orchestra l'intero processo di addestramento.
    """
    # --- IPERPARAMETRI E CONFIGURAZIONE ---
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    IMG_CHANNELS = 3
    NUM_CLASSES = 3

    INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    BATCH_SIZE = 8
    EPOCHS = 5
    LEARNING_RATE = 1e-4
    VALIDATION_SPLIT = 0.2

    # --- PATH DEI DATI ---
    IMAGE_DIR = "../images/Immagini/"
    MASK_DIR = "../images/Maschere/"
    MODEL_SAVE_PATH = "unet_segmentation_model.keras"

    # 1. CARICAMENTO E SUDDIVISIONE AUTOMATICA DEI DATI
    print("Caricamento e suddivisione dei dati...")

    # Cerca le immagini (invariato)
    image_files = sorted(
        [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    # MODIFICATO: Cerca le maschere, incluse le estensioni .tif e .tiff
    mask_files = sorted([os.path.join(MASK_DIR, f) for f in os.listdir(MASK_DIR) if
                         f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])

    print(f"File immagine trovati: {len(image_files)}")
    print(f"File maschera trovati: {len(mask_files)}")

    # Assicurati che ogni immagine abbia una maschera corrispondente
    assert len(image_files) == len(mask_files), "Il numero di immagini e maschere non corrisponde."

    # Il resto dello script continua da qui...
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
        image_files, mask_files, test_size=VALIDATION_SPLIT, random_state=42
    )

    print(f"Trovati {len(image_files)} campioni in totale.")
    print(f"Suddivisione in {len(train_img_paths)} campioni di training e {len(val_img_paths)} di validazione.")

    # 2. CREAZIONE DEI DATASET
    print("Creazione dei dataset TensorFlow...")
    train_dataset = create_dataset_from_files(
        train_img_paths, train_mask_paths, (IMG_HEIGHT, IMG_WIDTH), BATCH_SIZE, NUM_CLASSES, augment=True
    )
    val_dataset = create_dataset_from_files(
        val_img_paths, val_mask_paths, (IMG_HEIGHT, IMG_WIDTH), BATCH_SIZE, NUM_CLASSES, augment=False
    )
    print("Dataset creati.")

    # 3. COSTRUZIONE DEL MODELLO
    print("Costruzione del modello U-Net...")
    model = build_unet(INPUT_SHAPE, NUM_CLASSES)
    model.summary()
    print("Modello costruito.")

    # 4. COMPILAZIONE DEL MODELLO
    print("Compilazione del modello...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=[ArgmaxMeanIoU(num_classes=NUM_CLASSES)],
    )
    print("Modello compilato.")

    # 5. CALLBACKS PER IL TRAINING
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor="val_loss", verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir="logs"),
    ]

    # 6. ADDESTRAMENTO DEL MODELLO
    print("Inizio dell'addestramento...")
    model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks
    )
    print("Addestramento completato.")


# Questo blocco assicura che la funzione main() venga eseguita
# solo quando lo script viene lanciato direttamente.
if __name__ == '__main__':
    main()