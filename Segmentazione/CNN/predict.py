# Segmentazione/CNN/predict.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from skimage.io import imread


# --- CLASSE DELLA METRICA PERSONALIZZATA ---
# Dobbiamo includerla perché il modello è stato salvato con questa metrica
class ArgmaxMeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.clip_by_value(tf.cast(y_true, 'int32'), 0, self.num_classes - 1)
        return super().update_state(y_true, y_pred, sample_weight)


# --- CONFIGURAZIONE ---
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)
NUM_CLASSES = 3

# --- PATH DEI DATI E DEL MODELLO ---
IMAGE_DIR = "../images/Immagini/"
MASK_DIR = "../images/Maschere/"
MODEL_PATH = "unet_segmentation_model.keras"  # Il modello salvato da train.py


def load_for_prediction(image_path, mask_path):
    """Carica e pre-processa una singola coppia per la visualizzazione."""
    # Carica immagine originale per la visualizzazione
    original_image = imread(image_path)

    # Carica e processa l'immagine per darla in input al modello
    image = tf.convert_to_tensor(original_image, dtype=tf.float32)
    image = tf.image.resize(image, IMAGE_SIZE) / 255.0
    image = tf.expand_dims(image, axis=0)  # Aggiunge la dimensione del batch

    # Carica la maschera reale per il confronto
    true_mask = imread(mask_path)
    true_mask = tf.convert_to_tensor(true_mask, dtype=tf.int32)
    true_mask = tf.image.resize(tf.expand_dims(true_mask, axis=-1), IMAGE_SIZE, method='nearest')
    true_mask = tf.squeeze(true_mask, axis=-1)

    return original_image, image, true_mask


def display(display_list, titles):
    """Funzione per visualizzare le immagini."""
    plt.figure(figsize=(15, 5))
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(titles[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.show()


def main():
    print(f"Caricamento del modello da: {MODEL_PATH}")
    # Carica il modello, specificando la metrica personalizzata
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={'ArgmaxMeanIoU': ArgmaxMeanIoU}
    )
    print("Modello caricato.")

    # Prendi gli stessi dati di validazione usati nel training
    image_files = sorted(
        [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    mask_files = sorted([os.path.join(MASK_DIR, f) for f in os.listdir(MASK_DIR) if
                         f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])

    _, val_img_paths, _, val_mask_paths = train_test_split(
        image_files, mask_files, test_size=0.2, random_state=42
    )

    # Mostra le predizioni per un po' di immagini di validazione
    for i in range(min(5, len(val_img_paths))):  # Mostra fino a 5 immagini
        original_img, input_img, true_mask = load_for_prediction(val_img_paths[i], val_mask_paths[i])

        # Fai la predizione
        pred_mask_prob = model.predict(input_img)
        pred_mask = tf.argmax(pred_mask_prob, axis=-1)[0]  # Rimuovi la dimensione del batch

        print(f"\n--- Visualizzazione Immagine {i + 1} ---")
        display(
            [original_img, true_mask, pred_mask],
            ['Immagine Originale', 'Maschera Reale', 'Maschera Predetta']
        )


if __name__ == '__main__':
    main()