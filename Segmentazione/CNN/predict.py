# Segmentazione/CNN/predict.py

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# --- CONFIGURAZIONE ---
IMG_HEIGHT = 256
IMG_WIDTH = 256
NUM_CLASSES = 3
MODEL_PATH = "unet_segmentation_model.keras" # Path al modello salvato

# Carica il modello addestrato
print("Caricamento del modello...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Modello caricato.")

def preprocess_image(image_path):
    """
    Carica e pre-processa una singola immagine per la predizione.
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = img / 255.0
    # Aggiunge la dimensione del batch (da [H, W, C] a [1, H, W, C])
    img = tf.expand_dims(img, axis=0)
    return img

def create_mask(pred_mask):
    """
    Converte la maschera di predizione (con probabilità) in una maschera
    di segmentazione con valori interi (0, 1, 2).
    """
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[0]  # Rimuove la dimensione del batch
    return pred_mask.numpy()

def display(display_list):
    """
    Visualizza immagine originale, maschera reale e maschera predetta.
    """
    plt.figure(figsize=(15, 5))
    title = ["Input Image", "Predicted Mask"]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
    plt.show()

# --- ESEGUI PREDIZIONE ---
# Sostituisci con il path di un'immagine di test
TEST_IMAGE_PATH = "../../data/val_images/imageX.png" # Esempio

# Pre-processa l'immagine
input_image = preprocess_image(TEST_IMAGE_PATH)

# Esegui la predizione
pred_mask_prob = model.predict(input_image)
pred_mask = create_mask(pred_mask_prob)

# Visualizza i risultati
display([input_image[0], pred_mask[..., tf.newaxis]])