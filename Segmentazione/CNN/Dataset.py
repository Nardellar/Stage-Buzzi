# Segmentazione/CNN/dataset.py

import tensorflow as tf
import numpy as np
from skimage.io import imread  # <--- Usiamo la libreria scikit-image


def load_image_and_mask(image_path, mask_path, image_size):
    """
    Carica un'immagine e una maschera, assicurando che le forme siano corrette.
    """
    # Carica l'immagine (invariato)
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, image_size)
    image.set_shape([image_size[0], image_size[1], 3])
    image = tf.cast(image, tf.float32) / 255.0

    # Carica la maschera TIFF usando una funzione Python e tf.py_function
    def _load_tiff(path):
        # Usiamo scikit-image, che è molto affidabile con i formati scientifici
        path = path.numpy().decode('utf-8')
        mask_array = imread(path)
        return mask_array.astype(np.int32)

    # tf.py_function esegue la nostra funzione di caricamento
    mask = tf.py_function(_load_tiff, [mask_path], tf.int32)

    # Ridimensiona la maschera
    # Per il resize, la maschera ha bisogno di una dimensione per il canale
    mask.set_shape([None, None])
    mask = tf.expand_dims(mask, axis=-1)  # Forma -> [H, W, 1]
    mask = tf.image.resize(mask, image_size, method='nearest')

    # Rimuovi la dimensione del canale per la loss function
    # Forma finale -> [H, W], che è ciò che serve
    mask = tf.squeeze(mask, axis=-1)

    # Assicura che TensorFlow conosca la forma finale
    mask.set_shape([image_size[0], image_size[1]])

    return image, mask


def augment_data(image, mask):
    """
    Applica la stessa trasformazione casuale sia all'immagine che alla maschera.
    """
    # Per l'augmentation, la maschera ha di nuovo bisogno della dimensione del canale
    mask_with_channel = tf.expand_dims(mask, axis=-1)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask_with_channel = tf.image.flip_left_right(mask_with_channel)

    # Rimuovi di nuovo la dimensione del canale dalla maschera
    mask = tf.squeeze(mask_with_channel, axis=-1)

    return image, mask


def create_dataset_from_files(image_files, mask_files, image_size, batch_size, augment=False):
    """
    Crea il dataset finale, gestendo la forma della maschera.
    """
    dataset = tf.data.Dataset.from_tensor_slices((image_files, mask_files))

    # 1. Carica e ridimensiona
    dataset = dataset.map(
        lambda img, msk: load_image_and_mask(img, msk, image_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # 2. Applica augmentation se richiesto
    if augment:
        dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)

    # 3. Crea i batch e ottimizza il caricamento
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset