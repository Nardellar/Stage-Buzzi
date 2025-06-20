
import tensorflow as tf
from tensorflow.keras import layers

def conv_block(inputs, num_filters):
    """
    Blocco convoluzionale standard: Conv2D -> ReLU -> Conv2D -> ReLU.
    """
    x = layers.Conv2D(num_filters, 3, padding="same")(inputs)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(num_filters, 3, padding="same")(x)
    x = layers.Activation("relu")(x)
    return x

def encoder_block(inputs, num_filters):
    """
    Blocco dell'encoder: Blocco convoluzionale seguito da MaxPooling.
    Restituisce sia l'output del blocco conv che l'output del pooling.
    """
    skip = conv_block(inputs, num_filters)
    p = layers.MaxPooling2D((2, 2))(skip)
    return skip, p

def decoder_block(inputs, skip_features, num_filters):
    """
    Blocco del decoder: Up-convolution, concatenazione con skip connection
    e blocco convoluzionale.
    """
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape, num_classes):
    """
    Costruisce l'intera architettura U-Net.
    """
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder (Contracting Path)
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # Bridge (Ponte)
    bridge = conv_block(p4, 1024)

    # Decoder (Expansive Path)
    d1 = decoder_block(bridge, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    # Output Layer
    # Usa una convoluzione 1x1 per mappare ogni pixel alla classe desiderata.
    # L'attivazione 'softmax' è per la classificazione multi-classe.
    outputs = layers.Conv2D(num_classes, 1, padding="same", activation="softmax")(d4)

    model = tf.keras.Model(inputs, outputs, name="U-Net")
    return model

if __name__ == '__main__':
    # Stampa un riassunto del modello per verificarne la struttura
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    IMG_CHANNELS = 3
    NUM_CLASSES = 3 # Sfondo, Elemento 1, Elemento 2

    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    model = build_unet(input_shape, NUM_CLASSES)
    model.summary()