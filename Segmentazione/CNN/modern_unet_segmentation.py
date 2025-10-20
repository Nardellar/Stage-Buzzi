"""
Implementazione Moderna U-Net per Segmentazione
Approccio end-to-end più efficace del CNN + XGBoost per la segmentazione.
NOTA MIA: CONSIGLIATO DA IA MA APPUNTO è UN APPROCCIO END-TO-END(?) NON CON CLASSIFCIATORE
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB3, ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import cv2
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import albumentations as A


class ModernUNet:
    """
    U-Net moderna con encoder EfficientNet/ResNet per segmentazione end-to-end.
    Molto più efficace del approccio CNN + XGBoost.
    """
    
    def __init__(self, 
                 encoder_name='efficientnet_b3',
                 input_shape=(512, 512, 3),
                 num_classes=5,  # 5 classi utili (1-5, escluse pixel non etichettati)
                 dropout_rate=0.3):
        
        self.encoder_name = encoder_name
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model = None
        
        # Percorsi
        self.image_dir = "../images/Immagini/"
        self.mask_dir = "../images/Maschere/"
        
    def _create_encoder(self, input_tensor):
        """Crea l'encoder basato su EfficientNet o ResNet."""
        
        if self.encoder_name.startswith('efficientnet'):
            if self.encoder_name == 'efficientnet_b3':
                base_model = EfficientNetB3(
                    weights='imagenet',
                    include_top=False,
                    input_tensor=input_tensor
                )
                # Livelli per skip connections
                skip_layers = ['block2a_expand_activation', 'block3a_expand_activation', 
                              'block4a_expand_activation', 'block6a_expand_activation']
                
        elif self.encoder_name.startswith('resnet'):
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_tensor=input_tensor
            )
            # Livelli per skip connections
            skip_layers = ['conv2_block3_out', 'conv3_block4_out', 
                          'conv4_block6_out', 'conv5_block3_out']
        
        else:
            raise ValueError(f"Encoder non supportato: {self.encoder_name}")
        
        # Rendi trainable per fine-tuning
        for layer in base_model.layers[-50:]:  # Ultimi 50 layer trainable
            layer.trainable = True
            
        return base_model, skip_layers
    
    def _decoder_block(self, x, skip, filters, name):
        """Blocco decoder con skip connection."""
        
        # Upsampling
        x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
        
        # Concatenazione con skip connection
        if skip is not None:
            x = layers.Concatenate()([x, skip])
        
        # Convoluzioni
        x = layers.Conv2D(filters, 3, padding='same', name=f'{name}_conv1')(x)
        x = layers.BatchNormalization(name=f'{name}_bn1')(x)
        x = layers.Activation('relu', name=f'{name}_relu1')(x)
        
        x = layers.Conv2D(filters, 3, padding='same', name=f'{name}_conv2')(x)
        x = layers.BatchNormalization(name=f'{name}_bn2')(x)
        x = layers.Activation('relu', name=f'{name}_relu2')(x)
        
        return x
    
    def build_model(self):
        """Costruisce il modello U-Net completo."""
        
        # Input
        inputs = layers.Input(shape=self.input_shape, name='input')
        
        # Encoder
        encoder, skip_layer_names = self._create_encoder(inputs)
        
        # Bottleneck
        x = encoder.output
        x = layers.Conv2D(512, 3, padding='same', name='bottleneck_conv1')(x)
        x = layers.BatchNormalization(name='bottleneck_bn1')(x)
        x = layers.Activation('relu', name='bottleneck_relu1')(x)
        x = layers.Dropout(self.dropout_rate, name='bottleneck_dropout')(x)
        
        x = layers.Conv2D(512, 3, padding='same', name='bottleneck_conv2')(x)
        x = layers.BatchNormalization(name='bottleneck_bn2')(x)
        x = layers.Activation('relu', name='bottleneck_relu2')(x)
        
        # Skip connections
        skip_connections = []
        for layer_name in skip_layer_names:
            skip_connections.append(encoder.get_layer(layer_name).output)
        
        # Decoder
        x = self._decoder_block(x, skip_connections[3], 256, 'decoder4')
        x = self._decoder_block(x, skip_connections[2], 128, 'decoder3')
        x = self._decoder_block(x, skip_connections[1], 64, 'decoder2')
        x = self._decoder_block(x, skip_connections[0], 32, 'decoder1')
        
        # Output finale
        x = layers.Conv2D(32, 3, padding='same', name='final_conv')(x)
        x = layers.BatchNormalization(name='final_bn')(x)
        x = layers.Activation('relu', name='final_relu')(x)
        
        outputs = layers.Conv2D(self.num_classes, 1, activation='softmax', name='output')(x)
        
        # Crea il modello
        self.model = Model(inputs=inputs, outputs=outputs, name='ModernUNet')
        
        print(f"Modello {self.encoder_name} U-Net creato!")
        print(f"Parametri totali: {self.model.count_params():,}")
        
        return self.model
    
    def load_data(self):
        """Carica e preprocessa i dati."""
        
        print("Caricamento dati...")
        
        # Carica immagini
        train_images = []
        image_files = sorted(glob.glob(os.path.join(self.image_dir, "*.png")))
        
        for img_path in image_files:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.input_shape[:2])
            img = img / 255.0
            train_images.append(img)
        
        # Carica maschere
        train_masks = []
        mask_files = sorted(glob.glob(os.path.join(self.mask_dir, "*.tif")))
        
        for mask_path in mask_files:
            mask = Image.open(mask_path)
            mask = np.array(mask)
            mask = cv2.resize(mask, self.input_shape[:2], interpolation=cv2.INTER_NEAREST)
            
            # Rimappa le etichette: 0 -> 0 (pixel non etichettati), 1-5 -> 0-4
            mask_remapped = np.zeros_like(mask)
            for i in range(1, 6):  # Classi 1-5
                mask_remapped[mask == i] = i - 1  # 1->0, 2->1, 3->2, 4->3, 5->4
            
            # One-hot encoding per le classi utili
            mask_onehot = tf.keras.utils.to_categorical(mask_remapped, num_classes=self.num_classes)
            train_masks.append(mask_onehot)
        
        self.train_images = np.array(train_images)
        self.train_masks = np.array(train_masks)
        
        print(f"Immagini caricate: {self.train_images.shape}")
        print(f"Maschere caricate: {self.train_masks.shape}")
        
    def get_augmentation_pipeline(self):
        """Crea pipeline di data augmentation."""
        
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        ])
    
    def train(self, 
              epochs=100,
              batch_size=4,
              learning_rate=1e-4,
              validation_split=0.2,
              use_augmentation=True):
        """Addestra il modello."""
        
        if self.model is None:
            self.build_model()
        
        # Split dei dati
        X_train, X_val, y_train, y_val = train_test_split(
            self.train_images, self.train_masks, 
            test_size=validation_split, random_state=42
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        
        # Compila il modello
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', self._dice_coefficient]
        )
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                'best_unet_model.h5',
                monitor='val_dice_coefficient',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_dice_coefficient',
                mode='max',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Data augmentation
        if use_augmentation:
            augmentation = self.get_augmentation_pipeline()
            
            def data_generator(X, y, batch_size, augment=False):
                while True:
                    indices = np.random.choice(len(X), batch_size, replace=False)
                    batch_X = X[indices]
                    batch_y = y[indices]
                    
                    if augment:
                        augmented_X = []
                        augmented_y = []
                        
                        for i in range(batch_size):
                            augmented = augmentation(image=batch_X[i], mask=batch_y[i])
                            augmented_X.append(augmented['image'])
                            augmented_y.append(augmented['mask'])
                        
                        batch_X = np.array(augmented_X)
                        batch_y = np.array(augmented_y)
                    
                    yield batch_X, batch_y
            
            train_gen = data_generator(X_train, y_train, batch_size, augment=True)
            val_gen = data_generator(X_val, y_val, batch_size, augment=False)
            
            steps_per_epoch = len(X_train) // batch_size
            validation_steps = len(X_val) // batch_size
            
        else:
            train_gen = None
            val_gen = None
            steps_per_epoch = None
            validation_steps = None
        
        # Training
        if train_gen is not None:
            history = self.model.fit(
                train_gen,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_gen,
                validation_steps=validation_steps,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        else:
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        
        return history
    
    def _dice_coefficient(self, y_true, y_pred, smooth=1e-6):
        """Calcola il coefficiente Dice per la segmentazione."""
        
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    
    def predict_image(self, image_path):
        """Predice la segmentazione per una singola immagine."""
        
        if self.model is None:
            raise ValueError("Modello non addestrato.")
        
        # Carica e preprocessa l'immagine
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_shape[:2])
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Predici
        prediction = self.model.predict(img, verbose=0)
        prediction = np.argmax(prediction[0], axis=-1)
        
        return prediction
    
    def visualize_prediction(self, image_path, save_path=None):
        """Visualizza la predizione."""
        
        # Carica immagine originale
        original_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        original_img = cv2.resize(original_img, self.input_shape[:2])
        
        # Predici
        predicted_mask = self.predict_image(image_path)
        
        # Visualizza
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(original_img)
        axes[0].set_title('Immagine Originale')
        axes[0].axis('off')
        
        axes[1].imshow(predicted_mask, cmap='tab10')
        axes[1].set_title('Segmentazione Predetta')
        axes[1].axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
        return predicted_mask


def main():
    """Esempio di utilizzo."""
    
    # Crea e addestra il modello
    unet = ModernUNet(
        encoder_name='efficientnet_b3',  # Più moderno di VGG16
        input_shape=(512, 512, 3),
        num_classes=5
    )
    
    # Carica i dati
    unet.load_data()
    
    # Addestra
    history = unet.train(
        epochs=50,
        batch_size=2,  # Ridotto per gestire la memoria
        learning_rate=1e-4
    )
    
    print("Training completato!")
    
    # Test su un'immagine
    test_image = "../images/Immagini/1579--03.png"
    if os.path.exists(test_image):
        unet.visualize_prediction(test_image, 'unet_prediction.png')


if __name__ == "__main__":
    main()


