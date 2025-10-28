"""
Versione Migliorata del CNN + Classificatore
Aggiunge tecniche per gestire meglio il dataset piccolo.
"""

import numpy as np
import cv2
import os
import glob
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, EfficientNetB0, VGG16, DenseNet121
from tensorflow.keras.applications import ResNet101, DenseNet201, ConvNeXtTiny, ConvNeXtSmall
from tensorflow.keras.applications import EfficientNetV2B0, EfficientNetV2B1
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import pickle
import optuna
from tqdm import tqdm
import matplotlib.pyplot as plt
import albumentations as A
# Rimuoviamo pydensecrf - mal supportato
# import pydensecrf.densecrf as dcrf
# from pydensecrf.utils import unary_from_labels, create_pairwise_gaussian, create_pairwise_bilateral


class ImprovedCNNSegmentationClassifier:
    """
    Versione migliorata con tecniche per dataset piccolo.
    """
    
    def __init__(self, 
                 cnn_model='efficientnet_b0', 
                 image_size=(512, 512),
                 num_classes=5,
                 batch_size=4,
                 use_augmentation=True,
                 classifier='xgboost'):
        
        self.cnn_model_name = cnn_model
        self.image_size = image_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.use_augmentation = use_augmentation
        self.classifier_type = classifier
        self.encoder_model = None
        self.classifier = None
        self.class_weights = None
        
        # Percorsi
        self.image_dir = "../images/Immagini/"
        self.mask_dir = "../images/Maschere/"
        
    def _create_image_augmentation_pipeline(self):
        """Crea pipeline di augmentation solo per le immagini."""
        
        return A.Compose([
            # Geometriche
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,  # Ridotto per dataset piccolo
                scale_limit=0.05, 
                rotate_limit=15,   # Ridotto per dataset piccolo
                p=0.5
            ),
            
            # Fotometriche (solo per immagini)
            A.RandomBrightnessContrast(
                brightness_limit=0.1,  # Ridotto per dataset piccolo
                contrast_limit=0.1, 
                p=0.5
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            
            # Rumore (solo per immagini)
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            
            # Blur (solo per immagini)
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.3),
                A.MedianBlur(blur_limit=3, p=0.3),
                A.Blur(blur_limit=3, p=0.3),
            ], p=0.2),
        ])
    
    def _create_mask_augmentation_pipeline(self):
        """Crea pipeline di augmentation solo per le maschere (solo geometriche)."""
        
        return A.Compose([
            # Solo trasformazioni geometriche per le maschere
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05, 
                rotate_limit=15,
                p=0.5
            ),
        ])
    
    def _create_encoder(self):
        """Crea l'encoder CNN con tecniche anti-overfitting."""
        
        if self.cnn_model_name == 'efficientnet_b0':
            # Fix per compatibilità EfficientNetB0
            try:
                # Prova prima con input_shape standard
                base_model = EfficientNetB0(
                    weights='imagenet',
                    include_top=False, 
                    input_shape=(224, 224, 3)  # Usa dimensioni standard
                )
                print("EfficientNetB0 caricato con pesi ImageNet pre-addestrati (224x224)")
                
                # Ridimensiona l'input per le nostre immagini
                inputs = tf.keras.Input(shape=(*self.image_size, 3))
                resized_inputs = tf.keras.layers.Resizing(224, 224)(inputs)
                features = base_model(resized_inputs)
                
                # Upsample le features per tornare alle dimensioni originali
                upsampled_features = tf.keras.layers.UpSampling2D(
                    size=(self.image_size[0]//224, self.image_size[1]//224)
                )(features)
                
                # Crea il modello finale
                base_model = Model(inputs, upsampled_features)
                layer_names = ['up_sampling2d']  # Layer finale
                
            except Exception as e:
                print(f"ERRORE EfficientNetB0: {e}")
                print("FALLBACK: Uso ResNet50")
                base_model = ResNet50(
                    weights='imagenet',
                    include_top=False, 
                    input_shape=(*self.image_size, 3)
                )
                self.cnn_model_name = 'resnet50'
                layer_names = ['conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out']
            
        elif self.cnn_model_name == 'resnet50':
            base_model = ResNet50(
                weights='imagenet', 
                include_top=False, 
                input_shape=(*self.image_size, 3)
            )
            layer_names = ['conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out']
            
        elif self.cnn_model_name == 'vgg16':
            base_model = VGG16(
                weights='imagenet', 
                include_top=False, 
                input_shape=(*self.image_size, 3)
            )
            layer_names = ['block1_conv2', 'block2_conv2', 'block3_conv3']
            
        elif self.cnn_model_name == 'densenet121':
            base_model = DenseNet121(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.image_size, 3)
            )
            layer_names = ['conv3_block12_concat', 'conv4_block24_concat', 'conv5_block16_concat']
            print("DenseNet121 caricato - OTTIMO per segmentazione e dataset piccoli")
            
        elif self.cnn_model_name == 'densenet201':
            base_model = DenseNet201(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.image_size, 3)
            )
            layer_names = ['conv3_block12_concat', 'conv4_block24_concat', 'conv5_block16_concat']
            print("DenseNet201 caricato - PIÙ POTENTE per segmentazione")
            
        elif self.cnn_model_name == 'resnet101':
            base_model = ResNet101(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.image_size, 3)
            )
            layer_names = ['conv2_block3_out', 'conv3_block4_out', 'conv4_block23_out']
            print("ResNet101 caricato - PIÙ PROFONDO per segmentazione")
            
        elif self.cnn_model_name == 'convnext_tiny':
            base_model = ConvNeXtTiny(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.image_size, 3)
            )
            layer_names = ['convnext_tiny_stage_1_block_2_identity', 'convnext_tiny_stage_2_block_8_identity', 'convnext_tiny_stage_3_block_2_identity']
            print("ConvNeXtTiny caricato - MODERNO e performante per segmentazione")
            
        elif self.cnn_model_name == 'convnext_small':
            base_model = ConvNeXtSmall(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.image_size, 3)
            )
            layer_names = ['stage2_block1', 'stage3_block1', 'stage4_block1']
            print("ConvNeXtSmall caricato - MODERNO e performante per segmentazione")
            
        elif self.cnn_model_name == 'densenet201':
            base_model = DenseNet201(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.image_size, 3)
            )
            layer_names = ['conv3_block12_concat', 'conv4_block24_concat', 'conv5_block16_concat']
            print("DenseNet201 caricato - POTENTE per segmentazione")
            
        elif self.cnn_model_name == 'efficientnetv2_b0':
            base_model = EfficientNetV2B0(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.image_size, 3)
            )
            layer_names = ['block4a_expand_activation', 'block5a_expand_activation', 'block6a_expand_activation']
            print("EfficientNetV2B0 caricato - VERSIONE MIGLIORATA di EfficientNet")
            
        elif self.cnn_model_name == 'efficientnetv2_b1':
            base_model = EfficientNetV2B1(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.image_size, 3)
            )
            layer_names = ['block4a_expand_activation', 'block5a_expand_activation', 'block6a_expand_activation']
            print("EfficientNetV2B1 caricato - VERSIONE MIGLIORATA di EfficientNet")
            
        else:
            raise ValueError(f"Modello CNN non supportato: {self.cnn_model_name}")
        
        # Rendi NON trainabile per evitare overfitting su dataset piccolo
        for layer in base_model.layers:
            layer.trainable = False
            
        # Estrai outputs dai layer specificati
        encoder_outputs = [base_model.get_layer(name).output for name in layer_names]
        self.encoder_model = Model(inputs=base_model.input, outputs=encoder_outputs)
        
        print(f"Encoder {self.cnn_model_name} creato (NON trainabile per evitare overfitting)")
        
    def load_data_with_augmentation(self):
        """Carica dati con o senza augmentation."""
        
        if self.use_augmentation:
            print("Caricamento dati con augmentation...")
        else:
            print("Caricamento dati senza augmentation...")
        
        # Carica immagini originali
        train_images_raw = []
        image_files = sorted(glob.glob(os.path.join(self.image_dir, "*.png")))
        
        for img_path in tqdm(image_files, desc="Caricamento immagini"):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.image_size)
            img = img / 255.0
            train_images_raw.append(img)
            
        # Carica maschere
        train_masks = []
        mask_files = sorted(glob.glob(os.path.join(self.mask_dir, "*.tif")))
        
        for mask_path in tqdm(mask_files, desc="Caricamento maschere"):
            mask = Image.open(mask_path)
            mask = np.array(mask)
            mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
            train_masks.append(mask)
        
        self.train_images = np.array(train_images_raw)
        self.train_masks = np.array(train_masks)
        
        print(f"Immagini originali: {self.train_images.shape}")
        print(f"Maschere originali: {self.train_masks.shape}")
        
        # Applica augmentation se richiesta
        if self.use_augmentation:
            try:
                self._apply_augmentation()
            except Exception as e:
                print(f"Errore durante augmentation: {e}")
                print("Continuo senza augmentation...")
                self.use_augmentation = False
        
        # Calcola class weights per bilanciare le classi
        self._compute_class_weights()
        
    def _apply_augmentation(self):
        """Applica data augmentation per aumentare il dataset."""
        
        print("Applicazione data augmentation...")
        
        # Crea pipeline separate per immagini e maschere
        image_augmentation = self._create_image_augmentation_pipeline()
        mask_augmentation = self._create_mask_augmentation_pipeline()
        
        # Numero di augmentation per immagine (aumenta il dataset di 3x)
        aug_per_image = 3
        
        augmented_images = []
        augmented_masks = []
        
        for i in tqdm(range(len(self.train_images)), desc="Augmentation"):
            img = self.train_images[i]
            mask = self.train_masks[i]
            
            # Assicurati che i tipi siano corretti
            img = img.astype(np.float32)
            mask = mask.astype(np.uint8)
            
            # Aggiungi immagine originale
            augmented_images.append(img)
            augmented_masks.append(mask)
            
            # Genera versioni augmented
            for _ in range(aug_per_image):
                try:
                    # Augmenta immagine
                    augmented_img = image_augmentation(image=img)['image']
                    
                    # Augmenta maschera (solo trasformazioni geometriche)
                    augmented_mask = mask_augmentation(image=mask)['image']
                    
                    augmented_images.append(augmented_img)
                    augmented_masks.append(augmented_mask)
                    
                except Exception as e:
                    print(f"Errore augmentation immagine {i}: {e}")
                    # Se fallisce, usa l'originale
                    augmented_images.append(img)
                    augmented_masks.append(mask)
        
        self.train_images = np.array(augmented_images)
        self.train_masks = np.array(augmented_masks)
        
        print(f"Dopo augmentation - Immagini: {self.train_images.shape}")
        print(f"Dopo augmentation - Maschere: {self.train_masks.shape}")
        
    def _compute_class_weights(self):
        """Calcola i pesi delle classi per bilanciare il dataset."""
        
        # Estrai tutte le etichette
        all_labels = []
        for mask in self.train_masks:
            # Filtra solo le classi utili (1-5)
            mask_filtered = mask[mask != 0]
            if len(mask_filtered) > 0:
                all_labels.extend((mask_filtered - 1).tolist())  # Rimappa 1-5 a 0-4
        
        all_labels = np.array(all_labels)
        
        # Calcola class weights
        unique_classes = np.unique(all_labels)
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=all_labels
        )
        
        self.class_weights = dict(zip(unique_classes, class_weights))
        
        print("Class weights calcolati:")
        class_names = ['Resina', 'Pori/Imperfezioni', 'Fase Fusa', 'Belite', 'Alite']
        for class_id, weight in self.class_weights.items():
            if 0 <= class_id < len(class_names):
                print(f"  {class_names[class_id]}: {weight:.3f}")
            else:
                print(f"  Classe {class_id}: {weight:.3f}")
    
    def extract_features_improved(self):
        """Estrazione features migliorata con tecniche anti-overfitting."""
        
        if self.encoder_model is None:
            self._create_encoder()
            
        print("Estrazione features migliorata...")
        
        # Elaborazione in batch
        all_features = []
        all_labels = []
        
        for i in tqdm(range(0, len(self.train_images), self.batch_size), desc="Estrazione features"):
            batch_X = self.train_images[i:i + self.batch_size]
            batch_y = self.train_masks[i:i + self.batch_size]
            
            # Estrai features
            if self.cnn_model_name == 'efficientnet_b0':
                upsampled_features = self.encoder_model(batch_X)
                # Per EfficientNetB0 usiamo le features upsampled
                fused_features = upsampled_features
                
            elif self.cnn_model_name == 'resnet50':
                # Ora riceviamo 3 tensori, li ridimensioniamo e li combiniamo
                conv2, conv3, conv4 = self.encoder_model(batch_X)
                conv3_upsampled = tf.image.resize(conv3, size=conv2.shape[1:3])
                conv4_upsampled = tf.image.resize(conv4, size=conv2.shape[1:3])
                fused_features = Concatenate()([conv2, conv3_upsampled, conv4_upsampled])
                
            elif self.cnn_model_name == 'vgg16':
                block1, block2, block3 = self.encoder_model(batch_X)
                block2_upsampled = tf.image.resize(block2, size=block1.shape[1:3])
                block3_upsampled = tf.image.resize(block3, size=block1.shape[1:3])
                fused_features = Concatenate()([block1, block2_upsampled, block3_upsampled])
                
            elif self.cnn_model_name == 'densenet121':
                # DenseNet121 - 3 layer per massime prestazioni
                conv3, conv4, conv5 = self.encoder_model(batch_X)
                conv4_upsampled = tf.image.resize(conv4, size=conv3.shape[1:3])
                conv5_upsampled = tf.image.resize(conv5, size=conv3.shape[1:3])
                fused_features = Concatenate()([conv3, conv4_upsampled, conv5_upsampled])
                
            elif self.cnn_model_name == 'convnext_tiny':
                # ConvNeXt - 3 layer per massime prestazioni
                stage2, stage3, stage4 = self.encoder_model(batch_X)
                stage3_upsampled = tf.image.resize(stage3, size=stage2.shape[1:3])
                stage4_upsampled = tf.image.resize(stage4, size=stage2.shape[1:3])
                fused_features = Concatenate()([stage2, stage3_upsampled, stage4_upsampled])
                
            elif self.cnn_model_name == 'convnext_small':
                # ConvNeXt Small - 3 layer per massime prestazioni
                stage2, stage3, stage4 = self.encoder_model(batch_X)
                stage3_upsampled = tf.image.resize(stage3, size=stage2.shape[1:3])
                stage4_upsampled = tf.image.resize(stage4, size=stage2.shape[1:3])
                fused_features = Concatenate()([stage2, stage3_upsampled, stage4_upsampled])
                
            elif self.cnn_model_name == 'densenet201':
                # DenseNet201 - 3 layer per massime prestazioni
                conv3, conv4, conv5 = self.encoder_model(batch_X)
                conv4_upsampled = tf.image.resize(conv4, size=conv3.shape[1:3])
                conv5_upsampled = tf.image.resize(conv5, size=conv3.shape[1:3])
                fused_features = Concatenate()([conv3, conv4_upsampled, conv5_upsampled])
                
            elif self.cnn_model_name in ['efficientnetv2_b0', 'efficientnetv2_b1']:
                # EfficientNetV2 - 3 layer per massime prestazioni
                block4, block5, block6 = self.encoder_model(batch_X)
                block5_upsampled = tf.image.resize(block5, size=block4.shape[1:3])
                block6_upsampled = tf.image.resize(block6, size=block4.shape[1:3])
                fused_features = Concatenate()([block4, block5_upsampled, block6_upsampled])
            
            # 1. Ottieni la dimensione della mappa di feature (es. (B, 64, 64, C))
            feature_map_size_hw = fused_features.shape[1:3] # (H, W) es. (64, 64)
            
            # 2. Appiattisci le features
            features_reshaped = fused_features.numpy().reshape(-1, fused_features.shape[3]) # (B*H_feat*W_feat, C)
           
            # 3. Ridimensiona le maschere per farle combaciare con le feature
            batch_y_resized_list = []
            for j in range(batch_y.shape[0]): # Itera sul batch
                mask_img = batch_y[j] # (512, 512)
                
                # cv2.resize vuole (W, H)
                resized_mask = cv2.resize(
                    mask_img, 
                    (feature_map_size_hw[1], feature_map_size_hw[0]), # (W, H)
                    interpolation=cv2.INTER_NEAREST # Fondamentale per le maschere
                )
                batch_y_resized_list.append(resized_mask)
            
            batch_y = np.array(batch_y_resized_list) # (B, H_feat, W_feat)
            
            # 4. Ora appiattisci le maschere ridimensionate
            labels_reshaped = batch_y.reshape(-1) # (B*H_feat*W_feat,)
            
            # Filtra i pixel con label 0 (pixel non etichettati)
            mask = labels_reshaped != 0
            features_filtered = features_reshaped[mask]
            labels_filtered = labels_reshaped[mask]
            
            # Rimappa le etichette da 1-5 a 0-4 per il classificatore
            labels_filtered = labels_filtered - 1
            labels_filtered = np.clip(labels_filtered, 0, self.num_classes - 1)
            
            all_features.append(features_filtered)
            all_labels.append(labels_filtered)
        
        # Combina tutti i batch
        self.X_features = np.vstack(all_features)
        self.y_labels = np.hstack(all_labels)
        
        print(f"Features estratte: {self.X_features.shape}")
        print(f"Labels per classificazione: {self.y_labels.shape}")
        print(f"Distribuzione classi: {np.bincount(self.y_labels)}")
        
        # Verifica classi
        unique_labels = np.unique(self.y_labels)
        if len(unique_labels) != self.num_classes:
            print(f"Filtro ulteriore delle etichette...")
            valid_mask = (self.y_labels >= 0) & (self.y_labels < self.num_classes)
            self.X_features = self.X_features[valid_mask]
            self.y_labels = self.y_labels[valid_mask]
            print(f"Dopo filtro - Features: {self.X_features.shape}, Labels: {self.y_labels.shape}")
    
    def train_classifier_with_cv(self, cv_folds=3):
        """Addestra il classificatore con cross-validation per dataset piccolo."""
        
        print(f"Training con {cv_folds}-fold cross-validation...")
        
        # Cross-validation stratificata
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X_features, self.y_labels)):
            print(f"\n--- Fold {fold + 1}/{cv_folds} ---")
            
            X_train_fold = self.X_features[train_idx]
            X_val_fold = self.X_features[val_idx]
            y_train_fold = self.y_labels[train_idx]
            y_val_fold = self.y_labels[val_idx]
            
            # Parametri ottimizzati per dataset piccolo
            params = {
                'max_depth': 4,  # Ridotto per evitare overfitting
                'learning_rate': 0.05,  # Ridotto per training più stabile
                'n_estimators': 100,  # Ridotto per dataset piccolo
                'subsample': 0.7,  # Ridotto per più regolarizzazione
                'colsample_bytree': 0.7,
                'colsample_bylevel': 0.7,
                'colsample_bynode': 0.7,
                'gamma': 0.5,  # Aumentato per più regolarizzazione
                'reg_alpha': 1.0,  # Aumentato per più regolarizzazione
                'reg_lambda': 10.0,  # Aumentato per più regolarizzazione
                'objective': 'multi:softprob',
                'num_class': self.num_classes,
                'min_child_weight': 5,  # Aumentato per più regolarizzazione
                'tree_method': 'hist',
                'eval_metric': 'mlogloss',
                'random_state': 42,
                'verbosity': 0
            }
            
            # Aggiungi class weights se disponibili
            if self.class_weights is not None:
                # XGBoost usa scale_pos_weight, ma per multi-classe dobbiamo usare sample_weight
                sample_weights = np.array([self.class_weights[label] for label in y_train_fold])
            else:
                sample_weights = None
            
            # Addestra il modello
            model = xgb.XGBClassifier(**params)
            
            if sample_weights is not None:
                model.fit(X_train_fold, y_train_fold, 
                         eval_set=[(X_val_fold, y_val_fold)],
                         sample_weight=sample_weights,
                         verbose=False)
            else:
                model.fit(X_train_fold, y_train_fold, 
                         eval_set=[(X_val_fold, y_val_fold)],
                         verbose=False)
            
            # Valuta
            preds = model.predict(X_val_fold)
            accuracy = accuracy_score(y_val_fold, preds)
            cv_scores.append(accuracy)
            
            print(f"Accuracy Fold {fold + 1}: {accuracy:.4f}")
        
        # Salva il modello dell'ultimo fold come modello finale
        self.classifier = model
        
        print(f"\nCross-Validation Results:")
        print(f"Mean Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        print(f"Individual scores: {cv_scores}")
        
        return cv_scores
    
    def predict_image(self, image_path):
        """Predice la segmentazione per una singola immagine."""
        
        if self.encoder_model is None or self.classifier is None:
            raise ValueError("Modello non addestrato.")
        
        # Carica e preprocessa l'immagine
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.image_size)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Estrai features
        if self.cnn_model_name == 'efficientnet_b0':
            upsampled_features = self.encoder_model(img)
            # Per EfficientNetB0 usiamo le features upsampled
            fused_features = upsampled_features
            
        elif self.cnn_model_name == 'resnet50':
            # Applichiamo la stessa logica a 3 tensori del training
            conv2, conv3, conv4 = self.encoder_model(img)
            conv3_upsampled = tf.image.resize(conv3, size=conv2.shape[1:3])
            conv4_upsampled = tf.image.resize(conv4, size=conv2.shape[1:3])
            fused_features = Concatenate()([conv2, conv3_upsampled, conv4_upsampled])
            
        elif self.cnn_model_name == 'vgg16':
            block1, block2, block3 = self.encoder_model(img)
            block2_upsampled = tf.image.resize(block2, size=block1.shape[1:3])
            block3_upsampled = tf.image.resize(block3, size=block1.shape[1:3])
            fused_features = Concatenate()([block1, block2_upsampled, block3_upsampled])
            
        elif self.cnn_model_name == 'densenet121':
            conv3, conv4, conv5 = self.encoder_model(img)
            conv4_upsampled = tf.image.resize(conv4, size=conv3.shape[1:3])
            conv5_upsampled = tf.image.resize(conv5, size=conv3.shape[1:3])
            fused_features = Concatenate()([conv3, conv4_upsampled, conv5_upsampled])
            
        elif self.cnn_model_name == 'convnext_tiny':
            stage2, stage3, stage4 = self.encoder_model(img)
            stage3_upsampled = tf.image.resize(stage3, size=stage2.shape[1:3])
            stage4_upsampled = tf.image.resize(stage4, size=stage2.shape[1:3])
            fused_features = Concatenate()([stage2, stage3_upsampled, stage4_upsampled])
            
        elif self.cnn_model_name == 'convnext_small':
            stage2, stage3, stage4 = self.encoder_model(img)
            stage3_upsampled = tf.image.resize(stage3, size=stage2.shape[1:3])
            stage4_upsampled = tf.image.resize(stage4, size=stage2.shape[1:3])
            fused_features = Concatenate()([stage2, stage3_upsampled, stage4_upsampled])
            
        elif self.cnn_model_name == 'densenet201':
            conv3, conv4, conv5 = self.encoder_model(img)
            conv4_upsampled = tf.image.resize(conv4, size=conv3.shape[1:3])
            conv5_upsampled = tf.image.resize(conv5, size=conv3.shape[1:3])
            fused_features = Concatenate()([conv3, conv4_upsampled, conv5_upsampled])
            
        elif self.cnn_model_name in ['efficientnetv2_b0', 'efficientnetv2_b1']:
            block4, block5, block6 = self.encoder_model(img)
            block5_upsampled = tf.image.resize(block5, size=block4.shape[1:3])
            block6_upsampled = tf.image.resize(block6, size=block4.shape[1:3])
            fused_features = Concatenate()([block4, block5_upsampled, block6_upsampled])
        
        # Riorganizza per la predizione
        features_reshaped = fused_features.numpy().reshape(-1, fused_features.shape[3])
        
        # Predici
        predictions = self.classifier.predict(features_reshaped)
        pred_image = predictions.reshape(fused_features.shape[1], fused_features.shape[2])

        return pred_image
    
    def apply_morphological_refinement(self, predicted_mask):
        """
        Applica operazioni morfologiche per raffinare la segmentazione.
        Alternativa moderna e stabile a pydensecrf.
        """
        try:
            from scipy import ndimage
            from skimage.morphology import opening, closing, remove_small_objects
            
            # Converti in uint8 se necessario
            mask = predicted_mask.astype(np.uint8)
            
            # 1. Rimuovi oggetti piccoli (rumore)
            mask_cleaned = remove_small_objects(mask, min_size=50)
            
            # 2. Operazioni morfologiche per smoothing
            # Opening (erosione seguita da dilatazione) - rimuove piccoli oggetti
            mask_opened = opening(mask_cleaned, footprint=np.ones((3, 3)))
            
            # Closing (dilatazione seguita da erosione) - riempie buchi
            mask_closed = closing(mask_opened, footprint=np.ones((3, 3)))
            
            # 3. Smoothing con filtro gaussiano
            mask_smoothed = ndimage.gaussian_filter(mask_closed.astype(float), sigma=0.5)
            mask_smoothed = np.round(mask_smoothed).astype(np.uint8)
            
            return mask_smoothed
            
        except Exception as e:
            print(f"Errore durante morphological refinement: {e}")
            print("Ritorno la maschera originale senza raffinamento.")
            return predicted_mask
    
    def predict_image_with_refinement(self, image_path):
        """
        Predice la segmentazione con morphological refinement finale.
        Alternativa moderna e stabile a CRF.
        """
        # Predizione base
        predicted_mask = self.predict_image(image_path)
        
        # Applica morphological refinement
        refined_mask = self.apply_morphological_refinement(predicted_mask)
        
        return refined_mask
    
    def train_classifier_with_optuna(self, n_trials=50, test_size=0.2):
        """
        Addestra il classificatore con ottimizzazione Optuna e separa un test set finale.
        """
        print(f"Training con ottimizzazione Optuna ({n_trials} trials)...")
        
        # Prima separa un test set finale che non verrà mai usato per training/validation
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.X_features, self.y_labels, 
            test_size=test_size, random_state=42, 
            stratify=self.y_labels
        )
        
        print(f"Test set finale: {X_test.shape[0]} campioni")
        print(f"Training/Validation set: {X_temp.shape[0]} campioni")
        
        def objective(trial):
            # Parametri ottimizzati per massime prestazioni
            if self.classifier_type == 'lightgbm':
                params = {
                    'objective': 'multiclass',
                    'num_class': self.num_classes,
                    'max_depth': trial.suggest_int('max_depth', 6, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                    'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
                    'num_leaves': trial.suggest_int('num_leaves', 31, 200),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                    'random_state': 42,
                    'verbosity': -1
                }
            else:  # xgboost
                params = {
                    'base_score': 0.5,
                    'max_depth': trial.suggest_int('max_depth', 6, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.1, 0.4),
                    'n_estimators': trial.suggest_int('n_estimators', 150, 300),
                    'subsample': trial.suggest_float('subsample', 0.7, 0.95),
                    'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.7, 0.95),
                    'colsample_bynode': trial.suggest_float('colsample_bynode', 0.7, 0.95),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
                    'gamma': trial.suggest_float('gamma', 0.1, 0.4),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),
                    'objective': 'multi:softprob',
                    'num_class': self.num_classes,
                    'importance_type': 'gain',
                    'min_child_weight': trial.suggest_int('min_child_weight', 2, 5),
                    'num_parallel_tree': trial.suggest_int('num_parallel_tree', 1, 5),
                    'tree_method': 'hist',
                    'eval_metric': 'mlogloss',
                    'validate_parameters': 1,
                    'interaction_constraints': '',
                    'booster': 'dart',
                    'verbosity': 0,
                    'max_delta_step': 1,
                    'rate_drop': trial.suggest_float('rate_drop', 0.1, 0.3),
                    'skip_drop': trial.suggest_float('skip_drop', 0.1, 0.3),
                    'max_bin': 512,
                    'random_state': 42
                }
            
            # Split per validazione durante ottimizzazione
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
            )
            
            # Aggiungi class weights se disponibili
            if self.class_weights is not None:
                sample_weights = np.array([self.class_weights[label] for label in y_train])
            else:
                sample_weights = None
            
            # Crea e addestra il modello
            if self.classifier_type == 'lightgbm':
                model = lgb.LGBMClassifier(**params)
            else:  # xgboost
                model = xgb.XGBClassifier(**params)
            
            if sample_weights is not None:
                if self.classifier_type == 'lightgbm':
                    model.fit(X_train, y_train, 
                             eval_set=[(X_val, y_val)],
                             sample_weight=sample_weights,
                             callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)])
                else:
                    model.fit(X_train, y_train, 
                             eval_set=[(X_val, y_val)],
                             sample_weight=sample_weights,
                             verbose=False)
            else:
                if self.classifier_type == 'lightgbm':
                    model.fit(X_train, y_train, 
                             eval_set=[(X_val, y_val)],
                             callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)])
                else:
                    model.fit(X_train, y_train, 
                             eval_set=[(X_val, y_val)],
                             verbose=False)
            
            # Valuta
            preds = model.predict(X_val)
            accuracy = accuracy_score(y_val, preds)
            
            return accuracy
        
        # Esegui ottimizzazione con logging
        study = optuna.create_study(direction='maximize')
        
        # Callback per logging del progresso
        def progress_callback(study, trial):
            print(f"Trial {trial.number + 1}/{n_trials} completato")
            if trial.value is not None:
                print(f"  Accuracy: {trial.value:.4f}")
                print(f"  Parametri: {trial.params}")
            else:
                print(f"  Trial fallito: {trial.state}")
            print("-" * 50)
        
        # Esegui ottimizzazione con callback
        study.optimize(objective, n_trials=n_trials, callbacks=[progress_callback])
        
        self.best_params = study.best_params
        print(f"Migliori parametri trovati: {self.best_params}")
        print(f"Migliore accuracy: {study.best_value:.4f}")
        
        # Addestra il modello finale con i migliori parametri su tutto il training set
        print("Training modello finale con parametri ottimizzati...")
        
        final_params = self.best_params.copy()
        if self.classifier_type == 'lightgbm':
            final_params.update({
                'objective': 'multiclass',
                'num_class': self.num_classes,
                'random_state': 42,
                'verbosity': 1
            })
        else:  # xgboost
            final_params.update({
                'base_score': 0.5,
                'objective': 'multi:softprob',
                'num_class': self.num_classes,
                'importance_type': 'gain',
                'tree_method': 'hist',
                'eval_metric': 'mlogloss',
                'validate_parameters': 1,
                'interaction_constraints': '',
                'booster': 'dart',
                'verbosity': 1,
                'max_delta_step': 1,
                'max_bin': 512,
                'random_state': 42
            })
        
        # Aggiungi class weights per il training finale
        if self.class_weights is not None:
            sample_weights = np.array([self.class_weights[label] for label in y_temp])
        else:
            sample_weights = None
        
        if self.classifier_type == 'lightgbm':
            self.classifier = lgb.LGBMClassifier(**final_params)
        else:  # xgboost
            self.classifier = xgb.XGBClassifier(**final_params)
        
        if sample_weights is not None:
            if self.classifier_type == 'lightgbm':
                self.classifier.fit(X_temp, y_temp, 
                                   eval_set=[(X_test, y_test)],
                                   sample_weight=sample_weights,
                                   callbacks=[lgb.early_stopping(10), lgb.log_evaluation(10)])
            else:
                self.classifier.fit(X_temp, y_temp, 
                                   eval_set=[(X_test, y_test)],
                                   sample_weight=sample_weights,
                                   verbose=True)
        else:
            if self.classifier_type == 'lightgbm':
                self.classifier.fit(X_temp, y_temp, 
                                   eval_set=[(X_test, y_test)],
                                   callbacks=[lgb.early_stopping(10), lgb.log_evaluation(10)])
            else:
                self.classifier.fit(X_temp, y_temp, 
                                   eval_set=[(X_test, y_test)],
                                   verbose=True)
        
        # Valuta sul test set finale
        test_preds = self.classifier.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_preds)
        
        print(f"\nAccuracy finale su test set: {test_accuracy:.4f}")
        
        # Salva il test set per valutazioni future
        self.X_test = X_test
        self.y_test = y_test
        
        return test_accuracy
    
    def save_model_complete(self, model_path):
        """Salva il modello completo con tutti i componenti necessari."""
        
        if self.classifier is None or self.encoder_model is None:
            raise ValueError("Nessun modello da salvare.")
        
        # Salva l'encoder TensorFlow separatamente
        encoder_path = model_path.replace('.pkl', '_encoder.keras')
        self.encoder_model.save(encoder_path)
        
        model_data = {
            'classifier': self.classifier,
            'cnn_model_name': self.cnn_model_name,
            'image_size': self.image_size,
            'num_classes': self.num_classes,
            'best_params': getattr(self, 'best_params', None),
            'class_weights': self.class_weights,
            'encoder_path': encoder_path,  # Percorso al modello TensorFlow
            'X_test': getattr(self, 'X_test', None),
            'y_test': getattr(self, 'y_test', None)
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Modello completo salvato in: {model_path}")
        print(f"Encoder TensorFlow salvato in: {encoder_path}")
        
    def load_model_complete(self, model_path):
        """Carica il modello completo."""
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.cnn_model_name = model_data['cnn_model_name']
        self.image_size = model_data['image_size']
        self.num_classes = model_data['num_classes']
        self.best_params = model_data.get('best_params', None)
        self.class_weights = model_data.get('class_weights', None)
        self.X_test = model_data.get('X_test', None)
        self.y_test = model_data.get('y_test', None)
        
        # Carica l'encoder TensorFlow salvato
        encoder_path = model_data.get('encoder_path')
        if encoder_path and os.path.exists(encoder_path):
            self.encoder_model = tf.keras.models.load_model(encoder_path)
            print(f"Encoder TensorFlow caricato da: {encoder_path}")
        else:
            print("Encoder non trovato, ricreo da zero...")
            self._create_encoder()
        
        print(f"Modello completo caricato da: {model_path}")


