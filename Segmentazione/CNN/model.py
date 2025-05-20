import glob
import cv2
import tensorflow as tf
import os
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Conv2D, UpSampling2D
from PIL import Image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Concatenate
from keras.applications import ResNet50

import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


path_images = "/content/drive/MyDrive/Segmentazione/Immagini"
path_masks = "/content/drive/MyDrive/Segmentazione/Maschere"

train_images_raw = []

for directory_path in glob.glob(path_images):
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        #img = cv2.resize(img, (SIZE_Y, SIZE_X))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img / 127.5 - 1  # Normalizzazione tra -1 e 1
        train_images_raw.append(img)
        #train_labels.append(label)
#Convert list to array for machine learning processing
train_images = np.array(train_images_raw)
print("immagini: " , train_images.shape)

train_masks = []
for directory_path in glob.glob(path_masks):
    for mask_path in glob.glob(os.path.join(directory_path, "*.tif")):
        mask = Image.open(mask_path)
        mask = np.array(mask)
        #mask = mask.astype(np.uint8)
        #mask = cv2.resize(mask, (SIZE_Y, SIZE_X))
        #mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
        train_masks.append(mask)
        #train_labels.append(label)
#Convert list to array for machine learning processing
train_masks = np.array(train_masks)
print("maschere: " , train_masks.shape)

#Use customary x_train and y_train variables
X_train = train_images
y_train = train_masks


# Funzione per dividere i dati in batch
def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

# Parametri di input
batch_size = 3  # Scegli un batch size appropriato

# Carica il modello VGG16 senza strati fully connected
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE_X, SIZE_Y, 3))
for layer in base_model.layers:
    layer.trainable = False

layer_names = ['block1_conv2', 'block2_conv2']
encoder_outputs = [base_model.get_layer(name).output for name in layer_names]
encoder_model = Model(inputs=base_model.input, outputs=encoder_outputs)

# Elaborazione in batch
all_features = []
all_labels = []

for batch_X, batch_y in zip(batch_generator(X_train, batch_size), batch_generator(y_train, batch_size)):
    block1, block2 = encoder_model(batch_X)
    block2_upsampled = tf.image.resize(block2, size=block1.shape[1:3])
    fused_features = Concatenate()([block1, block2_upsampled])

    # Converti a numpy e ridimensiona
    features_reshaped = fused_features.numpy().reshape(-1, fused_features.shape[3])
    labels_reshaped = batch_y.reshape(-1)

    # Filtra i pixel con label 0
    mask = labels_reshaped != 0
    features_filtered = features_reshaped[mask]
    labels_filtered = labels_reshaped[mask]

    # Accoda i risultati
    all_features.append(features_filtered)
    all_labels.append(labels_filtered)

# Combina i batch elaborati
X_for_RF = np.vstack(all_features)
Y_for_RF = np.hstack(all_labels) - 1  # Sottrai 1 se necessario


# OTTIMIZZAZIONE DEI PARAMETRI PER TRAINING DEL MODELLO

X_train, X_val, y_train, y_val = train_test_split(X_for_RF, Y_for_RF, test_size=0.2, random_state=42)

def objective(trial):
    # Definizione dei parametri da ottimizzare
    param = {
        'base_score':0.5,
        'max_depth': trial.suggest_int('max_depth',6,8),  # Profondità dell'albero
        'learning_rate': trial.suggest_float('learning_rate', 0.1,0.4),  # Tasso di apprendimento
        'n_estimators': trial.suggest_int('n_estimators',150,300),  # Numero di alberi
        'subsample': trial.suggest_float('subsample',0.7,0.95),  # Percentuale di dati usati
        'colsample_bylevel': trial.suggest_float('colsample_bylevel',0.7,0.95),  # Ridotto per evitare overfitting
        'colsample_bynode': trial.suggest_float('colsample_bynode',0.7,0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree',0.7,0.95),  # Percentuale di feature usate
        'gamma': trial.suggest_float('gamma',0.1,0.4),  # Regolarizzazione per split
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),
        'objective': 'multi:softprob',  # Classificazione multi-classe
        'num_class': len(set(Y_for_RF)),  # Numero di classi
        'importance_type':'gain',
        'min_child_weight': trial.suggest_int('min_child_weight',2,5),
        'num_parallel_tree':trial.suggest_int('num_parallel_tree',1,5),
        'tree_method':'hist',
        'device':'cuda',
        'eval_metric':'mlogloss',
        'validate_parameters':1,
        'interaction_constraints':'',
        'booster':'dart',
        'verbosity' : 1,
        'max_delta_step': 1,
        'rate_drop': trial.suggest_float('rate_drop', 0.1,0.3),  # Parametro specifico di DART
        'skip_drop': trial.suggest_float('skip_drop', 0.1,0.3),  # Parametro specifico di DART
        'max_bin': 512
    }

    # Creazione del modello con i parametri suggeriti
    model = xgb.XGBClassifier(**param, random_state=42)

    # Addestramento
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Predizione e valutazione
    preds = model.predict(X_val)
    accuracy = accuracy_score(y_val, preds)

    return accuracy  # Optuna cercherà di massimizzare questo valore


study = optuna.create_study(direction="maximize")  # Maximizing accuracy
study.optimize(objective, n_trials=20)  # Esegui x iterazioni

print("Best params:", study.best_params)
print("Best value:", study.best_value)  # Migliore accuratezza


from optuna.visualization import plot_param_importances

plot_param_importances(study).show()


from optuna.visualization import plot_optimization_history

plot_optimization_history(study).show()

best_params = study.best_params
final_model = xgb.XGBClassifier(**best_params, random_state=42)
final_model.fit(X_train, y_train)

# ADDESTRAMENTO DEL MODELLO CON PARAMETRI OTTIMIZZATI

X_train, X_val, y_train, y_val = train_test_split(X_for_RF, Y_for_RF, test_size=0.2, random_state=42)

param = {
    'base_score': 0.5,
    'max_depth': 8,  # profondità
    'learning_rate': 0.35,  # Tasso di apprendimento
    'n_estimators': 250,  # numero di iterazioni
    'subsample': 0.91,  # dati usati
    'colsample_bytree': 0.8,  # feature usate
    'colsample_bylevel': 0.8,
    'colsample_bynode': 0.8,
    'gamma': 0.13,
    'reg_alpha': 0.8,
    'reg_lambda': 8.7,
    'objective': 'multi:softprob',
    'num_class': len(set(Y_for_RF)),
    'min_child_weight': 2,  # Incrementato per meno split
    'num_parallel_tree': 1,
    'tree_method': 'gpu_hist',  # Metodo più rapido
    'booster': 'dart',
    'rate_drop': 0.12,  # Parametro specifico di DART
    'skip_drop': 0.28,  # Parametro specifico di DART
    'verbosity': 2,
    'max_delta_step': 1,
    'importance_type': 'gain',
    'validate_parameters': 1,
    'interaction_constraints': '',
    'eval_metric': 'mlogloss',
    'max_bin': 512,
}
final_model = xgb.XGBClassifier(**param, random_state=42)
final_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],  # Dati di validazione
    verbose=True)



# VALUTAZIONE DELLE METRICHE DELLE CLASSI

X_train, X_val, y_train, y_val = train_test_split(X_for_RF, Y_for_RF, test_size=0.2, random_state=42)

prediction = final_model.predict(X_val)

#Using built in keras function
import numpy as np
from keras.metrics import MeanIoU
from sklearn.metrics import confusion_matrix

num_classes = 5
IOU_keras = MeanIoU(num_classes=num_classes)
IOU_keras.update_state(y_val, prediction)
print("Mean IoU =", IOU_keras.result().numpy())

# Calculate confusion matrix to get IoU for each class
cm = confusion_matrix(y_val, prediction, labels=np.arange(num_classes))

# Calculate IoU for each class
# Note: This assumes classes are labeled 0, 1, 2, ...
class_iou = []
for i in range(num_classes):
    tp = cm[i, i]
    fp = cm[i, :].sum() - tp
    fn = cm[:, i].sum() - tp
    iou = tp / (tp + fp + fn)
    class_iou.append(iou)
    print(f"IoU Class {i} =", iou)
