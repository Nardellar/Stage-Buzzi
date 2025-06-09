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


# Define image size constants
SIZE_X = 256  # Define your desired image size
SIZE_Y = 256  # Define your desired image size

path_images = "../images/Immagini"
path_masks = "../images/Maschere"

train_images_raw = []

print("Loading and preprocessing images...")
for directory_path in glob.glob(path_images):
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE_Y, SIZE_X))  # Added resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Fixed color conversion
        img = img / 127.5 - 1  # Normalization between -1 and 1
        train_images_raw.append(img)

# Convert list to array for machine learning processing
train_images = np.array(train_images_raw)
print("Images shape:", train_images.shape)

train_masks = []
print("Loading and preprocessing masks...")
for directory_path in glob.glob(path_masks):
    for mask_path in glob.glob(os.path.join(directory_path, "*.tif")):
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale
        mask = np.array(mask)
        mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation=cv2.INTER_NEAREST) # Added resize
        train_masks.append(mask)

# Convert list to array for machine learning processing
train_masks = np.array(train_masks)
print("Masks shape:", train_masks.shape)

# Use customary x_train and y_train variables
X_train = train_images
y_train = train_masks


# Function to split data into batches
def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


# Input parameters
batch_size = 3  # Choose an appropriate batch size

print("Loading VGG16 model...")
# Load VGG16 model without fully connected layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE_X, SIZE_Y, 3))
for layer in base_model.layers:
    layer.trainable = False

layer_names = ['block1_conv2', 'block2_conv2']
encoder_outputs = [base_model.get_layer(name).output for name in layer_names]
encoder_model = Model(inputs=base_model.input, outputs=encoder_outputs)

# Batch processing
print("Processing features in batches...")
all_features = []
all_labels = []

# Check if we have data to process
if len(X_train) == 0 or len(y_train) == 0:
    raise ValueError("No training data found. Check your input paths.")

try:
    for batch_X, batch_y in zip(batch_generator(X_train, batch_size), batch_generator(y_train, batch_size)):
        print(f"Processing batch of shape {batch_X.shape}")
        block1, block2 = encoder_model(batch_X)
        block2_upsampled = tf.image.resize(block2, size=block1.shape[1:3])
        fused_features = Concatenate()([block1, block2_upsampled])

        # Convert to numpy and reshape
        features_reshaped = fused_features.numpy().reshape(-1, fused_features.shape[3])
        labels_reshaped = batch_y.reshape(-1)

        # Filter pixels with label 0
        mask = labels_reshaped != 0
        features_filtered = features_reshaped[mask]
        labels_filtered = labels_reshaped[mask]

        # Append results
        if len(features_filtered) > 0:  # Only append if we have data
            all_features.append(features_filtered)
            all_labels.append(labels_filtered)
except Exception as e:
    print(f"Error during batch processing: {e}")
    # Try with a smaller batch or check if memory is sufficient
    print("Try reducing batch_size or image dimensions if you're facing memory issues.")
    raise

# Combine processed batches
if len(all_features) == 0 or len(all_labels) == 0:
    raise ValueError("No features extracted. Check your data and processing logic.")

X_for_RF = np.vstack(all_features)
Y_for_RF = np.hstack(all_labels) - 1  # Subtract 1 if necessary

print(f"Features shape: {X_for_RF.shape}, Labels shape: {Y_for_RF.shape}")
print(f"Number of classes: {len(set(Y_for_RF))}")

# Check for GPU availability for XGBoost
gpu_available = tf.test.is_gpu_available()
tree_method = 'gpu_hist' if gpu_available else 'hist'
print(f"Using tree_method: {tree_method}")

# OPTIMIZATION OF PARAMETERS FOR MODEL TRAINING
print("Splitting data for training and validation...")
X_train, X_val, y_train, y_val = train_test_split(X_for_RF, Y_for_RF, test_size=0.2, random_state=42)

def objective(trial):
    # Definition of parameters to optimize
    param = {
        'base_score': 0.5,
        'max_depth': trial.suggest_int('max_depth', 6, 8),  # Tree depth
        'learning_rate': trial.suggest_float('learning_rate', 0.1, 0.4),  # Learning rate
        'n_estimators': trial.suggest_int('n_estimators', 150, 300),  # Number of trees
        'subsample': trial.suggest_float('subsample', 0.7, 0.95),  # Percentage of data used
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.7, 0.95),  # Reduced to avoid overfitting
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.7, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),  # Percentage of features used
        'gamma': trial.suggest_float('gamma', 0.1, 0.4),  # Regularization for split
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),
        'objective': 'multi:softprob',  # Multi-class classification
        'num_class': len(set(Y_for_RF)),  # Number of classes
        'importance_type': 'gain',
        'min_child_weight': trial.suggest_int('min_child_weight', 2, 5),
        'num_parallel_tree': trial.suggest_int('num_parallel_tree', 1, 5),
        'tree_method': tree_method,
        'eval_metric': 'mlogloss',
        'validate_parameters': 1,
        'booster': 'dart',
        'verbosity': 1,
        'max_delta_step': 1,
        'rate_drop': trial.suggest_float('rate_drop', 0.1, 0.3),  # DART specific parameter
        'skip_drop': trial.suggest_float('skip_drop', 0.1, 0.3),  # DART specific parameter
        'max_bin': 512
    }

    # Creation of model with suggested parameters
    model = xgb.XGBClassifier(**param, random_state=42)

    # Training
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Prediction and evaluation
    preds = model.predict(X_val)
    accuracy = accuracy_score(y_val, preds)

    return accuracy  # Optuna will try to maximize this value


print("Starting Optuna optimization...")
study = optuna.create_study(direction="maximize")  # Maximizing accuracy
study.optimize(objective, n_trials=20)  # Run x iterations

print("Best params:", study.best_params)
print("Best value:", study.best_value)  # Best accuracy


try:
    from optuna.visualization import plot_param_importances
    plot_param_importances(study).show()

    from optuna.visualization import plot_optimization_history
    plot_optimization_history(study).show()
except ImportError:
    print("Optuna visualization not available. Install 'plotly' for visualization.")

best_params = study.best_params
# Add fixed parameters
best_params.update({
    'base_score': 0.5,
    'objective': 'multi:softprob',
    'num_class': len(set(Y_for_RF)),
    'tree_method': tree_method,
    'eval_metric': 'mlogloss',
    'validate_parameters': 1,
    'max_delta_step': 1,
    'importance_type': 'gain',
    'max_bin': 512,
    'verbosity': 2,
    'booster': 'dart'
})

print("Training final model with optimized parameters...")
final_model = xgb.XGBClassifier(**best_params, random_state=42)
final_model.fit(X_train, y_train)

# EVALUATION OF CLASS METRICS
print("Evaluating model...")
prediction = final_model.predict(X_val)

# Using built-in keras function
from keras.metrics import MeanIoU
from sklearn.metrics import confusion_matrix

num_classes = len(set(Y_for_RF))
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
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    class_iou.append(iou)
    print(f"IoU Class {i} =", iou)

print("Processing completed successfully!")

# Save the model
model_filename = "xgboost_segmentation_model.json"
final_model.save_model(model_filename)
print(f"Model saved to {model_filename}")
