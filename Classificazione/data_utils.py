import os
import zipfile
from pathlib import Path
from collections import defaultdict
import random

import gdown
import numpy as np
import tensorflow as tf


def download_and_extract(dataset_dir: Path, zip_name: Path, gdrive_id: str) -> None:
    """Download and extract dataset if not already present."""
    if not dataset_dir.exists():
        print("⬇️ Scaricamento del dataset da Google Drive...")
        url = f"https://drive.google.com/uc?id={gdrive_id}"
        gdown.download(url, str(zip_name), quiet=False)

        print("📦 Estrazione in corso...")
        with zipfile.ZipFile(zip_name, "r") as zip_ref:
            zip_ref.extractall(path=dataset_dir.parent)
        os.remove(zip_name)
        print("✅ Dataset pronto!")


def standardize_dataset(train_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset | None = None):
    images_list = []
    for images, _ in train_dataset:
        images_list.append(images.numpy())
    all_train_images = np.concatenate(images_list, axis=0)
    mean = np.mean(all_train_images, axis=(0, 1, 2))
    std = np.std(all_train_images, axis=(0, 1, 2))

    def standardize_batch(images, labels):
        return (tf.cast(images, tf.float32) - mean) / std, labels

    standardized_train = train_dataset.map(standardize_batch)
    if validation_dataset is not None:
        standardized_validation = validation_dataset.map(standardize_batch)
        return standardized_train, standardized_validation
    return standardized_train


def normalize_dataset(train_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset | None = None):
    images_list = []
    for images, _ in train_dataset:
        images_list.append(images.numpy())
    all_train_images = np.concatenate(images_list, axis=0)
    min_val = np.min(all_train_images, axis=(0, 1, 2))
    max_val = np.max(all_train_images, axis=(0, 1, 2))

    def normalize_batch(images, labels):
        return (tf.cast(images, tf.float32) - min_val) / (max_val - min_val), labels

    normalized_train = train_dataset.map(normalize_batch)
    if validation_dataset is not None:
        normalized_validation = validation_dataset.map(normalize_batch)
        return normalized_train, normalized_validation
    return normalized_train


def remap_labels(mapping):
    def map_fn(images, labels):
        def map_label(l):
            result = []
            for v in l:
                if isinstance(v, bytes):
                    key = v.decode("utf-8")
                elif isinstance(v, (str, np.str_)):
                    key = v
                elif isinstance(v, (np.integer, int)):
                    if v in mapping:
                        key = int(v)
                    else:
                        class_names = list(mapping.keys())
                        key = class_names[v] if 0 <= v < len(class_names) else None
                else:
                    key = None
                result.append(mapping.get(key, -1))
            return np.array(result, dtype=np.int32)

        labels = tf.numpy_function(map_label, [labels], tf.int32)
        labels.set_shape([None])
        return images, labels

    return map_fn


def balance_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Balance classes via oversampling."""
    class_data = defaultdict(list)
    for image, label in dataset.unbatch():
        class_index = int(label.numpy())
        class_data[class_index].append((image, label))

    max_count = max(len(samples) for samples in class_data.values())
    balanced_samples = []
    for samples in class_data.values():
        repeated = samples.copy()
        while len(repeated) < max_count:
            repeated.extend(random.sample(samples, min(max_count - len(repeated), len(samples))))
        balanced_samples.extend(repeated)

    random.shuffle(balanced_samples)
    images, labels = zip(*balanced_samples)
    images = tf.stack(images)
    labels = tf.convert_to_tensor(labels)
    return tf.data.Dataset.from_tensor_slices((images, labels)).batch(32)
