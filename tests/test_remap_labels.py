import os
import sys
import tensorflow as tf

# Allow importing the local package without installing it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common import dataset_organization


def test_remap_labels_basic():
    # Create a minimal dataset with two images and string labels
    images = tf.zeros((2, 2, 2, 3))
    labels = tf.constant(['A', 'B'])
    dataset = tf.data.Dataset.from_tensor_slices((images, labels)).batch(2)

    mapping = {'A': 0, 'B': 1}
    mapped = dataset.map(dataset_organization.remap_labels(mapping))

    for _, out_labels in mapped:
        assert out_labels.numpy().tolist() == [0, 1]
