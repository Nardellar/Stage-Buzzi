import os
import sys
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Classificazione import dataset_organization


def test_balance_dataset_even_counts():
    images = tf.zeros((3, 2, 2, 3))
    labels = tf.constant([0, 1, 1])
    ds = tf.data.Dataset.from_tensor_slices((images, labels)).batch(1)

    balanced = dataset_organization.balance_dataset(ds)
    counts = {}
    for _, lbls in balanced.unbatch():
        label = int(lbls.numpy())
        counts[label] = counts.get(label, 0) + 1
    assert counts[0] == counts[1]
