import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def get_dataset():
    """
        Method used to generate the dataset
    """

    # Numbers of row per class
    row_per_class = 100
    # Generate rowsy


    # Numbers of row per class
    row_per_class = 100
    # Generate rows
    sick = np.random.randn(row_per_class, 2) + np.array([-2, -2])
    sick2 = np.random.randn(row_per_class, 2) + np.array([2, 2])

    healthy = np.random.randn(row_per_class, 2) + np.array([-2, 2])
    healthy2 = np.random.randn(row_per_class, 2) + np.array([2, -2])

    features = np.vstack([sick, sick2, healthy, healthy2])
    targets = np.concatenate(
        (np.zeros(row_per_class * 2), np.zeros(row_per_class * 2) + 1))

    targets.reshape(-1, 1)

    return features, targets

if __name__ == '__main__':
    # Dataset
    features, targets = get_dataset()

    tf_features = tf.placeholder(tf.float32, shape=[None, 2])
    tf_targets = tf.placeholder(tf.float32, shape=[None, 1])

    pass 
