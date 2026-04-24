import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(data_dir, img_size=(64, 64), batch_size=32):
    """Load and preprocess Medical MNIST dataset from directory."""
    class_names = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    image_paths = []
    labels = []

    for i, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                image_paths.append(os.path.join(class_path, img_name))
                labels.append(i)

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    def process_path(file_path, label):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=1)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img, img  # Autoencoder: target = input

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.map(process_path).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
    test_ds = test_ds.map(process_path).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds, class_names
