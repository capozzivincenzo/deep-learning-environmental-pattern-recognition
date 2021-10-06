import tensorflow as tf
from tensorflow.keras import layers


def Conv1DWeather(input_shape, num_classes):
    return tf.keras.Sequential([
        layers.Conv1D(64, 3, activation='relu', padding="same", use_bias=False, input_shape=(input_shape)),
        layers.BatchNormalization(),
        layers.Conv1D(128, 3, activation='relu', padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dense(num_classes)  # Softmax is added on compile for numerical stability
    ])
