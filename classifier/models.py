"""
Script containing modules for creating models.
"""

import tensorflow.keras as keras
from config import NUM_GENRES


def create_logreg(input_shape, num_genres=NUM_GENRES):
    """
    Create a logistic regression model.

    Inputs:

    input_shape: tuple, shape of non-batch dimensions of X_train, X_valid, and X_test.
    num_genres: default NUM_GENRES, the number of genres.

    Output:
    model: a logistic regression model.
    """
    model = keras.Sequential(
        [
            keras.layers.Flatten(input_shape=input_shape),
            keras.layers.Dense(
                num_genres,
                activation="softmax",
                kernel_regularizer=keras.regularizers.l2(0.01),
            ),
        ]
    )

    return model


def create_CNN(input_shape, filters=32, num_genres=NUM_GENRES):
    """
    Create a convolutional neural network object.

    Inputs:

    input_shape: tuple, shape of non-batch dimensions of X_train, X_valid, and X_test.
    filters: int, default 32. Number of filters for each convolutional layer.
    num_genres: default NUM_GENRES, the number of genres.

    Output:

    model: a convolutional neural network model object
    """

    model = keras.Sequential()

    # conv layer 1
    model.add(
        keras.layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=input_shape,
            kernel_regularizer=keras.regularizers.l2(0.001),
        )
    )
    model.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same"))
    model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dropout(0.2))

    # conv layer 2
    model.add(
        keras.layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(0.001),
        )
    )
    model.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same"))
    model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dropout(0.2))

    # conv layer 3
    model.add(
        keras.layers.Conv2D(
            filters=filters,
            kernel_size=(2, 2),
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(0.001),
        )
    )
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
    model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dropout(0.2))

    # flatten and dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3))

    # softmax layer
    model.add(keras.layers.Dense(num_genres, activation="softmax"))

    return model
