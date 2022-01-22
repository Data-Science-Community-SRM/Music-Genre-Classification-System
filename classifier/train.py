""" Train a logistic regression model on the spectrograms """
import os
import warnings

import tensorflow.keras as keras

from config import JSON_PATH, MODEL_DIR
from load_data import load_data, load_mappings
from evaluation_utilities import (
    get_classification_report,
    get_confusion_matrix,
    plot_history,
)
from models import create_logreg, create_CNN

warnings.filterwarnings(action="ignore")


if __name__ == "__main__":

    # load data; set CNN=True to add channel axis to X.
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_data(JSON_PATH, CNN=True)

    # create the model
    input_shape = X_train.shape[1:]
    print(f"input shape: {input_shape}")

    ## logistic reg
    # model = create_logreg(input_shape)

    ## CNN
    model = create_CNN(input_shape)

    # compile the model
    optim = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optim, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # view model summary
    model.summary()

    # train the model
    history = model.fit(
        X_train, y_train, validation_data=(X_valid, y_valid), epochs=30, batch_size=32
    )

    # evaluate the model
    loss, acc = model.evaluate(X_test, y_test, verbose=1)

    print(f"Test accuracy: {acc}, test loss: {loss}")

    # view reports
    get_confusion_matrix(model, X_test, y_test)
    get_classification_report(
        model, X_test, y_test, target_names=load_mappings(JSON_PATH)
    )

    # plot history
    plot_history(history)

    # save model (optional)
    model_dir = os.path.join(MODEL_DIR, "model.h5")
    model.save(model_dir)
