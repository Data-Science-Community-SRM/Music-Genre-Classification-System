"""
Evaluation utilities.

plot_history: view the training accuracy and loss.
get_confusion_matrix: prints out a confusion matrix for the test data.
get_classification_report: prints out a classification report for the test data.
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report


def plot_history(history):
    """
    Plots the training history.

    Inputs:
    history, a history object output from calling model.fit() in the Keras API.

    Outputs:
    None
    """
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Valid"], loc="upper left")
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Valid"], loc="upper right")
    plt.show()


def get_confusion_matrix(model, X_test, y_test):
    """
    Make a confusion matrix from the model.
    
    Inputs:

    model: a (trained) Keras model object.
    X_test: a numpy array of test inputs
    y_test: a numpy array of test labels (integers between 0 and NUM_GENRES). 

    Outputs:
    None
    """

    y_pred = np.argmax(model.predict(X_test), axis=1)

    print(confusion_matrix(y_test, y_pred))


def get_classification_report(model, X_test, y_test, target_names):
    """
    Make a classification report for the model.
    
    Inputs:

    model: a (trained) Keras model object.
    X_test: a numpy array of test inputs
    y_test: a numpy array of test labels (integers between 0 and NUM_GENRES).
    target_names: list of genres to map labels back to genre names.

    Outputs:
    None
    """

    y_pred = np.argmax(model.predict(X_test), axis=1)

    print(classification_report(y_test, y_pred, target_names=target_names))
