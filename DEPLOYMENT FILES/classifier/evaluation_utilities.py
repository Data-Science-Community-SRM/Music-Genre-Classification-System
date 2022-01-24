"""
Evaluation utilities.

"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report


def plot_history(history):
    """
    training history.

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
    confusion matrix from the model.
    
    
    """

    y_pred = np.argmax(model.predict(X_test), axis=1)

    print(confusion_matrix(y_test, y_pred))


def get_classification_report(model, X_test, y_test, target_names):
    """
    classification report for the model.
    
    
    """

    y_pred = np.argmax(model.predict(X_test), axis=1)

    print(classification_report(y_test, y_pred, target_names=target_names))
