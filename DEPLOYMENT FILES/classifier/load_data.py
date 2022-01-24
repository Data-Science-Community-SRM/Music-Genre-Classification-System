"""
Modules that load the data and process the train/test split.
"""
import json

import numpy as np
from sklearn.model_selection import train_test_split

from config import JSON_PATH, DATA_OPTION


def load_mappings(json_path):
    """
    get the mappings from the json file.
    
    Inputs:

    json_path: str, path to json file to be loaded

    Outputs:

    List of genres.
    """

    with open(json_path, "r") as fp:
        data = json.load(fp)

    return data["mappings"]


def load_data(json_path, test_split=0.1, valid_split=0.2, CNN=False):
    """
    Load in the dataset and create train/validation/test splits.

    Inputs:

    json_path: str, path to json file to be loaded
    test_split: float, default 0.1. Fraction (between 0 and 1) of the data to be split into the test set.
    valid_split: float, default, 0.2. Fraction (between 0 and 1) of the train set to be split into the validation set.
    CNN: bool, default False. If True, adds a channel dimension to X_train/X_valid/X_test so as to be compatible with a 
    convolutional neural network architecture.

    Outputs:

    X_train, X_test, X_valid: numpy arrays of processed data (MFCCs or melspectrograms).
    y_train, y_test, y_valid: numpy arrays of integers (targets).
    """

    with open(json_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data[DATA_OPTION])
    y = np.array(data["labels"])

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=1
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=valid_split, random_state=2
    )

    if CNN:
        X_train = X_train[..., np.newaxis]
        X_valid = X_valid[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

    return X_train, X_valid, X_test, y_train, y_valid, y_test


if __name__ == "__main__":

    X_train, X_valid, X_test, y_train, y_valid, y_test = load_data(JSON_PATH)

    print(
        f"""
        X_train shape: {X_train.shape}
        X_valid shape: {X_valid.shape}
        X_test shape: {X_test.shape}

        y_train shape: {y_train.shape}
        y_valid shape: {y_valid.shape}
        y_test shape: {y_test.shape}

        Sample X: {X_train[0]}

        Sample y: {y_train[0]}
        """
    )

