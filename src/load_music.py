# Python ≥3.5 is required
import sys

assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
import ast
import pandas as pd

assert sklearn.__version__ >= "0.20"


# TensorFlow ≥2.0 is required
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow import keras

assert tf.__version__ >= "2.0"

# Common imports
import numpy as np
import os

from pydub import AudioSegment


def get_audio_path(audio_dir, track_id):
    """
    Return the path to the mp3 given the directory where the audio is stored
    and the track ID.
    Examples
    --------
    >>> import utils
    >>> AUDIO_DIR = os.environ.get('AUDIO_DIR')
    >>> utils.get_audio_path(AUDIO_DIR, 2)
    '../data/fma_small/000/000002.mp3'
    """
    tid_str = "{:06d}".format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + ".mp3")


def _decode_mp3(mp3_path):
    mp3_path = mp3_path.numpy().decode("utf-8")
    mp3_audio = AudioSegment.from_file(mp3_path, format="mp3")
    mp3_audio.set_channels(1)
    return mp3_audio.get_array_of_samples()


def load_meta(filepath):

    filename = os.path.basename(filepath)
    if "tracks" in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [
            ("track", "tags"),
            ("album", "tags"),
            ("artist", "tags"),
            ("track", "genres"),
            ("track", "genres_all"),
        ]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [
            ("track", "date_created"),
            ("track", "date_recorded"),
            ("album", "date_created"),
            ("album", "date_released"),
            ("artist", "date_created"),
            ("artist", "active_year_begin"),
            ("artist", "active_year_end"),
        ]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ("small", "medium", "large")
        try:
            tracks["set", "subset"] = tracks["set", "subset"].astype(
                "category", categories=SUBSETS, ordered=True
            )
        except (ValueError, TypeError):
            # the categories and ordered arguments were removed in pandas 0.25
            tracks["set", "subset"] = tracks["set", "subset"].astype(
                pd.CategoricalDtype(categories=SUBSETS, ordered=True)
            )

        COLUMNS = [
            ("track", "genre_top"),
            ("track", "license"),
            ("album", "type"),
            ("album", "information"),
            ("artist", "bio"),
        ]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype("category")

        return tracks


def by_track(numbers, dataset_root="./fma/fma_small/fma_small"):
    df = np.empty(2)
    indices = np.empty(1)
    for x in numbers:
        i, m = _decode_mp3(get_audio_path(dataset_root, x))
        np.append(indices, i)
        np.append(df, m)
    return indices, df


def dataset_from_path(numbers, dataset_root="./fma/fma_small/fma_small/"):
    dataset = tfio.audio.AudioIODataset.from_tensor_slices(
        np.array([get_audio_path(dataset_root, n) for n in numbers])
    )
    dataset = dataset.map(
        lambda path: tf.py_function(func=_decode_mp3, inp=[path], Tout=tf.float32)
    )
    return dataset
