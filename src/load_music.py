# Python ≥3.5 is required
import sys

assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn

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


def _decode_mp3(mp3_path):
    mp3_path = mp3_path.numpy().decode("utf-8")
    mp3_audio = AudioSegment.from_file(mp3_path, format="mp3")
    return mp3_audio.get_array_of_samples()


def dataset_from_path(dataset_root="./fma/fma_small/fma_small/*/*.mp3"):
    dataset = tfio.audio.AudioIODataset.list_files(dataset_root)
    dataset = dataset.map(
        lambda path: tf.py_function(func=_decode_mp3, inp=[path], Tout=tf.float32)
    )
    return dataset
