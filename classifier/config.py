"""
Holds all of the global variables.
"""
import os
import math

# data directory folders
DATA_DIR = "./data"
MODEL_DIR = "./models"

# mfcc or melspectrogram?
DATA_OPTION = "mfcc"

# processed data location
JSON_PATH = os.path.join(DATA_DIR, DATA_OPTION + ".json")

# dataset properties
NUM_GENRES = 10

# waveform properties
SAMPLE_RATE = 22050
DURATION = 25  # seconds; some files are not quite 30 seconds.
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
HOP_LENGTH = 512
N_FFT = 2048

# Segmenting Constants
NUM_SEGMENTS = 10  # split each track into this many pieces
NUM_SAMPLES_PER_SEGMENT = int(
    SAMPLES_PER_TRACK / NUM_SEGMENTS
)  # number of samples in each segment
EXPECTED_SEGMENT_LENGTH = math.ceil(NUM_SAMPLES_PER_SEGMENT / HOP_LENGTH)

# for MFCCs
N_MFCC = 13

# for melspecs
N_MELS = 128
