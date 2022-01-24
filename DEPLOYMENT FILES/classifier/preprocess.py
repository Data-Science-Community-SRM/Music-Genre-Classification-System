"""
processes data by extracting MFCCs or melspectrograms and store in a json file.
"""

import os
import json

import numpy as np
import librosa

from config import (
    DATA_DIR,
    SAMPLE_RATE,
    JSON_PATH,
    DATA_OPTION,
    NUM_SEGMENTS,
    NUM_SAMPLES_PER_SEGMENT,
    EXPECTED_SEGMENT_LENGTH,
    N_FFT,
    N_MFCC,
    HOP_LENGTH,
    N_MELS,
)


def process_file(
    file_path,
    num_segments=NUM_SEGMENTS,
    num_samples_per_segment=NUM_SAMPLES_PER_SEGMENT,
    option=DATA_OPTION,
    n_mfcc=N_MFCC,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    fmax=SAMPLE_RATE // 2,
):

    waveform, _ = librosa.load(file_path)

    feature_list = []

    for s in range(num_segments):
        start_sample = num_samples_per_segment * s
        end_sample = start_sample + num_samples_per_segment

        if option == "mfcc":
            mfcc = librosa.feature.mfcc(
                waveform[start_sample:end_sample],
                n_mfcc=n_mfcc,
                n_fft=n_fft,
                hop_length=hop_length,
            )

            feature_to_export = mfcc.T

        elif option == "melspectrogram":
            melspec = librosa.feature.melspectrogram(
                waveform[start_sample:end_sample],
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
                fmax=fmax,
            )
            feature_to_export = melspec.T

        else:
            raise ValueError("option needs to be either melspectrogram or mfcc.")

        # store segment if it has expected length and if it is non-zero
        if (len(feature_to_export) == EXPECTED_SEGMENT_LENGTH) and np.any(
            feature_to_export
        ):
            feature_list.append(feature_to_export.tolist())

    return feature_list


def save_mfcc(dataset_path, json_path, option=DATA_OPTION):
    """
    Save the mfccs or melspectrograms in a json file. Also split each sample up in to chunks.

    option: can be either "mfcc" or "melspectrogram".
    """

    # dictionary to store data
    data = {"mappings": [], "labels": [], option: []}

    # loop through all genres
    for i, (dirpath, _, filenames) in enumerate(os.walk(dataset_path)):

        # ensure not in root level
        if dirpath is not dataset_path:

            # save the semantic (genre) label
            genre = dirpath.split("/")[-1]
            data["mappings"].append(genre)

            print(f"Processing {genre}")

            # process files for specific genre
            for f in filenames:

                # load audio file
                file_path = os.path.join(dirpath, f)

                # get the mfccs or melspectrograms
                feature_list = process_file(file_path)

                # add them all to the feature_list
                data[option].extend(feature_list)

                for _ in range(len(feature_list)):
                    data["labels"].append(i - 1)

                print(f"Processed {f}, label {i-1}")

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(DATA_DIR, JSON_PATH)
