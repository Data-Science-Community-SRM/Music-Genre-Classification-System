import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)

GENRES = [
    "Electronic",
    "Experimental",
    "Folk",
    "Hip-Hop",
    "Instrumental",
    "International",
    "Pop",
    "Rock",
]  # Length 8


class FmaLoader(Dataset):
    """
    :param split: must be either "training", "validation" or "test"; anything else and you get an empty dataset
    :param subset: "small" by default
    """

    def __init__(self, audio_root, tracks_file, split, blacklist=None, subset="small"):
        self.tracks = load.load_meta(tracks_file)
        self.audio_root = audio_root
        self.subset = self.tracks["set", "subset"] <= subset
        blacklist = blacklist or [
            98565,
            98567,
            98569,
            99134,
            108925,
            133297,
            17631,
            17632,
            17633,
            17634,
            17635,
            17636,
            17637,
            29350,
            29351,
            29355,
            54568,
            54576,
            54578,
            55783,
        ]
        self.subset[blacklist] = False
        self.split = self.tracks["set", "split"] == split
        self.rows = tracks.index[self.subset & self.split]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        label = self.tracks["track", "genre_top"].loc[self.rows[item]]
        # We ignore the sample rate of the song, the model can choose that for itself
        waveform, _ = torchaudio.load(
            load.get_audio_path(self.audio_root, train[item]), num_frames=44100 * 29
        )
        return waveform.mean(0), genres.index(label)
