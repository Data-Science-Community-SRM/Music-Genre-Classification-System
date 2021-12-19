import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)


class Conv_2d(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        shape=(1, 4),
        padding=(2, 0),
        pooling=(1, 4),
        dropout=0.1,
    ):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, shape, padding=padding)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(pooling)
        self.dropout = nn.Dropout(dropout)

    def forward(self, wav):
        out = self.conv(wav)
        out = self.bn(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.dropout(out)
        return out


class Gramophone(nn.Module):
    def __init__(
        self,
        num_channels=16,
        sample_rate=44100,
        n_fft=2048,
        f_min=0.0,
        f_max=11025.0,
        num_mels=128,
        num_classes=8,
    ):
        super(Gramophone, self).__init__()

        # mel spectrogram
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            f_min=f_min,
            f_max=f_max,
            n_mels=num_mels,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.input_bn = nn.BatchNorm2d(1)

        # convolutional layers
        self.layer1 = Conv_2d(1, num_channels, pooling=(5, 1))
        self.layer2 = Conv_2d(num_channels, num_channels, pooling=(5, 1))
        self.layer3 = Conv_2d(num_channels, num_channels * 2, pooling=(5, 1))
        self.layer4 = Conv_2d(num_channels * 2, num_channels * 2, pooling=(5, 1))
        self.layer5 = Conv_2d(num_channels * 2, num_channels * 4, pooling=(5, 1))
        self.ap = nn.AdaptiveMaxPool2d((64, 64))

        # dense layers
        self.dense1 = nn.Linear(num_channels * 4, num_channels * 4)
        self.dense_bn = nn.BatchNorm1d(num_channels * 4)
        self.dense2 = nn.Linear(num_channels * 4, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, wav):
        # input Preprocessing
        out = self.melspec(wav)
        out = self.amplitude_to_db(out)

        # input batch normalization
        out = out.unsqueeze(1)
        out = self.input_bn(out)

        # convolutional layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.ap(out)

        # reshape. (batch_size, num_channels, 1, 1) -> (batch_size, num_channels)
        out = out.reshape(len(out), -1)

        # dense layers
        out = self.dense1(out)
        out = self.dense_bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.dense2(out)

        return out
