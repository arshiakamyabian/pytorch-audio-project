# dataset file is responsible for loading and converting audio into spectrograms
import torch
import torchaudio
import torchaudio.transforms
import numpy as np
import os
from torch.utils.data import Dataset

class MusicDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.is_train = True
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=35)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
        self.files = []
        self.labels = []

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050,
            n_mels=128,
            hop_length=512,
            n_fft=2048
        )

        # sorted so order is always same!
        self.genres = sorted([
            g for g in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, g))
        ])

        for label, genre in enumerate(self.genres):
            genre_folder = os.path.join(data_path, genre)
            for song_file in os.listdir(genre_folder):
                if not song_file.endswith('.wav'):
                    continue
                song_path = os.path.join(genre_folder, song_file)
                self.files.append(song_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        waveform, sample_rate = torchaudio.load(self.files[index])

        # convert stereo to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        spectrogram = self.mel_spectrogram(waveform)
        spectrogram = self.db_transform(spectrogram)

        # take middle chunk instead of beginning!
        total_width = spectrogram.shape[2]
        mid = total_width // 2
        spectrogram = spectrogram[:, :, mid:mid+256]

        if self.is_train:
            spectrogram = self.time_mask(spectrogram)
            spectrogram = self.freq_mask(spectrogram)

        label = self.labels[index]
        return spectrogram, label 
