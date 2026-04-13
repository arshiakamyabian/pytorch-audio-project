import torch
import torchaudio
from model import MusicGenreClassifier

device = torch.device("cpu")
model = MusicGenreClassifier(10)
model.load_state_dict(
    torch.load("models/pytorch_audio_project.pth", map_location=device)
)
model.eval()
print("model loaded successfully")

genres = sorted(
    [
        "blues",
        "classical",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
        "rock",
    ]
)

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=22050, n_mels=128, hop_length=512, n_fft=2048
)
db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)


def predict(file_path):
    waveform, sample_rate = torchaudio.load(file_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != 22050:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=22050
        )
        waveform = resampler(waveform)

    spectrogram = mel_spectrogram(waveform)
    spectrogram = db_transform(spectrogram)

    total_width = spectrogram.shape[2]

    chunks = []

    chunks.append(spectrogram[:, :, :256])

    mid = total_width // 2
    chunks.append(spectrogram[:, :, mid : mid + 256])

    end = total_width - 256
    chunks.append(spectrogram[:, :, end : end + 256])

    all_outputs = []
    with torch.no_grad():
        for chunk in chunks:
            input_data = chunk.unsqueeze(0)
            output = model(input_data)
            all_outputs.append(output)

    final_output = torch.stack(all_outputs).mean(dim=0)
    print("Raw scores:", final_output)
    _, index = torch.max(final_output, 1)

    return genres[index.item()]


my_song = "/Users/arshiakamyabian/Downloads/bombinsound-trap-512482.mp3"
result = predict(my_song)
print("Genre of song is:", result)
