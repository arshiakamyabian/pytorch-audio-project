import torch
import torch.nn as nn

class MusicGenreClassifier(nn.Module):
    def __init__(self, num_genre):
        super().__init__()

        # --- Convolution layers ---
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # --- Pooling ---
        self.pool = nn.MaxPool2d(2, 2)

        # --- IMPORTANT: Adaptive pooling ---
        self.gap = nn.AdaptiveAvgPool2d((4, 4))

        # --- Fully connected ---
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_genre)

        # --- Other layers ---
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        # Conv Block 1
        x = self.pool(self.relu(self.bn1(self.conv1(x))))

        # Conv Block 2
        x = self.pool(self.relu(self.bn2(self.conv2(x))))

        # Conv Block 3
        x = self.pool(self.relu(self.bn3(self.conv3(x))))

        # 🔥 THIS IS THE IMPORTANT PART
        x = self.gap(x)  # → makes output (batch, 64, 4, 4)

        # Flatten
        x = x.view(x.size(0), -1)  # → (batch, 1024)

        # Fully connected
        x = self.dropout(x)
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x