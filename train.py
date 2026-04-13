import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MusicDataset
from model import MusicGenreClassifier
import os

DATA_PATH = "data/genres_original"
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
NUM_GENRES = 10

dataset = MusicDataset(DATA_PATH)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size]
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = MusicGenreClassifier(NUM_GENRES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, labels in train_loader:
        x = x.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    scheduler.step()
    print("Epoch {}/{} | Loss: {:.4f}".format(epoch + 1, EPOCHS, avg_loss))

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for x, labels in test_loader:
        x = x.to(device)
        labels = labels.to(device)
        outputs = model(x)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print("Test Accuracy: {:.1f}%".format(accuracy))

if not os.path.exists("models"):
    os.makedirs("models")

torch.save(model.state_dict(), "models/pytorch_audio_project.pth")
