import torch

print("PyTorch version:", torch.__version__)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)
