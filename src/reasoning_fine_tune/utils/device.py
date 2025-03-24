import torch

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
if torch.mps.is_available():
    DEVICE = torch.device("mps")
