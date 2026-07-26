import torch
import torchvision
import pandas as pd
import numpy as np

print("PyTorch Version :", torch.__version__)
print("Torchvision Version :", torchvision.__version__)
print("Pandas Version :", pd.__version__)
print("NumPy Version :", np.__version__)

print("\nCUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Using CPU")