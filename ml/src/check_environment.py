import torch
import torchvision
import pandas as pd
import numpy as np

from utils.paths import *

print("=" * 60)
print("ENVIRONMENT CHECK")
print("=" * 60)

print("PyTorch Version      :", torch.__version__)
print("Torchvision Version  :", torchvision.__version__)
print("Pandas Version       :", pd.__version__)
print("NumPy Version        :", np.__version__)

print("\nCUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Using CPU")

print("\n" + "=" * 60)
print("PROJECT STRUCTURE CHECK")
print("=" * 60)

print(f"Project Root       : {PROJECT_ROOT}")
print(f"Data Directory     : {DATA_DIR}")
print(f"Raw Data Directory : {RAW_DATA_DIR}")
print(f"Image Directory    : {IMAGE_DIR}")
print(f"CSV File           : {DATA_ENTRY_CSV}")
print(f"Checkpoint Folder  : {CHECKPOINT_DIR}")
print(f"Output Folder      : {OUTPUT_DIR}")

print("\nPath Exists Check")
print("-" * 60)

print(f"Data Folder Exists        : {DATA_DIR.exists()}")
print(f"Raw Data Exists           : {RAW_DATA_DIR.exists()}")
print(f"Image Folder Exists       : {IMAGE_DIR.exists()}")
print(f"CSV Exists                : {DATA_ENTRY_CSV.exists()}")
print(f"Checkpoint Folder Exists  : {CHECKPOINT_DIR.exists()}")
print(f"Output Folder Exists      : {OUTPUT_DIR.exists()}")