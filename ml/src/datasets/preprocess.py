"""
preprocess.py

This script is responsible for:
1. Locating the NIH ChestX-ray14 dataset
2. Reading the metadata
3. Verifying the dataset
4. Preparing the data for the patient-wise split
"""

from pathlib import Path
import pandas as pd

# ==========================================
# Dataset Paths
# ==========================================

# Kaggle dataset path
KAGGLE_DATASET_PATH = Path("/kaggle/input/datasets/ksw2000/chestxray14/images")

# Local dataset path (update this later if needed)
LOCAL_DATASET_PATH = Path("../../data/chestxray14/images")

# Choose which one to use
DATASET_PATH = (
    KAGGLE_DATASET_PATH
    if KAGGLE_DATASET_PATH.exists()
    else LOCAL_DATASET_PATH
)

# ==========================================
# Check dataset path
# ==========================================

print("=" * 50)
print("Dataset Path")
print("=" * 50)

print(DATASET_PATH)
print()

if DATASET_PATH.exists():
    print("✅ Dataset found!")
else:
    print("❌ Dataset NOT found!")

# ==========================================
# Find all X-ray images
# ==========================================

image_paths = sorted(DATASET_PATH.glob("images_*/images/*.png"))

print("\n" + "=" * 50)
print("Image Information")
print("=" * 50)

print(f"Total images found: {len(image_paths)}")

# ==========================================
# Display a few sample image names
# ==========================================

print("\nFirst 10 image filenames:")

for image_path in image_paths[:10]:
    print(image_path.name)