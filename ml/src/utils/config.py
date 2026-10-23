"""
Project-wide configuration.
"""

import torch

# ==========================
# Device
# ==========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# Dataset
# ==========================

IMAGE_SIZE = 224

BATCH_SIZE = 16

NUM_WORKERS = 2

NUM_CLASSES = 14

RANDOM_SEED = 42

# ==========================
# Dataset split CSVs
# ==========================

TRAIN_CSV = "data/splits/train.csv"
VAL_CSV = "data/splits/val.csv"
TEST_CSV = "data/splits/test.csv"

# ==========================
# Training
# ==========================

LEARNING_RATE = 1e-4

EPOCHS = 5

# ==========================
# Checkpoints
# ==========================

from pathlib import Path

CHECKPOINT_DIR = Path("checkpoints")