"""
config.py

Project configuration.
"""

from pathlib import Path
import torch

# ==========================
# Device
# ==========================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==========================
# Training
# ==========================

BATCH_SIZE = 16

LEARNING_RATE = 1e-4

EPOCHS = 5

NUM_CLASSES = 14

# ==========================
# CSV files
# ==========================

TRAIN_CSV = "../../data/splits/train.csv"

VAL_CSV = "../../data/splits/val.csv"

TEST_CSV = "../../data/splits/test.csv"

# ==========================
# Save model
# ==========================

CHECKPOINT_DIR = Path("../../checkpoints")

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)