"""
train.py

Main training script.
"""

import torch
import torch.nn as nn

from models.model import get_model
from datasets.dataloader import get_dataloaders


def main():

    # --------------------------
    # Device
    # --------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # --------------------------
    # Data
    # --------------------------
    train_loader, val_loader, test_loader = get_dataloaders(
        train_csv="../../data/splits/train.csv",
        val_csv="../../data/splits/val.csv",
        test_csv="../../data/splits/test.csv",
        batch_size=16,
    )

    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    # --------------------------
    # Model
    # --------------------------
    model = get_model()

    model = model.to(device)

    # --------------------------
    # Loss
    # --------------------------
    criterion = nn.BCEWithLogitsLoss()

    print("\nEverything loaded successfully!")
    print(model)


if __name__ == "__main__":
    main()