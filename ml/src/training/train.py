"""
train.py

Main training script.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from configs.config import (
    DEVICE,
    BATCH_SIZE,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    LEARNING_RATE,
    EPOCHS,
    CHECKPOINT_DIR,
)

from models.model import get_model
from datasets.dataloader import get_dataloaders

from training.trainer import train_one_epoch
from training.validate import validate


def main():

    print(f"Using device: {DEVICE}")

    # --------------------------
    # Data
    # --------------------------
    train_loader, val_loader, test_loader = get_dataloaders(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        test_csv=TEST_CSV,
        batch_size=BATCH_SIZE,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # --------------------------
    # Model
    # --------------------------
    model = get_model()
    model = model.to(DEVICE)

    # --------------------------
    # Loss
    # --------------------------
    criterion = nn.BCEWithLogitsLoss()

    # --------------------------
    # Optimizer
    # --------------------------
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_loss = float("inf")

    # --------------------------
    # Training Loop
    # --------------------------
    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            DEVICE,
        )

        val_loss = validate(
            model,
            val_loader,
            criterion,
            DEVICE,
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            save_path = CHECKPOINT_DIR / "best_model.pth"

            torch.save(
                model.state_dict(),
                save_path,
            )

            print(f"✅ Saved best model to {save_path}")


if __name__ == "__main__":
    main()