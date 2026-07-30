"""
evaluate.py

Evaluate the trained model on the test dataset.
"""

import torch
import torch.nn as nn

from src.models.model import get_model
from src.datasets.dataloader import get_dataloaders
from src.utils.config import (
    DEVICE,
    BATCH_SIZE,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    CHECKPOINT_DIR,
)


def evaluate():

    print(f"Using device: {DEVICE}")

    # -------------------------
    # Load Test Data
    # -------------------------
    _, _, test_loader = get_dataloaders(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        test_csv=TEST_CSV,
        batch_size=BATCH_SIZE,
    )

    print(f"Test batches: {len(test_loader)}")

    # -------------------------
    # Load Model
    # -------------------------
    model = get_model()

    checkpoint = CHECKPOINT_DIR / "best_model.pth"

    model.load_state_dict(
        torch.load(checkpoint, map_location=DEVICE)
    )

    model.to(DEVICE)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    running_loss = 0.0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.float().to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

    test_loss = running_loss / len(test_loader)

    print(f"\nTest Loss: {test_loss:.4f}")


if __name__ == "__main__":
    evaluate()
    