"""
evaluate.py

Evaluate the trained model on the test dataset.
"""

import torch
import torch.nn as nn
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

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

    checkpoint = "/kaggle/input/datasets/priyanka0713/best-model/best_model.pth"

    model.load_state_dict(
        torch.load(checkpoint, map_location=DEVICE)
    )

    model.to(DEVICE)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    running_loss = 0.0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.float().to(DEVICE)

            outputs = model(images)

            probabilities = torch.sigmoid(outputs)

            all_predictions.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            loss = criterion(outputs, labels)

            running_loss += loss.item()

    test_loss = running_loss / len(test_loader)

        # Convert lists to NumPy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # Convert probabilities to binary predictions
    binary_predictions = (all_predictions >= 0.5).astype(int)

    # Compute metrics
    accuracy = accuracy_score(
        all_labels.flatten(),
        binary_predictions.flatten()
    )

    precision = precision_score(
        all_labels.flatten(),
        binary_predictions.flatten(),
        average="micro",
        zero_division=0,
    )

    recall = recall_score(
        all_labels.flatten(),
        binary_predictions.flatten(),
        average="micro",
        zero_division=0,
    )

    f1 = f1_score(
        all_labels.flatten(),
        binary_predictions.flatten(),
        average="micro",
        zero_division=0,
    )

    auroc = roc_auc_score(
        all_labels,
        all_predictions,
        average="macro",
    )

    print(f"\nTest Loss : {test_loss:.4f}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"AUROC     : {auroc:.4f}")


if __name__ == "__main__":
    evaluate()
    