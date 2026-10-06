"""
validate.py

Validation loop.
"""

import torch
from tqdm import tqdm


def validate(
    model,
    dataloader,
    criterion,
    device,
):
    model.eval()

    running_loss = 0.0

    with torch.no_grad():

        progress_bar = tqdm(
            dataloader,
            desc="Validation",
            leave=False,
        )

        for images, labels in progress_bar:

            images = images.to(device)
            labels = labels.float().to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(dataloader)

    return epoch_loss