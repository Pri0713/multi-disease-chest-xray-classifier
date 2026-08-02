"""
predict.py

Run inference on a single chest X-ray image.
"""

import torch
from PIL import Image
import sys

from src.models.model import get_model
from src.datasets.transforms import val_transform
from src.datasets.labels import DISEASE_LABELS
from src.utils.config import DEVICE

def predict(image_path):

    # Load model
    model = get_model()

    checkpoint = "/kaggle/input/datasets/priyanka0713/best-model/best_model.pth"

    model.load_state_dict(
        torch.load(checkpoint, map_location=DEVICE)
    )

    model.to(DEVICE)
    model.eval()

    # Load image
    image = Image.open(image_path).convert("RGB")

    image = val_transform(image)

    image = image.unsqueeze(0)

    image = image.to(DEVICE)
        # Run inference
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.sigmoid(outputs).cpu().numpy()[0]

    print("\nPredicted Diseases:\n")

    threshold = 0.5

    found = False

    for disease, prob in zip(DISEASE_LABELS, probabilities):
        if prob >= threshold:
            print(f"✓ {disease}: {prob:.4f}")
            found = True

    if not found:
        print("No disease detected.")
if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python -m src.inference.predict <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    predict(image_path)