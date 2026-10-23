import sys
from pathlib import Path

import torch
from PIL import Image

# Allow backend to import the existing ML package
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_DIR = PROJECT_ROOT / "ml"

if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from src.models.model import get_model
from src.datasets.transforms import val_transform
from src.datasets.labels import DISEASE_LABELS
from src.utils.config import DEVICE


class PredictionService:
    def __init__(self):
        self.device = DEVICE

        self.model = get_model(pretrained=False)

        checkpoint = PROJECT_ROOT / "ml" / "checkpoints" / "best_model.pth"

        self.model.load_state_dict(
            torch.load(checkpoint, map_location=self.device)
        )

        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_path: str):
        image = Image.open(image_path).convert("RGB")

        image = val_transform(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)

        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]

        results = []

        for disease, probability in zip(DISEASE_LABELS, probabilities):
            results.append({
                "disease": disease,
                "probability": float(probability),
            })

        return results


prediction_service = PredictionService()