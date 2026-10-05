"""
export_onnx.py

Export the trained DenseNet121 chest X-ray classifier to ONNX format.
"""

import torch

from src.models.model import get_model
from src.utils.config import DEVICE


CHECKPOINT_PATH = "checkpoints/best_model.pth"
OUTPUT_PATH = "checkpoints/chest_xray_classifier.onnx"


def export_model():

    # Create model architecture
    model = get_model(pretrained=False)

    # Load trained weights
    model.load_state_dict(
        torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
        )
    )

    model.to(DEVICE)
    model.eval()

    # Dummy input matching the model's expected image shape
    dummy_input = torch.randn(
        1, 3, 224, 224,
        device=DEVICE,
    )

    # Export model
    torch.onnx.export(
        model,
        dummy_input,
        OUTPUT_PATH,
        input_names=["image"],
        output_names=["predictions"],
        opset_version=18,
    )

    print(f"ONNX model exported to: {OUTPUT_PATH}")


if __name__ == "__main__":
    export_model()