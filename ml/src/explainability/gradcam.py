"""
gradcam.py

Generate Grad-CAM visualization for a chest X-ray.
"""

import sys
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from src.models.model import get_model
from src.datasets.transforms import val_transform
from src.utils.config import DEVICE


class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, target):

        self.model.zero_grad()

        target.backward(retain_graph=True)

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(
            activations.shape[1:],
            device=activations.device,
        )

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)

        cam -= cam.min()

        cam /= (cam.max() + 1e-8)

        return cam.cpu().numpy()


def generate_gradcam(image_path):

    model = get_model()

    checkpoint = "/kaggle/input/datasets/priyanka0713/best-model/best_model.pth"

    model.load_state_dict(
        torch.load(checkpoint, map_location=DEVICE)
    )

    model.to(DEVICE)
    model.eval()

    target_layer = model.features.denseblock4.denselayer16.conv2

    gradcam = GradCAM(model, target_layer)

    image = Image.open(image_path).convert("RGB")

    original = np.array(image)

    image_tensor = val_transform(image).unsqueeze(0).to(DEVICE)

    outputs = model(image_tensor)

    class_idx = outputs.argmax(dim=1)

    heatmap = gradcam.generate(outputs[0, class_idx])

    heatmap = cv2.resize(
        heatmap,
        (original.shape[1], original.shape[0])
    )

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET,
    )

    overlay = cv2.addWeighted(
        original,
        0.6,
        heatmap,
        0.4,
        0,
    )

    plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python -m src.explainability.gradcam <image_path>")
        sys.exit(1)

    generate_gradcam(sys.argv[1])