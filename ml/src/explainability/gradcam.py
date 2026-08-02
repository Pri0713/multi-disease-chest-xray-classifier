"""
gradcam.py

Generate Grad-CAM visualization for a chest X-ray.
"""

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

        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(
            self.save_activation
        )

        self.target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

        def generate(self, class_idx):

        # Backpropagate for the selected class
        self.model.zero_grad()

        class_idx.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))

        # Weighted sum of feature maps
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

    # Load model
    model = get_model()

    checkpoint = "/kaggle/input/datasets/priyanka0713/best-model/best_model.pth"

    model.load_state_dict(
        torch.load(checkpoint, map_location=DEVICE)
    )

    model.to(DEVICE)
    model.eval()

    # Last convolutional layer of DenseNet-121
    target_layer = model.features.denseblock4.denselayer16.conv2

    gradcam = GradCAM(
        model,
        target_layer,
    )

    # Load image
    image = Image.open(image_path).convert("RGB")

    original_image = np.array(image)

    image_tensor = val_transform(image)

    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

        # Forward pass
    outputs = model(image_tensor)

    # Predicted class
    class_idx = outputs[0].argmax()

    # Generate heatmap
    heatmap = gradcam.generate(outputs[0, class_idx])

    # Resize heatmap
    heatmap = cv2.resize(
        heatmap,
        (original_image.shape[1], original_image.shape[0])
    )

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET,
    )

    # Overlay heatmap on image
    overlay = cv2.addWeighted(
        original_image,
        0.6,
        heatmap,
        0.4,
        0,
    )

    # Display
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original X-ray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM")
    plt.axis("off")

    plt.show()

    import sys

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python -m src.explainability.gradcam <image_path>")
        sys.exit(1)

    generate_gradcam(sys.argv[1])