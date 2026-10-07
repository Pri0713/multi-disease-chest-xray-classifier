from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from backend.app.services.prediction_service import prediction_service
from src.datasets.transforms import val_transform
from src.datasets.labels import DISEASE_LABELS


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
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

        for i, weight in enumerate(weights):
            cam += weight * activations[i]

        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= cam.max() + 1e-8

        return cam.detach().cpu().numpy()


class GradCAMService:
    def __init__(self, prediction_service):
        self.prediction_service = prediction_service
        self.model = prediction_service.model
        self.device = prediction_service.device

        self.target_layer = (
            self.model.features.denseblock4.denselayer16.conv2
        )

        self.gradcam = GradCAM(
            self.model,
            self.target_layer,
        )

    def generate(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        original = np.array(image)

        image_tensor = (
            val_transform(image)
            .unsqueeze(0)
            .to(self.device)
        )

        outputs = self.model(image_tensor)

        class_idx = outputs.argmax(dim=1)
        class_index = class_idx.item()
        class_name = DISEASE_LABELS[class_index]

        heatmap = self.gradcam.generate(
            outputs[0, class_index]
        )

        heatmap = cv2.resize(
            heatmap,
            (original.shape[1], original.shape[0]),
        )

        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET,
        )

        original_bgr = cv2.cvtColor(
            original,
            cv2.COLOR_RGB2BGR,
        )

        overlay = cv2.addWeighted(
            original_bgr,
            0.6,
            heatmap,
            0.4,
            0,
        )

        success, encoded_image = cv2.imencode(
            ".png",
            overlay,
        )

        if not success:
            raise RuntimeError(
                "Failed to encode Grad-CAM image."
            )

        return class_name, encoded_image.tobytes()


gradcam_service = GradCAMService(prediction_service)