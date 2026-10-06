"""
model.py

DenseNet-121 model for multi-label Chest X-ray classification.
"""

import torch.nn as nn
from torchvision import models


def get_model(num_classes=14, pretrained=True):
    """
    Returns a DenseNet-121 model for multi-label classification.
    """

    # Load pretrained DenseNet121
    weights = (
        models.DenseNet121_Weights.DEFAULT
        if pretrained
        else None
    )

    model = models.densenet121(weights=weights)

    # Number of features coming into classifier
    in_features = model.classifier.in_features

    # Replace classifier
    model.classifier = nn.Linear(
        in_features,
        num_classes,
    )

    return model