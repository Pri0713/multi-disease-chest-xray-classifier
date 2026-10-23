"""
Utilities for handling disease labels in the NIH ChestX-ray14 dataset.
"""

DISEASE_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

NUM_CLASSES = len(DISEASE_LABELS)


def get_disease_labels():
    """
    Return the list of disease labels.
    """
    return DISEASE_LABELS.copy()