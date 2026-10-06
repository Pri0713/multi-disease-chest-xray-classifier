"""
Central location for all project paths.

Using pathlib ensures our code works across Windows, macOS, and Linux.
"""

from pathlib import Path

# Project root (ml/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

# Image directory
IMAGE_DIR = RAW_DATA_DIR / "images"

# CSV files
DATA_ENTRY_CSV = RAW_DATA_DIR / "Data_Entry_2017.csv"
BBOX_CSV = RAW_DATA_DIR / "BBox_List_2017.csv"

# Model and output folders
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Configuration folder
CONFIG_DIR = PROJECT_ROOT / "configs"