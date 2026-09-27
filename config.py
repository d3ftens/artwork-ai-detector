"""
config.py
---------
Central place for project settings. Edit this file first before training.

Your dataset should look like this (you provide train/ and test/ yourself):

dataset/
├── train/
│   ├── ai_generated/   <- put AI-generated artwork images here
│   └── real/           <- put real human-made paintings here
└── test/
    ├── ai_generated/
    └── real/

If your folder names are different (e.g. "fake" / "human"), just change
CLASS_NAMES below to match EXACTLY (same spelling/case as your folders).
"""

import os
import torch

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "artwork_detector.pth")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
GRADCAM_DIR = os.path.join(BASE_DIR, "static", "gradcam")
REPORTS_DIR = os.path.join(BASE_DIR, "static", "reports")
HISTORY_FILE = os.path.join(BASE_DIR, "prediction_history.json")

# ---- Class names ----
# IMPORTANT: These must exactly match your two subfolder names inside
# dataset/train/ and dataset/test/. Order doesn't matter — the training
# script reads it automatically from your folders and saves it into the
# checkpoint, but keep this in sync for predict.py / app.py fallback.
CLASS_NAMES = ["ai_generated", "real"]

# ---- Image / training settings ----
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15   # fraction of TRAIN_DIR held out for validation
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 5
LOW_CONFIDENCE_THRESHOLD = 60.0  # percent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

for d in (MODEL_DIR, UPLOAD_DIR, GRADCAM_DIR, REPORTS_DIR):
    os.makedirs(d, exist_ok=True)
