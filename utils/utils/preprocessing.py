"""
utils/preprocessing.py
-----------------------
Image transforms (augmentation for training, plain normalization for
validation/testing/inference) and a helper to load a single image for
prediction.
"""

from PIL import Image
import random
import torch
from torchvision import transforms

import config


import io


class RandomDownUpsample:
    """Randomly downsizes then upsizes an image, blurring away source-pipeline fingerprints."""
    def __init__(self, size, p=0.5):
        self.size = size
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            small = img.resize((self.size // 2, self.size // 2))
            img = small.resize((self.size, self.size))
        return img


class RandomJPEGCompression:
    """Randomly re-compresses an image through JPEG, at a random quality level."""
    def __init__(self, p=0.5, min_quality=30, max_quality=80):
        self.p = p
        self.min_quality = min_quality
        self.max_quality = max_quality

    def __call__(self, img):
        if random.random() < self.p:
            quality = random.randint(self.min_quality, self.max_quality)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            img = Image.open(buffer).convert("RGB")
        return img


def get_train_transforms():
    """Augmented transform pipeline used only for the training set."""
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomCrop(config.IMAGE_SIZE, padding=8, padding_mode="reflect"),
        RandomDownUpsample(config.IMAGE_SIZE, p=0.5),
        RandomJPEGCompression(p=0.6),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])
def get_eval_transforms():
    """Plain transform (resize + normalize only) for validation/test/inference."""
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


def load_image_for_prediction(image_path):
    """Load a single image from disk and apply eval transforms.
    Returns a batched tensor of shape (1, 3, H, W) ready for the model,
    plus the original PIL image (useful for Grad-CAM overlay)."""
    image = Image.open(image_path).convert("RGB")
    tensor = get_eval_transforms()(image).unsqueeze(0)
    return tensor, image
