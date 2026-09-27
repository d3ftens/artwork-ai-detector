"""
predict.py
----------
Loads the trained model and runs a prediction on a single image, with
Grad-CAM to visualize which parts of the artwork most influenced the
decision.

Can be used standalone:
    python predict.py path/to/image.jpg
or imported by app.py.
"""

import sys
import os

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm

import config
from utils.preprocessing import load_image_for_prediction

_model = None
_class_names = None


def build_inference_model(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model():
    """Loads the model once and caches it (used by the Flask app)."""
    global _model, _class_names
    if _model is not None:
        return _model, _class_names

    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {config.MODEL_PATH}. "
            f"Run 'python train_model.py' first."
        )

    checkpoint = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
    class_names = checkpoint.get("class_names", config.CLASS_NAMES)

    model = build_inference_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config.DEVICE)
    model.eval()

    _model, _class_names = model, class_names
    return model, class_names


class GradCAM:
    """Minimal Grad-CAM implementation hooked onto ResNet18's last conv block (layer4)."""

    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer = model.layer4[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()

        gradients = self.gradients[0]      # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        weights = gradients.mean(dim=(1, 2))  # (C,)

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = torch.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam.cpu().numpy()


def overlay_heatmap(original_image, cam, out_path, alpha=0.45):
    """Resize the CAM to the image size and blend it as a heatmap overlay."""
    heatmap = Image.fromarray(np.uint8(cm.jet(cam) * 255)).convert("RGB")
    heatmap = heatmap.resize(original_image.size)
    blended = Image.blend(original_image.convert("RGB"), heatmap, alpha=alpha)
    blended.save(out_path)


def predict_image(image_path, save_gradcam_path=None):
    """
    Returns a dict:
        {
            "predicted_class": str,
            "confidence": float (0-100),
            "low_confidence": bool,
            "probabilities": {class_name: float, ...},
            "gradcam_path": str or None
        }
    """
    model, class_names = load_model()
    input_tensor, original_image = load_image_for_prediction(image_path)
    input_tensor = input_tensor.to(config.DEVICE)

    outputs = model(input_tensor)
    probs = torch.softmax(outputs, dim=1)[0]
    confidence, pred_idx = torch.max(probs, dim=0)
    pred_idx = pred_idx.item()
    confidence = confidence.item() * 100

    result = {
        "predicted_class": class_names[pred_idx],
        "confidence": round(confidence, 2),
        "low_confidence": confidence < config.LOW_CONFIDENCE_THRESHOLD,
        "probabilities": {
            class_names[i]: round(probs[i].item() * 100, 2) for i in range(len(class_names))
        },
        "gradcam_path": None,
    }

    if save_gradcam_path:
        try:
            gradcam = GradCAM(model)
            cam = gradcam.generate(input_tensor, pred_idx)
            overlay_heatmap(original_image, cam, save_gradcam_path)
            result["gradcam_path"] = save_gradcam_path
        except Exception as e:
            # Grad-CAM is a bonus visualization; never fail the prediction over it
            print(f"Grad-CAM generation failed: {e}")

    return result


def explanation_for(predicted_class):
    """A short, honest, non-overclaiming explanation shown on the result page."""
    if predicted_class == "ai_generated":
        return ("The model detected texture, brushstroke, and pattern characteristics "
                "commonly associated with AI-generated artwork, such as unusually uniform "
                "textures or repeating fine detail.")
    return ("The model detected texture and brushstroke irregularities more typical of "
            "human-made paintings, such as inconsistent stroke pressure and organic detail.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/image.jpg")
        sys.exit(1)

    result = predict_image(sys.argv[1], save_gradcam_path="gradcam_output.png")
    print(f"\nPrediction: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']}%")
    if result["low_confidence"]:
        print("Low confidence prediction.")
    print(f"Probabilities: {result['probabilities']}")
