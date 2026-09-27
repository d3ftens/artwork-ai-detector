"""
train_model.py
---------------
Trains a ResNet18 (transfer learning) to classify artwork as
"ai_generated" vs "real" using your own dataset in dataset/train and
dataset/test.

Usage:
    python train_model.py

What it does:
    1. Loads dataset/train (splits it into train/validation, 85/15 by default)
    2. Loads dataset/test (held out, used only for final evaluation)
    3. Builds a ResNet18 pretrained on ImageNet, replaces the final layer
       with a 2-class head, freezes early layers, fine-tunes later layers
    4. Trains with early stopping + LR scheduling, saving the best checkpoint
       after every epoch that improves (so interrupting mid-run is safe)
    5. Evaluates on the test set: accuracy, precision, recall, F1,
       confusion matrix, ROC curve, PR curve
    6. Saves plots into static/reports/ and the model into model/artwork_detector.pth
"""

import copy
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models

import config
from utils.preprocessing import get_train_transforms, get_eval_transforms
from utils.evaluation import (
    compute_metrics, plot_confusion_matrix, plot_training_curves,
    plot_roc_curve, plot_precision_recall_curve
)


def build_dataloaders():
    train_full = datasets.ImageFolder(config.TRAIN_DIR, transform=get_train_transforms())
    class_names = train_full.classes  # alphabetical order, e.g. ['ai_generated', 'real']

    # Split TRAIN_DIR into train/validation with a fixed seed for reproducibility
    val_size = int(len(train_full) * config.VALIDATION_SPLIT)
    train_size = len(train_full) - val_size
    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    train_subset, val_subset = random_split(train_full, [train_size, val_size], generator=generator)

    # Validation must use eval transforms (no augmentation), so wrap a second
    # ImageFolder instance with eval transforms and reuse the same indices.
    val_full = datasets.ImageFolder(config.TRAIN_DIR, transform=get_eval_transforms())
    val_subset.dataset = val_full

    test_dataset = datasets.ImageFolder(config.TEST_DIR, transform=get_eval_transforms())

    print(f"Classes found: {class_names}")
    print(f"Training images:   {len(train_subset)}")
    print(f"Validation images: {len(val_subset)}")
    print(f"Test images:       {len(test_dataset)}")
    for cls in class_names:
        n_train = sum(1 for _, y in train_full.samples if train_full.classes[y] == cls)
        print(f"  '{cls}': {n_train} images in dataset/train")

    train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, class_names


def build_model(num_classes=2):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last residual block (layer4) for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace the final fully-connected layer (always trainable)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model.to(config.DEVICE)


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss, running_correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for inputs, labels in loader:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            if is_train:
                optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()
            preds = outputs.argmax(dim=1)
            running_loss += loss.item() * inputs.size(0)
            running_correct += (preds == labels).sum().item()
            total += inputs.size(0)
    return running_loss / total, running_correct / total


def train():
    torch.manual_seed(config.RANDOM_SEED)
    train_loader, val_loader, test_loader, class_names = build_dataloaders()
    model = build_model(num_classes=len(class_names))

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print("\nStarting training...\n")
    for epoch in range(1, config.NUM_EPOCHS + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start
        print(f"Epoch {epoch}/{config.NUM_EPOCHS}  "
              f"({elapsed:.1f}s)  "
              f"Training Loss: {train_loss:.4f}  "
              f"Validation Loss: {val_loss:.4f}  "
              f"Validation Accuracy: {val_acc * 100:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": best_state,
                "class_names": class_names,
            }, config.MODEL_PATH)
            print(f"  -> New best model saved to {config.MODEL_PATH}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs "
                      f"(no improvement for {config.EARLY_STOPPING_PATIENCE} epochs).")
                break

    # Restore best weights before final evaluation
    model.load_state_dict(best_state)
    print(f"\nBest model is saved at {config.MODEL_PATH}")

    plot_training_curves(history, f"{config.REPORTS_DIR}/training_curves.png")

    # ---- Final evaluation on held-out test set ----
    print("\nEvaluating on test set...")
    model.eval()
    all_labels, all_preds, all_scores = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(config.DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_scores.extend(probs[:, 1].cpu().numpy())  # prob of class index 1

    metrics = compute_metrics(all_labels, all_preds)
    print(f"\nTest Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"Test Precision: {metrics['precision']*100:.2f}%")
    print(f"Test Recall:    {metrics['recall']*100:.2f}%")
    print(f"Test F1-score:  {metrics['f1']*100:.2f}%")

    plot_confusion_matrix(all_labels, all_preds, class_names, f"{config.REPORTS_DIR}/confusion_matrix.png")
    roc_auc = plot_roc_curve(all_labels, all_scores, f"{config.REPORTS_DIR}/roc_curve.png")
    plot_precision_recall_curve(all_labels, all_scores, f"{config.REPORTS_DIR}/pr_curve.png")
    print(f"ROC AUC: {roc_auc:.3f}")

    metrics["roc_auc"] = roc_auc
    with open(f"{config.REPORTS_DIR}/test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nAll plots saved to {config.REPORTS_DIR}/")
    print("Done. You can now run: python app.py")


if __name__ == "__main__":
    train()