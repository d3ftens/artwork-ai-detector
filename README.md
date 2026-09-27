# Artwork Authenticity Detector

A Flask web app that predicts whether an uploaded piece of digital artwork
is **AI-generated** or a **real human-made painting**, using a ResNet18
fine-tuned with transfer learning. Built for a BSCS thesis/demo project.

## What's included

- `train_model.py` — trains the model on your dataset, evaluates it, saves plots
- `predict.py` — loads the trained model, predicts on one image, generates a Grad-CAM heatmap
- `app.py` — the Flask website (upload, result, history pages)
- `config.py` — all settings in one place (paths, class names, hyperparameters)
- `utils/preprocessing.py` — image transforms
- `utils/evaluation.py` — metrics and plotting

## 1. Install dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

## 2. Add your dataset

Put your images into this exact structure (you already have train/test —
just make sure the class subfolders match):

```
dataset/
├── train/
│   ├── ai_generated/    <- your AI-generated artwork images
│   └── real/            <- your real human-made painting images
└── test/
    ├── ai_generated/
    └── real/
```

If your class folder names are different (e.g. `fake` / `human`), open
`config.py` and update `CLASS_NAMES` to match. The training script also
reads folder names automatically, so as long as train and test use the
same two folder names, it will work either way.

There's no separate `validation/` folder needed — `train_model.py`
automatically holds out 15% of `dataset/train` for validation (change
`VALIDATION_SPLIT` in `config.py` if you want a different split).

## 3. Train the model

```bash
python train_model.py
```

This will:
- Load and split your dataset (prints image counts per class)
- Fine-tune a ResNet18 (ImageNet pretrained, early layers frozen, `layer4` + final layer trainable)
- Train for up to 25 epochs with early stopping (patience = 5) and an LR scheduler
- Save the best checkpoint to `model/artwork_detector.pth`
- Evaluate on `dataset/test` and save to `static/reports/`:
  - `training_curves.png` (loss + accuracy)
  - `confusion_matrix.png`
  - `roc_curve.png`
  - `pr_curve.png`
  - `test_metrics.json` (accuracy, precision, recall, F1, ROC AUC)

Console output looks like:

```
Epoch 3/25  (14.2s)  Training Loss: 0.41  Validation Loss: 0.29  Validation Accuracy: 91.2%
```

## 4. Run the website

```bash
python app.py
```

Open **http://127.0.0.1:5000**. Upload a JPG/PNG artwork and you'll get:
- A verdict (AI-Generated Artwork / Real Painting)
- A confidence percentage and progress bar
- A Grad-CAM heatmap showing which regions influenced the prediction
- A short plain-language explanation
- A "low confidence" flag if confidence is under 60%

Visit `/history` to see past predictions (timestamps, thumbnails, results —
stored in `prediction_history.json`).

## 5. Predict from the command line (optional)

```bash
python predict.py path/to/some_artwork.jpg
```

## Folder reference

| Path | Purpose |
|---|---|
| `dataset/train`, `dataset/test` | Your images (you provide these) |
| `model/artwork_detector.pth` | Saved model weights + class names, created after training |
| `static/uploads/` | Images uploaded through the website |
| `static/gradcam/` | Generated Grad-CAM heatmaps |
| `static/reports/` | Training/evaluation plots and metrics |
| `prediction_history.json` | Log of past predictions (auto-created) |

## Notes on accuracy

For the model to learn genuine AI-vs-real artifacts (rather than just
learning "these are landscapes, those are portraits"), try to include
similar subjects/styles in both classes — e.g. AI-generated portraits
alongside real portraits, not just AI abstract art vs. real landscapes.

This is a research/demo tool, not a certified authentication service —
the UI intentionally avoids claiming perfect accuracy.

## Troubleshooting

- **"No trained model found"** on the website → run `python train_model.py` first.
- **CUDA not found** → the code automatically falls back to CPU; training will just be slower.
- **Class mismatch errors** → make sure `dataset/train` and `dataset/test` use the exact same two subfolder names.
- **Out of memory during training** → lower `BATCH_SIZE` in `config.py` (e.g. to 16).
