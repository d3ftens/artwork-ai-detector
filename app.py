"""
app.py
------
Flask website for the artwork AI-detector. Upload an image, get a
prediction (AI-generated vs real), a confidence score, a Grad-CAM
heatmap, and a short explanation.

Run with: python app.py
Then open http://127.0.0.1:5000
"""

import json
import os
import uuid
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

import config
from predict import predict_image, explanation_for

app = Flask(__name__)
app.secret_key = "dev-secret-key-change-this-if-deploying"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS


def load_history():
    if os.path.exists(config.HISTORY_FILE):
        with open(config.HISTORY_FILE, "r") as f:
            return json.load(f)
    return []


def save_history_entry(entry):
    history = load_history()
    history.insert(0, entry)
    history = history[:50]  # keep the last 50 predictions
    with open(config.HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


@app.route("/")
def index():
    model_ready = os.path.exists(config.MODEL_PATH)
    return render_template("index.html", model_ready=model_ready)


@app.route("/predict", methods=["POST"])
def predict():
    if not os.path.exists(config.MODEL_PATH):
        flash("No trained model found yet. Run 'python train_model.py' first.")
        return redirect(url_for("index"))

    if "file" not in request.files or request.files["file"].filename == "":
        flash("Please choose an image to upload.")
        return redirect(url_for("index"))

    file = request.files["file"]
    if not allowed_file(file.filename):
        flash("Unsupported file type. Please upload a JPG, JPEG, or PNG.")
        return redirect(url_for("index"))

    ext = file.filename.rsplit(".", 1)[1].lower()
    uid = uuid.uuid4().hex[:10]
    saved_filename = f"{uid}.{ext}"
    saved_path = os.path.join(config.UPLOAD_DIR, saved_filename)
    file.save(saved_path)

    gradcam_filename = f"{uid}_gradcam.png"
    gradcam_path = os.path.join(config.GRADCAM_DIR, gradcam_filename)

    try:
        result = predict_image(saved_path, save_gradcam_path=gradcam_path)
    except Exception as e:
        flash(f"Prediction failed: {e}")
        return redirect(url_for("index"))

    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": saved_filename,
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
    }
    save_history_entry(entry)

    return render_template(
        "result.html",
        image_url=url_for("static", filename=f"uploads/{saved_filename}"),
        gradcam_url=url_for("static", filename=f"gradcam/{gradcam_filename}") if result["gradcam_path"] else None,
        prediction=result["predicted_class"],
        confidence=result["confidence"],
        low_confidence=result["low_confidence"],
        probabilities=result["probabilities"],
        explanation=explanation_for(result["predicted_class"]),
        timestamp=entry["timestamp"],
    )


@app.route("/history")
def history():
    return render_template("history.html", history=load_history())


@app.route("/api/history")
def api_history():
    return jsonify(load_history())


if __name__ == "__main__":
    app.run(debug=True)
