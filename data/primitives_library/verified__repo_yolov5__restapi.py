import argparse
import io
import os
import secrets

from flask import Flask, request
from PIL import Image
from werkzeug.exceptions import RequestEntityTooLarge

DETECTION_URL = "/v1/object-detection/<model>"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"}
MAX_IMAGE_SIZE = 16 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_SIZE
models = {}


@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(_):
    """Return a JSON error for uploads rejected by Flask before request parsing."""
    return {"error": "File too large. Maximum size is 16 MB."}, 413


@app.route(DETECTION_URL, methods=["POST"])
def predict(model):
    """Predict and return object detections in JSON format given an image and model name via a Flask REST API POST request."""
    configured_key = os.getenv("API_KEY")
    submitted_key = request.headers.get("X-API-Key", "")

    if configured_key and not secrets.compare_digest(submitted_key.encode(), configured_key.encode()):
        return {"error": "Unauthorized"}, 401

    upload = request.files.get("image")
    if not upload:
        return {"error": "No image file provided"}, 400

    name = upload.filename or ""
    suffix = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if suffix not in ALLOWED_EXTENSIONS:
        return {"error": "Invalid file type. Allowed types: " + ", ".join(sorted(ALLOWED_EXTENSIONS))}, 400

    contents = upload.read(MAX_IMAGE_SIZE + 1)
    if len(contents) > MAX_IMAGE_SIZE:
        return {"error": "File too large. Maximum size is 16 MB."}, 413

    try:
        with Image.open(io.BytesIO(contents)) as candidate:
            candidate.verify()
    except Exception:
        return {"error": "Invalid image file"}, 400

    image = Image.open(io.BytesIO(contents))

    if model not in models:
        return {"error": "Model not found. Available models: " + ", ".join(sorted(models))}, 404

    prediction = models[model](image, size=640)
    return prediction.pandas().xyxy[0].to_json(orient="records")


if __name__ == "__main__":
    import torch

    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    parser.add_argument("--model", nargs="+", default=["yolov5s"], help="model(s) to run, i.e. --model yolov5n yolov5s")
    opt = parser.parse_args()

    for name in opt.model:
        models[name] = torch.hub.load("ultralytics/yolov5", name, force_reload=True, skip_validation=True)

    app.run(host="127.0.0.1", port=opt.port)