# app/inference.py
import os
import json
import urllib.request
from pathlib import Path

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------------- Labels ----------------
with open(Path(__file__).with_name("labels.json"), "r") as f:
    LABELS = json.load(f)

# ---------------- Robust model loader ----------------
BASE_DIR = Path(__file__).parent
MODELS_DIR = (BASE_DIR / "../models").resolve()

# Change this if your file is .h5 instead:
PREFERRED = MODELS_DIR / "fer_mnet_4cls.keras"

def _find_latest_model():
    # prefer Keras/HD5 files by mtime
    cands = sorted(
        list(MODELS_DIR.glob("*.keras")) + list(MODELS_DIR.glob("*.h5")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if cands:
        return cands[0]
    # fallback to SavedModel directory
    dirs = sorted(
        [d for d in MODELS_DIR.iterdir() if d.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for d in dirs:
        if (d / "saved_model.pb").exists():
            return d
    return None

def _load_model():
    target = PREFERRED if PREFERRED.exists() else _find_latest_model()
    if target is None:
        raise FileNotFoundError(f"No model found in {MODELS_DIR}. Train and save one first.")
    print(f"[inference] Loading model: {target}")
    return tf.keras.models.load_model(str(target))

MODEL = _load_model()

# ---------------- Face detector (OpenCV DNN SSD) ----------------
PROTOTXT = BASE_DIR / "deploy.prototxt"
CAFFEMODEL = BASE_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

_URLS = {
    "prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
}

def _ensure_detector():
    try:
        if not PROTOTXT.exists():
            urllib.request.urlretrieve(_URLS["prototxt"], PROTOTXT)
        if not CAFFEMODEL.exists():
            urllib.request.urlretrieve(_URLS["caffemodel"], CAFFEMODEL)
    except Exception as e:
        print("[inference] Warning: could not download detector weights:", e)

_ensure_detector()
DNN = cv2.dnn.readNetFromCaffe(str(PROTOTXT), str(CAFFEMODEL)) if PROTOTXT.exists() and CAFFEMODEL.exists() else None

# Eye cascade for simple alignment
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def _align_face(gray, box):
    x, y, w, h = box
    roi = gray[y:y+h, x:x+w]
    eyes = EYE_CASCADE.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[0])[:2]
        (x1, y1, w1, h1) = eyes[0]
        (x2, y2, w2, h2) = eyes[1]
        p1 = (x1 + w1 // 2, y1 + h1 // 2)
        p2 = (x2 + w2 // 2, y2 + h2 // 2)
        dy, dx = (p2[1] - p1[1]), (p2[0] - p1[0])
        angle = np.degrees(np.arctan2(dy, dx))
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        roi = cv2.warpAffine(roi, M, (w, h), flags=cv2.INTER_LINEAR)
    return roi

def _detect_faces_dnn(frame_bgr, conf=0.6):
    if DNN is None:
        return []
    h, w = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(frame_bgr, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    DNN.setInput(blob)
    det = DNN.forward()
    boxes = []
    for i in range(det.shape[2]):
        score = float(det[0, 0, i, 2])
        if score < conf:
            continue
        x1, y1, x2, y2 = (det[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes

def _preprocess_face(bgr, box):
    x, y, w, h = box
    face = bgr[y:y+h, x:x+w]
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = _align_face(gray, (0, 0, w, h))
    gray = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype("float32") / 255.0
    # Model expects [0,1] and has a Rescaling(2, -1) layer inside
    return np.expand_dims(rgb, axis=0)  # (1,96,96,3)

def predict_from_bgr(frame_bgr):
    faces = _detect_faces_dnn(frame_bgr, conf=0.6)
    if len(faces) == 0:
        return {"success": True, "face": None, "top": None, "probs": {lbl: 0.0 for lbl in LABELS}}
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    inp = _preprocess_face(frame_bgr, (x, y, w, h))
    preds = MODEL.predict(inp, verbose=0)[0]
    probs = {LABELS[i]: float(preds[i]) for i in range(len(LABELS))}
    top_idx = int(np.argmax(preds))
    return {
        "success": True,
        "face": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
        "top": {"label": LABELS[top_idx], "score": float(preds[top_idx])},
        "probs": probs,
    }

def decode_image(file_bytes: bytes):
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img
