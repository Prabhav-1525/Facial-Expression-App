# app/inference.py
import json
from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf

# ---------------- Paths & labels ----------------
BASE_DIR = Path(__file__).parent
MODELS_DIR = (BASE_DIR / "../models").resolve()
LABELS = json.loads((BASE_DIR / "labels.json").read_text())

# Prefer this filename; otherwise pick newest .keras/.h5 or SavedModel dir
PREFERRED = MODELS_DIR / "fer_mnet_4cls.keras"

def _find_latest_model():
    cands = sorted(
        list(MODELS_DIR.glob("*.keras")) + list(MODELS_DIR.glob("*.h5")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if cands:
        return cands[0]
    for d in sorted([p for p in MODELS_DIR.iterdir() if p.is_dir()],
                    key=lambda p: p.stat().st_mtime, reverse=True):
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
try:
    _ = MODEL.predict(np.zeros((1, 96, 96, 3), dtype="float32"), verbose=0)
    print("[inference] Model warm-up done.")
except Exception as e:
    print("[inference] Warm-up skipped:", e)

# ---------------- Face detectors ----------------
# Use local, vendored DNN files (commit them to app/)
PROTOTXT = BASE_DIR / "deploy.prototxt"
CAFFEMODEL = BASE_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

DNN = None
if PROTOTXT.exists() and CAFFEMODEL.exists():
    try:
        DNN = cv2.dnn.readNetFromCaffe(str(PROTOTXT), str(CAFFEMODEL))
        print("[inference] DNN face detector loaded.")
    except Exception as e:
        print("[inference] Failed to load DNN detector:", e)

# Haar fallback so we’re never face-blind in prod
HAAR_FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def _detect_faces_haar(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    rects = HAAR_FACE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return [(x, y, w, h) for (x, y, w, h) in rects]

def _detect_faces_dnn(frame_bgr, conf=0.30):  # ↓ lowered from 0.6 to 0.30
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

def _align_face(gray, box):
    # box may contain numpy types; coerce and guard
    x, y, w, h = [int(v) for v in box]
    if w <= 0 or h <= 0:
        return gray  # nothing to do

    # Crop ROI safely
    H, W = gray.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    roi = gray[y0:y1, x0:x1].copy()

    # Detect eyes inside ROI
    eyes = EYE_CASCADE.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

    if len(eyes) >= 2:
        # pick two left-most eyes
        eyes = sorted(eyes, key=lambda e: e[0])[:2]
        (x1e, y1e, w1e, h1e), (x2e, y2e, w2e, h2e) = eyes
        p1 = (x1e + w1e // 2, y1e + h1e // 2)
        p2 = (x2e + w2e // 2, y2e + h2e // 2)

        dy, dx = float(p2[1] - p1[1]), float(p2[0] - p1[0])
        angle = float(np.degrees(np.arctan2(dy, dx)))

        # center must be Python floats
        cx, cy = float(roi.shape[1] * 0.5), float(roi.shape[0] * 0.5)
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        aligned = cv2.warpAffine(roi, M, (roi.shape[1], roi.shape[0]), flags=cv2.INTER_LINEAR)
        return aligned

    return roi

def _preprocess_face(bgr, box):
    x, y, w, h = box
    face = bgr[y:y+h, x:x+w]
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = _align_face(gray, (0, 0, w, h))
    gray = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype("float32") / 255.0
    # Model expects [0,1]; Rescaling(2,-1) is inside the model
    return np.expand_dims(rgb, axis=0)  # (1,96,96,3)

def predict_from_bgr(frame_bgr):
    faces = _detect_faces_dnn(frame_bgr, conf=0.30)
    if not faces:
        faces = _detect_faces_haar(frame_bgr)  # fallback

    if not faces:
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
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

