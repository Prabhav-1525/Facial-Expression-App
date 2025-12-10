# streamlit_app.py  — Snapshot-only (no WebRTC)

import io
import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

# ---------- App/Model paths ----------
BASE_DIR = Path(__file__).parent
APP_DIR = BASE_DIR / "app"
MODELS_DIR = BASE_DIR / "models"
LABELS_PATH = APP_DIR / "labels.json"
PREFERRED_MODEL = MODELS_DIR / "fer_mnet_4cls.keras"   # change if needed

# ---------- UI ----------
st.set_page_config(page_title="Facial Expression Recognition (Snapshots)", layout="centered")
st.title("Facial Expression Recognition (FER2013 → MobileNetV2)")
st.caption("Snapshot-based demo using OpenCV DNN face detection + Keras model (angry, happy, sad, neutral).")

# ---------- Labels ----------
if LABELS_PATH.exists():
    LABELS = json.loads(LABELS_PATH.read_text())
else:
    LABELS = ["angry", "happy", "sad", "neutral"]  # fallback

# ---------- Caching: model + detector ----------
@st.cache_resource(show_spinner="Loading model & detector…")
def load_model_and_detector():
    # Load model
    if PREFERRED_MODEL.exists():
        model = tf.keras.models.load_model(str(PREFERRED_MODEL))
    else:
        # pick the newest .keras/.h5 if preferred missing
        cands = sorted(
            list(MODELS_DIR.glob("*.keras")) + list(MODELS_DIR.glob("*.h5")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not cands:
            raise FileNotFoundError(f"No model file found in {MODELS_DIR}")
        model = tf.keras.models.load_model(str(cands[0]))

    # Warm-up (avoids first-prediction latency)
    try:
        _ = model.predict(np.zeros((1, 96, 96, 3), dtype="float32"), verbose=0)
    except Exception:
        pass

    # Ensure detector weights (local or auto-download)
    prototxt = APP_DIR / "deploy.prototxt"
    caffemodel = APP_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
    if not prototxt.exists() or not caffemodel.exists():
        import urllib.request
        URLS = {
            "prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            "caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        }
        try:
            if not prototxt.exists():
                urllib.request.urlretrieve(URLS["prototxt"], prototxt)
            if not caffemodel.exists():
                urllib.request.urlretrieve(URLS["caffemodel"], caffemodel)
        except Exception as e:
            st.warning(f"Could not download face detector weights: {e}")

    dnn = None
    if (APP_DIR / "deploy.prototxt").exists() and (APP_DIR / "res10_300x300_ssd_iter_140000.caffemodel").exists():
        dnn = cv2.dnn.readNetFromCaffe(str(APP_DIR / "deploy.prototxt"),
                                       str(APP_DIR / "res10_300x300_ssd_iter_140000.caffemodel"))
    return model, dnn

MODEL, DNN = load_model_and_detector()
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# ---------- Helpers ----------
def _align_face(gray, box):
    x, y, w, h = box
    roi = gray[y:y + h, x:x + w]
    eyes = EYE_CASCADE.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[0])[:2]
        (x1, y1, w1, h1) = eyes[0]
        (x2, y2, w2, h2) = eyes[1]
        p1 = (x1 + w1 // 2, y1 + h1 // 2)
        p2 = (x2 + w2 // 2, y2 + h2 // 2)
        dy, dx = (p2[1] - p1[1]), (p2[0] - p1[0])
        angle = float(np.degrees(np.arctan2(dy, dx)))
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        roi = cv2.warpAffine(roi, M, (w, h), flags=cv2.INTER_LINEAR)
    return roi

def _detect_faces_dnn(frame_bgr, conf=0.6):
    if DNN is None:
        return []
    h, w = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(frame_bgr, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0), swapRB=False, crop=False)
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

def _preprocess_face_for_model(bgr, box):
    x, y, w, h = box
    face = bgr[y:y + h, x:x + w]
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = _align_face(gray, (0, 0, w, h))
    gray = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype("float32") / 255.0
    return np.expand_dims(rgb, axis=0)  # (1, 96, 96, 3)

def predict_on_image(bgr):
    faces = _detect_faces_dnn(bgr, conf=0.6)
    if len(faces) == 0:
        return None, None, {"note": "No face detected."}

    # choose largest face
    box = max(faces, key=lambda b: b[2] * b[3])
    inp = _preprocess_face_for_model(bgr, box)
    preds = MODEL.predict(inp, verbose=0)[0]
    probs = {LABELS[i]: float(preds[i]) for i in range(len(LABELS))}
    top_idx = int(np.argmax(preds))
    top_label = LABELS[top_idx]
    top_score = float(preds[top_idx])

    # annotate
    x, y, w, h = box
    vis = bgr.copy()
    cv2.rectangle(vis, (x, y), (x + w, y + h), (122, 162, 247), 2)
    cv2.putText(vis, f"{top_label}: {top_score:.2f}", (x, max(0, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (122, 162, 247), 2, cv2.LINE_AA)
    return vis, probs, {"box": {"x": x, "y": y, "w": w, "h": h}, "top": {"label": top_label, "score": top_score}}

def bytes_to_bgr(file_bytes: bytes):
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

# ---------- EMA smoothing across snapshots (optional) ----------
alpha = st.slider("Smoothing (EMA α)", 0.0, 1.0, 0.60, 0.05, help="Smoothing across your last predictions.")
if "ema_probs" not in st.session_state:
    st.session_state.ema_probs = None

def ema(curr: dict, prev: dict, a: float):
    if prev is None:
        return curr
    out = {}
    for k in curr.keys():
        out[k] = a * curr[k] + (1.0 - a) * prev.get(k, 0.0)
    return out

# ---------- Input widgets ----------
st.subheader("Capture a snapshot")
c1, c2 = st.columns(2)
with c1:
    cam_img = st.camera_input("Camera (click “Take Photo”)", label_visibility="collapsed")
with c2:
    uploaded = st.file_uploader("Or upload a photo", type=["jpg", "jpeg", "png"])

run = st.button("Run on last snapshot", type="primary")
reset = st.button("Reset smoothing")

if reset:
    st.session_state.ema_probs = None
    st.toast("Smoothing reset.", icon="✅")

# Determine source image
source_bytes = None
source_label = None
if cam_img is not None:
    source_bytes = cam_img.getvalue()
    source_label = "Camera snapshot"
elif uploaded is not None:
    source_bytes = uploaded.read()
    source_label = f"Uploaded: {uploaded.name}"

# If user pressed Run, do inference
if run:
    if not source_bytes:
        st.warning("Please capture a photo or upload an image first.")
    else:
        bgr = bytes_to_bgr(source_bytes)
        if bgr is None:
            st.error("Could not decode the image.")
        else:
            vis, probs, meta = predict_on_image(bgr)
            if vis is None:
                st.info("No face detected. Try better lighting and a frontal view.")
            else:
                # smoothing
                st.session_state.ema_probs = ema(probs, st.session_state.ema_probs, alpha)
                final_probs = st.session_state.ema_probs or probs

                # Show annotated image
                st.subheader("Result")
                st.caption(source_label)
                st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

                # Probabilities
                st.write("Probabilities")
                st.dataframe(
                    {
                        "emotion": list(final_probs.keys()),
                        "probability": [round(final_probs[k], 4) for k in final_probs.keys()],
                    },
                    hide_index=True,
                )

                # Top label
                top_em = max(final_probs, key=lambda k: final_probs[k])
                st.success(f"Top: **{top_em}** ({final_probs[top_em]*100:.1f}%)")
else:
    st.info("Take a photo or upload an image, then click **Run on last snapshot**.")

st.divider()
st.caption("Tip: good lighting and a centered, frontal face help detection and accuracy.")
