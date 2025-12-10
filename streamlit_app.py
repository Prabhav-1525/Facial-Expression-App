# streamlit_app.py
import os
import json
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
    VideoProcessorBase,
)

# =============== UI ===============
st.set_page_config(page_title="Facial Expression Recognition", layout="wide")
st.title("Facial Expression Recognition\n(📦 FER2013 → 🧠 MobileNetV2)")

st.markdown(
    "Real-time webcam demo using OpenCV DNN face detection + a fine-tuned Keras model "
    "(classes: **angry**, **happy**, **sad**, **neutral**)."
)

# =============== TURN / WebRTC config ===============
def _get_secret(name: str, default: str = "") -> str:
    # Prefer Streamlit Secrets (cloud), fall back to env (local)
    val = ""
    try:
        val = st.secrets.get(name, "")
    except Exception:
        pass
    if not val:
        val = os.getenv(name, default)
    return val

TURN_USERNAME = _get_secret("TURN_USERNAME")
TURN_PASSWORD = _get_secret("TURN_PASSWORD")
USE_TURN = bool(TURN_USERNAME and TURN_PASSWORD)

RTC_CONFIGURATION: RTCConfiguration = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
}
if USE_TURN:
    RTC_CONFIGURATION["iceServers"].append(
        {
            "urls": [
                "turn:global.relay.metered.ca:80",
                "turn:global.relay.metered.ca:443",
                "turns:global.relay.metered.ca:443?transport=tcp",
            ],
            "username": TURN_USERNAME,
            "credential": TURN_PASSWORD,
        }
    )
    # Force relay to avoid UDP-blocked environments
    RTC_CONFIGURATION["iceTransportPolicy"] = "relay"
else:
    st.warning(
        "TURN credentials not found. On Streamlit Cloud, WebRTC usually needs a TURN relay.\n\n"
        "Add **TURN_USERNAME** and **TURN_PASSWORD** in **App → Settings → Secrets** "
        "(e.g., free key from metered.ca/openrelay).",
        icon="⚠️",
    )

# =============== Model & Detector (global singletons) ===============
BASE_DIR = Path(__file__).parent.resolve()
APP_DIR = (BASE_DIR / "app").resolve()  # reuse the same files as Flask version, if present
MODELS_DIR = (BASE_DIR / "models").resolve()

LABELS_PATH = APP_DIR / "labels.json"
if not LABELS_PATH.exists():
    # Fallback to local labels
    LABELS = ["angry", "happy", "sad", "neutral"]
else:
    with open(LABELS_PATH, "r") as f:
        LABELS = json.load(f)

# Prefer Keras format
PREFERRED_MODEL = MODELS_DIR / "fer_mnet_4cls.keras"

def _find_latest_model() -> Optional[Path]:
    cands = sorted(
        list(MODELS_DIR.glob("*.keras")) + list(MODELS_DIR.glob("*.h5")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if cands:
        return cands[0]
    # SavedModel directory fallback
    dirs = sorted(
        [d for d in MODELS_DIR.iterdir() if d.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for d in dirs:
        if (d / "saved_model.pb").exists():
            return d
    return None

@st.cache_resource(show_spinner=True)
def load_keras_model():
    import tensorflow as tf
    target = PREFERRED_MODEL if PREFERRED_MODEL.exists() else _find_latest_model()
    if target is None:
        raise FileNotFoundError(f"No model found in {MODELS_DIR}. Train and upload one.")
    st.write(f"[model] Loading: `{target}`")
    model = tf.keras.models.load_model(str(target))
    # Warm-up to allocate kernels
    try:
        _ = model.predict(np.zeros((1, 96, 96, 3), dtype="float32"), verbose=0)
        st.write("[model] Warm-up done.")
    except Exception as e:
        st.write(f"[model] Warm-up skipped: {e}")
    return model

@st.cache_resource(show_spinner=True)
def load_face_detector():
    prototxt = APP_DIR / "deploy.prototxt"
    caffemodel = APP_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
    urls = {
        "prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    }
    for p, key in [(prototxt, "prototxt"), (caffemodel, "caffemodel")]:
        if not p.exists():
            try:
                urllib.request.urlretrieve(urls[key], p)
            except Exception as e:
                st.warning(f"[detector] Could not download {p.name}: {e}")
    if prototxt.exists() and caffemodel.exists():
        net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
        st.write("[detector] OpenCV DNN face detector loaded.")
        return net
    st.error("Face detector weights missing. Upload `deploy.prototxt` and "
             "`res10_300x300_ssd_iter_140000.caffemodel` into `app/`.")
    return None

MODEL = load_keras_model()
DNN = load_face_detector()
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# =============== Pre/Post utilities ===============
def detect_faces_dnn(frame_bgr: np.ndarray, conf: float = 0.6) -> list[Tuple[int, int, int, int]]:
    if DNN is None:
        return []
    h, w = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame_bgr, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False
    )
    DNN.setInput(blob)
    det = DNN.forward()
    boxes: list[Tuple[int, int, int, int]] = []
    for i in range(det.shape[2]):
        score = float(det[0, 0, i, 2])
        if score < conf:
            continue
        x1, y1, x2, y2 = (det[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
    return boxes

def align_face(gray: np.ndarray, box_wh: Tuple[int, int, int, int]) -> np.ndarray:
    # box_wh in local ROI coordinates: (0,0,w,h)
    _, _, w, h = box_wh
    roi = gray.copy()
    eyes = EYE_CASCADE.detectMultiScale(
        roi, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
    )
    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[0])[:2]
        (x1, y1, w1, h1) = map(int, eyes[0])
        (x2, y2, w2, h2) = map(int, eyes[1])
        p1 = (int(x1 + w1 // 2), int(y1 + h1 // 2))
        p2 = (int(x2 + w2 // 2), int(y2 + h2 // 2))
        dy, dx = (p2[1] - p1[1]), (p2[0] - p1[0])
        angle = float(np.degrees(np.arctan2(dy, dx)))
        center = (int(w // 2), int(h // 2))
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        roi = cv2.warpAffine(roi, M, (int(w), int(h)), flags=cv2.INTER_LINEAR)
    return roi

def preprocess_face(bgr: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = [int(v) for v in box]
    face = bgr[y : y + h, x : x + w]
    if face.size == 0:
        # fallback to center crop if bad box
        H, W = bgr.shape[:2]
        cx, cy = W // 2, H // 2
        s = min(H, W) // 2
        face = bgr[cy - s : cy + s, cx - s : cx + s]
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = align_face(gray, (0, 0, gray.shape[1], gray.shape[0]))
    gray = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype("float32") / 255.0
    return np.expand_dims(rgb, axis=0)  # (1,96,96,3)

def softmax_to_dict(pred: np.ndarray) -> dict:
    return {LABELS[i]: float(pred[i]) for i in range(len(LABELS))}

def ema(curr: dict, prev: Optional[dict], alpha: float) -> dict:
    if prev is None:
        return curr
    out = {}
    for k in curr.keys():
        out[k] = alpha * curr[k] + (1.0 - alpha) * prev.get(k, 0.0)
    return out

# =============== Streamlit controls ===============
alpha = st.slider("Smoothing (EMA α)", min_value=0.0, max_value=1.0, value=0.60, step=0.05)

# =============== WebRTC video processor ===============
class FERVideoProcessor(VideoProcessorBase):
    def __init__(self, alpha_val: float = 0.6):
        self.alpha = float(alpha_val)
        self.prev: Optional[dict] = None

    def _predict(self, frame_bgr: np.ndarray) -> Tuple[dict, Optional[Tuple[int, int, int, int]]]:
        faces = detect_faces_dnn(frame_bgr, conf=0.60)
        if not faces:
            return ({lbl: 0.0 for lbl in LABELS}, None)
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        inp = preprocess_face(frame_bgr, (x, y, w, h))
        preds = MODEL.predict(inp, verbose=0)[0]
        probs = softmax_to_dict(preds)
        return probs, (x, y, w, h)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        probs, box = self._predict(img)
        smoothed = ema(probs, self.prev, self.alpha)
        self.prev = smoothed

        # Draw overlays
        vis = img.copy()
        if box:
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (122, 162, 247), 2)

        # Top label from smoothed scores
        lbl = max(smoothed, key=smoothed.get)
        score = smoothed[lbl]
        cv2.putText(
            vis,
            f"{lbl}: {score*100:.1f}%",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return av.VideoFrame.from_ndarray(vis, format="bgr24")

# =============== Start WebRTC ===============
st.subheader("Live webcam (WebRTC)")
ctx = webrtc_streamer(
    key="fer-demo",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=lambda: FERVideoProcessor(alpha_val=alpha),
)

# Connection status panel
st.subheader("Connection status")
if ctx and hasattr(ctx, "peer_connection") and ctx.peer_connection:
    pc = ctx.peer_connection
    st.write("iceConnectionState:", getattr(pc, "iceConnectionState", None))
    st.write("connectionState:", getattr(pc, "connectionState", None))
else:
    st.write("Waiting for peer connection…")

# =============== Snapshot fallback ===============
st.markdown("---")
st.subheader("Snapshot fallback (works without WebRTC)")
snap = st.camera_input("Take a snapshot (then deselect to retake)")
if snap is not None:
    # Decode to OpenCV
    bytes_data = snap.getvalue()
    arr = np.frombuffer(bytes_data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    probs, box = FERVideoProcessor(alpha_val=alpha)._predict(bgr)
    lbl = max(probs, key=probs.get)
    st.write("Predicted:", f"**{lbl}** ({probs[lbl]*100:.1f}%)")
    st.json({k: round(v, 4) for k, v in probs.items()})
    if box:
        x, y, w, h = [int(v) for v in box]
        cv2.rectangle(bgr, (x, y), (x + w, y + h), (122, 162, 247), 2)
    st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Detection")
