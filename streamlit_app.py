# streamlit_app.py
# Full Streamlit app with WebRTC (TURN-ready), async processing, connection status,
# and a snapshot fallback. Works with TensorFlow 2.16.1 on Python 3.11.

import os

# Disable file watching on Streamlit Cloud to avoid inotify limits
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

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

# Your inference function & labels come from the Flask app's module
from app.inference import predict_from_bgr

# ---------------------------- Streamlit UI ----------------------------
st.set_page_config(page_title="Facial Expression Recognition", layout="wide")
st.title("Facial Expression Recognition\n(FER2013 → MobileNetV2)")

st.markdown(
    "Real-time webcam demo using OpenCV DNN face detection + Keras model "
    "(angry, happy, sad, neutral)."
)

alpha = st.slider("Smoothing (EMA α)", 0.0, 1.0, 0.60, 0.05, help="Applies EMA to probabilities.")

# ---------------------------- WebRTC / TURN ---------------------------
# Read TURN credentials from environment (recommended for secrets)
TURN_USERNAME = os.getenv("TURN_USERNAME", "").strip()
TURN_PASSWORD = os.getenv("TURN_PASSWORD", "").strip()
USE_TURN = bool(TURN_USERNAME and TURN_PASSWORD)

# Base config with public STUN
RTC_CONFIGURATION: dict = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
}

# If TURN creds provided, add TURN and force relay to avoid UDP flakiness on some hosts
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
    RTC_CONFIGURATION["iceTransportPolicy"] = "relay"

# ------------------------ Video Processor (webrtc) --------------------
class FERVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.alpha = 0.60
        self.prev_probs = None  # for EMA smoothing

    def _ema(self, curr: dict) -> dict:
        """Apply EMA to a dict of probabilities keyed by label."""
        if self.prev_probs is None:
            self.prev_probs = curr
            return curr
        out = {k: self.alpha * curr[k] + (1.0 - self.alpha) * self.prev_probs[k] for k in curr}
        self.prev_probs = out
        return out

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert frame to BGR
        bgr = frame.to_ndarray(format="bgr24")

        # Run inference (handles detection + preprocessing internally)
        out = predict_from_bgr(bgr)

        vis = bgr.copy()
        face = out.get("face")
        probs = out.get("probs", {})

        # Smooth probabilities for stability
        if probs:
            probs = self._ema(probs)

        # Draw overlays
        if face:
            x, y, w, h = face["x"], face["y"], face["w"], face["h"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (80, 160, 255), 2)

            # Determine top label from smoothed probs if available
            if probs:
                top_lbl = max(probs.items(), key=lambda kv: kv[1])[0]
                top_score = probs[top_lbl] * 100.0
                label = f"{top_lbl} {top_score:.1f}%"
            elif out.get("top"):
                label = f'{out["top"]["label"]} {out["top"]["score"] * 100:.1f}%'
            else:
                label = "face"

            cv2.putText(
                vis,
                label,
                (x, max(0, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (80, 160, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                vis,
                "No face",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (160, 160, 160),
                2,
                cv2.LINE_AA,
            )

        return av.VideoFrame.from_ndarray(vis, format="bgr24")


# ---------------------------- WebRTC Component ------------------------
ctx = webrtc_streamer(
    key="fer-demo",
    mode=WebRtcMode.SENDRECV,  # enum (not a string)
    rtc_configuration=RTC_CONFIGURATION,  # dict is fine
    media_stream_constraints={
        "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
        "audio": False,
    },
    video_processor_factory=FERVideoProcessor,
    async_processing=True,     # don't block UI thread
    sendback_audio=False,
)

# Live-tune alpha on the processor
if ctx and ctx.video_processor:
    ctx.video_processor.alpha = float(alpha)

# ---------------------------- Connection Status -----------------------
st.subheader("Connection status")
if ctx is None:
    st.warning("Component not initialized.")
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("state:", getattr(ctx, "state", None))
        st.write("playing:", getattr(ctx, "playing", None))
    pc = getattr(ctx, "peer_connection", None)
    if pc is not None:
        with col2:
            st.write("signalingState:", getattr(pc, "signalingState", None))
            st.write("iceConnectionState:", getattr(pc, "iceConnectionState", None))
        with col3:
            st.write("connectionState:", getattr(pc, "connectionState", None))
            st.caption(
                "If ICE stays at 'checking/failed', ensure TURN credentials are set "
                "via environment variables TURN_USERNAME and TURN_PASSWORD."
            )

# ---------------------------- Snapshot Fallback -----------------------
st.divider()
st.subheader("Fallback: Snapshot mode")
snap = st.camera_input("If live video doesn’t start, use a snapshot")
if snap is not None:
    # Decode JPEG to BGR
    arr = np.frombuffer(snap.getvalue(), np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    out = predict_from_bgr(bgr)
    vis = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    f = out.get("face")
    if f:
        x, y, w, h = f["x"], f["y"], f["w"], f["h"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (80, 160, 255), 2)
        if out.get("top"):
            cv2.putText(
                vis,
                f'{out["top"]["label"]} {out["top"]["score"]*100:.1f}%',
                (x, max(0, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (80, 160, 255),
                2,
                cv2.LINE_AA,
            )

    st.image(vis, caption="Snapshot prediction", use_column_width=True)
    if "probs" in out:
        st.json(out["probs"])

st.caption("Tip: use good lighting and look straight at the camera for best results.")
