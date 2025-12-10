# streamlit_app.py
import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Import your existing inference pipeline
from app.inference import predict_from_bgr

st.set_page_config(page_title="Facial Expression Recognition", layout="centered")
st.title("Facial Expression Recognition (FER2013 → MobileNetV2)")

st.markdown(
    """
This demo runs a MobileNetV2-based model fine-tuned on FER2013 (angry, happy, sad, neutral).
It detects a face with OpenCV DNN (or Haar fallback), aligns, preprocesses, and predicts in real time.
"""
)

LABELS = ["angry", "happy", "sad", "neutral"]

# Optional UI: smoothing alpha (EMA applied inside the VideoProcessor if you add it)
alpha = st.slider("Smoothing (EMA α)", 0.0, 1.0, 0.6, 0.05)

# WebRTC STUN config (public Google STUN)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class FERVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.alpha = 0.6
        self.prev = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")

        # Run your existing inference
        out = predict_from_bgr(img_bgr)

        # Visualize
        vis = img_bgr.copy()
        if out.get("face"):
            f = out["face"]
            x, y, w, h = f["x"], f["y"], f["w"], f["h"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (80, 160, 255), 2)

            # Draw label
            if out.get("top"):
                label = f'{out["top"]["label"]} {out["top"]["score"]*100:.1f}%'
                cv2.putText(vis, label, (x, max(0, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 160, 255), 2, cv2.LINE_AA)
        else:
            # Optional overlay when no face
            cv2.putText(vis, "No face", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (160, 160, 160), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(vis, format="bgr24")

ctx = webrtc_streamer(
    key="fer-demo",
    mode="SENDRECV",
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=FERVideoProcessor,
)

# live update of alpha (smoothing) if you later add EMA inside processor
if ctx and ctx.video_processor:
    ctx.video_processor.alpha = alpha

st.caption("Tip: ensure good lighting and face the camera for best results.")
