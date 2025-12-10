# streamlit_app.py
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
import av
import cv2
import numpy as np
import streamlit as st

# 👇 include VideoProcessorBase here
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
    VideoProcessorBase,
)

from app.inference import predict_from_bgr

st.set_page_config(page_title="Facial Expression Recognition", layout="centered")
st.title("Facial Expression Recognition (FER2013 → MobileNetV2)")

st.markdown(
    "Real-time webcam demo using OpenCV DNN face detection + Keras model (angry, happy, sad, neutral)."
)

LABELS = ["angry", "happy", "sad", "neutral"]
alpha = st.slider("Smoothing (EMA α)", 0.0, 1.0, 0.6, 0.05)

# Public Google STUN server
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        # TURN over TCP/TLS (replace with your real creds)
        {
            "urls": [
                "turn:global.relay.metered.ca:80",
                "turn:global.relay.metered.ca:443",
                "turns:global.relay.metered.ca:443?transport=tcp",
            ],
            "username": "<YOUR_TURN_USERNAME>",
            "credential": "<YOUR_TURN_PASSWORD>",
        },
    ],
    # Force using TURN only; avoids flaky UDP paths on Streamlit Cloud
    "iceTransportPolicy": "relay",
}
class FERVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.alpha = 0.6
        self.prev = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        bgr = frame.to_ndarray(format="bgr24")
        out = predict_from_bgr(bgr)

        vis = bgr.copy()
        face = out.get("face")
        if face:
            x, y, w, h = face["x"], face["y"], face["w"], face["h"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (80, 160, 255), 2)
            if out.get("top"):
                label = f'{out["top"]["label"]} {out["top"]["score"]*100:.1f}%'
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

ctx = webrtc_streamer(
    key="fer-demo",
    mode=WebRtcMode.SENDRECV,              # enum, not string
    rtc_configuration=RTC_CONFIGURATION,    # dict is fine
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=FERVideoProcessor,
)

if ctx and ctx.video_processor:
    ctx.video_processor.alpha = alpha

st.caption("Tip: use good lighting and look straight at the camera for best results.")
