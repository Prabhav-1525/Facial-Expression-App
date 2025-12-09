# Facial Expression Recognition App (FER2013 → Flask)

Real-time emotion recognition (happy, sad, angry, neutral) from webcam frames, using TensorFlow/Keras and OpenCV.

## 1) Setup

### Python
- Python 3.9+ recommended
- Create venv and install deps:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt