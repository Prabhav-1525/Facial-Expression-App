# ---- keep your imports as-is ----
import cv2
import numpy as np

# Eye cascade (already in your file)
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def _align_face(gray: np.ndarray, box):
    # Unpack and force Python ints
    x, y, w, h = [int(v) for v in box]
    # Defensive: if anything is invalid, return the input ROI
    if w <= 0 or h <= 0:
        return gray

    roi = gray[y:y+h, x:x+w]
    # If ROI is empty for any reason, bail out gracefully
    if roi.size == 0:
        return gray

    # Detect eyes for a quick alignment
    eyes = EYE_CASCADE.detectMultiScale(
        roi, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
    )
    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[0])[:2]
        (x1, y1, w1, h1) = [int(v) for v in eyes[0]]
        (x2, y2, w2, h2) = [int(v) for v in eyes[1]]
        p1 = (x1 + w1 // 2, y1 + h1 // 2)
        p2 = (x2 + w2 // 2, y2 + h2 // 2)
        dy = float(p2[1] - p1[1])
        dx = float(p2[0] - p1[0])
        angle = float(np.degrees(np.arctan2(dy, dx)))

        # OpenCV wants native Python floats here
        cx = float(w) / 2.0
        cy = float(h) / 2.0
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        aligned = cv2.warpAffine(roi, M, (int(w), int(h)), flags=cv2.INTER_LINEAR)
        return aligned

    return roi


def _preprocess_face_for_model(bgr: np.ndarray, box):
    # Unpack/validate bbox and crop
    x, y, w, h = [int(v) for v in box]
    if w <= 0 or h <= 0:
        # Return a neutral dummy tensor that won’t crash the model
        dummy = np.zeros((1, 96, 96, 3), dtype="float32")
        return dummy

    face = bgr[y:y+h, x:x+w]
    if face.size == 0:
        dummy = np.zeros((1, 96, 96, 3), dtype="float32")
        return dummy

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = _align_face(gray, (0, 0, w, h))  # align in the local face coords
    gray = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype("float32") / 255.0
    return np.expand_dims(rgb, axis=0)  # (1,96,96,3)
