from flask import Flask, render_template, request, jsonify
from .inference import predict_from_bgr, decode_image 
import traceback

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.post("/predict")
def predict():
    try:
        if "frame" in request.files:
            data = request.files["frame"].read()
        elif "image" in request.files:
            data = request.files["image"].read()
        else:
            return jsonify({"success": False, "error": "No file field (frame/image)"}), 400

        img = decode_image(data)
        if img is None:
            return jsonify({"success": False, "error": "Decode failed"}), 400

        res = predict_from_bgr(img)
        return jsonify(res)

    except Exception as e:
        print("[server] /predict error:", e)
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    # For dev only; use a production WSGI server for deployment
    app.run(host='0.0.0.0', port=5000, debug=True)