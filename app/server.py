from flask import Flask, render_template, request, jsonify
from .inference import predict_from_bgr, decode_image 

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.post('/predict')
def predict():
    if 'frame' not in request.files:
        return jsonify({"success": False, "error": "No frame"}), 400


    file = request.files['frame']
    img = decode_image(file.read())
    if img is None:
        return jsonify({"success": False, "error": "Bad image"}), 400


    result = predict_from_bgr(img)
    return jsonify(result)


if __name__ == '__main__':
    # For dev only; use a production WSGI server for deployment
    app.run(host='0.0.0.0', port=5000, debug=True)