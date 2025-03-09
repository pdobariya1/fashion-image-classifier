from PIL import Image
from flask import Flask, request, jsonify

from predict import load_model, predict


# initialize the flask app
app = Flask(__name__)


# Load model
model = load_model()


# Prediction route
@app.route("/predict", methods=["POST"])
def prediction():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    image = Image.open(file).convert("RGB")
    
    prediction = predict(image, model)
    
    return jsonify(prediction)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)