from flask import Flask, request, jsonify
import requests
from flask_cors import CORS

app = Flask(__name__)

# Configure CORS to allow frontend requests from localhost:4200
CORS(app, resources={r"/predict": {"origins": "http://localhost:4200"}})

MODEL_ENDPOINTS = {
    "svm": "http://localhost:5001/predict",  # SVM service
    "vgg": "http://localhost:5002/predict",  # VGG service
}


@app.route("/predict", methods=["POST"])
def predict():
    print(f"Received request with method: {request.method}")  # Log de la méthode
    model_choice = request.form.get("model")
    file = request.files.get("songFile")
    print(f"Received request with method: {file}")

    if not file:
        return jsonify({"error": "No file part in the request"}), 400

    # Log de la réception des données
    print(f"Model choice: {model_choice}")
    print(f"Received file: {file.filename}")

    response = requests.post(MODEL_ENDPOINTS[model_choice], files={"file": file})
    return jsonify(response.json())


if __name__ == "__main__":
    app.run(port=5000, debug=True)
