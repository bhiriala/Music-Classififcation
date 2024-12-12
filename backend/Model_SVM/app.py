from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import pickle
import os

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})
app.config["UPLOAD_FOLDER"] = "static/uploads"
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])
model_filename = "svm_genre_classifier.pkl"
with open(model_filename, "rb") as file:
    clf = pickle.load(file)

genres = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]


def predict_genre(file_path, clf):
    hop_length = 512
    n_fft = 2048
    n_mels = 128

    signal, rate = librosa.load(file_path)
    S = librosa.feature.melspectrogram(
        y=signal, sr=rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    S_DB = librosa.power_to_db(S, ref=np.max)
    S_DB = S_DB.flatten()[:1200]

    genre_label = clf.predict([S_DB])[0]
    return genres[genre_label]


@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if not file.filename.endswith(".wav"):
        return jsonify({"error": "File must be a .wav file"}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    try:
        predicted_genre = predict_genre(filepath, clf)
        result = {"genre": predicted_genre}
    except Exception as e:
        result = {"error": str(e)}
    os.remove(filepath)

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
