from flask import Flask, request, jsonify
from flask_cors import CORS  # Pour activer CORS
import librosa
import numpy as np
import os
import joblib

app = Flask(__name__)
CORS(app)  # Activer CORS pour toutes les routes

# Charger le modèle SVM
model = joblib.load("model.pkl")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]  # Aligné avec le frontend

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = "./temp_audio.wav"
    file.save(file_path)

    try:
        # Extraire les caractéristiques
        y, sr = librosa.load(file_path, sr=None)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

        # Moyenne des caractéristiques
        spectral_centroid_avg = np.mean(spectral_centroids)
        spectral_rolloff_avg = np.mean(spectral_rolloff)

        # Préparer les caractéristiques pour le modèle
        features = np.array([spectral_centroid_avg, spectral_rolloff_avg]).reshape(
            1, -1
        )

        # Prédiction
        predicted_genre = model.predict(features)[0]

        result = {
            "genre": predicted_genre,  # Aligner la clé avec `response.genre` dans le frontend
        }
    except Exception as e:
        app.logger.error(f"Erreur lors du traitement du fichier : {str(e)}")
        result = {"error": f"Error processing audio: {str(e)}"}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    return jsonify(result)


if __name__ == "__main__":
    app.run(port=5001, debug=True)
