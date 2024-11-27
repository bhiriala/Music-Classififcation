from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

app = Flask(__name__)
CORS(app)  # Autoriser les requêtes Cross-Origin pour résoudre les problèmes CORS

# Charger le modèle VGG
model = load_model("music_genre_model.h5")

# Genres de musique pris en charge
GENRES = [
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


def preprocess_audio(file_path):
    try:
        # Charger l'audio et appliquer les prétraitements
        y, sr = librosa.load(file_path, sr=22050)
        y, _ = librosa.effects.trim(y)

        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=128, fmax=8000
        )
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Vérifier si le spectrogramme est vide
        if mel_spectrogram.shape[1] == 0:
            raise ValueError("Spectrogram is empty. Invalid audio file.")

        # Pad les séquences pour correspondre à l'entrée du modèle
        padded_mel = pad_sequences(
            [mel_spectrogram.T], maxlen=43, padding="post", truncating="post"
        )

        return np.expand_dims(
            padded_mel, axis=-1
        )  # Ajouter une dimension pour le modèle
    except Exception as e:
        raise ValueError(f"Error during audio preprocessing: {str(e)}")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]  # Utilisation correcte de la clé "file"

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Validation de l'extension du fichier
    if not file.filename.endswith(".wav"):
        return (
            jsonify({"error": "Invalid file format. Please upload a .wav file."}),
            400,
        )

    file_path = "./temp_audio.wav"
    file.save(file_path)

    try:
        # Prétraitement de l'audio
        audio_features = preprocess_audio(file_path)

        # Prédiction
        prediction = model.predict(audio_features)
        predicted_genre = GENRES[np.argmax(prediction)]

        result = {
            "genre": predicted_genre,  # Aligner la clé avec `response.genre` dans le frontend
        }
    except ValueError as e:
        result = {"error": str(e)}  # Erreur spécifique de prétraitement
    except Exception as e:
        result = {"error": f"Unexpected error: {str(e)}"}
    finally:
        # Suppression du fichier temporaire
        if os.path.exists(file_path):
            os.remove(file_path)

    return jsonify(result)


if __name__ == "__main__":
    app.run(port=5002, debug=True)
