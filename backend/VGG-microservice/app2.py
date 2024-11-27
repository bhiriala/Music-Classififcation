from flask import Flask, request, jsonify
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

app = Flask(__name__)

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
    # Charger l'audio et appliquer les prétraitements
    y, sr = librosa.load(file_path, sr=22050)
    y, _ = librosa.effects.trim(y)

    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Pad les séquences pour correspondre à l'entrée du modèle
    padded_mel = pad_sequences(
        [mel_spectrogram.T], maxlen=43, padding="post", truncating="post"
    )

    return np.expand_dims(padded_mel, axis=-1)  # Ajouter une dimension pour le modèle


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["songFile"]  # Remplacez 'file' par 'songFile' ici aussi

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

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
    except Exception as e:
        result = {"error": f"Error processing audio: {str(e)}"}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    return jsonify(result)


if __name__ == "__main__":
    app.run(port=5002, debug=True)
