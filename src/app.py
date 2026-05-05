"""
app.py
Spam Detection Classifier — Real-Time Inference API

Exposes a REST endpoint for real-time spam classification.
Loads the trained model and tokeniser once at startup and
scores incoming text with sub-50ms average latency.

Author: Aryan Patil

Usage:
    python src/app.py
    curl -X POST http://localhost:5000/predict \
         -H "Content-Type: application/json" \
         -d '{"text": "Congratulations! You have won a free prize. Click here now."}'
"""

import pickle
import time
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH      = "models/spam_classifier.h5"
TOKENIZER_PATH  = "models/tokenizer.pkl"
MAX_SEQUENCE_LEN = 200
SPAM_THRESHOLD   = 0.5     # Probability above which a message is classified spam

app = Flask(__name__)

# ── Load model and tokeniser at startup ──────────────────────────────────────
print("Loading model and tokeniser...")
model     = load_model(MODEL_PATH)
tokeniser = pickle.load(open(TOKENIZER_PATH, "rb"))
stemmer   = PorterStemmer()

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))
print("Ready.")


def preprocess(text: str) -> str:
    """
    Applies the same preprocessing pipeline used during training.
    Must match train.py exactly to avoid train/serve skew.

    @param  text    str     Raw input text from the API request
    @return str             Cleaned, stemmed, stopword-filtered text string
    """
    text   = text.lower()
    text   = "".join([ch for ch in text if ch.isalnum() or ch == " "])
    tokens = text.split()
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)


def score(text: str) -> dict:
    """
    Runs the full inference pipeline on a single text input.
    Preprocesses, tokenises, pads, and scores the text using the loaded model.

    @param  text    str     Raw text to classify
    @return dict            Result dict with keys:
                              - label (str):       "spam" or "ham"
                              - confidence (float): model probability score (0.0–1.0)
                              - latency_ms (float): end-to-end inference time in milliseconds
    """
    start     = time.time()
    cleaned   = preprocess(text)
    sequence  = tokeniser.texts_to_sequences([cleaned])
    padded    = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LEN, padding="post", truncating="post")
    prob      = float(model.predict(padded, verbose=0)[0][0])
    label     = "spam" if prob >= SPAM_THRESHOLD else "ham"
    latency   = round((time.time() - start) * 1000, 2)

    return {"label": label, "confidence": round(prob, 4), "latency_ms": latency}


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Accepts a JSON body with a 'text' field and returns a spam classification.

    @param  None    (reads from Flask request context)
    @return JSON    {"label": str, "confidence": float, "latency_ms": float}
                    or {"error": str} with HTTP 400 on bad input
    """
    data = request.get_json(force=True)
    if not data or "text" not in data:
        return jsonify({"error": "Request body must contain a 'text' field."}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "'text' field cannot be empty."}), 400

    result = score(text)
    return jsonify(result), 200


@app.route("/health", methods=["GET"])
def health():
    """
    GET /health
    Simple liveness check — confirms the API and model are loaded and responding.

    @param  None
    @return JSON    {"status": "ok"}
    """
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
