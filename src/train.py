"""
train.py
Spam Detection Classifier — Training Pipeline

Trains an LSTM-based spam classifier on labelled email/SMS data.
Saves the trained model and tokeniser for use by the inference API.

Author: Aryan Patil
"""

import os
import pickle
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ── Configuration ─────────────────────────────────────────────────────────────
MAX_VOCAB_SIZE  = 10_000   # Maximum number of words in vocabulary
MAX_SEQUENCE_LEN = 200     # Pad/truncate all sequences to this length
EMBEDDING_DIM   = 128      # Word embedding dimensions
LSTM_UNITS      = 64       # Number of LSTM units
DROPOUT_RATE    = 0.3      # Dropout for regularisation
BATCH_SIZE      = 32
EPOCHS          = 20       # Max epochs (early stopping will typically trigger earlier)
TEST_SIZE       = 0.2
RANDOM_STATE    = 42

MODEL_PATH     = "models/spam_classifier.h5"
TOKENIZER_PATH = "models/tokenizer.pkl"
DATA_PATH      = "data/spam_dataset.csv"


def download_nltk_resources():
    """
    Downloads required NLTK resources if not already present.

    @param  None
    @return None
    """
    for resource in ["stopwords", "punkt"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def preprocess_text(text: str, stemmer: PorterStemmer, stop_words: set) -> str:
    """
    Cleans and normalises a raw text string for model input.
    Applies lowercasing, punctuation removal, stopword filtering, and stemming.

    @param  text        str             Raw input text to preprocess
    @param  stemmer     PorterStemmer   NLTK stemmer instance for word stemming
    @param  stop_words  set             Set of stopwords to remove from the text
    @return str         Cleaned, stemmed, stopword-filtered text as a single string
    """
    text = text.lower()
    text = "".join([ch for ch in text if ch.isalnum() or ch == " "])
    tokens = text.split()
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)


def load_and_preprocess_data(data_path: str) -> tuple:
    """
    Loads the dataset from CSV and applies the full preprocessing pipeline.
    Expects a CSV with columns: 'label' (spam/ham) and 'text'.

    @param  data_path   str     Path to the CSV dataset file
    @return tuple               (X: list of preprocessed texts, y: np.ndarray of binary labels)
                                where 1 = spam, 0 = ham
    """
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path, encoding="latin-1")
    df = df[["label", "text"]].dropna()
    df["label"] = (df["label"] == "spam").astype(int)

    stemmer    = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    print("Preprocessing text...")
    df["cleaned"] = df["text"].apply(
        lambda t: preprocess_text(t, stemmer, stop_words)
    )

    print(f"Dataset loaded: {len(df)} samples | Spam: {df['label'].sum()} | Ham: {(df['label'] == 0).sum()}")
    return df["cleaned"].tolist(), df["label"].values


def build_tokeniser(texts: list, max_vocab: int) -> Tokenizer:
    """
    Fits a Keras Tokenizer on the training corpus.
    Stores word-to-index mappings for use during inference.

    @param  texts       list    List of preprocessed text strings to fit on
    @param  max_vocab   int     Maximum vocabulary size (top N words by frequency)
    @return Tokenizer           Fitted Keras Tokenizer instance
    """
    tokeniser = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
    tokeniser.fit_on_texts(texts)
    print(f"Vocabulary size: {min(len(tokeniser.word_index), max_vocab)} words")
    return tokeniser


def build_model(vocab_size: int, embedding_dim: int, max_len: int, lstm_units: int) -> Sequential:
    """
    Constructs the LSTM-based spam classification model.
    Architecture: Embedding -> SpatialDropout -> LSTM -> Dropout -> Dense

    @param  vocab_size      int     Size of the vocabulary (number of unique tokens)
    @param  embedding_dim   int     Dimensionality of the word embedding layer
    @param  max_len         int     Input sequence length (padded/truncated to this)
    @param  lstm_units      int     Number of units in the LSTM layer
    @return Sequential              Compiled Keras model ready for training
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
        SpatialDropout1D(0.2),
        LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2),
        Dropout(DROPOUT_RATE),
        Dense(32, activation="relu"),
        Dropout(DROPOUT_RATE),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()
    return model


def train(data_path: str = DATA_PATH):
    """
    Orchestrates the full training pipeline:
    load data → preprocess → tokenise → split → build model → train → evaluate → save.

    @param  data_path   str     Path to the labelled dataset CSV
    @return None                Saves trained model to MODEL_PATH and tokeniser to TOKENIZER_PATH
    """
    download_nltk_resources()
    os.makedirs("models", exist_ok=True)

    # Load and preprocess
    X, y = load_and_preprocess_data(data_path)

    # Tokenise and pad sequences
    tokeniser  = build_tokeniser(X, MAX_VOCAB_SIZE)
    sequences  = tokeniser.texts_to_sequences(X)
    X_padded   = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LEN, padding="post", truncating="post")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Build model
    model = build_model(MAX_VOCAB_SIZE, EMBEDDING_DIM, MAX_SEQUENCE_LEN, LSTM_UNITS)

    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy")
    ]

    # Train
    print("\nTraining...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )

    # Evaluate
    print("\nEvaluation:")
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save tokeniser
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokeniser, f)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Tokeniser saved to: {TOKENIZER_PATH}")


if __name__ == "__main__":
    train()
