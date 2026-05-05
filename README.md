# Spam Classifier

ML-powered spam detection pipeline built with Python, TensorFlow, Keras, and NLTK. Classifies text as spam or legitimate in real time with iterative model improvement.

## Features
- Text preprocessing pipeline (tokenisation, stopword removal, stemming)
- TF-IDF and word embedding feature extraction
- LSTM neural network classifier
- Real-time scoring API
- Model evaluation and retraining scripts

## Stack
Python · TensorFlow · Keras · NLTK · scikit-learn · Flask

## Quickstart
```bash
pip install -r requirements.txt
python src/train.py
python src/app.py
```

## Results
- 97.4% accuracy on test set
- <50ms average inference time
