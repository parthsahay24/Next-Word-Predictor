"""
Prediction Module for Next Word Predictor

Loads a trained LSTM model and generates next word predictions
with configurable temperature and top-k filtering.

Usage:
    python predict.py --text "to be or not"
    python predict.py --text "shall i compare" --top_k 10
"""

import argparse

import torch
import torch.nn.functional as F

import config
from model import NextWordLSTM
from dataset import Vocabulary, clean_text, tokenize


class NextWordPredictor:
    """
    Inference wrapper for the trained LSTM model.

    Loads model weights and vocabulary, then generates
    next word predictions with probability scores.
    """

    def __init__(self, model_path=None, vocab_path=None, device=None):
        self.model_path = model_path or config.MODEL_PATH
        self.vocab_path = vocab_path or config.VOCAB_PATH
        self.device = device or torch.device("cpu")

        # Load vocabulary
        self.vocab = Vocabulary.load(self.vocab_path)

        # Load model
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)

        self.model = NextWordLSTM(
            vocab_size=checkpoint["vocab_size"],
            embedding_dim=checkpoint["embedding_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"],
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print(f"[Predictor] Model loaded (trained for {checkpoint['epoch']} epochs, loss: {checkpoint['loss']:.4f})")

    def predict(self, text, top_k=None, temperature=None):
        """
        Predict the next word given input text.

        Args:
            text:        Input text string
            top_k:       Number of top predictions to return
            temperature: Sampling temperature (lower = more conservative)

        Returns:
            List of (word, probability) tuples sorted by probability
        """
        top_k = top_k or config.TOP_K
        temperature = temperature or config.TEMPERATURE

        # Preprocess input
        cleaned = clean_text(text)
        tokens = tokenize(cleaned)

        if len(tokens) == 0:
            return []

        # Take last seq_length tokens (or pad if shorter)
        seq_length = config.SEQUENCE_LENGTH

        if len(tokens) < seq_length:
            # Pad with PAD tokens at the beginning
            padding = [config.SEQUENCE_LENGTH] * (seq_length - len(tokens))
            encoded = padding + self.vocab.encode_sequence(tokens)
        else:
            # Take the last seq_length tokens
            encoded = self.vocab.encode_sequence(tokens[-seq_length:])

        # Convert to tensor
        input_tensor = torch.tensor([encoded], dtype=torch.long).to(self.device)

        # Forward pass (no gradients needed)
        with torch.no_grad():
            output, _ = self.model(input_tensor)

        # Apply temperature scaling
        logits = output[0] / temperature

        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)

        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, top_k)

        predictions = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            word = self.vocab.decode(idx)
            # Skip special tokens
            if word not in ("<PAD>", "<UNK>"):
                predictions.append({"word": word, "probability": round(prob * 100, 2)})

        return predictions

    def predict_and_format(self, text, top_k=None, temperature=None):
        """Predict and return a formatted string."""
        predictions = self.predict(text, top_k, temperature)

        if not predictions:
            return "No predictions available. Try entering more text."

        lines = [f'\nPredictions for: "{text}"\n']
        for i, pred in enumerate(predictions, 1):
            bar = "█" * int(pred["probability"] / 2)
            lines.append(f"  {i}. {pred['word']:<15} {pred['probability']:>6.2f}%  {bar}")

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Next Word Predictor — Inference")
    parser.add_argument("--text", type=str, required=True, help="Input text to predict next word for")
    parser.add_argument("--top_k", type=int, default=config.TOP_K, help="Number of predictions")
    parser.add_argument("--temperature", type=float, default=config.TEMPERATURE, help="Sampling temperature")
    args = parser.parse_args()

    predictor = NextWordPredictor()
    result = predictor.predict_and_format(args.text, args.top_k, args.temperature)
    print(result)


if __name__ == "__main__":
    main()
