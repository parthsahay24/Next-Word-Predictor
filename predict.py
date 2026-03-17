"""
Prediction Module for Next Word Predictor
Loads an ONNX model and generates next word predictions
with configurable temperature and top-k filtering.
Usage:
    python predict.py --text "to be or not"
    python predict.py --text "shall i compare" --top_k 10
"""
import argparse
import numpy as np
import onnxruntime as ort
import config
from dataset import Vocabulary, clean_text, tokenize
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()
class NextWordPredictor:
    """
    Inference wrapper using ONNX Runtime (lightweight, ~50MB vs PyTorch's ~400MB).
    Loads ONNX model weights and vocabulary, then generates
    next word predictions with probability scores.
    """
    def __init__(self, model_path=None, vocab_path=None):
        onnx_path = model_path or config.MODEL_PATH.replace(".pth", ".onnx")
        self.vocab_path = vocab_path or config.VOCAB_PATH
        # Load vocabulary
        self.vocab = Vocabulary.load(self.vocab_path)
        print(f"[Vocabulary] Loaded {len(self.vocab.word2idx)} words from {self.vocab_path}")
        # Load ONNX model (CPU only — no PyTorch needed at runtime)
        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"]
        )
        print(f"[Predictor] ONNX model loaded from {onnx_path}")
    def predict(self, text, top_k=None, temperature=None):
        """
        Predict the next word given input text.
        Args:
            text:        Input text string
            top_k:       Number of top predictions to return
            temperature: Sampling temperature (lower = more conservative)
        Returns:
            List of dicts with 'word' and 'probability' keys
        """
        top_k = top_k or config.TOP_K
        temperature = temperature or config.TEMPERATURE
        # Preprocess input
        cleaned = clean_text(text)
        tokens = tokenize(cleaned)
        if len(tokens) == 0:
            return []
        seq_length = config.SEQUENCE_LENGTH
        if len(tokens) < seq_length:
            padding = [config.SEQUENCE_LENGTH] * (seq_length - len(tokens))
            encoded = padding + self.vocab.encode_sequence(tokens)
        else:
            encoded = self.vocab.encode_sequence(tokens[-seq_length:])
        # Run ONNX inference
        input_array = np.array([encoded], dtype=np.int64)
        logits = self.session.run(["logits"], {"input": input_array})[0][0]
        # Apply temperature scaling and softmax
        probs = softmax(logits / temperature)
        # Get top-k predictions
        top_indices = np.argsort(probs)[::-1][:top_k]
        predictions = []
        for idx in top_indices:
            word = self.vocab.decode(int(idx))
            if word not in ("<PAD>", "<UNK>"):
                predictions.append({
                    "word": word,
                    "probability": round(float(probs[idx]) * 100, 2)
                })
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
