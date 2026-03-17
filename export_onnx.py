"""
Export trained LSTM model to ONNX format for lightweight inference.

Run this ONCE locally before deploying:
    python export_onnx.py

Produces: checkpoints/next_word_lstm.onnx
This replaces PyTorch (~400MB RAM) with onnxruntime (~50MB RAM) at deploy time.
"""

import torch
import config
from model import NextWordLSTM

# ── Load checkpoint ───────────────────────────────────────────────────────────
checkpoint = torch.load(config.MODEL_PATH, map_location="cpu", weights_only=True)

model = NextWordLSTM(
    vocab_size=checkpoint["vocab_size"],
    embedding_dim=checkpoint["embedding_dim"],
    hidden_dim=checkpoint["hidden_dim"],
    num_layers=checkpoint["num_layers"],
    dropout=checkpoint["dropout"],
).cpu()

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"Loaded model: vocab={checkpoint['vocab_size']}, "
      f"embed={checkpoint['embedding_dim']}, hidden={checkpoint['hidden_dim']}, "
      f"layers={checkpoint['num_layers']}, epochs={checkpoint['epoch']}, "
      f"loss={checkpoint['loss']:.4f}")

# ── Create dummy input ────────────────────────────────────────────────────────
seq_len = config.SEQUENCE_LENGTH
dummy_input = torch.zeros((1, seq_len), dtype=torch.long)

# ── Export to ONNX ────────────────────────────────────────────────────────────
onnx_path = config.MODEL_PATH.replace(".pth", ".onnx")

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "logits": {0: "batch_size"},
    },
)

print(f"\n✅ Exported to: {onnx_path}")

# ── Quick sanity check ────────────────────────────────────────────────────────
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
dummy_np = np.zeros((1, seq_len), dtype=np.int64)
logits = sess.run(["logits"], {"input": dummy_np})[0]
print(f"✅ ONNX sanity check passed — output shape: {logits.shape}")
