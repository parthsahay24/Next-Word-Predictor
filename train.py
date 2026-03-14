"""
Training Script for Next Word Predictor

Trains the LSTM language model on the text corpus and saves:
    - Model checkpoint (.pth)
    - Vocabulary mapping (vocab.json)
    - Training history (training_history.json)

Usage:
    python train.py
"""

import os
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from model import NextWordLSTM
from dataset import prepare_data


def train():
    """Main training loop."""


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Next Word Predictor — Training")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")


    dataset, vocab = prepare_data()

    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    print(f"\n[Training] Batches per epoch: {len(dataloader)}")
    print(f"[Training] Vocabulary size: {vocab.size}")


    model = NextWordLSTM(
        vocab_size=vocab.size,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total parameters: {total_params:,}")
    print(f"[Model] Trainable parameters: {trainable_params:,}")


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )


    history: dict[str, list] = {"epoch": [], "loss": [], "perplexity": [], "lr": [], "time": []}
    best_loss = float("inf")

    print(f"\n{'─'*60}")
    print(f"  Starting training for {config.EPOCHS} epochs")
    print(f"{'─'*60}\n")

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, (sequences, targets) in enumerate(dataloader):
            sequences = sequences.to(device)
            targets = targets.to(device)

            # Forward pass
            output, _ = model(sequences)
            loss = criterion(output, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()


            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        elapsed = time.time() - epoch_start
        current_lr = float(optimizer.param_groups[0]["lr"])

        scheduler.step(avg_loss)

        history["epoch"].append(epoch)
        history["loss"].append(round(avg_loss, 4))
        history["perplexity"].append(round(perplexity, 2))
        history["lr"].append(current_lr)
        history["time"].append(round(elapsed, 2))


        print(
            f"  Epoch [{epoch:>2}/{config.EPOCHS}]  "
            f"Loss: {avg_loss:.4f}  "
            f"Perplexity: {perplexity:.2f}  "
            f"LR: {current_lr:.6f}  "
            f"Time: {elapsed:.1f}s"
        )


        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_size": vocab.size,
                    "embedding_dim": config.EMBEDDING_DIM,
                    "hidden_dim": config.HIDDEN_DIM,
                    "num_layers": config.NUM_LAYERS,
                    "dropout": config.DROPOUT,
                    "epoch": epoch,
                    "loss": best_loss,
                },
                config.MODEL_PATH,
            )
            print(f"           ✓ Best model saved (loss: {best_loss:.4f})")


    vocab.save(config.VOCAB_PATH)

    with open(config.HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[Training] History saved to {config.HISTORY_PATH}")


    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Best Loss: {best_loss:.4f}")
    print(f"  Best Perplexity: {torch.exp(torch.tensor(best_loss)).item():.2f}")
    print(f"  Model saved to: {config.MODEL_PATH}")
    print(f"  Vocabulary saved to: {config.VOCAB_PATH}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    train()
