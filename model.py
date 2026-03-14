"""
LSTM Language Model for Next Word Prediction

Architecture:
    Input → Embedding → LSTM (2 layers) → Dropout → Linear → Softmax

The model learns contextual word relationships by predicting
the next word given a sequence of preceding words.
"""

import torch
import torch.nn as nn


class NextWordLSTM(nn.Module):
    """
    A multi-layer LSTM language model for next word prediction.

    Args:
        vocab_size:     Size of the vocabulary
        embedding_dim:  Dimension of word embeddings
        hidden_dim:     LSTM hidden state dimension
        num_layers:     Number of stacked LSTM layers
        dropout:        Dropout probability for regularization
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(NextWordLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding layer: maps word indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM: processes sequences and captures long-range dependencies
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Dropout for regularization before the output layer
        self.dropout = nn.Dropout(dropout)

        # Output layer: maps hidden state to vocabulary probabilities
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # Initialize weights for better convergence
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for stable training."""
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-init_range, init_range)

    def forward(self, x, hidden=None):
        """
        Forward pass through the model.

        Args:
            x:      Input tensor of word indices [batch_size, seq_len]
            hidden: Optional tuple of (h_0, c_0) for LSTM

        Returns:
            output: Logits over vocabulary [batch_size, vocab_size]
            hidden: Updated LSTM hidden state
        """
        batch_size = x.size(0)

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)

        # Embed input words: [batch, seq_len] → [batch, seq_len, embed_dim]
        embeds = self.embedding(x)

        # Pass through LSTM: captures sequential patterns
        lstm_out, hidden = self.lstm(embeds, hidden)

        # Use only the last time step's output for prediction
        lstm_out = lstm_out[:, -1, :]  # [batch, hidden_dim]

        # Apply dropout and project to vocabulary space
        out = self.dropout(lstm_out)
        out = self.fc(out)  # [batch, vocab_size]

        return out, hidden

    def init_hidden(self, batch_size, device):
        """Initialize hidden state and cell state with zeros."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)
