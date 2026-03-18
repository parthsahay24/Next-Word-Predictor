"""
Dataset processing for Next Word Prediction

Handles:
    1. Text loading and cleaning
    2. Vocabulary building with frequency filtering
    3. N-gram sequence generation
    4. PyTorch Dataset for DataLoader
"""

import json
import re
import os
from collections import Counter

# torch is only needed for training (NextWordDataset), not for inference.
# Wrap in try/except so prediction-only servers (no torch installed) still work.
try:
    import torch
    from torch.utils.data import Dataset as _Dataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    _Dataset = object  # fallback base so class definition doesn't fail

import config


# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


class Vocabulary:
    """
    Bidirectional word ↔ index mapping with frequency filtering.

    Builds vocabulary from a text corpus, keeping only words
    that appear at least `min_freq` times.
    """

    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = Counter()
        self.size = 0

        # Reserve indices for special tokens
        self._add_word(PAD_TOKEN)
        self._add_word(UNK_TOKEN)

    def _add_word(self, word):
        """Add a single word to the vocabulary."""
        if word not in self.word2idx:
            self.word2idx[word] = self.size
            self.idx2word[self.size] = word
            self.size += 1

    def build_from_text(self, text):
        """
        Build vocabulary from tokenized text.

        Args:
            text: List of words (already tokenized)
        """
        self.word_count = Counter(text)

        for word, count in self.word_count.most_common():
            if count >= self.min_freq:
                self._add_word(word)

        print(f"[Vocabulary] Total unique words: {len(self.word_count)}")
        print(f"[Vocabulary] Words after filtering (min_freq={self.min_freq}): {self.size}")

    def encode(self, word):
        """Convert word to index. Returns UNK for unknown words."""
        return self.word2idx.get(word, self.word2idx[UNK_TOKEN])

    def decode(self, idx):
        """Convert index to word."""
        return self.idx2word.get(idx, UNK_TOKEN)

    def encode_sequence(self, words):
        """Encode a list of words to indices."""
        return [self.encode(w) for w in words]

    def decode_sequence(self, indices):
        """Decode a list of indices to words."""
        return [self.decode(i) for i in indices]

    def save(self, path):
        """Save vocabulary to JSON file."""
        data = {
            "word2idx": self.word2idx,
            "idx2word": {str(k): v for k, v in self.idx2word.items()},
            "size": self.size,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Vocabulary] Saved to {path}")

    @classmethod
    def load(cls, path):
        """Load vocabulary from JSON file."""
        vocab = cls()
        with open(path, "r") as f:
            data = json.load(f)
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(k): v for k, v in data["idx2word"].items()}
        vocab.size = data["size"]
        print(f"[Vocabulary] Loaded {vocab.size} words from {path}")
        return vocab


def clean_text(text):
    """
    Clean and normalize raw text for training.

    - Converts to lowercase
    - Keeps only letters and spaces (removes punctuation)
    - Collapses multiple spaces
    """
    text = text.lower()
    # Only keep letters and spaces (strips punctuation to ensure pure word tokens)
    text = re.sub(r"[^a-z\s]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    """Simple whitespace tokenizer."""
    return text.split()


def load_corpus(path):
    """Load and preprocess a text corpus."""
    print(f"[Dataset] Loading corpus from {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    cleaned = clean_text(raw_text)
    tokens = tokenize(cleaned)
    print(f"[Dataset] Corpus loaded: {len(tokens)} tokens")
    return tokens


class NextWordDataset(_Dataset):
    """
    PyTorch Dataset for next word prediction.

    Creates (input_sequence, target_word) pairs using a sliding
    window of size `seq_length` over the tokenized text.

    Example (seq_length=4):
        Text: "to be or not to be"
        Sample 1: input=["to","be","or","not"], target="to"
        Sample 2: input=["be","or","not","to"], target="be"
    """

    def __init__(self, tokens, vocab, seq_length):
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required for training. Install it with: pip install torch")
        self.vocab = vocab
        self.seq_length = seq_length
        self.sequences = []
        self.targets = []

        # Encode all tokens
        encoded = vocab.encode_sequence(tokens)

        # Create sliding window sequences
        for i in range(len(encoded) - seq_length):
            seq = encoded[i : i + seq_length]
            target = encoded[i + seq_length]
            self.sequences.append(seq)
            self.targets.append(target)

        print(f"[Dataset] Created {len(self.sequences)} training sequences (window={seq_length})")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.long),
        )


def prepare_data():
    """
    Full data preparation pipeline.

    Returns:
        dataset: NextWordDataset ready for DataLoader
        vocab:   Vocabulary object
    """
    tokens = load_corpus(config.CORPUS_PATH)

    vocab = Vocabulary(min_freq=config.MIN_WORD_FREQ)
    vocab.build_from_text(tokens)

    dataset = NextWordDataset(tokens, vocab, config.SEQUENCE_LENGTH)

    return dataset, vocab
