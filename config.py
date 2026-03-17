"""
Configuration for Next Word Predictor
All hyperparameters and file paths in one place.
"""

import os

#  Paths 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
CORPUS_PATH = os.path.join(DATA_DIR, "tech_corpus.txt")
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "next_word_lstm.pth")
VOCAB_PATH = os.path.join(CHECKPOINT_DIR, "vocab.json")
HISTORY_PATH = os.path.join(CHECKPOINT_DIR, "training_history.json")

# Model Hyperparameters 
SEQUENCE_LENGTH = 10         
EMBEDDING_DIM = 128          
HIDDEN_DIM = 512             
NUM_LAYERS = 2               
DROPOUT = 0.3                

#  Training Hyperparameters 
BATCH_SIZE = 64              
LEARNING_RATE = 0.001        
EPOCHS = 15                 
MIN_WORD_FREQ = 3            

# Prediction 
TOP_K = 5                   
TEMPERATURE = 0.8            

# Flask 
FLASK_HOST = "0.0.0.0"
FLASK_PORT = int(os.environ.get("PORT", 5001))
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
