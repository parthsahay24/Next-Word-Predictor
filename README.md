# Next Word Predictor - NLP LSTM AI Project

![Banner](https://img.shields.io/badge/AI-NLP-blue?style=for-the-badge) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)

An end-to-end Machine Learning web application that predicts the next likely word in a sentence based on context. Built from scratch using a multi-layer **Long Short-Term Memory (LSTM)** neural network in PyTorch, served via a **Flask JSON API**, and wrapped in a premium, real-time glassmorphism frontend.

---

## 🚀 Features
*   **Custom LSTM Architecture:** Trained entirely from scratch rather than relying on pre-trained black-box models. 
*   **Context Awareness:** Uses a sliding n-gram context window to understand preceding words.
*   **Temperature Scaling:** Interactive UI controls allow tweaking the "creativity" of the model in real-time.
*   **Full Stack AI:** Implements the complete pipeline: Data Preprocessing → PyTorch Training → Checkpoint Saving → Flask Inference API → Frontend Consumption.

## 🧠 Model Architecture

```
[Input Text] → Tokenization → Vocabulary Mapping 
     ↓
[Word Indices] → nn.Embedding (Dense Vectors)
     ↓
nn.LSTM (2 Layers, 256 Hidden Dim) → captures sequential dependencies
     ↓
nn.Dropout (Regularization)
     ↓
nn.Linear → Logits over Vocabulary
     ↓
Softmax + Temperature Scaling → Top-K Probability Output
```

## 🛠️ Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/next-word-predictor.git
cd next-word-predictor
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train the Model (Required for first run)**
The project comes with a sample dataset (William Shakespeare corpus) located in `data/shakespeare.txt`.
```bash
python train.py
```
*This will train the LSTM, output loss/perplexity metrics per epoch, and save `next_word_lstm.pth` in the `checkpoints/` folder.*

4. **Launch the Web Application**
```bash
python app.py
```

5. **Interact**
Open your browser and navigate to `http://localhost:5000`. Start typing to see real-time AI predictions!

---
*Built as a portfolio project demonstrating applied Deep Learning and Natural Language Processing capabilities.*
