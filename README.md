# Next Word Predictor — NLP LSTM AI Project

![AI](https://img.shields.io/badge/AI-NLP-blue?style=for-the-badge) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white) ![ONNX](https://img.shields.io/badge/ONNX-Runtime-gray?style=for-the-badge&logo=onnx&logoColor=white) ![Render](https://img.shields.io/badge/Deployed-Render-46E3B7?style=for-the-badge)

An end-to-end Machine Learning web application that predicts the next likely word in a sentence based on context. Built from scratch using a multi-layer **Long Short-Term Memory (LSTM)** neural network in PyTorch, exported to **ONNX** for lightweight inference, served via a **Flask JSON API**, and wrapped in a premium real-time glassmorphism frontend.

🌐 **Live Demo:** [next-word-predictor-yk51.onrender.com](https://next-word-predictor-yk51.onrender.com)


---

## 🚀 Features

- **Custom LSTM Architecture:** Trained entirely from scratch — no pre-trained black-box models.
- **Context Awareness:** Uses a sliding n-gram context window to understand preceding words.
- **Temperature Scaling:** Interactive UI controls allow tweaking the "creativity" of predictions in real-time.
- **ONNX Inference:** Model exported to ONNX Runtime for deployment — eliminates PyTorch's ~400MB runtime footprint.
- **Full Stack AI:** Complete pipeline: Data Preprocessing → PyTorch Training → ONNX Export → Flask API → Frontend.

## 🧠 Model Architecture

```
[Input Text] → Tokenization → Vocabulary Mapping
     ↓
[Word Indices] → nn.Embedding (128-dim Dense Vectors)
     ↓
nn.LSTM (2 Layers, 512 Hidden Dim) → captures sequential dependencies
     ↓
nn.Dropout (Regularization)
     ↓
nn.Linear → Logits over Vocabulary (2472 words)
     ↓
Softmax + Temperature Scaling → Top-K Probability Output
```

## 🛠️ Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/parthsahay24/Next-Word-Predictor.git
cd Next-Word-Predictor
```

2. **Create a virtual environment and install dependencies**
```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install torch              # Only needed for training/export (not on server)
```

3. **Train the model** *(skip if using the pre-trained checkpoint)*
```bash
python train.py
```
*Trains the LSTM for 15 epochs, saves `checkpoints/next_word_lstm.pth` and `checkpoints/vocab.json`.*

4. **Export to ONNX** *(skip if `checkpoints/next_word_lstm.onnx` already exists)*
```bash
pip install onnx
python export_onnx.py
```

5. **Run the app locally**
```bash
python app.py
```
Open `http://localhost:5001` and start typing to see real-time predictions.

## 📁 Project Structure

```
├── app.py              # Flask web server & API
├── model.py            # LSTM model definition (PyTorch)
├── dataset.py          # Vocabulary + data preprocessing
├── train.py            # Training script
├── predict.py          # Inference using ONNX Runtime
├── export_onnx.py      # Converts .pth → .onnx for deployment
├── config.py           # All hyperparameters and paths
├── requirements.txt    # Server dependencies (no torch)
├── Procfile            # Gunicorn startup command for Render
├── runtime.txt         # Pins Python 3.10 for Render
├── checkpoints/
│   ├── next_word_lstm.pth   # Trained PyTorch weights
│   ├── next_word_lstm.onnx  # ONNX model (used in production)
│   └── vocab.json           # Vocabulary mapping
├── data/
│   └── tech_corpus.txt      # Training corpus
├── static/             # CSS, JS
└── templates/          # HTML templates
```

---

*Built as a portfolio project demonstrating applied Deep Learning and Natural Language Processing capabilities.*
