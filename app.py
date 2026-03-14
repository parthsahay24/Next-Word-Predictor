import os
from flask import Flask, request, jsonify, render_template

import config
from predict import NextWordPredictor

app = Flask(__name__)

# Load predictor globally on startup
print("Initializing ML model...")
try:
    predictor = NextWordPredictor()
    model_loaded = True
except Exception as e:
    print(f"Warning: Model could not be loaded. Please train the model first. Error: {e}")
    predictor = None
    model_loaded = False


@app.route("/")
def index():
    """Render the main UI."""
    return render_template("index.html", model_loaded=model_loaded)


@app.route("/api/predict", methods=["POST"])
def predict_endpoint():
    """API endpoint for next word prediction."""
    if not predictor:
        return jsonify({"error": "Model not loaded. Please run train.py first."}), 503

    data = request.json
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"predictions": []})

    top_k = data.get("top_k", config.TOP_K)
    temperature = data.get("temperature", config.TEMPERATURE)

    try:
        predictions = predictor.predict(text, top_k=top_k, temperature=temperature)
        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG)
