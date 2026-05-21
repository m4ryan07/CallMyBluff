"""
app.py — CallMyBluff Flask Backend
"""

import os, sys, joblib, pandas as pd
from flask import Flask, render_template, request, jsonify

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.features import StylometricTransformer, FEATURE_FUNCTIONS

app = Flask(__name__)

# ── Load artifacts once at startup ────────────────────────────────────────────
model      = joblib.load("models/best_model.joblib")
scaler     = joblib.load("models/scaler.joblib")
meta       = joblib.load("models/model_meta.joblib")
benchmarks = joblib.load("models/benchmarks.joblib")


def predict(text: str):
    X = StylometricTransformer().fit_transform(pd.Series([text]))
    X = scaler.transform(X)
    label = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0].tolist()
    features = {name: round(fn(text), 4) for name, fn in FEATURE_FUNCTIONS.items()}
    bench = {
        feat: {
            "human": round(vals.get(0, 0), 4),
            "ai":    round(vals.get(1, 0), 4),
        }
        for feat, vals in benchmarks.items()
    }
    return {
        "label":      label,
        "human_prob": round(proba[0] * 100, 1),
        "ai_prob":    round(proba[1] * 100, 1),
        "features":   features,
        "benchmarks": bench,
        "word_count": len(text.split()),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text or len(text.split()) < 30:
        return jsonify({"error": "Please paste at least 30 words."}), 400
    return jsonify(predict(text))


if __name__ == "__main__":
    app.run(debug=True, port=5050)