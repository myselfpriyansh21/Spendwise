"""
SpendWise Backend
------------------
Stateless Flask API providing:
  1. POST /api/analyze  -> ML-based (or heuristic) anomaly detection on transactions
  2. POST /api/chat     -> AI budgeting coach proxy via OpenRouter
  3. GET  /             -> health check

No database. All persistent state lives in the browser via localStorage.
The frontend is expected to run separately (e.g. opened directly in a browser
or served by any static file server) and talk to this API.
"""

import os
import statistics
from collections import defaultdict

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct:free").strip()
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "service": "SpendWise API",
        "ai_configured": bool(OPENROUTER_API_KEY),
    })


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------
def fallback_anomaly_detection(transactions):
    """Simple heuristic used when there are too few transactions for ML:
    flag an expense if it's more than 2.5x its category's average."""
    by_category = defaultdict(list)
    for t in transactions:
        by_category[t.get("category", "Other")].append(t)

    flagged = []
    for category, txs in by_category.items():
        amounts = [float(t.get("amount", 0)) for t in txs]
        if not amounts:
            continue
        avg = sum(amounts) / len(amounts)
        if avg <= 0:
            continue
        for t in txs:
            amount = float(t.get("amount", 0))
            if amount > 2.5 * avg:
                flagged.append({
                    "id": t.get("id"),
                    "reason": (
                        f"This expense is unusual for your typical {category} "
                        f"spending — it's noticeably higher than what you usually spend "
                        f"in this category."
                    ),
                })
    return flagged


def isolation_forest_anomaly_detection(transactions):
    """Use IsolationForest on transaction amounts when we have enough data."""
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import IsolationForest

    df = pd.DataFrame(transactions)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)

    model = IsolationForest(contamination=0.1, random_state=42)
    amounts = df[["amount"]].values
    predictions = model.fit_predict(amounts)  # -1 = anomaly, 1 = normal
    scores = model.decision_function(amounts)

    flagged = []
    for idx, pred in enumerate(predictions):
        if pred == -1:
            row = df.iloc[idx]
            category = row.get("category", "Other")
            flagged.append({
                "id": row.get("id"),
                "reason": (
                    f"This expense is unusual for your typical spending pattern in "
                    f"{category} — it stands out from your usual transaction sizes. "
                    f"No judgment, just something worth a second look!"
                ),
                "score": round(float(scores[idx]), 4),
            })
    return flagged


@app.route("/api/analyze", methods=["POST"])
def analyze_transactions():
    try:
        data = request.get_json(force=True, silent=True) or {}
        transactions = data.get("transactions", [])

        if not isinstance(transactions, list) or len(transactions) == 0:
            return jsonify({"flagged": [], "method": "none", "message": "No transactions to analyze."})

        # Normalize amounts, drop rows with unusable data
        clean_transactions = []
        for t in transactions:
            try:
                amt = float(t.get("amount", 0))
            except (TypeError, ValueError):
                continue
            clean_transactions.append({**t, "amount": amt})

        if len(clean_transactions) < 6:
            flagged = fallback_anomaly_detection(clean_transactions)
            method = "heuristic"
        else:
            try:
                flagged = isolation_forest_anomaly_detection(clean_transactions)
                method = "isolation_forest"
            except Exception:
                # If sklearn/pandas hiccup for any reason, fall back gracefully
                flagged = fallback_anomaly_detection(clean_transactions)
                method = "heuristic_fallback"

        return jsonify({"flagged": flagged, "method": method})

    except Exception as e:
        return jsonify({"flagged": [], "method": "error", "error": str(e)}), 200


# ---------------------------------------------------------------------------
# AI Coach Chat
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are the SpendWise Coach, an encouraging and practical budgeting mentor "
    "for college students. You are warm, concise, and never judgmental about "
    "spending habits. Base your advice ONLY on the financial summary provided "
    "to you in the user's message — do not invent numbers. Give specific, "
    "actionable, student-friendly tips (e.g. meal prepping, student discounts, "
    "subscription audits). Keep responses short: 3-5 sentences, or a short "
    "bullet list when listing tips."
)


@app.route("/api/chat", methods=["POST"])
def chat_with_coach():
    if not OPENROUTER_API_KEY:
        return jsonify({
            "reply": (
                "The AI Coach isn't fully set up yet — an OpenRouter API key hasn't "
                "been configured on the server. Add OPENROUTER_API_KEY to your .env "
                "file to unlock personalized coaching. In the meantime, keep an eye "
                "on your biggest spending category and try setting a small weekly cap!"
            ),
            "ai_configured": False,
        })

    try:
        data = request.get_json(force=True, silent=True) or {}
        question = data.get("question", "").strip()
        summary = data.get("summary", {})

        if not question:
            return jsonify({"reply": "Ask me anything about your spending — I'm happy to help!"})

        user_content = (
            f"Here is my current financial summary: {summary}\n\n"
            f"My question: {question}"
        )

        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "max_tokens": 400,
            },
            timeout=20,
        )

        if response.status_code != 200:
            return jsonify({
                "reply": (
                    "The AI Coach had trouble reaching the model provider just now. "
                    "Please try again in a moment."
                ),
                "ai_configured": True,
                "error": response.text[:300],
            }), 200

        payload = response.json()
        reply = (
            payload.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        if not reply:
            reply = "I couldn't generate a response just now — please try asking again."

        return jsonify({"reply": reply, "ai_configured": True})

    except requests.exceptions.RequestException as e:
        return jsonify({
            "reply": "The AI Coach couldn't connect right now. Please check your connection and try again.",
            "ai_configured": True,
            "error": str(e),
        }), 200
    except Exception as e:
        return jsonify({
            "reply": "Something went wrong on the coach's end. Please try again.",
            "ai_configured": True,
            "error": str(e),
        }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)