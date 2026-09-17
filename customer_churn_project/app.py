"""
Flask REST API for the Telco Customer Churn prediction model.

Endpoint
--------
POST /predict
    Accepts a single customer's raw information as JSON, applies the exact
    same preprocessing/feature-engineering used during training (via the
    saved scikit-learn Pipeline), and returns the churn prediction and
    probability.

    Example response:
        {
            "prediction": "Yes",
            "churn_probability": 0.82
        }

Run locally:
    python app.py
    # or, for production-style serving:
    waitress-serve --port=5000 app:app   (Windows)
    gunicorn -w 2 -b 0.0.0.0:5000 app:app  (Linux/Mac)
"""
from __future__ import annotations

import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from churn_features import RAW_FEATURE_COLUMNS  # single source of truth for expected fields

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "churn_model.pkl")

app = Flask(__name__)

# Load the trained pipeline (feature engineering + preprocessing + classifier) once at startup.
try:
    model = joblib.load(MODEL_PATH)
    MODEL_LOAD_ERROR = None
except Exception as exc:  # noqa: BLE001
    model = None
    MODEL_LOAD_ERROR = str(exc)

# Fields that MUST be present in every request (all raw model inputs are required;
# missing optional-service fields like "MultipleLines" should still be sent explicitly,
# e.g. "No phone service", to mirror the training data schema).
REQUIRED_FIELDS = RAW_FEATURE_COLUMNS

# Categorical fields and the values the model was trained on (used for input validation).
VALID_CATEGORIES = {
    "gender": {"Male", "Female"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    },
}

NUMERIC_INPUT_FIELDS = {"SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"}


def validate_payload(payload: dict) -> list[str]:
    """Validate a raw customer JSON payload. Returns a list of error messages (empty if valid)."""
    errors: list[str] = []

    if not isinstance(payload, dict):
        return ["Request body must be a JSON object."]

    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        errors.append(f"Missing required field(s): {', '.join(missing)}")

    for field in NUMERIC_INPUT_FIELDS:
        if field in payload and payload[field] is not None:
            try:
                float(payload[field])
            except (TypeError, ValueError):
                errors.append(f"Field '{field}' must be numeric, got: {payload[field]!r}")

    for field, allowed_values in VALID_CATEGORIES.items():
        if field in payload and payload[field] is not None:
            if payload[field] not in allowed_values:
                errors.append(
                    f"Field '{field}' has invalid value {payload[field]!r}. "
                    f"Allowed values: {sorted(allowed_values)}"
                )

    if "SeniorCitizen" in payload and payload["SeniorCitizen"] is not None:
        try:
            if int(payload["SeniorCitizen"]) not in (0, 1):
                errors.append("Field 'SeniorCitizen' must be 0 or 1.")
        except (TypeError, ValueError):
            pass  # already reported as non-numeric above

    return errors


@app.route("/", methods=["GET"])
def health_check():
    """Simple health-check / index endpoint."""
    return jsonify({
        "status": "ok" if model is not None else "error",
        "message": "Telco Customer Churn Prediction API",
        "model_loaded": model is not None,
        "model_load_error": MODEL_LOAD_ERROR,
        "endpoints": {"predict": "POST /predict"},
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Predict churn for a single customer record supplied as JSON."""
    if model is None:
        return jsonify({
            "error": "Model is not loaded on the server.",
            "details": MODEL_LOAD_ERROR,
        }), 500

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({
            "error": "Invalid or missing JSON body. Send a JSON object with customer fields."
        }), 400

    validation_errors = validate_payload(payload)
    if validation_errors:
        return jsonify({"error": "Invalid input.", "details": validation_errors}), 400

    try:
        # Build a single-row DataFrame with the exact raw columns the pipeline expects.
        input_df = pd.DataFrame([{col: payload.get(col) for col in RAW_FEATURE_COLUMNS}])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]  # probability of class "1" (Churn = Yes)

        response = {
            "prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": round(float(probability), 4),
        }
        return jsonify(response), 200

    except Exception as exc:  # noqa: BLE001
        return jsonify({
            "error": "Failed to generate prediction.",
            "details": str(exc),
        }), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
