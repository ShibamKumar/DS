"""
Shared feature-engineering utilities for the Telco Customer Churn project.

This module is imported by BOTH the training notebook (notebook/churn_analysis.ipynb)
and the Flask API (app.py) so that *exactly* the same preprocessing / feature
engineering logic is applied to the training data and to any new/unseen data
sent to the API. Keeping this logic in a standalone, importable module (instead
of defining it inline inside the notebook) is what makes it possible to:

  1. Pickle the fitted scikit-learn Pipeline (custom transformer classes must
     live in an importable module to be unpickled later).
  2. Guarantee there is a single source of truth for preprocessing, so the
     model can be applied consistently to brand-new customer records without
     risk of train/serve skew or data leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Columns that represent optional / value-added services a customer can subscribe to.
ADDON_SERVICE_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# The raw input columns the model expects (everything from the original dataset
# except the identifier `customerID` and the target `Churn`).
RAW_FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

# Feature groups fed into the ColumnTransformer (after feature engineering runs).
NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "NumAddOnServices",
    "AvgMonthlyCharge",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "TenureGroup",
]


def _tenure_group(tenure: pd.Series) -> pd.Series:
    """Bucket `tenure` (months) into customer lifecycle stages."""
    bins = [-1, 12, 24, 48, 60, np.inf]
    labels = ["0-12", "13-24", "25-48", "49-60", "61-72+"]
    return pd.cut(tenure, bins=bins, labels=labels)


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds engineered features used by the churn model.

    Engineered features
    --------------------
    1. TenureGroup (categorical):
       Buckets the continuous `tenure` variable into 5 lifecycle stages
       (0-12, 13-24, 25-48, 49-60, 61-72+ months). Churn risk is highly
       non-linear with tenure (very high in the first year, then drops
       sharply), so an explicit bucket lets a shallow Decision Tree capture
       that pattern with a single split instead of several numeric splits.

    2. NumAddOnServices (numeric, 0-6):
       Count of value-added services the customer has subscribed to
       (OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
       StreamingTV, StreamingMovies). This acts as a proxy for customer
       "stickiness" / engagement - customers who have bundled in more
       services tend to be more embedded in the ecosystem and less likely
       to churn.

    3. AvgMonthlyCharge (numeric):
       TotalCharges / tenure (fallback to MonthlyCharges for brand-new
       customers with tenure == 0). Comparing this historical average to
       the current MonthlyCharges highlights customers whose bill has
       recently increased relative to what they've paid on average - a
       common churn trigger.

    Implemented as a scikit-learn compatible transformer so it is the first
    step of the training Pipeline and is therefore applied automatically
    (and identically) to any new data passed to the API at inference time.
    """

    def fit(self, X: pd.DataFrame, y=None):  # noqa: D401, ARG002
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Coerce fields that may arrive as strings (e.g. from JSON/CSV) to numeric.
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
        X["MonthlyCharges"] = pd.to_numeric(X["MonthlyCharges"], errors="coerce")
        X["tenure"] = pd.to_numeric(X["tenure"], errors="coerce")
        X["SeniorCitizen"] = pd.to_numeric(X["SeniorCitizen"], errors="coerce")

        # Brand-new customers (tenure == 0) have no historical TotalCharges yet.
        X["TotalCharges"] = X["TotalCharges"].fillna(0)

        # --- Feature 1: TenureGroup ---
        X["TenureGroup"] = _tenure_group(X["tenure"]).astype(str)

        # --- Feature 2: NumAddOnServices ---
        addon_flags = X[ADDON_SERVICE_COLS].apply(lambda col: (col == "Yes").astype(int))
        X["NumAddOnServices"] = addon_flags.sum(axis=1)

        # --- Feature 3: AvgMonthlyCharge ---
        safe_tenure = X["tenure"].replace(0, np.nan)
        avg_charge = X["TotalCharges"] / safe_tenure
        X["AvgMonthlyCharge"] = avg_charge.fillna(X["MonthlyCharges"])

        return X
