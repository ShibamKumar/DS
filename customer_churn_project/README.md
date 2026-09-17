# Customer Churn Prediction — Telco Customer Churn

End-to-end data science solution to predict customer churn for a telecommunications
company, so the retention team can proactively engage at-risk customers.

**Workflow**: Business Problem → Data → Preparation → EDA → Feature Engineering →
Model → Evaluation → Interpretation → Saved Model → API

## Project Structure

```
customer_churn_project/
│
├── data/
│   ├── TelcoCustomerChurn.csv                  # raw dataset
│   └── TelcoCustomerChurn - Data Dictionary.csv
├── notebook/
│   └── churn_analysis.ipynb                    # full analysis, EDA, modelling, evaluation
├── model/
│   └── churn_model.pkl                         # saved final scikit-learn Pipeline
├── churn_features.py                           # shared feature-engineering module (used by both notebook and API)
├── app.py                                       # Flask REST API (POST /predict)
├── requirements.txt
├── sample_request.json                          # example API request payload
└── README.md
```

## Model Summary

- **Algorithm**: Decision Tree Classifier (scikit-learn), wrapped in a full
  `Pipeline` (feature engineering → preprocessing → classifier).
- **Train/Test split**: 70:30, stratified on `Churn`, `random_state=42`.
- **Final model** (tuned via `GridSearchCV`): `criterion="entropy"`, `max_depth=4`,
  `min_samples_leaf=30`, `class_weight="balanced"`.
- **Test performance**: Accuracy 0.749, Precision 0.518, Recall 0.758, F1 0.615,
  ROC-AUC 0.828.
- **Engineered features**: `TenureGroup`, `NumAddOnServices`, `AvgMonthlyCharge`
  (see [`churn_features.py`](churn_features.py) and the notebook's Section 3 for
  full rationale).

Full analysis, visualizations, business insights, model comparison, and
interpretation are documented in [`notebook/churn_analysis.ipynb`](notebook/churn_analysis.ipynb).

## Setup

### 1. Create and activate a virtual environment (recommended)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. (Optional) Re-run the notebook

Open [`notebook/churn_analysis.ipynb`](notebook/churn_analysis.ipynb) in VS Code /
Jupyter, select the `.venv` kernel, and run all cells. This will regenerate
`model/churn_model.pkl` from scratch. A pre-trained model is already included,
so this step is optional if you just want to run the API.

## Running the API

From the `customer_churn_project` directory:

```powershell
python app.py
```

The server starts on `http://127.0.0.1:5000` by default. You should see a
confirmation that the model loaded successfully by visiting the root URL:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/"
```

### Endpoint: `POST /predict`

Accepts a single customer's raw information as JSON (see
[`sample_request.json`](sample_request.json)) and returns the churn prediction
and probability. All preprocessing (missing-value handling, feature
engineering, encoding) is applied automatically inside the saved pipeline —
just send the raw fields.

**Required JSON fields** (all 19 raw columns from the original dataset,
excluding `customerID` and `Churn`):

`gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract,
PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges`

#### Sample request

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" `
  -Method POST -InFile "sample_request.json" -ContentType "application/json"
```

Or with `curl`:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

Request body ([`sample_request.json`](sample_request.json)):

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 5,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 90.05,
  "TotalCharges": "451.25"
}
```

#### Sample response

```json
{
  "prediction": "Yes",
  "churn_probability": 0.8335
}
```

### Error handling

- **Missing fields** or **invalid values** (e.g. non-numeric `tenure`, an
  unrecognized `gender` category) return `HTTP 400` with a `details` array
  describing every validation issue found:

  ```json
  {
    "error": "Invalid input.",
    "details": [
      "Missing required field(s): SeniorCitizen, Partner, ...",
      "Field 'tenure' must be numeric, got: 'abc'",
      "Field 'gender' has invalid value 'Unknown'. Allowed values: ['Female', 'Male']"
    ]
  }
  ```

- **Malformed/non-JSON body** returns `HTTP 400`.
- **Model not loaded / internal errors** return `HTTP 500`.

## Notes on Data Leakage Prevention

All preprocessing (imputation, one-hot encoding) and feature engineering
(`ChurnFeatureEngineer` in [`churn_features.py`](churn_features.py)) are
implemented as scikit-learn-compatible transformers inside a single `Pipeline`,
fit **only** on the training split. The identical fitted transformations are
then applied to the test set and to any new data sent to the API — ensuring
consistent, leakage-free behaviour between training and production.
