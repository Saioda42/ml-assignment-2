# Adult Income Prediction - ML Pipeline

A complete machine learning pipeline for predicting whether an adult earns >$50K/year. Compares two models (MLP with embeddings and XGBoost) deployed via FastAPI with an interactive web frontend.

## Project Overview

**Dataset:** Adult Income (UCI Machine Learning Repository)
- **Samples:** 48,842
- **Task:** Binary classification (income >50K or ≤50K)
- **Features:** 14 (mix of categorical and numerical)

**Models:**
1. **MLP (PyTorch)** - Neural network with embedding layers for categorical features
2. **XGBoost** - Gradient boosting tree model with early stopping

## Results Comparison

| Model | Accuracy | F1 Score | AUC-ROC |
|-------|----------|----------|---------|
| MLP | 0.81 | 0.71 | 0.86 |
| XGBoost | 0.85 | 0.77 | 0.91 |

**Key Finding:** XGBoost outperforms the MLP on this dataset, particularly in capturing the relationship between features and high-income predictions.

## Installation & Setup

### Prerequisites
- Python 3.9+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install fastapi uvicorn torch scikit-learn xgboost pandas numpy joblib
```

## Running the Application

### Start the API Server

```bash
uvicorn app.main:app --reload
```

The server will start at `http://127.0.0.1:8000`

### Access the Frontend

Open your browser and go to:
```
http://127.0.0.1:8000
```

You'll see an interactive form to make predictions with both models simultaneously.

## API Endpoints

### GET `/`
Serves the interactive web frontend.

### GET `/valid-countries`
Returns list of valid countries from training data (for form validation).

**Response:**
```json
{
  "countries": ["United-States", "China", "India", ...]
}
```

### GET `/education-levels`
Returns education level mappings (1-15).

**Response:**
```json
{
  "education_levels": {
    "1": "Preschool",
    "2": "1st-4th",
    ...
    "13": "Masters",
    ...
  }
}
```

### POST `/predict/xgboost`
Make a prediction using the XGBoost model.

**Request:**
```json
{
  "age": 35,
  "workclass": "Private",
  "fnlwgt": 100000,
  "education_num": 13,
  "marital_status": "Married-civ-spouse",
  "occupation": "Exec-managerial",
  "relationship": "Husband",
  "race": "White",
  "sex": "Male",
  "capital_gain": 5000,
  "capital_loss": 0,
  "hours_per_week": 40,
  "native_country": "United-States"
}
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.81
}
```
(1 = >50K, 0 = ≤50K)

### POST `/predict/mlp`
Make a prediction using the MLP model (same request/response format as XGBoost).

## Input Features

All input fields are required:

| Field | Type | Valid Range/Options |
|-------|------|-------------------|
| age | integer | 0-100 |
| workclass | string | Private, Self-emp-not-inc, Federal-gov, etc. |
| fnlwgt | integer | > 0 |
| education_num | integer | 1-15 (see education-levels endpoint) |
| marital_status | string | Married-civ-spouse, Never-married, Divorced, etc. |
| occupation | string | Exec-managerial, Prof-specialty, Tech-support, etc. |
| relationship | string | Husband, Wife, Not-in-family, Own-child, etc. |
| race | string | White, Black, Asian-Pac-Islander, etc. |
| sex | string | Male, Female |
| capital_gain | integer | ≥ 0 |
| capital_loss | integer | ≥ 0 |
| hours_per_week | integer | 0-100+ |
| native_country | string | Valid country from /valid-countries endpoint |

## Project Structure

```
assignment-2/
├── app/
│   ├── main.py              # FastAPI application
│   ├── schemas.py           # Pydantic schemas
│   ├── preprocessing.py     # Data preprocessing & model loading
│   ├── models/
│   │   ├── xgb_model.pkl    # XGBoost model
│   │   ├── mlp_model.pt     # PyTorch MLP weights
│   │   ├── scaler.pkl       # StandardScaler for numerical features
│   │   └── label_encoders.pkl # LabelEncoders for categorical features
│   └── static/
│       └── index.html       # Interactive web frontend
├── notebooks/
│   └── pipeline.ipynb       # Full ML pipeline notebook
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Data Preprocessing

1. **Missing Values:** Handled during EDA
2. **Categorical Encoding:** LabelEncoder for all categorical features
3. **Numerical Scaling:** StandardScaler (fit on training set only)
4. **Train/Val Split:** 80/20 split with stratification on target
5. **Class Imbalance:** Addressed using `pos_weight` in XGBoost

## Model Architecture

### MLP (PyTorch)
- **Embedding Layers:** Categorical features → learned embeddings
- **Concatenation:** Embeddings + scaled numerical features
- **Hidden Layers:** [128, 64] with ReLU activation
- **Dropout:** Applied for regularization
- **Output:** Sigmoid for binary classification

### XGBoost
- **Base Learner:** Trees (max_depth=6)
- **Early Stopping:** Monitored on validation set
- **Learning Rate:** 0.1
- **n_estimators:** 200 (stopped early)
- **pos_weight:** Tuned for class imbalance

## Frontend Features

- **Interactive Form:** Fill in all features and see predictions instantly
- **Dual Model Comparison:** Both XGBoost and MLP predictions shown side-by-side
- **Probability Visualization:** Bar chart comparing model confidence
- **Dynamic Dropdowns:** Country and education level lists loaded from API
- **Responsive Design:** Works on desktop and mobile

## Example Usage

### Via Browser Frontend
1. Go to http://127.0.0.1:8000
2. Fill in the form with sample data
3. Click "Predicera (Båda modeller)"
4. See predictions and comparison chart

### Via Python Requests

```python
import requests
import json

data = {
    "age": 35,
    "workclass": "Private",
    "fnlwgt": 100000,
    "education_num": 13,
    "marital_status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 5000,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
}

# Get XGBoost prediction
xgb_response = requests.post("http://127.0.0.1:8000/predict/xgboost", json=data)
print("XGBoost:", xgb_response.json())

# Get MLP prediction
mlp_response = requests.post("http://127.0.0.1:8000/predict/mlp", json=data)
print("MLP:", mlp_response.json())
```

### Via cURL

```bash
curl -X POST "http://127.0.0.1:8000/predict/xgboost" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "workclass": "Private",
    "fnlwgt": 100000,
    "education_num": 13,
    "marital_status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 5000,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
  }'
```

## Key Learnings

1. **XGBoost > MLP** for this tabular data - trees capture non-linear patterns better
2. **Feature Importance** - Age, education level, and hours per week were top predictors
3. **Class Imbalance** - Properly handling the imbalanced target distribution was crucial
4. **Embeddings** - Neural networks benefit from learned categorical embeddings, but tree models don't need them
5. **Deployment** - Same preprocessing pipeline must be used for all predictions

## Troubleshooting

### "ValueError: y contains previously unseen labels"
- Ensure the input country/occupation/workclass is from the valid list
- Use `/valid-countries` endpoint to get current valid options

### "Feature names mismatch" warnings
- These are safe to ignore - they occur due to StandardScaler fitting details
- Predictions are still correct

### Models not loading
- Check that model files exist in `app/models/`
- Ensure XGBoost and PyTorch are installed correctly

## References

- **Dataset:** [UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- **PyTorch:** https://pytorch.org/
- **XGBoost:** https://xgboost.readthedocs.io/
- **FastAPI:** https://fastapi.tiangolo.com/

## Author

Assignment 2: Tabular ML Pipeline
Part of "Practical Machine Learning for Programmers"
