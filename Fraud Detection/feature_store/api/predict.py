import json
import sys
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from ab_testing import choose_model_for_customer
except ImportError:
    from feature_store.ab_testing import choose_model_for_customer

MODEL_PATHS = {
    "model_a": BASE_DIR / "model_a.pkl",
    "model_b": BASE_DIR / "model_b.pkl",
    "fraud_model": BASE_DIR / "fraud_model.pkl",
}
FEATURE_PATH = BASE_DIR / "streaming" / "streaming_features.json"


def load_model_for_route(customer_id, force_model=None):
    model_name = choose_model_for_customer(customer_id, force_model=force_model)
    candidate = MODEL_PATHS.get(model_name)

    if candidate and candidate.exists():
        return model_name, joblib.load(candidate)

    legacy = MODEL_PATHS["fraud_model"]
    if legacy.exists():
        return "fraud_model", joblib.load(legacy)

    raise FileNotFoundError(
        "No trained model found. Train the models first with train_model.py or save fraud_model.pkl."
    )


def predict_fraud(customer_id, force_model=None):
    customer_id = str(customer_id)
    with open(FEATURE_PATH, "r") as f:
        features = json.load(f)

    if customer_id not in features:
        return {"error": f"Customer {customer_id} not found"}

    customer = features[customer_id]
    input_df = pd.DataFrame({
        "transaction_amount": [customer["average_amount"]],
        "account_balance": [customer["total_amount"]],
        "is_high_risk": [1 if customer["average_amount"] > 2000 else 0],
    })

    model_name, model = load_model_for_route(customer_id, force_model=force_model)
    prediction = int(model.predict(input_df)[0])

    return {
        "customer_id": customer_id,
        "model_used": model_name,
        "transaction_count": customer["transaction_count"],
        "average_amount": customer["average_amount"],
        "total_amount": customer["total_amount"],
        "prediction": "High Risk" if prediction == 1 else "Low Risk",
    }
