import json
import os
import joblib
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# fraud_model.pkl is inside feature_store
MODEL_PATH = os.path.join(BASE_DIR, "..", "fraud_model.pkl")

# streaming_features.json is inside streaming
FEATURE_PATH = os.path.join( BASE_DIR, "..", "streaming", "streaming_features.json")               
 

def predict_fraud(customer_id):
    customer_id = str(customer_id)
    with open(FEATURE_PATH, "r") as f:
        features = json.load(f)

    if customer_id not in features:
        return { "error": f"Customer {customer_id} not found"}

    customer = features[customer_id]
    input_df = pd.DataFrame({
        "transaction_amount": [customer["average_amount"]],
        "account_balance": [customer["total_amount"]],
        "is_high_risk": [
            1 if customer["average_amount"] > 2000
            else 0
        ]
    })

    model = joblib.load(MODEL_PATH)

    prediction = model.predict(input_df)[0]

    return {
        "customer_id": customer_id,
        "transaction_count": customer["transaction_count"],
        "average_amount": customer["average_amount"],
        "total_amount": customer["total_amount"],
        "prediction":
            "High Risk"
            if prediction == 1
            else "Low Risk"
    }