# Train a simple model using the features from the feature store and save it for later use in inference.
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier


print("\n-----------------------------------------------------")
print("Loading features from feature store")
print("-----------------------------------------------------\n")

df = pd.read_parquet("feature_repo/fraud_transactions.parquet")
print(df)

# Feature selection and target variable
X = df[[ "transaction_amount", "account_balance", "is_high_risk"]]
y = [1, 0, 1, 0, 1]


# Train a simple model using the features from the feature store and save it for later use in inference.
print("\n-----------------------------------------------------")        
print("Training Fraud Model")
print("-----------------------------------------------------\n")

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)


# Save the model for later use in inference
joblib.dump(model, "fraud_model.pkl")
print("Model saved successfully.")
print("File: fraud_model.pkl")

