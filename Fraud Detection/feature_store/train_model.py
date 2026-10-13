# Train a fraud detection model using Optuna hyperparameter tuning
# and save the best model for later inference.

import joblib
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
import mlflow


print("\n-----------------------------------------------------")
print("Loading features from feature store")
print("-----------------------------------------------------\n")

# Read feature dataset from feature store
# Replace the path if your dataset location differs
try:
    df = pd.read_parquet("feature_repo/fraud_transactions.parquet")
except FileNotFoundError:
    df = pd.read_csv("..\\feature pipeline\\data\\fraud_data.csv")

print(df.head())

# Feature selection and target variable
# Accept a few common target column names if available.
target_col = None
for candidate in ("is_fraud", "is_high_risk", "label", "fraud"):
    if candidate in df.columns:
        target_col = candidate
        break

if target_col is None:
    raise ValueError("The dataset must contain a target column (is_fraud / is_high_risk / label / fraud).")

feature_cols = [col for col in df.columns if col != target_col]
X = df[feature_cols]
y = df[target_col]

# Keep only numeric features for sklearn; drop datetime/object cols and obvious IDs
X = X.select_dtypes(include=["number"]).copy()
id_cols = [c for c in X.columns if c.lower().endswith("id") or "id" in c.lower()]
X = X.drop(columns=[c for c in id_cols if c in X.columns], errors="ignore")

try:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
except ValueError:
    # Fallback when stratified split isn't possible (e.g., very small classes)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=None,
    )


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "random_state": 42,
    }

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_valid)[:, 1]
    score = roc_auc_score(y_valid, proba)
    return score


print("\n-----------------------------------------------------")
print("Running Optuna Hyperparameter Tuning")
print("-----------------------------------------------------\n")
# Guard: small or imbalanced datasets can make stratified tuning and ROC AUC unstable.
class_counts = y.value_counts()
if len(y) < 20 or class_counts.min() < 2:
    print("Dataset too small or imbalanced for hyperparameter search; using defaults.")
    best_params = {
        "n_estimators": 100,
        "max_depth": 7,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
        "class_weight": None,
    }
else:
    study = optuna.create_study(direction="maximize", study_name="fraud_detection_rf")
    study.optimize(objective, n_trials=30)
    best_params = study.best_params
    print("Best trial score:", study.best_value)
    print("Best parameters:", best_params)

print("\n-----------------------------------------------------")
print("Training final model with best parameters")
print("-----------------------------------------------------\n")

final_model = RandomForestClassifier(**best_params, random_state=42)
final_model.fit(X_train, y_train)

# Save the model for later use in inference
joblib.dump(final_model, "fraud_model.pkl")
print("Model saved successfully.")
print("File: fraud_model.pkl")

# MLflow tracking: use central server if MLFLOW_TRACKING_URI is set; otherwise local ./mlruns
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
else:
    # Use local mlruns directory in repo root
    repo_root = Path(__file__).resolve().parents[2]
    mlflow.set_tracking_uri(f"file://{repo_root / 'mlruns'}")

mlflow.set_experiment("fraud_detection")
with mlflow.start_run():
    # Log best params
    for k, v in best_params.items():
        mlflow.log_param(k, v)
    # Evaluate on validation set and log metric
    valid_proba = final_model.predict_proba(X_valid)[:, 1]
    final_auc = roc_auc_score(y_valid, valid_proba)
    mlflow.log_metric("validation_roc_auc", float(final_auc))
    # Log model artifact
    mlflow.log_artifact("fraud_model.pkl")
    print(f"Logged best params and validation AUC ({final_auc}) to MLflow")

# If BentoML is available, save the model into BentoML model store for serving
try:
    import bentoml
    from bentoml._internal.models import Model

    try:
        # Save sklearn model under the name 'fraud_model'
        bentoml.sklearn.save_model("fraud_model", final_model)
        print("Registered model with BentoML as 'fraud_model'.")
    except Exception as e:
        print("BentoML model registration failed:", e)
except Exception:
    # BentoML not installed; skip registration
    pass

