import pandas as pd
import numpy as np
import joblib
import optuna
from tqdm import tqdm

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, classification_report
)
from sklearn.base import BaseEstimator, ClassifierMixin

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier


# --------------------------------------------------------------------------
# 1. Manual Voting Ensemble Class with TQDM Progress Bar
# --------------------------------------------------------------------------
class ManualVotingEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators, weights=None):
        self.estimators = estimators
        self.weights = weights if weights is not None else [1.0] * len(estimators)
        self.fitted_estimators_ = []

    def fit(self, X, y):
        self.fitted_estimators_ = []
        for est in tqdm(self.estimators, desc="Fitting base estimators"):
            model = est[1] if isinstance(est, tuple) else est
            model.fit(X, y)
            self.fitted_estimators_.append(model)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        total_weight = sum(self.weights)
        proba_sum = None
        for w, estimator in zip(self.weights, self.fitted_estimators_):
            p = estimator.predict_proba(X)
            if proba_sum is None:
                proba_sum = w * p
            else:
                proba_sum += w * p
        return proba_sum / total_weight

    def predict(self, X):
        avg_proba = self.predict_proba(X)
        return np.argmax(avg_proba, axis=1)

    def _more_tags(self):
        return {"estimator_type": "classifier", "allow_nan": True, "X_types": ["2darray"]}


# --------------------------------------------------------------------------
# 2. Load and Prepare Dataset
# --------------------------------------------------------------------------
def load_and_prepare_data(file_path):
    data = pd.read_csv(file_path)
    print("Dataset shape:", data.shape)

    X = data.drop(columns=["Class"])
    y = data["Class"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, encoder, scaler


# --------------------------------------------------------------------------
# 3. SMOTE for Class Balancing
# --------------------------------------------------------------------------
def balance_dataset(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print("Balanced Dataset Shape:", X_train_balanced.shape)
    return X_train_balanced, y_train_balanced


# --------------------------------------------------------------------------
# 4. Define Base Models
# --------------------------------------------------------------------------
def define_base_models():
    xgb_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
    )
    lgb_model = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    cat_model = CatBoostClassifier(
        iterations=150,
        depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=0,
    )

    base_estimators = [
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('cat', cat_model)
    ]
    return base_estimators


# --------------------------------------------------------------------------
# 5. Build Manual Voting Ensemble (Soft Voting)
# --------------------------------------------------------------------------
def build_manual_voting_ensemble(base_estimators, meta_weight=2.0):
    weights = [meta_weight] + [1] * (len(base_estimators) - 1)
    ensemble = ManualVotingEnsemble(estimators=base_estimators, weights=weights)
    return ensemble


# --------------------------------------------------------------------------
# 6. Train, Evaluate, and Save Ensemble Model
# --------------------------------------------------------------------------
def train_evaluate_save(model, X_train, X_test, y_train, y_test, encoder, scaler, model_path, encoder_path, scaler_path):
    print("\n=== Training Manual Voting Ensemble ===")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')

    print("\n=== Manual Voting Ensemble Results ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Precision: {precision:.4f}")

    y_pred_decoded = encoder.inverse_transform(y_pred)
    y_test_decoded = encoder.inverse_transform(y_test)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_decoded, y_pred_decoded))

    print("\nClassification Report:")
    print(classification_report(y_test_decoded, y_pred_decoded))

    print(f"\nSaving the Manual Voting Ensemble model to: {model_path}")
    joblib.dump(model, model_path)
    print("Model saved successfully!")

    print(f"\nSaving the Label Encoder to: {encoder_path}")
    joblib.dump(encoder, encoder_path)
    print("Label Encoder saved successfully!")

    print(f"\nSaving the Scaler to: {scaler_path}")
    joblib.dump(scaler, scaler_path)
    print("Scaler saved successfully!")


# --------------------------------------------------------------------------
# 7. Main Execution
# --------------------------------------------------------------------------
if __name__ == "__main__":
    file_path = r"C:\CODES\AI\MalwareAPK\updated_filtered_dataset.csv"
    model_save_path = "blended_ensemble_model.pkl"
    encoder_save_path = "label_encoder.pkl"
    scaler_save_path = "scaler.pkl"

    # Load and prepare data
    X_train, X_test, y_train, y_test, encoder, scaler = load_and_prepare_data(file_path)

    # Balance the dataset with SMOTE
    X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)

    # Polynomial Features (degree=1 to avoid blow-up in feature size)
    poly = PolynomialFeatures(degree=1, interaction_only=False, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_balanced)
    X_test_poly = poly.transform(X_test)

    # Define the base models
    base_estimators = define_base_models()

    # Build a manual ensemble with custom weights
    manual_ensemble = build_manual_voting_ensemble(base_estimators)

    # Train, evaluate, and save
    train_evaluate_save(
        manual_ensemble,
        X_train_poly,
        X_test_poly,
        y_train_balanced,
        y_test,
        encoder,
        scaler,
        model_save_path,
        encoder_save_path,
        scaler_save_path
    )
