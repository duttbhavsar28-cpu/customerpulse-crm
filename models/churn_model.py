"""
models/churn_model.py
=====================
Machine learning churn prediction model using Scikit-learn.
Features: RFM, tenure, interaction history, support complaints, negative outcomes.
Model: Random Forest Classifier with genuine train/test evaluation.
Outputs:
- Churn Probability (0.0 to 1.0)
- Risk Category: Low (0-30%), Medium (31-60%), High (61-100%)
- Real evaluation metrics: Accuracy, Precision, Recall, F1, Confusion Matrix
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "churn_metrics.joblib")

CURRENT_REF_DATE = datetime(2026, 9, 1)

FEATURE_COLS = [
    "recency_days",
    "frequency",
    "monetary_total",
    "monetary_avg",
    "tenure_days",
    "interaction_count",
    "support_tickets",
    "negative_interactions",
    "days_since_interaction",
    "avg_quantity",
]


def extract_customer_features(
    customers_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    ref_date: datetime = CURRENT_REF_DATE
) -> pd.DataFrame:
    """
    Extract aggregated RFM and engagement features per customer from raw relational tables.
    Avoids data leakage by using only behavioral indicators.
    """
    # 1. Customer tenure
    cust = customers_df[["customer_id", "customer_since", "customer_status"]].copy()
    cust["cust_since_dt"] = pd.to_datetime(cust["customer_since"])
    cust["tenure_days"] = (ref_date - cust["cust_since_dt"]).dt.days.clip(lower=1)

    # 2. Transaction aggregations (RFM)
    tx = transactions_df.copy()
    tx["tx_date_dt"] = pd.to_datetime(tx["transaction_date"])

    tx_agg = tx.groupby("customer_id").agg(
        last_tx_date=("tx_date_dt", "max"),
        frequency=("transaction_id", "count"),
        monetary_total=("amount", "sum"),
        monetary_avg=("amount", "mean"),
        avg_quantity=("quantity", "mean")
    ).reset_index()

    tx_agg["recency_days"] = (ref_date - tx_agg["last_tx_date"]).dt.days.clip(lower=0)
    tx_agg = tx_agg.drop(columns=["last_tx_date"])

    # 3. Interaction aggregations
    itx = interactions_df.copy()
    itx["int_date_dt"] = pd.to_datetime(itx["interaction_date"])

    itx_agg = itx.groupby("customer_id").agg(
        last_int_date=("int_date_dt", "max"),
        interaction_count=("interaction_id", "count"),
        support_tickets=("interaction_type", lambda s: (s == "Support Request").sum()),
        negative_interactions=("outcome", lambda s: (s == "Negative").sum())
    ).reset_index()

    itx_agg["days_since_interaction"] = (ref_date - itx_agg["last_int_date"]).dt.days.clip(lower=0)
    itx_agg = itx_agg.drop(columns=["last_int_date"])

    # Merge all features
    df = cust.merge(tx_agg, on="customer_id", how="left")
    df = df.merge(itx_agg, on="customer_id", how="left")

    # Handle customers with 0 transactions or 0 interactions
    df["recency_days"] = df["recency_days"].fillna(df["tenure_days"])
    df["frequency"] = df["frequency"].fillna(0)
    df["monetary_total"] = df["monetary_total"].fillna(0.0)
    df["monetary_avg"] = df["monetary_avg"].fillna(0.0)
    df["avg_quantity"] = df["avg_quantity"].fillna(0.0)
    df["interaction_count"] = df["interaction_count"].fillna(0)
    df["support_tickets"] = df["support_tickets"].fillna(0)
    df["negative_interactions"] = df["negative_interactions"].fillna(0)
    df["days_since_interaction"] = df["days_since_interaction"].fillna(df["tenure_days"])

    # Target variable: 1 if customer_status is Churned, 0 otherwise
    df["target_churn"] = (df["customer_status"] == "Churned").astype(int)

    return df


class ChurnPredictor:
    """Manages training, evaluation, persistence, and inference for customer churn prediction."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.metrics_path = METRICS_PATH
        self.model: Optional[RandomForestClassifier] = None
        self.metrics: Optional[Dict[str, Any]] = None
        self.feature_names: List[str] = FEATURE_COLS
        self._load_if_exists()

    def _load_if_exists(self) -> bool:
        if os.path.exists(self.model_path) and os.path.exists(self.metrics_path):
            try:
                self.model = joblib.load(self.model_path)
                self.metrics = joblib.load(self.metrics_path)
                return True
            except Exception:
                return False
        return False

    def train(
        self,
        customers_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Train Random Forest churn model on historical customer behavioral data.
        Evaluates real metrics on stratified 20% test holdout.
        """
        if not force and self.model is not None and self.metrics is not None:
            return self.metrics

        os.makedirs(MODEL_DIR, exist_ok=True)
        feat_df = extract_customer_features(customers_df, transactions_df, interactions_df)

        X = feat_df[self.feature_names]
        y = feat_df["target_churn"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        model = RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Real test set evaluations
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()  # [[TN, FP], [FN, TP]]

        importances = pd.DataFrame({
            "feature": self.feature_names,
            "importance": model.feature_importances_
        }).sort_values(by="importance", ascending=False).to_dict(orient="records")

        self.model = model
        self.metrics = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "confusion_matrix": cm,
            "feature_importances": importances,
            "trained_at": datetime.now().isoformat(),
            "n_samples": len(X),
            "test_samples": len(X_test)
        }

        # Persist model artifacts
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.metrics, self.metrics_path)

        return self.metrics

    def predict_single(self, feature_dict: Dict[str, float]) -> Tuple[float, str]:
        """Predict churn probability and risk tier for a single feature dictionary."""
        if self.model is None:
            raise ValueError("Churn model is not trained yet. Call train() first.")

        df_row = pd.DataFrame([feature_dict])[self.feature_names]
        prob = float(self.model.predict_proba(df_row)[0, 1])

        if prob <= 0.30:
            category = "Low"
        elif prob <= 0.60:
            category = "Medium"
        else:
            category = "High"

        return round(prob, 4), category

    def predict_batch(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate churn probability and risk category for a DataFrame of customer features.
        Returns DataFrame with customer_id, churn_probability, churn_risk_category.
        """
        if self.model is None:
            raise ValueError("Churn model is not trained yet. Call train() first.")

        X = features_df[self.feature_names]
        probs = self.model.predict_proba(X)[:, 1]

        res = features_df[["customer_id"]].copy()
        res["churn_probability"] = np.round(probs, 4)
        res["churn_risk_category"] = np.where(
            probs <= 0.30, "Low",
            np.where(probs <= 0.60, "Medium", "High")
        )
        return res
