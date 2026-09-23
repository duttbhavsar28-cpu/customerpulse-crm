"""
models/revenue_prediction.py
============================
Machine learning regression model predicting future 12-month customer revenue.
Algorithm: Gradient Boosting Regressor / Random Forest Regressor.
Calculates:
- Current Historical Revenue
- Predicted Future Revenue
- Revenue at Risk (Churn Probability * Predicted Revenue)
- Potential Expansion Revenue
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
REV_MODEL_PATH = os.path.join(MODEL_DIR, "revenue_model.joblib")
REV_METRICS_PATH = os.path.join(MODEL_DIR, "revenue_metrics.joblib")

REV_FEATURES = [
    "recency_days",
    "frequency",
    "monetary_total",
    "monetary_avg",
    "tenure_days",
    "interaction_count",
    "avg_quantity",
]


class RevenuePredictor:
    """Manages training, forecasting, and evaluation of customer future revenue."""

    def __init__(self, model_path: str = REV_MODEL_PATH):
        self.model_path = model_path
        self.metrics_path = REV_METRICS_PATH
        self.model: Optional[GradientBoostingRegressor] = None
        self.metrics: Optional[Dict[str, Any]] = None
        self.feature_names: List[str] = REV_FEATURES
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

    def train(self, features_df: pd.DataFrame, force: bool = False) -> Dict[str, Any]:
        """
        Train regression model on customer transactional velocity and historical spend.
        Constructs realistic target future 12-month revenue based on annualized run rate
        and recency penalty.
        """
        if not force and self.model is not None and self.metrics is not None:
            return self.metrics

        os.makedirs(MODEL_DIR, exist_ok=True)
        df = features_df.copy()

        # Construct realistic target: annualized run-rate adjusted for recency & active status
        annual_run_rate = (df["monetary_total"] / np.maximum(df["tenure_days"] / 365.25, 0.25))
        # Recency decay penalty: accounts inactive for 180+ days suffer severe revenue decay
        recency_factor = np.exp(-df["recency_days"] / 120.0)
        interaction_boost = 1.0 + (df["interaction_count"] * 0.04).clip(0, 0.35)
        target = annual_run_rate * recency_factor * interaction_boost
        # Add slight natural economic variance
        noise = np.random.normal(1.0, 0.08, size=len(df))
        df["target_future_revenue"] = np.maximum(0.0, target * noise).round(2)

        X = df[self.feature_names]
        y = df["target_future_revenue"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        reg = GradientBoostingRegressor(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=5,
            random_state=42
        )
        reg.fit(X_train, y_train)

        y_pred = reg.predict(X_test)
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))

        self.model = reg
        self.metrics = {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2_score": round(r2, 4),
            "trained_at": datetime.now().isoformat(),
            "n_samples": len(X),
            "test_samples": len(X_test)
        }

        joblib.dump(self.model, self.model_path)
        joblib.dump(self.metrics, self.metrics_path)

        return self.metrics

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate future 12-month revenue predictions and compute revenue-at-risk.
        Returns DataFrame with:
        - customer_id
        - current_revenue
        - predicted_revenue
        - revenue_at_risk
        - potential_revenue
        """
        if self.model is None:
            raise ValueError("Revenue model is not trained yet.")

        X = features_df[self.feature_names]
        preds = np.maximum(0.0, self.model.predict(X)).round(2)

        res = features_df[["customer_id"]].copy()
        res["current_revenue"] = features_df["monetary_total"].round(2)
        res["predicted_revenue"] = preds

        # Merge churn probability if available
        if "churn_probability" in features_df.columns:
            res["revenue_at_risk"] = (res["predicted_revenue"] * features_df["churn_probability"]).round(2)
        else:
            res["revenue_at_risk"] = 0.0

        # Potential expansion revenue (predicted + 15% upside for growth accounts)
        res["potential_revenue"] = (res["predicted_revenue"] * 1.18).round(2)

        return res
