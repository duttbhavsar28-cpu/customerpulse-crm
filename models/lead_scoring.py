"""
models/lead_scoring.py
======================
Machine learning lead scoring and prioritization model.
Predicts conversion probability (0-100) using Gradient Boosting / Random Forest.
Classifies:
- 80–100: High Priority
- 50–79: Medium Priority
- 0–49: Low Priority
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
LEAD_MODEL_PATH = os.path.join(MODEL_DIR, "lead_scoring_model.joblib")
LEAD_METRICS_PATH = os.path.join(MODEL_DIR, "lead_scoring_metrics.joblib")

CAT_COLS = ["industry", "source", "salesperson"]
NUM_COLS = ["budget", "engagement_score"]


class LeadScorer:
    """Manages training, scoring, priority assignment, and evaluation of sales leads."""

    def __init__(self, model_path: str = LEAD_MODEL_PATH):
        self.model_path = model_path
        self.metrics_path = LEAD_METRICS_PATH
        self.pipeline: Optional[Pipeline] = None
        self.metrics: Optional[Dict[str, Any]] = None
        self._load_if_exists()

    def _load_if_exists(self) -> bool:
        if os.path.exists(self.model_path) and os.path.exists(self.metrics_path):
            try:
                self.pipeline = joblib.load(self.model_path)
                self.metrics = joblib.load(self.metrics_path)
                return True
            except Exception:
                return False
        return False

    def train(self, leads_df: pd.DataFrame, force: bool = False) -> Dict[str, Any]:
        """
        Train Gradient Boosting pipeline to predict conversion probability.
        Target: 1 if status in ('Won', 'Qualified', 'Proposal Sent'), 0 otherwise.
        """
        if not force and self.pipeline is not None and self.metrics is not None:
            return self.metrics

        os.makedirs(MODEL_DIR, exist_ok=True)
        df = leads_df.copy()

        # Binary conversion target: 1 if converted or qualified high-intent
        df["target"] = df["status"].apply(lambda s: 1 if s in ("Won", "Qualified", "Proposal Sent", "Negotiation") else 0)

        X = df[CAT_COLS + NUM_COLS]
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), NUM_COLS),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS)
            ]
        )

        clf = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=4,
            random_state=42
        )

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        pipeline.fit(X_train, y_train)

        # Evaluate on holdout test set
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, y_prob))

        self.pipeline = pipeline
        self.metrics = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc,
            "trained_at": datetime.now().isoformat(),
            "n_samples": len(X),
            "test_samples": len(X_test)
        }

        joblib.dump(self.pipeline, self.model_path)
        joblib.dump(self.metrics, self.metrics_path)

        return self.metrics

    def score_leads(self, leads_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score a DataFrame of leads, returning:
        - lead_score (0-100)
        - priority ('High Priority', 'Medium Priority', 'Low Priority')
        - recommended_action
        """
        if self.pipeline is None:
            raise ValueError("Lead scoring model is not trained yet.")

        X = leads_df[CAT_COLS + NUM_COLS]
        probs = self.pipeline.predict_proba(X)[:, 1]
        scores = np.round(probs * 100.0, 1)

        result = leads_df.copy()
        result["lead_score"] = scores
        result["priority"] = np.where(
            scores >= 80.0, "High Priority",
            np.where(scores >= 50.0, "Medium Priority", "Low Priority")
        )

        def recommend_lead_action(row) -> str:
            score = row["lead_score"]
            status = row["status"]
            if score >= 80:
                if status in ("New", "Contacted"):
                    return "Direct Executive Phone Outreach & Schedule Demo within 24h"
                elif status == "Qualified":
                    return "Expedite Custom Solution Architecture & Enterprise Proposal"
                else:
                    return "Offer Executive Closing Concession & Schedule Final Sign-off"
            elif score >= 50:
                if status == "New":
                    return "Send Targeted Industry Case Study & Warm Follow-up Email"
                elif status in ("Contacted", "Qualified"):
                    return "Invite to Next Live Product Deep-Dive Webinar"
                else:
                    return "Nurture with Monthly ROI Calculator & Feature Teaser"
            else:
                return "Place into Low-Touch Automated Email Drip Campaign"

        result["recommended_action"] = result.apply(recommend_lead_action, axis=1)
        return result
