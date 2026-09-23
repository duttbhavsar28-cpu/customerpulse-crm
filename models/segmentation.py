"""
models/segmentation.py
======================
Unsupervised Customer Segmentation using K-Means Clustering on RFM & engagement behavior.
Includes automatic behavioral cluster naming, cluster profiling metrics, and 2D PCA projection.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from models.churn_model import extract_customer_features

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
MODEL_PATH = os.path.join(MODEL_DIR, "segmentation_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "segmentation_scaler.joblib")
MAPPING_PATH = os.path.join(MODEL_DIR, "segmentation_mapping.joblib")


SEGMENT_FEATURES = [
    "recency_days",
    "frequency",
    "monetary_total",
    "monetary_avg",
    "tenure_days",
    "interaction_count",
]


class CustomerSegmentation:
    """Manages K-Means clustering, automated segment naming, and customer profiling."""

    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.kmeans: Optional[KMeans] = None
        self.scaler: Optional[StandardScaler] = None
        self.cluster_labels_map: Dict[int, str] = {}
        self.pca: Optional[PCA] = None
        self._load_if_exists()

    def _load_if_exists(self) -> bool:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(MAPPING_PATH):
            try:
                self.kmeans = joblib.load(MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                self.cluster_labels_map = joblib.load(MAPPING_PATH)
                return True
            except Exception:
                return False
        return False

    def _auto_name_clusters(self, df_profile: pd.DataFrame) -> Dict[int, str]:
        """
        Dynamically assign intuitive business segment names based on cluster centroids:
        - High-Value Loyal
        - Growth Opportunity
        - New Customers
        - At-Risk Customers
        - Low Engagement
        """
        unassigned_clusters = list(df_profile["cluster"].values)
        mapping = {}

        # 1. High-Value Loyal: highest monetary_total
        loyal_cluster = df_profile.sort_values(by="monetary_total", ascending=False)["cluster"].iloc[0]
        mapping[int(loyal_cluster)] = "High-Value Loyal"
        unassigned_clusters.remove(loyal_cluster)

        # 2. At-Risk Customers: highest recency_days among remaining
        remaining_df = df_profile[df_profile["cluster"].isin(unassigned_clusters)]
        at_risk_cluster = remaining_df.sort_values(by="recency_days", ascending=False)["cluster"].iloc[0]
        mapping[int(at_risk_cluster)] = "At-Risk Customers"
        unassigned_clusters.remove(at_risk_cluster)

        # 3. New Customers: lowest tenure among remaining
        remaining_df = df_profile[df_profile["cluster"].isin(unassigned_clusters)]
        new_cust_cluster = remaining_df.sort_values(by="tenure_days", ascending=True)["cluster"].iloc[0]
        mapping[int(new_cust_cluster)] = "New Customers"
        unassigned_clusters.remove(new_cust_cluster)

        # 4. Low Engagement: lowest frequency among remaining
        remaining_df = df_profile[df_profile["cluster"].isin(unassigned_clusters)]
        low_eng_cluster = remaining_df.sort_values(by="frequency", ascending=True)["cluster"].iloc[0]
        mapping[int(low_eng_cluster)] = "Low Engagement"
        unassigned_clusters.remove(low_eng_cluster)

        # 5. Remaining: Growth Opportunity
        for c in unassigned_clusters:
            mapping[int(c)] = "Growth Opportunity"

        return mapping

    def fit(self, features_df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
        """
        Fit K-Means clustering model on customer features and assign descriptive names.
        Returns features_df enriched with 'cluster', 'customer_segment', 'pca_x', 'pca_y'.
        """
        os.makedirs(MODEL_DIR, exist_ok=True)
        X = features_df[SEGMENT_FEATURES].copy()

        # Log transform monetary to reduce skewness
        X["monetary_total_log"] = np.log1p(X["monetary_total"])
        X["frequency_log"] = np.log1p(X["frequency"])

        scale_cols = [
            "recency_days", "frequency_log", "monetary_total_log",
            "monetary_avg", "tenure_days", "interaction_count"
        ]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X[scale_cols])

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        # Compute centroids for automated naming
        temp_df = features_df.copy()
        temp_df["cluster"] = clusters
        cluster_summary = temp_df.groupby("cluster").agg({
            "monetary_total": "mean",
            "recency_days": "mean",
            "frequency": "mean",
            "tenure_days": "mean",
            "interaction_count": "mean"
        }).reset_index()

        self.cluster_labels_map = self._auto_name_clusters(cluster_summary)
        self.kmeans = kmeans
        self.scaler = scaler

        # 2D PCA for visual exploration
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)
        self.pca = pca

        # Save artifacts
        joblib.dump(self.kmeans, MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)
        joblib.dump(self.cluster_labels_map, MAPPING_PATH)

        enriched = features_df.copy()
        enriched["cluster"] = clusters
        enriched["customer_segment"] = [self.cluster_labels_map[c] for c in clusters]
        enriched["pca_x"] = coords[:, 0]
        enriched["pca_y"] = coords[:, 1]

        return enriched

    def get_segment_summary(self, enriched_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute summary metrics per segment:
        - Segment
        - Number of customers
        - Average revenue
        - Average order value
        - Average purchase frequency
        - Average churn probability (if present)
        """
        agg_dict = {
            "customer_id": "count",
            "monetary_total": "mean",
            "monetary_avg": "mean",
            "frequency": "mean",
            "recency_days": "mean"
        }
        if "churn_probability" in enriched_df.columns:
            agg_dict["churn_probability"] = "mean"

        summary = enriched_df.groupby("customer_segment").agg(agg_dict).reset_index()
        summary = summary.rename(columns={
            "customer_segment": "Segment",
            "customer_id": "Number of Customers",
            "monetary_total": "Average Revenue ($)",
            "monetary_avg": "Average Order Value ($)",
            "frequency": "Average Purchase Frequency",
            "recency_days": "Average Recency (Days)",
            "churn_probability": "Average Churn Probability"
        })

        return summary.sort_values(by="Average Revenue ($)", ascending=False)
