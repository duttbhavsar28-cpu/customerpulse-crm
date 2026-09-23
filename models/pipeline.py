"""
models/pipeline.py
==================
Unified Machine Learning & Customer Intelligence pipeline for CustomerPulse CRM.
Orchestrates:
1. Feature extraction from raw CRM tables
2. Churn prediction model training & inference
3. Unsupervised K-Means behavioral customer segmentation
4. Lead scoring model training & inference
5. Revenue forecasting regression model
6. Customer Health Score calculation
7. Next-Best-Action recommendation generation
8. Syncing predictions directly into SQLite 'predictions' table
"""

from datetime import datetime
from typing import Any, Dict, Optional
import pandas as pd

from analytics.customer_health import compute_customer_health_batch
from database.database import execute_query, get_db_path, insert_df, query_df
from models.churn_model import ChurnPredictor, extract_customer_features
from models.lead_scoring import LeadScorer
from models.revenue_prediction import RevenuePredictor
from models.segmentation import CustomerSegmentation
from recommendations.next_best_action import determine_next_best_action


def run_full_ml_pipeline(force_retrain: bool = False, db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute end-to-end Machine Learning and Customer Intelligence pipeline.
    """
    path = db_path or get_db_path()

    # 1. Fetch raw data
    cust_df = query_df("SELECT * FROM customers;", db_path=path)
    tx_df = query_df("SELECT * FROM transactions;", db_path=path)
    itx_df = query_df("SELECT * FROM interactions;", db_path=path)
    leads_df = query_df("SELECT * FROM leads;", db_path=path)

    if cust_df.empty or tx_df.empty:
        raise ValueError("Cannot run ML pipeline: customers or transactions table is empty.")

    # 2. Extract features
    features_df = extract_customer_features(cust_df, tx_df, itx_df)

    # 3. Churn Prediction Model
    churn_predictor = ChurnPredictor()
    churn_metrics = churn_predictor.train(cust_df, tx_df, itx_df, force=force_retrain)
    churn_preds = churn_predictor.predict_batch(features_df)

    # Merge churn probability into features
    features_df["churn_probability"] = churn_preds["churn_probability"]
    features_df["churn_risk_category"] = churn_preds["churn_risk_category"]

    # 4. Customer Segmentation (K-Means)
    segmenter = CustomerSegmentation(n_clusters=5)
    segmented_features = segmenter.fit(features_df, force=force_retrain)
    features_df["customer_segment"] = segmented_features["customer_segment"]
    features_df["pca_x"] = segmented_features["pca_x"]
    features_df["pca_y"] = segmented_features["pca_y"]

    # 5. Customer Health Score
    health_df = compute_customer_health_batch(features_df)
    features_df["health_score"] = health_df["health_score"]
    features_df["health_category"] = health_df["health_category"]

    # 6. Revenue Forecasting Model
    rev_predictor = RevenuePredictor()
    rev_metrics = rev_predictor.train(features_df, force=force_retrain)
    rev_preds = rev_predictor.predict(features_df)
    features_df["predicted_revenue"] = rev_preds["predicted_revenue"]
    features_df["revenue_at_risk"] = rev_preds["revenue_at_risk"]
    features_df["potential_revenue"] = rev_preds["potential_revenue"]

    # 7. Lead Scoring Model
    lead_scorer = LeadScorer()
    lead_metrics = lead_scorer.train(leads_df, force=force_retrain)

    # 8. Next-Best-Action Generation
    actions = []
    for _, row in features_df.iterrows():
        action_info = determine_next_best_action(row.to_dict())
        actions.append(action_info["title"])
    features_df["recommended_action"] = actions

    # 9. Sync to SQLite 'predictions' Table
    today_str = datetime.now().strftime("%Y-%m-%d")
    pred_table_df = pd.DataFrame({
        "customer_id": features_df["customer_id"],
        "churn_probability": features_df["churn_probability"],
        "customer_segment": features_df["customer_segment"],
        "health_score": features_df["health_score"],
        "predicted_revenue": features_df["predicted_revenue"],
        "recommended_action": features_df["recommended_action"],
        "prediction_date": today_str
    })

    # Clear and replace predictions table safely
    execute_query("DELETE FROM predictions;", db_path=path)
    insert_df("predictions", pred_table_df, if_exists="append", db_path=path)

    return {
        "churn_metrics": churn_metrics,
        "segment_summary": segmenter.get_segment_summary(features_df),
        "lead_metrics": lead_metrics,
        "revenue_metrics": rev_metrics,
        "total_predictions_stored": len(pred_table_df),
        "executed_at": datetime.now().isoformat()
    }
