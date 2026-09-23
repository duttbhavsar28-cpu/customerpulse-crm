"""
analytics/customer_health.py
============================
Transparent Customer Health Score calculation (0-100).
Components:
- Purchase Recency (0-25 pts): High points for recent orders (<30 days).
- Purchase Frequency (0-25 pts): Consistent order cadence.
- Monetary Value (0-20 pts): Historical spend tier.
- Interaction Engagement (0-20 pts): Recent meetings, calls, demos, positive outcomes.
- Support Penalty (-20 to 0 pts): Deductions for negative outcomes and open support tickets.
Tiers:
- 90–100: Excellent
- 70–89: Healthy
- 50–69: Needs Attention
- 0–49: Critical
"""

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd


def get_health_category(score: float) -> str:
    """Return categorical health tier from numerical score."""
    s = float(score)
    if s >= 90.0:
        return "Excellent"
    elif s >= 70.0:
        return "Healthy"
    elif s >= 50.0:
        return "Needs Attention"
    else:
        return "Critical"


def calculate_health_score(
    recency_days: float,
    frequency: float,
    monetary_total: float,
    interaction_count: float,
    support_tickets: float = 0.0,
    negative_interactions: float = 0.0
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate 0-100 Customer Health Score and detailed component breakdown.
    """
    # 1. Recency Points (Max 25)
    # < 30 days = 25 pts, decaying smoothly to 0 pts at 180+ days
    if recency_days <= 30:
        pts_recency = 25.0
    elif recency_days <= 60:
        pts_recency = 20.0
    elif recency_days <= 90:
        pts_recency = 15.0
    elif recency_days <= 150:
        pts_recency = 8.0
    elif recency_days <= 200:
        pts_recency = 3.0
    else:
        pts_recency = 0.0

    # 2. Frequency Points (Max 25)
    # > 10 txns = 25 pts, 6-10 = 20 pts, 3-5 = 14 pts, 1-2 = 7 pts, 0 = 0 pts
    if frequency >= 10:
        pts_freq = 25.0
    elif frequency >= 6:
        pts_freq = 20.0
    elif frequency >= 3:
        pts_freq = 14.0
    elif frequency >= 1:
        pts_freq = 7.0
    else:
        pts_freq = 0.0

    # 3. Monetary Points (Max 20)
    # > $15,000 = 20 pts, $5k-$15k = 16 pts, $1.5k-$5k = 12 pts, > $0 = 6 pts
    if monetary_total >= 15000:
        pts_monetary = 20.0
    elif monetary_total >= 5000:
        pts_monetary = 16.0
    elif monetary_total >= 1500:
        pts_monetary = 12.0
    elif monetary_total > 0:
        pts_monetary = 6.0
    else:
        pts_monetary = 0.0

    # 4. Engagement & Interaction Points (Max 20)
    # > 4 interactions = 20 pts, 2-4 = 14 pts, 1 = 8 pts, 0 = 2 pts
    if interaction_count >= 5:
        pts_engagement = 20.0
    elif interaction_count >= 3:
        pts_engagement = 15.0
    elif interaction_count >= 1:
        pts_engagement = 9.0
    else:
        pts_engagement = 2.0

    # 5. Complaints & Support Penalty (Subtract up to 25 pts)
    # 4 pts per negative interaction, 2 pts per support ticket
    penalty = (negative_interactions * 4.5) + (support_tickets * 1.5)
    pts_penalty = min(25.0, penalty)

    raw_score = (pts_recency + pts_freq + pts_monetary + pts_engagement) - pts_penalty
    final_score = float(np.clip(raw_score, 0.0, 100.0))

    breakdown = {
        "recency_points": pts_recency,
        "frequency_points": pts_freq,
        "monetary_points": pts_monetary,
        "engagement_points": pts_engagement,
        "complaint_penalty": pts_penalty,
        "final_score": round(final_score, 1),
        "health_category": get_health_category(final_score)
    }

    return round(final_score, 1), breakdown


def compute_customer_health_batch(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute health scores across an entire features DataFrame.
    Returns DataFrame with customer_id, health_score, health_category.
    """
    scores = []
    categories = []

    for _, row in features_df.iterrows():
        score, _ = calculate_health_score(
            recency_days=row.get("recency_days", 90),
            frequency=row.get("frequency", 1),
            monetary_total=row.get("monetary_total", 0.0),
            interaction_count=row.get("interaction_count", 0),
            support_tickets=row.get("support_tickets", 0),
            negative_interactions=row.get("negative_interactions", 0)
        )
        scores.append(score)
        categories.append(get_health_category(score))

    res = features_df[["customer_id"]].copy()
    res["health_score"] = scores
    res["health_category"] = categories
    return res
