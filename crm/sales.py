"""
crm/sales.py
============
Sales pipeline analytics, deal stage distributions, and conversion tracking.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
from database.database import query_df


def get_pipeline_stages(db_path: Optional[str] = None) -> pd.DataFrame:
    """
    Return deal count, total value, and average deal size grouped by pipeline stage.
    Ordered by typical sales stage progression.
    """
    stage_order = {
        "New": 1,
        "Contacted": 2,
        "Qualified": 3,
        "Proposal Sent": 4,
        "Negotiation": 5,
        "Won": 6,
        "Lost": 7
    }

    query = """
        SELECT
            status as stage,
            COUNT(lead_id) as deal_count,
            COALESCE(SUM(budget), 0.0) as total_pipeline_value,
            COALESCE(AVG(budget), 0.0) as avg_deal_size,
            COALESCE(AVG(engagement_score), 0.0) as avg_engagement
        FROM leads
        GROUP BY status;
    """
    df = query_df(query, db_path=db_path)
    if not df.empty:
        df["stage_order"] = df["stage"].map(lambda s: stage_order.get(s, 99))
        df = df.sort_values(by="stage_order").reset_index(drop=True)
    return df


def get_sales_velocity(db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Compute Sales Velocity metric:
    Sales Velocity = (Number of Opportunities * Win Rate * Average Deal Size) / Sales Cycle Length (Days)
    """
    stages_df = get_pipeline_stages(db_path=db_path)
    if stages_df.empty:
        return {
            "sales_velocity_daily": 0.0,
            "sales_velocity_monthly": 0.0,
            "win_rate": 0.0,
            "open_deals": 0,
            "avg_deal_size": 0.0,
            "avg_cycle_days": 45.0
        }

    total_closed = stages_df[stages_df["stage"].isin(["Won", "Lost"])]["deal_count"].sum()
    won_deals = stages_df[stages_df["stage"] == "Won"]["deal_count"].sum()
    win_rate = (won_deals / total_closed) if total_closed > 0 else 0.22

    open_deals_df = stages_df[~stages_df["stage"].isin(["Won", "Lost"])]
    open_deals = int(open_deals_df["deal_count"].sum())
    total_open_value = float(open_deals_df["total_pipeline_value"].sum())
    avg_deal_size = (total_open_value / open_deals) if open_deals > 0 else 25000.0

    avg_cycle_days = 42.0  # standard B2B enterprise sales cycle

    velocity_daily = (open_deals * win_rate * avg_deal_size) / avg_cycle_days
    velocity_monthly = velocity_daily * 30.0

    return {
        "sales_velocity_daily": round(velocity_daily, 2),
        "sales_velocity_monthly": round(velocity_monthly, 2),
        "win_rate": round(win_rate * 100.0, 1),
        "open_deals": open_deals,
        "total_open_value": round(total_open_value, 2),
        "avg_deal_size": round(avg_deal_size, 2),
        "avg_cycle_days": avg_cycle_days
    }
