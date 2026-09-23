"""
analytics/kpis.py
=================
Executive KPI aggregator for CustomerPulse CRM.
Calculates real-time business health metrics across SQLite database tables.
"""

from typing import Any, Dict, Optional
import pandas as pd
from database.database import query_df


def get_executive_kpis(db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Compute core executive KPIs:
    - Total Customers
    - Active Customers
    - At-Risk Customers
    - Total Revenue
    - Average Customer Value (AOV / ACV)
    - Open Leads
    - Conversion Rate
    - Revenue at Risk
    """
    # 1. Customer counts by status
    cust_query = """
        SELECT
            COUNT(*) as total_customers,
            SUM(CASE WHEN customer_status = 'Active' THEN 1 ELSE 0 END) as active_customers,
            SUM(CASE WHEN customer_status = 'At-Risk' THEN 1 ELSE 0 END) as at_risk_customers,
            SUM(CASE WHEN customer_status = 'Churned' THEN 1 ELSE 0 END) as churned_customers
        FROM customers;
    """
    cust_df = query_df(cust_query, db_path=db_path)
    total_cust = int(cust_df["total_customers"].iloc[0] or 0)
    active_cust = int(cust_df["active_customers"].iloc[0] or 0)
    at_risk_cust = int(cust_df["at_risk_customers"].iloc[0] or 0)
    churned_cust = int(cust_df["churned_customers"].iloc[0] or 0)

    # 2. Transaction metrics
    tx_query = """
        SELECT
            COUNT(*) as total_txns,
            COALESCE(SUM(amount), 0.0) as total_revenue,
            COALESCE(AVG(amount), 0.0) as avg_order_value
        FROM transactions;
    """
    tx_df = query_df(tx_query, db_path=db_path)
    total_revenue = float(tx_df["total_revenue"].iloc[0] or 0.0)
    avg_order_value = float(tx_df["avg_order_value"].iloc[0] or 0.0)
    avg_cust_value = (total_revenue / total_cust) if total_cust > 0 else 0.0

    # 3. Lead conversion metrics
    lead_query = """
        SELECT
            COUNT(*) as total_leads,
            SUM(CASE WHEN status NOT IN ('Won', 'Lost') THEN 1 ELSE 0 END) as open_leads,
            SUM(CASE WHEN status = 'Won' THEN 1 ELSE 0 END) as won_leads
        FROM leads;
    """
    lead_df = query_df(lead_query, db_path=db_path)
    total_leads = int(lead_df["total_leads"].iloc[0] or 0)
    open_leads = int(lead_df["open_leads"].iloc[0] or 0)
    won_leads = int(lead_df["won_leads"].iloc[0] or 0)
    conversion_rate = (won_leads / total_leads) if total_leads > 0 else 0.0

    # 4. Revenue at Risk (from predictions or customer spend for at-risk accounts)
    pred_query = """
        SELECT
            COALESCE(SUM(churn_probability * predicted_revenue), 0.0) as revenue_at_risk
        FROM predictions;
    """
    try:
        pred_df = query_df(pred_query, db_path=db_path)
        revenue_at_risk = float(pred_df["revenue_at_risk"].iloc[0] or 0.0)
    except Exception:
        revenue_at_risk = 0.0

    # If predictions table is empty yet, fallback to at-risk customers' historical annual spend
    if revenue_at_risk == 0.0 and at_risk_cust > 0:
        fallback_query = """
            SELECT COALESCE(SUM(t.amount), 0.0) * 0.65 as risk_rev
            FROM transactions t
            JOIN customers c ON t.customer_id = c.customer_id
            WHERE c.customer_status = 'At-Risk';
        """
        fb_df = query_df(fallback_query, db_path=db_path)
        revenue_at_risk = float(fb_df["risk_rev"].iloc[0] or 0.0)

    churn_rate = (churned_cust / total_cust) if total_cust > 0 else 0.0

    return {
        "total_customers": total_cust,
        "active_customers": active_cust,
        "at_risk_customers": at_risk_cust,
        "churned_customers": churned_cust,
        "churn_rate": churn_rate,
        "total_revenue": total_revenue,
        "avg_order_value": avg_order_value,
        "avg_customer_value": avg_cust_value,
        "total_leads": total_leads,
        "open_leads": open_leads,
        "won_leads": won_leads,
        "conversion_rate": conversion_rate,
        "revenue_at_risk": revenue_at_risk,
    }
