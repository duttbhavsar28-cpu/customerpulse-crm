"""
analytics/revenue.py
====================
Revenue intelligence and Customer Lifetime Value (CLV) analytics.
Provides:
- Transparent CLV formula and explanations
- Monthly revenue trends and velocity
- Industry, product category, and salesperson breakdown
- Top customer account leaderboards
"""

from typing import Any, Dict, Optional
import pandas as pd
from database.database import query_df


def calculate_clv(
    avg_order_value: float,
    purchase_frequency_per_year: float,
    churn_rate: float,
    gross_margin: float = 0.75
) -> Dict[str, Any]:
    """
    Calculate estimated business-wide Customer Lifetime Value (CLV).
    Formula:
        CLV = (AOV * Annual Purchase Frequency / Churn Rate) * Gross Margin
    """
    effective_churn = max(0.04, min(0.95, churn_rate if churn_rate > 0 else 0.15))
    annual_revenue_per_customer = avg_order_value * purchase_frequency_per_year
    customer_lifespan_years = 1.0 / effective_churn
    clv_val = annual_revenue_per_customer * customer_lifespan_years * gross_margin

    return {
        "clv_value": round(clv_val, 2),
        "annual_revenue_per_customer": round(annual_revenue_per_customer, 2),
        "customer_lifespan_years": round(customer_lifespan_years, 2),
        "effective_churn_rate": round(effective_churn, 4),
        "gross_margin": gross_margin,
        "formula": "CLV = (AOV × Annual Purchase Frequency ÷ Churn Rate) × Gross Margin"
    }


def get_monthly_revenue_trend(db_path: Optional[str] = None) -> pd.DataFrame:
    """Return month-by-month revenue and transaction volume."""
    query = """
        SELECT
            strftime('%Y-%m', transaction_date) as month,
            SUM(amount) as monthly_revenue,
            COUNT(transaction_id) as transaction_count,
            COUNT(DISTINCT customer_id) as active_buyers
        FROM transactions
        GROUP BY strftime('%Y-%m', transaction_date)
        ORDER BY month ASC;
    """
    df = query_df(query, db_path=db_path)
    if not df.empty:
        df["cumulative_revenue"] = df["monthly_revenue"].cumsum()
        df["revenue_growth_pct"] = df["monthly_revenue"].pct_change() * 100.0
    return df


def get_revenue_by_industry(db_path: Optional[str] = None) -> pd.DataFrame:
    """Return revenue, customer count, and AOV grouped by industry."""
    query = """
        SELECT
            c.industry,
            COUNT(DISTINCT c.customer_id) as customer_count,
            COALESCE(SUM(t.amount), 0.0) as total_revenue,
            COALESCE(AVG(t.amount), 0.0) as avg_transaction_size
        FROM customers c
        LEFT JOIN transactions t ON c.customer_id = t.customer_id
        GROUP BY c.industry
        ORDER BY total_revenue DESC;
    """
    return query_df(query, db_path=db_path)


def get_revenue_by_product(db_path: Optional[str] = None) -> pd.DataFrame:
    """Return revenue, total quantity, and category grouped by product."""
    query = """
        SELECT
            p.product_id,
            p.product_name,
            p.category,
            p.price,
            COALESCE(SUM(t.quantity), 0) as units_sold,
            COALESCE(SUM(t.amount), 0.0) as total_revenue
        FROM products p
        LEFT JOIN transactions t ON p.product_id = t.product_id
        GROUP BY p.product_id, p.product_name, p.category, p.price
        ORDER BY total_revenue DESC;
    """
    return query_df(query, db_path=db_path)


def get_revenue_by_salesperson(db_path: Optional[str] = None) -> pd.DataFrame:
    """Return pipeline conversion performance grouped by salesperson."""
    query = """
        SELECT
            salesperson,
            COUNT(lead_id) as total_leads_handled,
            SUM(CASE WHEN status = 'Won' THEN 1 ELSE 0 END) as won_deals,
            ROUND(100.0 * SUM(CASE WHEN status = 'Won' THEN 1 ELSE 0 END) / COUNT(lead_id), 1) as win_rate_pct,
            COALESCE(SUM(CASE WHEN status = 'Won' THEN budget ELSE 0.0 END), 0.0) as won_revenue,
            COALESCE(SUM(CASE WHEN status NOT IN ('Won', 'Lost') THEN budget ELSE 0.0 END), 0.0) as open_pipeline_value
        FROM leads
        GROUP BY salesperson
        ORDER BY won_revenue DESC;
    """
    return query_df(query, db_path=db_path)


def get_top_customers(limit: int = 10, db_path: Optional[str] = None) -> pd.DataFrame:
    """Return top customer accounts by total revenue."""
    query = f"""
        SELECT
            c.customer_id,
            c.name,
            c.company,
            c.industry,
            c.customer_status,
            COALESCE(SUM(t.amount), 0.0) as total_revenue,
            COUNT(t.transaction_id) as order_count,
            MAX(t.transaction_date) as last_purchase_date
        FROM customers c
        LEFT JOIN transactions t ON c.customer_id = t.customer_id
        GROUP BY c.customer_id, c.name, c.company, c.industry, c.customer_status
        ORDER BY total_revenue DESC
        LIMIT {int(limit)};
    """
    return query_df(query, db_path=db_path)
