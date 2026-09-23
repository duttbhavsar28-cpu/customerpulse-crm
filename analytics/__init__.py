"""Analytics package for CustomerPulse CRM."""
from analytics.customer_health import (
    calculate_health_score,
    compute_customer_health_batch,
    get_health_category,
)
from analytics.kpis import get_executive_kpis
from analytics.revenue import (
    calculate_clv,
    get_monthly_revenue_trend,
    get_revenue_by_industry,
    get_revenue_by_product,
    get_top_customers,
)

__all__ = [
    "calculate_health_score",
    "compute_customer_health_batch",
    "get_health_category",
    "get_executive_kpis",
    "calculate_clv",
    "get_monthly_revenue_trend",
    "get_revenue_by_industry",
    "get_revenue_by_product",
    "get_top_customers",
]
