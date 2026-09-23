"""Recommendations package for CustomerPulse CRM."""
from recommendations.next_best_action import (
    determine_next_best_action,
    recommend_products_for_customer,
    generate_batch_recommendations,
)

__all__ = [
    "determine_next_best_action",
    "recommend_products_for_customer",
    "generate_batch_recommendations",
]
