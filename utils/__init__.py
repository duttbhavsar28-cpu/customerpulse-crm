"""Utility package for CustomerPulse CRM."""
from utils.helpers import (
    validate_email,
    validate_phone,
    validate_positive_number,
    format_currency,
    format_percent,
    format_date,
    get_risk_badge,
    get_health_badge,
)
from utils.data_generator import generate_synthetic_data, load_or_generate_data

__all__ = [
    "validate_email",
    "validate_phone",
    "validate_positive_number",
    "format_currency",
    "format_percent",
    "format_date",
    "get_risk_badge",
    "get_health_badge",
    "generate_synthetic_data",
    "load_or_generate_data",
]
