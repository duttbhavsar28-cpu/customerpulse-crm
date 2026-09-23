"""
utils/helpers.py
================
Validation, formatting, and UI styling utilities for CustomerPulse CRM.
"""

import re
from datetime import datetime
from typing import Any, Optional, Tuple


def validate_email(email: str) -> bool:
    """Validate email format using standard regex pattern."""
    if not email or not isinstance(email, str):
        return False
    email = email.strip()
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(pattern, email))


def validate_phone(phone: str) -> bool:
    """Validate phone number format (supports international & standard formats)."""
    if not phone or not isinstance(phone, str):
        return False
    cleaned = re.sub(r"[\s\-\(\)\.]", "", phone.strip())
    # Require at least 7 digits, maximum 16 digits, optional leading +
    pattern = r"^\+?[0-9]{7,16}$"
    return bool(re.match(pattern, cleaned))


def validate_positive_number(val: Any) -> bool:
    """Validate that value is a non-negative number."""
    try:
        num = float(val)
        return num >= 0
    except (ValueError, TypeError):
        return False


def format_currency(amount: Optional[float], compact: bool = False) -> str:
    """Format float into professional currency string ($)."""
    if amount is None or (isinstance(amount, float) and (amount != amount)):  # NaN check
        return "$0.00"
    try:
        val = float(amount)
        if compact:
            if abs(val) >= 1_000_000:
                return f"${val / 1_000_000:.2f}M"
            elif abs(val) >= 1_000:
                return f"${val / 1_000:.1f}k"
        return f"${val:,.2f}"
    except (ValueError, TypeError):
        return "$0.00"


def format_percent(val: Optional[float], is_ratio: bool = True) -> str:
    """
    Format float into percentage string.
    If is_ratio=True, 0.456 -> 45.6%. If is_ratio=False, 45.6 -> 45.6%.
    """
    if val is None or (isinstance(val, float) and (val != val)):
        return "0.0%"
    try:
        num = float(val)
        if is_ratio:
            num *= 100.0
        return f"{num:.1f}%"
    except (ValueError, TypeError):
        return "0.0%"


def format_date(val: Any, output_format: str = "%b %d, %Y") -> str:
    """Format string or datetime object into consistent display string."""
    if not val:
        return "N/A"
    if isinstance(val, datetime):
        return val.strftime(output_format)
    try:
        dt = datetime.fromisoformat(str(val).replace("Z", ""))
        return dt.strftime(output_format)
    except Exception:
        # Fallback for YYYY-MM-DD
        try:
            dt = datetime.strptime(str(val)[:10], "%Y-%m-%d")
            return dt.strftime(output_format)
        except Exception:
            return str(val)


def get_risk_badge(prob: float) -> Tuple[str, str, str]:
    """
    Return (Category, Hex Color, Background Color) for churn probability.
    Categories: Low (0-30%), Medium (31-60%), High (61-100%).
    """
    try:
        p = float(prob)
        if p > 1.0:
            p /= 100.0  # normalize if passed as percentage
    except (ValueError, TypeError):
        p = 0.0

    if p <= 0.30:
        return ("Low Risk", "#10B981", "rgba(16, 185, 129, 0.15)")
    elif p <= 0.60:
        return ("Medium Risk", "#F59E0B", "rgba(245, 158, 11, 0.15)")
    else:
        return ("High Risk", "#EF4444", "rgba(239, 68, 68, 0.15)")


def get_health_badge(score: float) -> Tuple[str, str, str]:
    """
    Return (Status, Hex Color, Background Color) for Customer Health Score (0-100).
    Tiers: Excellent (90-100), Healthy (70-89), Needs Attention (50-69), Critical (0-49).
    """
    try:
        s = float(score)
    except (ValueError, TypeError):
        s = 0.0

    if s >= 90:
        return ("Excellent", "#10B981", "rgba(16, 185, 129, 0.15)")
    elif s >= 70:
        return ("Healthy", "#06B6D4", "rgba(6, 182, 212, 0.15)")
    elif s >= 50:
        return ("Needs Attention", "#F59E0B", "rgba(245, 158, 11, 0.15)")
    else:
        return ("Critical", "#EF4444", "rgba(239, 68, 68, 0.15)")


def get_lead_priority_badge(score: float) -> Tuple[str, str, str]:
    """
    Return (Priority, Hex Color, Background Color) for Lead Score (0-100).
    Tiers: High Priority (80-100), Medium Priority (50-79), Low Priority (0-49).
    """
    try:
        s = float(score)
    except (ValueError, TypeError):
        s = 0.0

    if s >= 80:
        return ("High Priority", "#10B981", "rgba(16, 185, 129, 0.15)")
    elif s >= 50:
        return ("Medium Priority", "#F59E0B", "rgba(245, 158, 11, 0.15)")
    else:
        return ("Low Priority", "#6B7280", "rgba(107, 114, 128, 0.15)")
