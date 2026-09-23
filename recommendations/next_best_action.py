"""
recommendations/next_best_action.py
===================================
Hybrid Rule + Machine Learning Next-Best-Action and Product Recommendation Engine.
No paid external APIs; runs completely locally.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from database.database import query_df


def determine_next_best_action(customer_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Determine the optimal next-best-action using machine learning outputs
    (churn probability, customer segment, health score, spend velocity).
    Returns dict with:
    - title
    - description
    - channel
    - priority ('Urgent', 'High', 'Medium', 'Standard')
    - expected_impact
    """
    churn_prob = float(customer_data.get("churn_probability", 0.2))
    health_score = float(customer_data.get("health_score", 75.0))
    monetary_total = float(customer_data.get("monetary_total", 0.0))
    segment = str(customer_data.get("customer_segment", "Growth Opportunity"))
    tenure_days = float(customer_data.get("tenure_days", 365))
    recency_days = float(customer_data.get("recency_days", 30))
    support_tickets = float(customer_data.get("support_tickets", 0))
    negative_interactions = float(customer_data.get("negative_interactions", 0))

    # Decision Matrix:
    # 1. Critical Escalation: High negative feedback or unresolved complaints
    if negative_interactions >= 2 or (support_tickets >= 3 and health_score < 50):
        return {
            "title": "Dispatch Senior Escalation Specialist & Executive CS Review",
            "description": "Customer has logged multiple critical support issues or negative feedback. Executive outreach is required within 24 hours to review SLAs and offer technical remediation.",
            "channel": "Direct Phone Call & Dedicated Meeting",
            "priority": "Urgent",
            "expected_impact": "Prevents immediate contract cancellation & restores trust"
        }

    # 2. High Churn Risk on High Revenue Account
    if churn_prob >= 0.60 and monetary_total >= 4000.0:
        return {
            "title": "Schedule Executive Retention Call & Value Realization Review",
            "description": "High-value account displaying severe churn indicators. Arrange a call with VP of Customer Success to present roadmap updates, renewal incentives, and custom concession options.",
            "channel": "Executive Video Conference",
            "priority": "Urgent",
            "expected_impact": f"Protects ${monetary_total:,.2f} in historical revenue"
        }

    # 3. High Churn Risk on Low Revenue Account
    if churn_prob >= 0.60:
        return {
            "title": "Enroll in Automated Reactivation Sequence & Discount Offer",
            "description": "Trigger an automated re-engagement email sequence featuring a 20% renewal discount and a survey requesting feedback on product challenges.",
            "channel": "Automated Email Drip",
            "priority": "High",
            "expected_impact": "Low-touch customer reactivation"
        }

    # 4. New Customer with High Engagement (<90 days tenure)
    if tenure_days <= 120 and health_score >= 65:
        return {
            "title": "Schedule 1-on-1 Onboarding & Feature Walkthrough Demo",
            "description": "New account showing strong early adoption. Guide customer through advanced workflow integrations and configure team administrator seats to maximize stickiness.",
            "channel": "Live Onboarding Demo",
            "priority": "High",
            "expected_impact": "Accelerates Time-to-Value and establishes long-term retention"
        }

    # 5. High Engagement but High Recency (Inactive recent buyer)
    if recency_days >= 60 and health_score >= 60:
        return {
            "title": "Send Personalized Reactivation Offer & Upgrade Incentive",
            "description": "Customer has not purchased recently despite strong historical engagement. Present an exclusive cross-sell bundle or promotional voucher tailored to their top category.",
            "channel": "Direct Sales Email",
            "priority": "Medium",
            "expected_impact": "Revives purchasing cadence before lapse"
        }

    # 6. High-Value Loyal Segment
    if segment == "High-Value Loyal" or (monetary_total >= 10000.0 and health_score >= 80):
        return {
            "title": "Invite to VIP Client Advisory Board & Offer Enterprise Perks",
            "description": "Champion account with excellent health and frequent spend. Offer early beta feature access, co-marketing case study spotlight, and executive sponsorship.",
            "channel": "VIP Account Management",
            "priority": "Standard",
            "expected_impact": "Strengthens advocacy and unlocks multi-year enterprise expansion"
        }

    # 7. Growth Opportunity Segment
    if segment == "Growth Opportunity" or health_score >= 70:
        return {
            "title": "Propose Cross-Sell Add-on Bundle & Expansion Volume Discount",
            "description": "Account is thriving and ready for expansion. Present the Revenue Intelligence and AI Lead Scoring add-on packages with tiered volume pricing.",
            "channel": "Account Manager Meeting",
            "priority": "Medium",
            "expected_impact": "Increases Net Revenue Retention (NRR) by 15-25%"
        }

    # 8. Default: Quarterly Health Check-in
    return {
        "title": "Trigger Automated Quarterly Product Digest & Health Check",
        "description": "Account is operating steadily. Share recent release notes, security updates, and a light 1-click satisfaction poll.",
        "channel": "Monthly Email Digest",
        "priority": "Standard",
        "expected_impact": "Maintains continuous brand awareness"
    }


def recommend_products_for_customer(customer_id: str, limit: int = 3, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Recommend complementary products for a customer based on past purchase categories
    and popular enterprise add-ons they have not purchased yet.
    """
    # 1. Fetch products already purchased by this customer
    purchased_query = """
        SELECT DISTINCT product_id
        FROM transactions
        WHERE customer_id = ?;
    """
    purchased_df = query_df(purchased_query, (customer_id,), db_path=db_path)
    purchased_ids = set(purchased_df["product_id"].tolist()) if not purchased_df.empty else set()

    # 2. Fetch customer top purchased category
    cat_query = """
        SELECT p.category, COUNT(*) as cnt
        FROM transactions t
        JOIN products p ON t.product_id = p.product_id
        WHERE t.customer_id = ?
        GROUP BY p.category
        ORDER BY cnt DESC
        LIMIT 1;
    """
    top_cat_df = query_df(cat_query, (customer_id,), db_path=db_path)
    top_cat = top_cat_df["category"].iloc[0] if not top_cat_df.empty else "Add-on Modules"

    # 3. Recommend complementary products not yet owned
    all_prods = query_df("SELECT * FROM products;", db_path=db_path)
    if all_prods.empty:
        return []

    # Prefer products in Add-on Modules or Professional Services if customer already has SaaS
    candidates = all_prods[~all_prods["product_id"].isin(purchased_ids)].copy()
    if candidates.empty:
        candidates = all_prods.copy()

    # Complementary logic: if bought SaaS, recommend Add-on or Training
    category_weights = {
        "Add-on Modules": 1.5,
        "Training & Support": 1.3,
        "Professional Services": 1.2,
        "SaaS Subscriptions": 1.0,
        "Hardware & Infrastructure": 0.8
    }
    candidates["weight"] = candidates["category"].map(lambda c: category_weights.get(c, 1.0))
    recommended = candidates.sort_values(by="weight", ascending=False).head(limit)

    results = []
    for _, row in recommended.iterrows():
        cat = row["category"]
        if cat == "Add-on Modules":
            reason = "High affinity complement to expand your existing software capabilities."
        elif cat == "Training & Support":
            reason = "Accelerates team onboarding and unlocks priority support SLAs."
        elif cat == "Professional Services":
            reason = "Tailored architecture review to guarantee seamless scalability."
        else:
            reason = "Top-rated complementary solution for organizations in your industry."

        results.append({
            "product_id": row["product_id"],
            "product_name": row["product_name"],
            "category": row["category"],
            "price": float(row["price"]),
            "reason": reason,
            "potential_value": float(row["price"])
        })

    return results


def generate_batch_recommendations(
    customers_features_df: pd.DataFrame,
    predictions_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate next-best-action recommendations for an entire customer dataset.
    Returns DataFrame matching SQLite 'predictions' schema:
    [customer_id, churn_probability, customer_segment, health_score, predicted_revenue, recommended_action, prediction_date]
    """
    merged = customers_features_df.merge(predictions_df, on="customer_id", how="left")
    actions = []
    
    for _, row in merged.iterrows():
        action_dict = determine_next_best_action(row.to_dict())
        actions.append(action_dict["title"])

    merged["recommended_action"] = actions
    return merged
