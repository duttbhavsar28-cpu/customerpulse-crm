"""
crm/customers.py
================
Customer relationship management operations:
- Full CRUD operations with input validation and parameterized SQL
- Customer 360 profile aggregation (financials, ML scores, history)
- Advanced multi-criteria search and filtering
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from database.database import execute_query, query_df
from utils.helpers import validate_email, validate_phone


def get_all_customers(
    search_query: Optional[str] = None,
    industry: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 1000,
    db_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch customers joined with predictions, with optional search and industry/status filters.
    """
    conditions = ["1=1"]
    params: List[Any] = []

    if search_query:
        term = f"%{search_query.strip()}%"
        conditions.append("(c.name LIKE ? OR c.company LIKE ? OR c.email LIKE ? OR c.customer_id LIKE ?)")
        params.extend([term, term, term, term])

    if industry and industry != "All Industries":
        conditions.append("c.industry = ?")
        params.append(industry)

    if status and status != "All Statuses":
        conditions.append("c.customer_status = ?")
        params.append(status)

    where_clause = " AND ".join(conditions)

    sql = f"""
        SELECT
            c.customer_id,
            c.name,
            c.company,
            c.industry,
            c.email,
            c.phone,
            c.location,
            c.customer_since,
            c.customer_status,
            COALESCE(p.churn_probability, 0.20) as churn_probability,
            COALESCE(p.customer_segment, 'Growth Opportunity') as customer_segment,
            COALESCE(p.health_score, 70.0) as health_score,
            COALESCE(p.predicted_revenue, 0.0) as predicted_revenue,
            COALESCE(p.recommended_action, 'Conduct Account Review') as recommended_action,
            COALESCE(t_agg.total_revenue, 0.0) as total_revenue,
            COALESCE(t_agg.order_count, 0) as total_purchases
        FROM customers c
        LEFT JOIN predictions p ON c.customer_id = p.customer_id
        LEFT JOIN (
            SELECT customer_id, SUM(amount) as total_revenue, COUNT(transaction_id) as order_count
            FROM transactions
            GROUP BY customer_id
        ) t_agg ON c.customer_id = t_agg.customer_id
        WHERE {where_clause}
        ORDER BY c.customer_id ASC
        LIMIT {int(limit)};
    """

    return query_df(sql, tuple(params), db_path=db_path)


def get_customer_profile(customer_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Retrieve comprehensive 360-degree customer profile:
    - Demographic info
    - Financial aggregates (total spend, AOV, order count, last order)
    - Health score and breakdown
    - Churn prediction & risk category
    - Transaction history
    - Interaction timeline
    """
    cust_df = query_df("SELECT * FROM customers WHERE customer_id = ?", (customer_id,), db_path=db_path)
    if cust_df.empty:
        return None

    cust_info = cust_df.iloc[0].to_dict()

    # Financial and transaction metrics
    tx_df = query_df(
        """
        SELECT
            t.transaction_id,
            t.transaction_date,
            p.product_name,
            p.category,
            t.quantity,
            t.amount
        FROM transactions t
        JOIN products p ON t.product_id = p.product_id
        WHERE t.customer_id = ?
        ORDER BY t.transaction_date DESC;
        """,
        (customer_id,),
        db_path=db_path
    )

    total_revenue = float(tx_df["amount"].sum()) if not tx_df.empty else 0.0
    total_purchases = len(tx_df)
    avg_order_value = float(tx_df["amount"].mean()) if total_purchases > 0 else 0.0
    last_purchase = str(tx_df["transaction_date"].iloc[0]) if total_purchases > 0 else "None"

    # Interaction history
    int_df = query_df(
        """
        SELECT
            interaction_id,
            interaction_date,
            interaction_type,
            outcome,
            notes
        FROM interactions
        WHERE customer_id = ?
        ORDER BY interaction_date DESC;
        """,
        (customer_id,),
        db_path=db_path
    )

    # Predictions
    pred_df = query_df(
        "SELECT * FROM predictions WHERE customer_id = ?",
        (customer_id,),
        db_path=db_path
    )

    if not pred_df.empty:
        pred_info = pred_df.iloc[0].to_dict()
    else:
        pred_info = {
            "churn_probability": 0.20,
            "customer_segment": "Growth Opportunity",
            "health_score": 75.0,
            "predicted_revenue": total_revenue * 1.1,
            "recommended_action": "Standard Account Review"
        }

    return {
        "customer": cust_info,
        "financials": {
            "total_revenue": total_revenue,
            "total_purchases": total_purchases,
            "avg_order_value": avg_order_value,
            "last_purchase": last_purchase,
        },
        "intelligence": pred_info,
        "transactions": tx_df,
        "interactions": int_df,
    }


def create_customer(
    name: str,
    company: str,
    industry: str,
    email: str,
    phone: str,
    location: str,
    customer_status: str = "Active",
    db_path: Optional[str] = None
) -> Tuple[bool, str, Optional[str]]:
    """
    Validate and insert a new customer into the database.
    Returns (success: bool, message: str, customer_id: Optional[str]).
    """
    if not name or not company or not email:
        return False, "Customer Name, Company, and Email are required.", None

    if not validate_email(email):
        return False, "Invalid email address format.", None

    if not validate_phone(phone):
        return False, "Invalid phone number format (must contain 7-15 digits).", None

    # Check for existing email
    exists = query_df("SELECT customer_id FROM customers WHERE email = ?", (email.strip(),), db_path=db_path)
    if not exists.empty:
        return False, f"A customer with email '{email}' already exists.", None

    # Generate sequential customer ID
    count_df = query_df("SELECT COUNT(*) as cnt FROM customers", db_path=db_path)
    new_num = int(count_df["cnt"].iloc[0] or 0) + 1
    new_cid = f"CUST-{new_num:05d}"
    
    # Ensure ID uniqueness
    while not query_df("SELECT customer_id FROM customers WHERE customer_id = ?", (new_cid,), db_path=db_path).empty:
        new_num += 1
        new_cid = f"CUST-{new_num:05d}"

    today_str = datetime.now().strftime("%Y-%m-%d")

    sql = """
        INSERT INTO customers (
            customer_id, name, company, industry, email, phone, location, customer_since, customer_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    try:
        execute_query(sql, (
            new_cid, name.strip(), company.strip(), industry.strip(),
            email.strip(), phone.strip(), location.strip(), today_str, customer_status
        ), db_path=db_path)

        # Default prediction entry
        pred_sql = """
            INSERT INTO predictions (
                customer_id, churn_probability, customer_segment, health_score,
                predicted_revenue, recommended_action, prediction_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
        """
        execute_query(pred_sql, (
            new_cid, 0.15, "New Customers", 85.0, 1500.0,
            "Schedule 1-on-1 Onboarding & Feature Walkthrough Demo", today_str
        ), db_path=db_path)

        return True, f"Customer {name} ({new_cid}) successfully registered.", new_cid
    except Exception as e:
        return False, f"Database error creating customer: {str(e)}", None


def update_customer(
    customer_id: str,
    name: str,
    company: str,
    industry: str,
    email: str,
    phone: str,
    location: str,
    customer_status: str,
    db_path: Optional[str] = None
) -> Tuple[bool, str]:
    """Update existing customer details."""
    if not name or not company or not email:
        return False, "Name, Company, and Email cannot be empty."

    if not validate_email(email):
        return False, "Invalid email format."

    if not validate_phone(phone):
        return False, "Invalid phone number."

    sql = """
        UPDATE customers
        SET name = ?, company = ?, industry = ?, email = ?, phone = ?, location = ?, customer_status = ?
        WHERE customer_id = ?;
    """
    try:
        rows = execute_query(sql, (
            name.strip(), company.strip(), industry.strip(), email.strip(),
            phone.strip(), location.strip(), customer_status, customer_id
        ), db_path=db_path)
        if rows > 0:
            return True, "Customer profile successfully updated."
        return False, "Customer ID not found."
    except Exception as e:
        return False, f"Failed to update customer: {str(e)}"


def delete_customer(customer_id: str, db_path: Optional[str] = None) -> Tuple[bool, str]:
    """Delete a customer and related records."""
    try:
        # Transactions and interactions cascade or delete manually
        execute_query("DELETE FROM predictions WHERE customer_id = ?", (customer_id,), db_path=db_path)
        execute_query("DELETE FROM transactions WHERE customer_id = ?", (customer_id,), db_path=db_path)
        execute_query("DELETE FROM interactions WHERE customer_id = ?", (customer_id,), db_path=db_path)
        rows = execute_query("DELETE FROM customers WHERE customer_id = ?", (customer_id,), db_path=db_path)
        if rows > 0:
            return True, f"Customer {customer_id} successfully deleted."
        return False, "Customer not found."
    except Exception as e:
        return False, f"Failed to delete customer: {str(e)}"
