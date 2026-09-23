"""
crm/interactions.py
===================
Customer interaction logging and chronological timeline generation.
Tracks touchpoints: Calls, Emails, Meetings, Demos, Support Requests, and Follow-ups.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from database.database import execute_query, query_df


def log_interaction(
    customer_id: str,
    interaction_type: str,
    outcome: str,
    notes: str,
    interaction_date: Optional[str] = None,
    db_path: Optional[str] = None
) -> Tuple[bool, str, Optional[str]]:
    """
    Log an interaction for a customer with outcome and notes.
    """
    valid_types = ['Phone Call', 'Email', 'Meeting', 'Demo', 'Support Request', 'Follow-up']
    valid_outcomes = ['Positive', 'Neutral', 'Negative', 'Resolved', 'Pending', 'Action Required']

    if interaction_type not in valid_types:
        return False, f"Invalid interaction type. Must be one of: {', '.join(valid_types)}", None

    if outcome not in valid_outcomes:
        return False, f"Invalid outcome. Must be one of: {', '.join(valid_outcomes)}", None

    # Check that customer exists
    cust_exists = query_df("SELECT customer_id FROM customers WHERE customer_id = ?", (customer_id,), db_path=db_path)
    if cust_exists.empty:
        return False, f"Customer {customer_id} does not exist.", None

    count_df = query_df("SELECT COUNT(*) as cnt FROM interactions", db_path=db_path)
    new_num = int(count_df["cnt"].iloc[0] or 0) + 1
    new_iid = f"INT-{new_num:06d}"
    
    # Ensure ID uniqueness
    while not query_df("SELECT interaction_id FROM interactions WHERE interaction_id = ?", (new_iid,), db_path=db_path).empty:
        new_num += 1
        new_iid = f"INT-{new_num:06d}"

    date_str = interaction_date or datetime.now().strftime("%Y-%m-%d")

    sql = """
        INSERT INTO interactions (
            interaction_id, customer_id, interaction_type, interaction_date, notes, outcome
        ) VALUES (?, ?, ?, ?, ?, ?);
    """
    try:
        execute_query(sql, (new_iid, customer_id, interaction_type, date_str, notes.strip(), outcome), db_path=db_path)
        return True, f"Interaction {new_iid} successfully logged.", new_iid
    except Exception as e:
        return False, f"Database error logging interaction: {str(e)}", None


def get_customer_timeline(customer_id: str, db_path: Optional[str] = None) -> pd.DataFrame:
    """
    Return chronological interaction timeline for a specific customer.
    """
    sql = """
        SELECT
            interaction_id,
            interaction_date,
            interaction_type,
            outcome,
            notes
        FROM interactions
        WHERE customer_id = ?
        ORDER BY interaction_date DESC;
    """
    return query_df(sql, (customer_id,), db_path=db_path)


def get_interaction_stats(db_path: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Return global breakdown of interactions by type and by outcome.
    """
    by_type = query_df("""
        SELECT interaction_type, COUNT(*) as count
        FROM interactions
        GROUP BY interaction_type
        ORDER BY count DESC;
    """, db_path=db_path)

    by_outcome = query_df("""
        SELECT outcome, COUNT(*) as count
        FROM interactions
        GROUP BY outcome
        ORDER BY count DESC;
    """, db_path=db_path)

    return {"by_type": by_type, "by_outcome": by_outcome}
