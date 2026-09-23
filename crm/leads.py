"""
crm/leads.py
============
Lead pipeline management, status tracking, qualification, and customer conversion.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from database.database import execute_query, query_df
from models.lead_scoring import LeadScorer
from utils.helpers import validate_positive_number


def get_all_leads(
    salesperson: Optional[str] = None,
    status: Optional[str] = None,
    source: Optional[str] = None,
    scored: bool = True,
    limit: int = 1500,
    db_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch leads with optional filters, sorted by lead score if scored=True.
    """
    conditions = ["1=1"]
    params: List[Any] = []

    if salesperson and salesperson != "All Salespeople":
        conditions.append("salesperson = ?")
        params.append(salesperson)

    if status and status != "All Statuses":
        conditions.append("status = ?")
        params.append(status)

    if source and source != "All Sources":
        conditions.append("source = ?")
        params.append(source)

    where_clause = " AND ".join(conditions)

    sql = f"""
        SELECT
            lead_id,
            company,
            industry,
            source,
            budget,
            engagement_score,
            status,
            created_date,
            salesperson
        FROM leads
        WHERE {where_clause}
        ORDER BY created_date DESC
        LIMIT {int(limit)};
    """

    df = query_df(sql, tuple(params), db_path=db_path)

    if scored and not df.empty:
        scorer = LeadScorer()
        if scorer.pipeline is not None:
            try:
                df = scorer.score_leads(df)
                df = df.sort_values(by="lead_score", ascending=False)
            except Exception:
                df["lead_score"] = df["engagement_score"]
                df["priority"] = "Medium Priority"
                df["recommended_action"] = "Follow up with client"
        else:
            # Fallback heuristic based on engagement
            df["lead_score"] = df["engagement_score"]
            df["priority"] = df["lead_score"].apply(
                lambda s: "High Priority" if s >= 80 else ("Medium Priority" if s >= 50 else "Low Priority")
            )
            df["recommended_action"] = "Follow up with client"

    return df


def create_lead(
    company: str,
    industry: str,
    source: str,
    budget: float,
    engagement_score: float,
    status: str = "New",
    salesperson: str = "Unassigned",
    db_path: Optional[str] = None
) -> Tuple[bool, str, Optional[str]]:
    """Insert a new sales lead."""
    if not company or not industry or not source:
        return False, "Company, Industry, and Source are required.", None

    if not validate_positive_number(budget):
        return False, "Budget must be a non-negative number.", None

    engagement = float(np.clip(engagement_score, 0.0, 100.0)) if "np" in globals() else max(0.0, min(100.0, float(engagement_score)))

    count_df = query_df("SELECT COUNT(*) as cnt FROM leads", db_path=db_path)
    new_num = int(count_df["cnt"].iloc[0] or 0) + 1
    new_lid = f"LEAD-{new_num:05d}"
    today_str = datetime.now().strftime("%Y-%m-%d")

    sql = """
        INSERT INTO leads (
            lead_id, company, industry, source, budget, engagement_score, status, created_date, salesperson
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    try:
        execute_query(sql, (
            new_lid, company.strip(), industry.strip(), source.strip(),
            float(budget), engagement, status, today_str, salesperson.strip()
        ), db_path=db_path)
        return True, f"Lead {new_lid} created successfully.", new_lid
    except Exception as e:
        return False, f"Failed to create lead: {str(e)}", None


def update_lead_status(lead_id: str, new_status: str, db_path: Optional[str] = None) -> Tuple[bool, str]:
    """Update sales stage of a lead."""
    valid_statuses = ['New', 'Contacted', 'Qualified', 'Proposal Sent', 'Negotiation', 'Won', 'Lost']
    if new_status not in valid_statuses:
        return False, f"Invalid status. Must be one of: {', '.join(valid_statuses)}"

    sql = "UPDATE leads SET status = ? WHERE lead_id = ?;"
    try:
        rows = execute_query(sql, (new_status, lead_id), db_path=db_path)
        if rows > 0:
            return True, f"Lead {lead_id} updated to '{new_status}'."
        return False, "Lead ID not found."
    except Exception as e:
        return False, f"Error updating lead status: {str(e)}"


def convert_lead_to_customer(
    lead_id: str,
    contact_name: str,
    email: str,
    phone: str,
    location: str,
    db_path: Optional[str] = None
) -> Tuple[bool, str, Optional[str]]:
    """
    Convert a qualified lead into an active customer account.
    Marks the lead as 'Won' and registers new customer in SQLite.
    """
    from crm.customers import create_customer
    from crm.interactions import log_interaction

    lead_df = query_df("SELECT * FROM leads WHERE lead_id = ?", (lead_id,), db_path=db_path)
    if lead_df.empty:
        return False, "Lead not found.", None

    lead = lead_df.iloc[0].to_dict()

    success, msg, new_cid = create_customer(
        name=contact_name,
        company=lead["company"],
        industry=lead["industry"],
        email=email,
        phone=phone,
        location=location,
        customer_status="Active",
        db_path=db_path
    )

    if not success:
        return False, f"Could not create customer: {msg}", None

    # Update lead status to Won
    update_lead_status(lead_id, "Won", db_path=db_path)

    # Log initial interaction
    log_interaction(
        customer_id=new_cid,
        interaction_type="Meeting",
        outcome="Positive",
        notes=f"Converted from Lead {lead_id} ({lead['source']}). Budget: ${lead['budget']:,.2f}.",
        db_path=db_path
    )

    return True, f"Lead successfully converted to Customer {new_cid}.", new_cid
