"""
utils/data_generator.py
=======================
Realistic synthetic data generation engine for CustomerPulse CRM.
Generates:
- 5,000 Customers with realistic personas and behavioral profiles
- 25,000+ Transactions adhering to Pareto spend distributions
- 2,000 Leads with realistic source/engagement conversion dynamics
- 10,000 Interactions (Support tickets, Demos, Calls, Meetings)
- 30 SaaS & Enterprise Products
"""

import os
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from database.database import get_connection, get_db_path, init_db, insert_df, query_df

# Fixed reference "current" date for consistent relative recency & tenure
CURRENT_DATE = datetime(2026, 9, 1)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

INDUSTRIES = [
    "SaaS & Cloud Software",
    "Financial Services & FinTech",
    "Healthcare & Life Sciences",
    "Manufacturing & Industrial",
    "Retail & E-Commerce",
    "Professional Services & Consulting",
    "Telecommunications",
    "Media & Entertainment",
]

LOCATIONS = [
    "New York, NY", "San Francisco, CA", "Austin, TX", "Chicago, IL",
    "Boston, MA", "Seattle, WA", "Denver, CO", "Atlanta, GA",
    "Toronto, Canada", "London, UK", "Berlin, Germany", "Sydney, Australia"
]

LEAD_SOURCES = [
    "Referral", "Webinar", "Organic Search", "LinkedIn Campaign",
    "Partner Channel", "Inbound Demo Request", "Cold Outreach"
]

SALESPERSONS = [
    "Sarah Jenkins", "Alex Rivera", "Michael Chang",
    "Emily Watson", "David Kim", "Priya Patel"
]

PRODUCT_CATALOG = [
    # SaaS Core
    ("PROD-001", "Pulse Core - Starter Tier", "SaaS Subscriptions", 49.0),
    ("PROD-002", "Pulse Core - Professional Tier", "SaaS Subscriptions", 149.0),
    ("PROD-003", "Pulse Core - Enterprise Annual", "SaaS Subscriptions", 1200.0),
    ("PROD-004", "Pulse Enterprise Suite", "SaaS Subscriptions", 4800.0),
    ("PROD-005", "Pulse Global Multi-Tenant", "SaaS Subscriptions", 12500.0),
    # Add-on Modules
    ("PROD-006", "Revenue Intelligence Module", "Add-on Modules", 299.0),
    ("PROD-007", "Predictive Churn Radar", "Add-on Modules", 399.0),
    ("PROD-008", "AI Lead Scoring Engine", "Add-on Modules", 350.0),
    ("PROD-009", "Automated Playbook Executor", "Add-on Modules", 250.0),
    ("PROD-010", "Custom Pipeline Builder", "Add-on Modules", 199.0),
    ("PROD-011", "Executive Analytics Suite", "Add-on Modules", 450.0),
    ("PROD-012", "Data Enrichment Connector", "Add-on Modules", 180.0),
    # Professional Services
    ("PROD-013", "Fast-Track Onboarding Package", "Professional Services", 2500.0),
    ("PROD-014", "Enterprise Architecture Review", "Professional Services", 6000.0),
    ("PROD-015", "Custom API Integration Service", "Professional Services", 4500.0),
    ("PROD-016", "Data Migration & Cleanse Sprint", "Professional Services", 3500.0),
    ("PROD-017", "Revenue Ops Consulting (40 hrs)", "Professional Services", 8000.0),
    ("PROD-018", "Quarterly Business Review Retainer", "Professional Services", 5000.0),
    # Training & Support
    ("PROD-019", "Dedicated TAM Support (Monthly)", "Training & Support", 950.0),
    ("PROD-020", "24/7 Priority SLA Upgrade", "Training & Support", 650.0),
    ("PROD-021", "Sales Team Certification Bootcamp", "Training & Support", 1800.0),
    ("PROD-022", "Admin Masterclass Workshop", "Training & Support", 1200.0),
    ("PROD-023", "Continuous Learning Portal Access", "Training & Support", 300.0),
    # Hardware & Appliances
    ("PROD-024", "Pulse On-Premise Gateway Server", "Hardware & Infrastructure", 7500.0),
    ("PROD-025", "Edge Telemetry Collector Box", "Hardware & Infrastructure", 2200.0),
    ("PROD-026", "High-Security HSM Key Module", "Hardware & Infrastructure", 3400.0),
    ("PROD-027", "Rackmount Analytics Accelerator", "Hardware & Infrastructure", 9800.0),
    ("PROD-028", "Redundant Power Backup Unit", "Hardware & Infrastructure", 1100.0),
    ("PROD-029", "Pulse Kiosk Field Terminal", "Hardware & Infrastructure", 1600.0),
    ("PROD-030", "Secure Enclave Hardware Vault", "Hardware & Infrastructure", 11500.0),
]

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
    "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
    "Matthew", "Margaret", "Anthony", "Betty", "Mark", "Sandra", "Donald", "Ashley",
    "Steven", "Dorothy", "Paul", "Kimberly", "Andrew", "Emily", "Joshua", "Donna",
    "Kenneth", "Michelle", "Kevin", "Carol", "Brian", "Amanda", "George", "Melissa",
    "Edward", "Deborah", "Ronald", "Stephanie", "Timothy", "Rebecca", "Jason", "Sharon",
    "Jeffrey", "Laura", "Ryan", "Cynthia", "Jacob", "Kathleen", "Gary", "Amy"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
    "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
    "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz", "Parker"
]

COMPANY_PREFIXES = [
    "Apex", "Nexus", "Vertex", "Quantum", "Hyperion", "Synapse", "Stratosphere", "Vanguard",
    "Omni", "Prism", "Aegis", "Catalyst", "Beacon", "Pinnacle", "Acro", "Meridian", "Solstice",
    "Horizon", "Cognitive", "Elevate", "Velocity", "Aura", "Zenith", "CoreTech", "Starlight"
]

COMPANY_SUFFIXES = [
    "Technologies", "Solutions", "Dynamics", "Systems", "Enterprises", "Innovations",
    "Global", "Analytics", "Capital", "Health", "Logistics", "Digital", "Labs", "Partners"
]


def generate_synthetic_data(
    num_customers: int = 5000,
    num_transactions: int = 25000,
    num_leads: int = 2000,
    num_interactions: int = 10000,
    seed: int = 42,
    db_path: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Generate complete synthetic dataset for CustomerPulse CRM with realistic correlations:
    - Persona-based purchasing frequency & amounts
    - Real support ticket outcome & recency correlation to churn
    - Lead source and engagement scores correlated to conversion
    """
    np.random.seed(seed)
    random.seed(seed)
    os.makedirs(DATA_DIR, exist_ok=True)
    db_file = db_path or get_db_path()
    init_db(db_file)

    # 1. GENERATE PRODUCTS
    products_df = pd.DataFrame(PRODUCT_CATALOG, columns=["product_id", "product_name", "category", "price"])

    # 2. GENERATE CUSTOMERS
    # Assign latent personas:
    # 0: High-Value Loyal (20%) - low churn, high spend, frequent
    # 1: Mid-Market Steady (30%) - moderate spend & frequency
    # 2: Growth Candidate (20%) - newer, high engagement, increasing frequency
    # 3: At-Risk / Problematic (15%) - high complaints, long recency, likely churn
    # 4: Dormant / Low Engagement (15%) - low spend, single purchase, inactive
    persona_choices = np.random.choice([0, 1, 2, 3, 4], size=num_customers, p=[0.20, 0.30, 0.20, 0.15, 0.15])

    customers_records = []
    used_emails = set()

    for i in range(num_customers):
        cid = f"CUST-{i+1:05d}"
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        name = f"{first} {last}"
        company = f"{random.choice(COMPANY_PREFIXES)} {random.choice(COMPANY_SUFFIXES)}"
        industry = random.choice(INDUSTRIES)
        location = random.choice(LOCATIONS)
        
        domain = company.lower().replace(" ", "").replace("&", "")[:12] + ".com"
        email = f"{first.lower()}.{last.lower()}{i+1}@{domain}"
        while email in used_emails:
            email = f"{first.lower()}.{last.lower()}{random.randint(100, 999)}@{domain}"
        used_emails.add(email)
        
        phone = f"+1-{random.randint(201, 989)}-{random.randint(200, 899)}-{random.randint(1000, 9999)}"

        persona = persona_choices[i]
        # Tenure logic based on persona
        if persona == 0:  # Loyal: longer tenure
            tenure_days = np.random.randint(300, 1800)
        elif persona == 2:  # Growth: newer
            tenure_days = np.random.randint(60, 450)
        elif persona == 3:  # At-risk: moderate tenure
            tenure_days = np.random.randint(200, 1200)
        else:
            tenure_days = np.random.randint(90, 1500)

        cust_since = (CURRENT_DATE - timedelta(days=int(tenure_days))).strftime("%Y-%m-%d")

        # Preliminary status; will be reconciled with transaction recency
        if persona == 0:
            status = "Active"
        elif persona == 1:
            status = "Active" if np.random.rand() > 0.15 else "At-Risk"
        elif persona == 2:
            status = "Active"
        elif persona == 3:
            status = "At-Risk" if np.random.rand() > 0.40 else "Churned"
        else:  # persona 4
            status = "Inactive" if np.random.rand() > 0.50 else "Churned"

        customers_records.append({
            "customer_id": cid,
            "name": name,
            "company": company,
            "industry": industry,
            "email": email,
            "phone": phone,
            "location": location,
            "customer_since": cust_since,
            "customer_status": status,
            "_persona": persona,
            "_tenure_days": tenure_days
        })

    customers_df = pd.DataFrame(customers_records)

    # 3. GENERATE TRANSACTIONS
    # Target ~25,000 transactions distributed according to persona
    # High-Value: 8-22 txns
    # Mid-Market: 4-10 txns
    # Growth: 2-7 txns
    # At-Risk: 2-5 txns (clustered in early tenure, none recent)
    # Dormant: 1-2 txns
    tx_records = []
    tx_counter = 1
    
    prod_ids = products_df["product_id"].values
    prod_prices = dict(zip(products_df["product_id"], products_df["price"]))
    prod_categories = dict(zip(products_df["product_id"], products_df["category"]))

    # High-value products for enterprise
    high_tier_prods = [p for p, c in prod_categories.items() if c in ("SaaS Subscriptions", "Professional Services") and prod_prices[p] >= 1000.0]
    mid_tier_prods = [p for p, c in prod_categories.items() if prod_prices[p] < 2000.0]

    for idx, row in customers_df.iterrows():
        cid = row["customer_id"]
        persona = row["_persona"]
        tenure = row["_tenure_days"]
        start_date = CURRENT_DATE - timedelta(days=int(tenure))

        if persona == 0:  # High-Value
            num_tx = np.random.randint(7, 20)
            max_recency = np.random.randint(5, 60)  # very recent
        elif persona == 1:  # Mid-Market
            num_tx = np.random.randint(3, 9)
            max_recency = np.random.randint(20, 110)
        elif persona == 2:  # Growth
            num_tx = np.random.randint(2, 6)
            max_recency = np.random.randint(5, 45)
        elif persona == 3:  # At-Risk
            num_tx = np.random.randint(2, 5)
            max_recency = np.random.randint(180, 400)  # stale!
        else:  # Dormant
            num_tx = np.random.randint(1, 3)
            max_recency = np.random.randint(220, 500)  # inactive

        # Generate timestamps between start_date and (CURRENT_DATE - max_recency)
        latest_allowed = max(start_date + timedelta(days=10), CURRENT_DATE - timedelta(days=int(max_recency)))
        if latest_allowed <= start_date:
            latest_allowed = start_date + timedelta(days=15)

        total_span = (latest_allowed - start_date).days
        if total_span <= 0:
            tx_offsets = [0] * num_tx
        else:
            tx_offsets = sorted(np.random.randint(0, total_span, size=num_tx))

        for offset in tx_offsets:
            tx_date = (start_date + timedelta(days=int(offset))).strftime("%Y-%m-%d")
            
            # Select product based on persona
            if persona == 0 and np.random.rand() > 0.4:
                pid = random.choice(high_tier_prods)
                qty = np.random.choice([1, 2, 5, 10], p=[0.5, 0.3, 0.15, 0.05])
            else:
                pid = random.choice(mid_tier_prods)
                qty = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])

            unit_price = prod_prices[pid]
            # Small random variation / volume discount (up to 10%)
            discount = 1.0 - (0.05 if qty > 2 else 0.0)
            amount = round(unit_price * qty * discount, 2)

            tx_records.append({
                "transaction_id": f"TXN-{tx_counter:06d}",
                "customer_id": cid,
                "product_id": pid,
                "quantity": int(qty),
                "amount": float(amount),
                "transaction_date": tx_date
            })
            tx_counter += 1

    transactions_df = pd.DataFrame(tx_records)

    # 4. GENERATE INTERACTIONS
    # Target ~10,000 interactions across customers
    # Support tickets with Negative outcomes correlate strongly with At-Risk persona
    int_records = []
    int_counter = 1
    int_types = ["Phone Call", "Email", "Meeting", "Demo", "Support Request", "Follow-up"]

    for idx, row in customers_df.iterrows():
        cid = row["customer_id"]
        persona = row["_persona"]
        tenure = row["_tenure_days"]
        start_date = CURRENT_DATE - timedelta(days=int(tenure))

        # Number of interactions per customer
        if persona == 0:  # Loyal
            k = np.random.choice([2, 3, 4], p=[0.5, 0.35, 0.15])
        elif persona == 2:  # Growth
            k = np.random.choice([2, 3, 5], p=[0.4, 0.4, 0.2])
        elif persona == 3:  # At-risk
            k = np.random.choice([2, 3, 4], p=[0.3, 0.4, 0.3])
        else:
            k = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])

        for _ in range(k):
            days_ago = np.random.randint(1, min(max(30, tenure), 500))
            int_date = (CURRENT_DATE - timedelta(days=int(days_ago))).strftime("%Y-%m-%d")

            if persona == 3:  # At-risk: more support complaints and negative/pending outcomes
                itype = np.random.choice(["Support Request", "Phone Call", "Email"], p=[0.6, 0.25, 0.15])
                outcome = np.random.choice(["Negative", "Action Required", "Pending", "Resolved"], p=[0.45, 0.30, 0.15, 0.10])
                notes = random.choice([
                    "Customer reported repeated latency issues in production.",
                    "Escalated billing discrepancy and requested immediate contract renegotiation.",
                    "User expressed dissatisfaction with recent feature update and integration delays.",
                    "Account champion left the company; new leadership evaluating competitors.",
                    "Support ticket regarding missing analytics data during executive presentation."
                ])
            elif persona == 0:  # Loyal
                itype = random.choice(["Meeting", "Demo", "Phone Call", "Follow-up"])
                outcome = np.random.choice(["Positive", "Resolved", "Neutral"], p=[0.75, 0.20, 0.05])
                notes = random.choice([
                    "Quarterly Business Review completed with stellar feedback.",
                    "Executive sponsor confirmed budget approval for annual renewal.",
                    "Discussed expansion into EMEA division; demo scheduled.",
                    "Customer praised customer success team response time and roadmap alignment.",
                    "Product advisory council attendance confirmed."
                ])
            elif persona == 2:  # Growth
                itype = random.choice(["Demo", "Follow-up", "Meeting", "Email"])
                outcome = np.random.choice(["Positive", "Action Required", "Neutral"], p=[0.65, 0.25, 0.10])
                notes = random.choice([
                    "Customer engaged actively during AI Lead Scoring demo; requested pricing quote.",
                    "Followed up on pilot milestone; positive sentiment across engineering team.",
                    "Requested security architecture whitepaper for upcoming enterprise tier upgrade."
                ])
            else:
                itype = random.choice(int_types)
                outcome = np.random.choice(["Neutral", "Resolved", "Positive", "Negative"], p=[0.4, 0.3, 0.2, 0.1])
                notes = "Standard check-in touchpoint regarding account health and general usage."

            int_records.append({
                "interaction_id": f"INT-{int_counter:06d}",
                "customer_id": cid,
                "interaction_type": itype,
                "interaction_date": int_date,
                "notes": notes,
                "outcome": outcome
            })
            int_counter += 1

    interactions_df = pd.DataFrame(int_records)

    # Reconcile customer status based on actual recency:
    last_tx_series = transactions_df.groupby("customer_id")["transaction_date"].max()
    for idx, row in customers_df.iterrows():
        cid = row["customer_id"]
        last_date_str = last_tx_series.get(cid, None)
        if last_date_str:
            days_since = (CURRENT_DATE - datetime.strptime(last_date_str, "%Y-%m-%d")).days
            if days_since > 180:
                customers_df.at[idx, "customer_status"] = "Churned" if row["_persona"] == 3 else "Inactive"
            elif days_since > 90 or row["_persona"] == 3:
                customers_df.at[idx, "customer_status"] = "At-Risk"
            else:
                customers_df.at[idx, "customer_status"] = "Active"

    # Drop internal helper columns before saving
    customers_df_clean = customers_df.drop(columns=["_persona", "_tenure_days"])

    # 5. GENERATE LEADS (2,000 leads)
    lead_records = []
    for i in range(num_leads):
        lid = f"LEAD-{i+1:05d}"
        company = f"{random.choice(COMPANY_PREFIXES)} {random.choice(COMPANY_SUFFIXES)}"
        industry = random.choice(INDUSTRIES)
        source = random.choice(LEAD_SOURCES)
        salesperson = random.choice(SALESPERSONS)

        # Correlate budget and engagement
        if source in ("Referral", "Inbound Demo Request"):
            budget = float(np.random.choice([15000, 35000, 60000, 120000], p=[0.3, 0.4, 0.2, 0.1]))
            engagement_score = float(np.clip(np.random.normal(78, 12), 30, 99))
            status_choices = ["Won", "Proposal Sent", "Negotiation", "Qualified", "Lost"]
            status = np.random.choice(status_choices, p=[0.35, 0.25, 0.20, 0.15, 0.05])
        elif source in ("Webinar", "LinkedIn Campaign"):
            budget = float(np.random.choice([8000, 20000, 45000], p=[0.5, 0.35, 0.15]))
            engagement_score = float(np.clip(np.random.normal(62, 15), 15, 95))
            status_choices = ["Qualified", "Contacted", "Proposal Sent", "Won", "Lost"]
            status = np.random.choice(status_choices, p=[0.30, 0.25, 0.20, 0.15, 0.10])
        else:  # Cold Outreach / Organic
            budget = float(np.random.choice([5000, 12000, 25000], p=[0.6, 0.3, 0.1]))
            engagement_score = float(np.clip(np.random.normal(42, 16), 5, 85))
            status_choices = ["New", "Contacted", "Lost", "Qualified", "Won"]
            status = np.random.choice(status_choices, p=[0.35, 0.30, 0.25, 0.07, 0.03])

        created_days_ago = np.random.randint(2, 360)
        created_date = (CURRENT_DATE - timedelta(days=int(created_days_ago))).strftime("%Y-%m-%d")

        lead_records.append({
            "lead_id": lid,
            "company": company,
            "industry": industry,
            "source": source,
            "budget": round(budget, 2),
            "engagement_score": round(engagement_score, 1),
            "status": status,
            "created_date": created_date,
            "salesperson": salesperson
        })

    leads_df = pd.DataFrame(lead_records)

    # 6. SAVE TO CSV
    products_df.to_csv(os.path.join(DATA_DIR, "products.csv"), index=False)
    customers_df_clean.to_csv(os.path.join(DATA_DIR, "customers.csv"), index=False)
    transactions_df.to_csv(os.path.join(DATA_DIR, "transactions.csv"), index=False)
    interactions_df.to_csv(os.path.join(DATA_DIR, "interactions.csv"), index=False)
    leads_df.to_csv(os.path.join(DATA_DIR, "leads.csv"), index=False)

    # 7. INSERT INTO SQLITE DB (with foreign key order safety)
    with get_connection(db_file) as conn:
        conn.execute("PRAGMA foreign_keys = OFF;")
        conn.execute("DELETE FROM predictions;")
        conn.execute("DELETE FROM transactions;")
        conn.execute("DELETE FROM interactions;")
        conn.execute("DELETE FROM leads;")
        conn.execute("DELETE FROM customers;")
        conn.execute("DELETE FROM products;")
        conn.execute("PRAGMA foreign_keys = ON;")

    insert_df("products", products_df, if_exists="append", db_path=db_file)
    insert_df("customers", customers_df_clean, if_exists="append", db_path=db_file)
    insert_df("transactions", transactions_df, if_exists="append", db_path=db_file)
    insert_df("interactions", interactions_df, if_exists="append", db_path=db_file)
    insert_df("leads", leads_df, if_exists="append", db_path=db_file)

    # Return dictionary of DataFrames
    return {
        "products": products_df,
        "customers": customers_df_clean,
        "transactions": transactions_df,
        "interactions": interactions_df,
        "leads": leads_df
    }


def load_or_generate_data(db_path: Optional[str] = None, force_regenerate: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Ensure database and CSV data are populated.
    If database has fewer than 100 customers or force_regenerate=True, triggers generator.
    """
    db_file = db_path or get_db_path()
    init_db(db_file)

    if not force_regenerate:
        try:
            cust_count = query_df("SELECT COUNT(*) as cnt FROM customers", db_path=db_file)["cnt"].iloc[0]
            if cust_count >= 500:
                return {
                    "products": query_df("SELECT * FROM products", db_path=db_file),
                    "customers": query_df("SELECT * FROM customers", db_path=db_file),
                    "transactions": query_df("SELECT * FROM transactions", db_path=db_file),
                    "interactions": query_df("SELECT * FROM interactions", db_path=db_file),
                    "leads": query_df("SELECT * FROM leads", db_path=db_file)
                }
        except Exception:
            pass

    return generate_synthetic_data(db_path=db_file)
