"""
app.py
======
CustomerPulse CRM: Python-only CRM and Customer Revenue Intelligence System.
Built with Streamlit, SQLite, Scikit-learn, Pandas, NumPy, and Plotly.
"""

import os
import sys

# Ensure local customerpulse-crm directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

from database.database import get_db_path, init_db, query_df
from utils.data_generator import load_or_generate_data, generate_synthetic_data
from models.pipeline import run_full_ml_pipeline
from models.segmentation import CustomerSegmentation
from pages.dashboard import render_dashboard
from pages.customers import render_customers
from pages.leads import render_leads
from pages.analytics import render_analytics
from pages.predictions import render_predictions
from crm.sales import get_pipeline_stages, get_sales_velocity
from recommendations.next_best_action import determine_next_best_action, recommend_products_for_customer
from utils.helpers import format_currency, format_percent, get_risk_badge, get_health_badge


# 1. Page Configuration
st.set_page_config(
    page_title="CustomerPulse CRM",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Modern Glassmorphic CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Layout Styling */
    .stApp {
        background-color: #0A0E17;
        color: #F8FAFC;
    }
    
    /* Top Header Bar */
    .top-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 0 20px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 25px;
    }
    
    .brand-title {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366F1 0%, #38BDF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(19, 27, 46, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    div[data-testid="stMetric"] label {
        color: #94A3B8 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
    }
    
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 800 !important;
    }
    
    /* Sidebar Polish */
    section[data-testid="stSidebar"] {
        background-color: #0D1322 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Tabs & Buttons */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        font-weight: 600;
        color: #94A3B8;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(99, 102, 241, 0.15) !important;
        color: #818CF8 !important;
        border-bottom: 2px solid #6366F1 !important;
    }
    
    /* Dataframe Styling */
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def ensure_system_initialized():
    """Ensure database, synthetic data, and ML models are initialized once at startup."""
    data = load_or_generate_data()
    # Check if predictions exist, else run full pipeline
    pred_count = query_df("SELECT COUNT(*) as cnt FROM predictions;")["cnt"].iloc[0]
    if pred_count == 0:
        run_full_ml_pipeline(force_retrain=True)
    return True


ensure_system_initialized()


# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 10px 0 20px 0;">
            <div style="font-size: 1.6rem; font-weight: 800; background: linear-gradient(135deg, #6366F1, #38BDF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                ⚡ CustomerPulse
            </div>
            <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 2px;">
                Revenue Intelligence CRM &bull; v2.0
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    nav_selection = st.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "👥 Customers",
            "🎯 Leads",
            "💼 Sales",
            "📈 Analytics",
            "🔮 Predictions",
            "🧩 Customer Segmentation",
            "⚡ Recommendations",
            "⚙️ Settings"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### 🟢 System Status")
    st.caption("Engine: **Python 3.14 + Scikit-learn**")
    st.caption("Database: **SQLite (WAL Mode)**")
    st.caption("Models: **All 4 Local Models Active**")


# --- VIEW ROUTING ---

# 1. DASHBOARD
if nav_selection == "📊 Dashboard":
    render_dashboard()

# 2. CUSTOMERS
elif nav_selection == "👥 Customers":
    render_customers()

# 3. LEADS
elif nav_selection == "🎯 Leads":
    render_leads()

# 4. SALES
elif nav_selection == "💼 Sales":
    st.markdown("## 💼 Sales Pipeline & Deal Velocity")
    st.markdown("Track deals across the sales lifecycle, monitor pipeline health, and analyze deal progress.")

    stages_df = get_pipeline_stages()
    velocity = get_sales_velocity()

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Total Active Pipeline", format_currency(velocity["total_open_value"], compact=True))
    with s2:
        st.metric("Open Opportunities", f"{velocity['open_deals']} deals")
    with s3:
        st.metric("Win Rate", f"{velocity['win_rate']}%")
    with s4:
        st.metric("Monthly Sales Velocity", format_currency(velocity["sales_velocity_monthly"], compact=True))

    st.markdown("---")

    col_pipe, col_funnel = st.columns([3, 2])
    with col_pipe:
        st.subheader("📊 Deals by Pipeline Stage")
        if not stages_df.empty:
            fig_stg = px.bar(
                stages_df,
                x="stage",
                y="deal_count",
                color="stage",
                text="deal_count",
                labels={"stage": "Pipeline Stage", "deal_count": "Number of Deals"}
            )
            fig_stg.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                height=320,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_stg, use_container_width=True)

    with col_funnel:
        st.subheader("💰 Total Pipeline Value by Stage")
        if not stages_df.empty:
            fig_val = px.pie(
                stages_df,
                names="stage",
                values="total_pipeline_value",
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Tealgrn
            )
            fig_val.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                height=320,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_val, use_container_width=True)

    st.subheader("📋 Pipeline Deals Register")
    leads_pipeline = query_df("""
        SELECT lead_id, company, industry, budget, status, engagement_score, salesperson, created_date
        FROM leads
        ORDER BY budget DESC
        LIMIT 200;
    """)
    leads_pipeline["budget"] = leads_pipeline["budget"].map(lambda b: f"${b:,.2f}")
    st.dataframe(
        leads_pipeline,
        use_container_width=True,
        column_config={
            "lead_id": "Lead ID",
            "company": "Company",
            "industry": "Industry",
            "budget": "Deal Value ($)",
            "status": "Pipeline Stage",
            "engagement_score": st.column_config.ProgressColumn("Engagement", min_value=0, max_value=100),
            "salesperson": "Owner",
            "created_date": "Created Date"
        },
        hide_index=True
    )

# 5. ANALYTICS
elif nav_selection == "📈 Analytics":
    render_analytics()

# 6. PREDICTIONS
elif nav_selection == "🔮 Predictions":
    render_predictions()

# 7. CUSTOMER SEGMENTATION
elif nav_selection == "🧩 Customer Segmentation":
    st.markdown("## 🧩 Unsupervised Customer Segmentation (K-Means)")
    st.markdown(
        "Behavioral customer segmentation powered by K-Means clustering on normalized RFM, tenure, and interaction metrics."
    )

    seg_summary_df = query_df("""
        SELECT
            p.customer_segment as 'Segment',
            COUNT(c.customer_id) as 'Number of Customers',
            ROUND(AVG(t_agg.total_spend), 2) as 'Average Revenue ($)',
            ROUND(AVG(t_agg.avg_order), 2) as 'Average Order Value ($)',
            ROUND(AVG(t_agg.freq), 1) as 'Average Purchase Frequency',
            ROUND(AVG(p.churn_probability) * 100, 1) as 'Average Churn Risk (%)',
            ROUND(AVG(p.health_score), 1) as 'Average Health Score'
        FROM customers c
        JOIN predictions p ON c.customer_id = p.customer_id
        LEFT JOIN (
            SELECT customer_id, SUM(amount) as total_spend, AVG(amount) as avg_order, COUNT(transaction_id) as freq
            FROM transactions
            GROUP BY customer_id
        ) t_agg ON c.customer_id = t_agg.customer_id
        GROUP BY p.customer_segment
        ORDER BY AVG(t_agg.total_spend) DESC;
    """)

    st.subheader("📊 Segment Behavioral Profiles & Benchmarks")
    if not seg_summary_df.empty:
        display_seg = seg_summary_df.copy()
        display_seg["Average Revenue ($)"] = display_seg["Average Revenue ($)"].map(lambda v: f"${v:,.2f}" if pd.notnull(v) else "$0.00")
        display_seg["Average Order Value ($)"] = display_seg["Average Order Value ($)"].map(lambda v: f"${v:,.2f}" if pd.notnull(v) else "$0.00")
        display_seg["Average Churn Risk (%)"] = display_seg["Average Churn Risk (%)"].map(lambda v: f"{v}%")

        st.dataframe(
            display_seg,
            use_container_width=True,
            column_config={
                "Average Health Score": st.column_config.ProgressColumn(min_value=0, max_value=100)
            },
            hide_index=True
        )

    st.markdown("---")

    # Visual Cluster Breakdown
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.subheader("🍰 Portfolio Share by Segment")
        fig_pie = px.pie(
            seg_summary_df,
            names="Segment",
            values="Number of Customers",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            height=340,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_s2:
        st.subheader("💵 Total Revenue Contribution by Segment")
        # Compute total revenue
        fig_rev_seg = px.bar(
            seg_summary_df,
            x="Segment",
            y="Average Revenue ($)",
            color="Segment",
            labels={"Segment": "Segment", "Average Revenue ($)": "Avg Revenue"},
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_rev_seg.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            height=340,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_rev_seg, use_container_width=True)

    # Segment Strategic Playbooks
    st.subheader("📘 Automated Segment Playbooks")
    playbooks = [
        ("🏆 High-Value Loyal", "#10B981", "Highest monetary value, high order cadence, low churn risk. Strategy: VIP advisory board, executive sponsorship, beta product access, multi-year enterprise renewals."),
        ("🚀 Growth Opportunity", "#6366F1", "Moderate spend, strong engagement velocity, high upside. Strategy: Propose cross-sell modules (Revenue Intelligence, AI Lead Scoring), expansion seats, and volume tiers."),
        ("🌱 New Customers", "#06B6D4", "Low tenure (<120 days), recent first transactions, active onboarding. Strategy: Dedicated TAM onboarding, admin certification masterclasses, rapid time-to-value."),
        ("⚠️ At-Risk Customers", "#EF4444", "High recency (no orders >90 days), spike in support tickets/complaints. Strategy: Immediate executive CS review, SLA health audit, personalized retention concession."),
        ("💤 Low Engagement", "#6B7280", "Low frequency, dormant interaction history, low historical basket size. Strategy: Automated quarterly product newsletters, self-service knowledge base, light re-engagement vouchers.")
    ]

    for title, color, desc in playbooks:
        st.markdown(
            f"""
            <div style="border-left: 4px solid {color}; background: rgba(30, 41, 59, 0.4); padding: 12px 16px; border-radius: 6px; margin-bottom: 10px;">
                <div style="font-weight: 700; color: {color}; font-size: 1rem;">{title}</div>
                <div style="font-size: 0.9rem; color: #CBD5E1; margin-top: 4px;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# 8. RECOMMENDATIONS
elif nav_selection == "⚡ Recommendations":
    st.markdown("## ⚡ Next-Best-Action & Recommendation Engine")
    st.markdown(
        "Prescriptive guidance combining rule-based heuristics with machine learning scores to trigger high-impact retention and expansion workflows."
    )

    r_tab1, r_tab2 = st.tabs(["🎯 Next-Best-Action Workflows", "🛍️ Product Cross-Sell Catalog"])

    with r_tab1:
        st.subheader("📋 Priority Customer Action Queue")
        action_df = query_df("""
            SELECT
                c.customer_id,
                c.company,
                c.industry,
                p.customer_segment,
                ROUND(p.health_score, 1) as health_score,
                ROUND(p.churn_probability * 100, 1) as churn_pct,
                p.recommended_action
            FROM customers c
            JOIN predictions p ON c.customer_id = p.customer_id
            WHERE p.churn_probability >= 0.50 OR p.health_score < 60 OR p.customer_segment = 'High-Value Loyal'
            ORDER BY p.churn_probability DESC
            LIMIT 200;
        """)

        if not action_df.empty:
            action_df["churn_pct"] = action_df["churn_pct"].map(lambda p: f"{p}%")
            st.dataframe(
                action_df,
                use_container_width=True,
                column_config={
                    "customer_id": "Customer ID",
                    "company": "Company",
                    "industry": "Industry",
                    "customer_segment": "Segment",
                    "health_score": st.column_config.ProgressColumn("Health Score", min_value=0, max_value=100),
                    "churn_pct": "Churn Probability",
                    "recommended_action": "Prescribed Next-Best-Action"
                },
                hide_index=True
            )
        else:
            st.info("No urgent actions currently queued.")

    with r_tab2:
        st.subheader("🛍️ Cross-Sell Product Catalog & Affinity Rules")
        products_df = query_df("SELECT * FROM products ORDER BY category, price DESC;")
        products_df["price"] = products_df["price"].map(lambda p: f"${p:,.2f}")
        st.dataframe(
            products_df,
            use_container_width=True,
            column_config={
                "product_id": "Product ID",
                "product_name": "Product Name",
                "category": "Category",
                "price": "Standard Price"
            },
            hide_index=True
        )

# 9. SETTINGS
elif nav_selection == "⚙️ Settings":
    st.markdown("## ⚙️ System Settings & Data Operations")
    st.markdown("Manage SQLite database records, trigger machine learning retraining, and inspect system telemetry.")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("🤖 Machine Learning Model Operations")
        st.write("Retrain all 4 local Scikit-learn models (Churn, Segmentation, Lead Scoring, Revenue) on current SQLite data.")
        if st.button("🔄 Retrain All ML Models Now", type="primary"):
            with st.spinner("Retraining Random Forest, Gradient Boosting, and K-Means models..."):
                metrics = run_full_ml_pipeline(force_retrain=True)
                st.success(f"✅ Models retrained successfully! Updated {metrics['total_predictions_stored']} customer predictions.")
                st.rerun()

    with c2:
        st.subheader("📦 Synthetic Data Operations")
        st.write("Regenerate synthetic dataset (5,000 customers, 25,000+ transactions, 2,000 leads, 10,000 interactions).")
        if st.button("⚠️ Force Regenerate Entire Dataset"):
            with st.spinner("Generating fresh synthetic dataset and running ML pipeline..."):
                generate_synthetic_data()
                run_full_ml_pipeline(force_retrain=True)
                st.success("✅ Dataset and models freshly generated!")
                st.rerun()

    st.markdown("---")
    st.subheader("🗄️ Database Table Counts & File Sizes")
    counts = {
        "Customers": query_df("SELECT COUNT(*) as cnt FROM customers;")["cnt"].iloc[0],
        "Transactions": query_df("SELECT COUNT(*) as cnt FROM transactions;")["cnt"].iloc[0],
        "Leads": query_df("SELECT COUNT(*) as cnt FROM leads;")["cnt"].iloc[0],
        "Interactions": query_df("SELECT COUNT(*) as cnt FROM interactions;")["cnt"].iloc[0],
        "Products": query_df("SELECT COUNT(*) as cnt FROM products;")["cnt"].iloc[0],
        "Predictions": query_df("SELECT COUNT(*) as cnt FROM predictions;")["cnt"].iloc[0],
    }

    t_cols = st.columns(6)
    for idx, (tbl, cnt) in enumerate(counts.items()):
        with t_cols[idx]:
            st.metric(tbl, f"{cnt:,}")

    db_path = get_db_path()
    if os.path.exists(db_path):
        size_mb = os.path.getsize(db_path) / (1024 * 1024)
        st.caption(f"Database File: `{db_path}` ({size_mb:.2f} MB)")
