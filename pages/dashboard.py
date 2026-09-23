"""
pages/dashboard.py
==================
Executive Revenue & Customer Intelligence Dashboard for CustomerPulse CRM.
Displays core business KPIs, revenue trends, churn risk distributions, and account leaderboards.
"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics.kpis import get_executive_kpis
from analytics.revenue import (
    get_monthly_revenue_trend,
    get_revenue_by_industry,
    get_revenue_by_product,
    get_top_customers,
)
from database.database import query_df
from utils.helpers import format_currency, format_percent


def render_dashboard():
    """Render the primary executive dashboard view."""
    st.markdown("## 📊 Executive Revenue & Customer Intelligence Dashboard")
    st.markdown(
        "Real-time synthesis of customer health, churn vulnerability, sales pipeline velocity, and revenue distribution."
    )

    kpis = get_executive_kpis()

    # --- TOP ROW KPI CARDS ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="Total Customers",
            value=f"{kpis['total_customers']:,}",
            delta=f"{kpis['active_customers']:,} Active",
        )
    with c2:
        st.metric(
            label="Total Revenue",
            value=format_currency(kpis["total_revenue"], compact=True),
            delta=f"AOV: {format_currency(kpis['avg_order_value'])}",
        )
    with c3:
        st.metric(
            label="Revenue at Risk",
            value=format_currency(kpis["revenue_at_risk"], compact=True),
            delta=f"{kpis['at_risk_customers']} Accounts At-Risk",
            delta_color="inverse",
        )
    with c4:
        st.metric(
            label="Lead Conversion Rate",
            value=format_percent(kpis["conversion_rate"]),
            delta=f"{kpis['open_leads']} Open Pipeline Deals",
        )

    st.markdown("---")

    # --- CHARTS ROW 1: REVENUE OVER TIME & CHURN RISK ---
    col_chart1, col_chart2 = st.columns([3, 2])

    with col_chart1:
        st.subheader("📈 Monthly Revenue & Trajectory")
        trend_df = get_monthly_revenue_trend()
        if not trend_df.empty:
            fig_rev = go.Figure()
            fig_rev.add_trace(
                go.Bar(
                    x=trend_df["month"],
                    y=trend_df["monthly_revenue"],
                    name="Monthly Revenue ($)",
                    marker_color="#3B82F6",
                    opacity=0.85,
                )
            )
            fig_rev.add_trace(
                go.Scatter(
                    x=trend_df["month"],
                    y=trend_df["cumulative_revenue"],
                    name="Cumulative Revenue ($)",
                    yaxis="y2",
                    line=dict(color="#10B981", width=3),
                )
            )
            fig_rev.update_layout(
                yaxis=dict(title="Monthly Revenue ($)", showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
                yaxis2=dict(title="Cumulative ($)", overlaying="y", side="right", showgrid=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                hovermode="x unified",
                height=350,
            )
            st.plotly_chart(fig_rev, use_container_width=True)
        else:
            st.info("No transaction history available yet.")

    with col_chart2:
        st.subheader("⚠️ Customer Churn Risk Tiers")
        risk_df = query_df("""
            SELECT
                CASE
                    WHEN churn_probability <= 0.30 THEN 'Low Risk (0-30%)'
                    WHEN churn_probability <= 0.60 THEN 'Medium Risk (31-60%)'
                    ELSE 'High Risk (61-100%)'
                END as risk_tier,
                COUNT(*) as customer_count
            FROM predictions
            GROUP BY risk_tier
            ORDER BY customer_count DESC;
        """)
        if not risk_df.empty:
            color_map = {
                "Low Risk (0-30%)": "#10B981",
                "Medium Risk (31-60%)": "#F59E0B",
                "High Risk (61-100%)": "#EF4444",
            }
            fig_risk = px.pie(
                risk_df,
                names="risk_tier",
                values="customer_count",
                hole=0.45,
                color="risk_tier",
                color_discrete_map=color_map,
            )
            fig_risk.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                height=350,
            )
            st.plotly_chart(fig_risk, use_container_width=True)
        else:
            st.info("Run predictions to view Churn Risk breakdown.")

    # --- CHARTS ROW 2: SEGMENTS & REVENUE BY INDUSTRY ---
    col_chart3, col_chart4 = st.columns(2)

    with col_chart3:
        st.subheader("🧩 Customer Behavioral Segments")
        seg_df = query_df("""
            SELECT customer_segment, COUNT(*) as count, AVG(health_score) as avg_health
            FROM predictions
            GROUP BY customer_segment
            ORDER BY count DESC;
        """)
        if not seg_df.empty:
            fig_seg = px.bar(
                seg_df,
                x="customer_segment",
                y="count",
                color="customer_segment",
                color_discrete_sequence=px.colors.qualitative.Prism,
                text="count",
                labels={"customer_segment": "Segment", "count": "Customer Count"},
            )
            fig_seg.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                height=320,
            )
            st.plotly_chart(fig_seg, use_container_width=True)
        else:
            st.info("No segmentation predictions available.")

    with col_chart4:
        st.subheader("🏢 Revenue by Industry")
        ind_df = get_revenue_by_industry()
        if not ind_df.empty:
            fig_ind = px.bar(
                ind_df.head(6),
                y="industry",
                x="total_revenue",
                orientation="h",
                color="total_revenue",
                color_continuous_scale="Viridis",
                labels={"industry": "Industry", "total_revenue": "Total Revenue ($)"},
            )
            fig_ind.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
                height=320,
            )
            st.plotly_chart(fig_ind, use_container_width=True)

    # --- CHARTS ROW 3: LEAD CONVERSION & PRODUCT REVENUE ---
    col_lead, col_prod = st.columns(2)

    with col_lead:
        st.subheader("🎯 Lead Acquisition Sources & Conversion")
        lead_source_df = query_df("""
            SELECT
                source,
                COUNT(lead_id) as total_leads,
                SUM(CASE WHEN status = 'Won' THEN 1 ELSE 0 END) as won_leads
            FROM leads
            GROUP BY source
            ORDER BY total_leads DESC;
        """)
        if not lead_source_df.empty:
            lead_source_df["conversion_rate"] = (
                lead_source_df["won_leads"] / lead_source_df["total_leads"]
            ) * 100.0
            fig_src = px.bar(
                lead_source_df,
                x="source",
                y=["total_leads", "won_leads"],
                barmode="group",
                color_discrete_map={"total_leads": "#64748B", "won_leads": "#10B981"},
                labels={"value": "Leads", "source": "Source", "variable": "Metric"},
            )
            fig_src.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=320,
            )
            st.plotly_chart(fig_src, use_container_width=True)

    with col_prod:
        st.subheader("📦 Top Revenue Products")
        prod_df = get_revenue_by_product()
        if not prod_df.empty:
            fig_prod = px.bar(
                prod_df.head(6),
                x="total_revenue",
                y="product_name",
                orientation="h",
                color="category",
                labels={"total_revenue": "Revenue ($)", "product_name": "Product"},
            )
            fig_prod.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=320,
            )
            st.plotly_chart(fig_prod, use_container_width=True)

    # --- TOP 10 ACCOUNTS LEADERBOARD ---
    st.subheader("🏆 Top 10 Customer Accounts by Revenue")
    top_cust = get_top_customers(limit=10)
    if not top_cust.empty:
        # Join with predictions for health and churn
        top_enriched = query_df("""
            SELECT
                c.customer_id,
                c.name,
                c.company,
                c.industry,
                c.customer_status,
                COALESCE(SUM(t.amount), 0.0) as total_revenue,
                COUNT(t.transaction_id) as order_count,
                p.customer_segment,
                ROUND(p.health_score, 1) as health_score,
                ROUND(p.churn_probability * 100, 1) as churn_pct
            FROM customers c
            LEFT JOIN transactions t ON c.customer_id = t.customer_id
            LEFT JOIN predictions p ON c.customer_id = p.customer_id
            GROUP BY c.customer_id
            ORDER BY total_revenue DESC
            LIMIT 10;
        """)
        top_enriched["total_revenue"] = top_enriched["total_revenue"].map(lambda x: f"${x:,.2f}")
        top_enriched["churn_pct"] = top_enriched["churn_pct"].map(lambda x: f"{x}%")
        st.dataframe(
            top_enriched,
            use_container_width=True,
            column_config={
                "customer_id": "Customer ID",
                "name": "Contact Name",
                "company": "Company",
                "industry": "Industry",
                "customer_status": "Status",
                "total_revenue": "Total Spend",
                "order_count": "Orders",
                "customer_segment": "Segment",
                "health_score": st.column_config.ProgressColumn("Health Score", min_value=0, max_value=100),
                "churn_pct": "Churn Risk"
            },
            hide_index=True
        )


if __name__ == "__main__":
    render_dashboard()
