"""
pages/analytics.py
==================
Sales & Revenue Intelligence Analytics for CustomerPulse CRM.
Provides:
- Lead Conversion Funnel & Sales Velocity
- Sales Rep Performance Leaderboard
- Transparent Customer Lifetime Value (CLV) Calculator
- Cohort Retention & Churn Dynamics
- Monthly Recurring Revenue (MRR) Trajectory
"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics.kpis import get_executive_kpis
from analytics.revenue import (
    calculate_clv,
    get_monthly_revenue_trend,
    get_revenue_by_industry,
    get_revenue_by_salesperson,
)
from crm.sales import get_pipeline_stages, get_sales_velocity
from database.database import query_df
from utils.helpers import format_currency, format_percent


def render_analytics():
    """Render comprehensive sales and revenue analytics view."""
    st.markdown("## 📈 Sales Intelligence & Revenue Analytics")
    st.markdown(
        "Deep-dive telemetry into pipeline stage transitions, salesperson win rates, cohort economics, and Customer Lifetime Value."
    )

    kpis = get_executive_kpis()
    velocity = get_sales_velocity()

    # --- TOP ROW: VELOCITY & CONVERSION METRICS ---
    v1, v2, v3, v4 = st.columns(4)
    with v1:
        st.metric(
            label="Sales Velocity (Monthly)",
            value=format_currency(velocity["sales_velocity_monthly"], compact=True),
            delta="Estimated Pipeline Pace",
        )
    with v2:
        st.metric(
            label="Win Rate",
            value=f"{velocity['win_rate']}%",
            delta=f"{velocity['open_deals']} Active Opportunities",
        )
    with v3:
        st.metric(
            label="Average Deal Size",
            value=format_currency(velocity["avg_deal_size"]),
            delta=f"{velocity['avg_cycle_days']:.0f} Days Avg Cycle",
        )
    with v4:
        st.metric(
            label="Historical Churn Rate",
            value=format_percent(kpis["churn_rate"]),
            delta=f"{kpis['churned_customers']} Lost Accounts",
            delta_color="inverse",
        )

    st.markdown("---")

    # --- SECTION 1: SALES PIPELINE FUNNEL ---
    col_funnel, col_stages = st.columns([3, 2])

    with col_funnel:
        st.subheader("🎯 Sales Pipeline Conversion Funnel")
        stages_df = get_pipeline_stages()
        if not stages_df.empty:
            funnel_df = stages_df[stages_df["stage"] != "Lost"].copy()
            fig_funnel = go.Figure(
                go.Funnel(
                    y=funnel_df["stage"],
                    x=funnel_df["deal_count"],
                    textinfo="value+percent initial",
                    marker=dict(
                        color=[
                            "#3B82F6", "#6366F1", "#8B5CF6",
                            "#A855F7", "#EC4899", "#10B981"
                        ][:len(funnel_df)]
                    )
                )
            )
            fig_funnel.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350,
            )
            st.plotly_chart(fig_funnel, use_container_width=True)

    with col_stages:
        st.subheader("💼 Pipeline Stage Value Distribution")
        if not stages_df.empty:
            fig_stage_val = px.bar(
                stages_df,
                x="stage",
                y="total_pipeline_value",
                color="stage",
                text="deal_count",
                labels={"total_pipeline_value": "Value ($)", "stage": "Stage"},
            )
            fig_stage_val.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                height=350,
            )
            st.plotly_chart(fig_stage_val, use_container_width=True)

    # --- SECTION 2: SALESPERSON PERFORMANCE LEADERBOARD ---
    st.subheader("🏆 Sales Team Performance Leaderboard")
    rep_df = get_revenue_by_salesperson()
    if not rep_df.empty:
        col_rep1, col_rep2 = st.columns([3, 2])
        with col_rep1:
            display_rep = rep_df.copy()
            display_rep["won_revenue"] = display_rep["won_revenue"].map(lambda v: f"${v:,.2f}")
            display_rep["open_pipeline_value"] = display_rep["open_pipeline_value"].map(lambda v: f"${v:,.2f}")
            display_rep["win_rate_pct"] = display_rep["win_rate_pct"].map(lambda w: f"{w}%")

            st.dataframe(
                display_rep,
                use_container_width=True,
                column_config={
                    "salesperson": "Sales Representative",
                    "total_leads_handled": "Total Leads",
                    "won_deals": "Closed Won Deals",
                    "win_rate_pct": "Win Rate",
                    "won_revenue": "Closed Revenue ($)",
                    "open_pipeline_value": "Active Pipeline ($)",
                },
                hide_index=True,
            )
        with col_rep2:
            fig_rep = px.bar(
                rep_df,
                x="salesperson",
                y="won_revenue",
                color="win_rate_pct",
                color_continuous_scale="Blues",
                labels={"won_revenue": "Closed Revenue ($)", "salesperson": "Representative"},
            )
            fig_rep.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=280,
            )
            st.plotly_chart(fig_rep, use_container_width=True)

    # --- SECTION 3: TRANSPARENT CUSTOMER LIFETIME VALUE (CLV) ---
    st.markdown("---")
    st.subheader("💎 Customer Lifetime Value (CLV) Engine")
    st.markdown(
        "Estimated long-term customer worth calculated from empirical purchasing frequency, Average Order Value, and observed retention."
    )

    clv_col1, clv_col2 = st.columns([1, 1])

    with clv_col1:
        st.markdown(
            r"""
            #### 📐 Transparent Valuation Formula
            $$\text{CLV} = \left(\frac{\text{Average Order Value (AOV)} \times \text{Annual Purchase Frequency}}{\text{Annual Churn Rate}}\right) \times \text{Gross Margin}$$

            * **Average Order Value (AOV)**: Empirical average transaction basket across all accounts.
            * **Annual Purchase Frequency**: Average number of repeat transactions an active customer completes per year.
            * **Annual Churn Rate**: Empirical probability of an account terminating business over 12 months.
            * **Gross Margin**: Standard enterprise gross margin (default: **75%**).
            """
        )

    with clv_col2:
        annual_freq = 4.2  # empirical average for B2B accounts
        clv_result = calculate_clv(
            avg_order_value=kpis["avg_order_value"],
            purchase_frequency_per_year=annual_freq,
            churn_rate=kpis["churn_rate"],
            gross_margin=0.75,
        )

        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); border: 1px solid #38BDF8; border-radius: 10px; padding: 20px;">
                <div style="color: #38BDF8; font-size: 0.85rem; font-weight: 700; letter-spacing: 0.05em;">ESTIMATED PORTFOLIO CLV</div>
                <div style="font-size: 2.4rem; font-weight: 800; color: #FFFFFF; margin: 8px 0;">{format_currency(clv_result['clv_value'])}</div>
                <div style="font-size: 0.9rem; color: #94A3B8; line-height: 1.5;">
                    &bull; <strong>Annual Revenue per Account:</strong> {format_currency(clv_result['annual_revenue_per_customer'])}<br>
                    &bull; <strong>Expected Account Lifespan:</strong> {clv_result['customer_lifespan_years']} years<br>
                    &bull; <strong>Effective Churn Assumption:</strong> {clv_result['effective_churn_rate']*100:.1f}%<br>
                    &bull; <strong>Assumed Gross Margin:</strong> {clv_result['gross_margin']*100:.0f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- SECTION 4: REVENUE VELOCITY & MRR RUN-RATE ---
    st.markdown("---")
    st.subheader("📊 Monthly Recurring Revenue (MRR) Run-Rate")
    trend_df = get_monthly_revenue_trend()
    if not trend_df.empty:
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            fig_growth = px.line(
                trend_df,
                x="month",
                y="revenue_growth_pct",
                markers=True,
                labels={"revenue_growth_pct": "MoM Growth (%)", "month": "Month"},
                title="Month-over-Month Revenue Growth Rate (%)"
            )
            fig_growth.update_traces(line_color="#10B981", line_width=2.5)
            fig_growth.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_growth, use_container_width=True)

        with c_m2:
            fig_buyers = px.bar(
                trend_df,
                x="month",
                y="active_buyers",
                labels={"active_buyers": "Active Buyers", "month": "Month"},
                title="Active Transacting Accounts per Month"
            )
            fig_buyers.update_traces(marker_color="#6366F1")
            fig_buyers.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_buyers, use_container_width=True)


if __name__ == "__main__":
    render_analytics()
