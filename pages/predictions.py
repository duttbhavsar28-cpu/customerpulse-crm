"""
pages/predictions.py
====================
AI Predictions Lab & Machine Learning Performance Center.
Presents genuine model evaluations (Confusion Matrix, Precision/Recall, Feature Importances, R²),
portfolio-wide revenue forecasts, and an interactive real-time What-If Risk Simulator.
"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

from models.churn_model import ChurnPredictor
from models.lead_scoring import LeadScorer
from models.revenue_prediction import RevenuePredictor
from analytics.customer_health import calculate_health_score
from recommendations.next_best_action import determine_next_best_action
from utils.helpers import format_currency, format_percent, get_risk_badge, get_health_badge
from database.database import query_df


def render_predictions():
    """Render the AI predictions lab and model evaluation suite."""
    st.markdown("## 🔮 Predictive Intelligence Lab & ML Evaluation")
    st.markdown(
        "Evaluate Scikit-learn model performance metrics and simulate real-time customer churn risks using what-if parameters."
    )

    # Initialize model objects
    churn_pred = ChurnPredictor()
    rev_pred = RevenuePredictor()
    lead_scorer = LeadScorer()

    tab_eval, tab_sim, tab_rev = st.tabs([
        "📊 Model Evaluation & Metrics",
        "🧪 Interactive What-If Risk Simulator",
        "💰 Portfolio Revenue Forecasting"
    ])

    with tab_eval:
        st.subheader("🤖 Churn Classification Model (Random Forest)")
        c_metrics = churn_pred.metrics

        if c_metrics:
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Test Accuracy", f"{c_metrics['accuracy']*100:.1f}%")
            with m2:
                st.metric("Precision", f"{c_metrics['precision']*100:.1f}%")
            with m3:
                st.metric("Recall", f"{c_metrics['recall']*100:.1f}%")
            with m4:
                st.metric("F1 Score", f"{c_metrics['f1_score']*100:.1f}%")

            col_cm, col_imp = st.columns(2)

            with col_cm:
                st.markdown("#### 🎯 Confusion Matrix (Test Set)")
                cm = c_metrics["confusion_matrix"]
                # cm = [[TN, FP], [FN, TP]]
                tn, fp = cm[0]
                fn, tp = cm[1]

                cm_data = [[tn, fp], [fn, tp]]
                fig_cm = px.imshow(
                    cm_data,
                    labels=dict(x="Predicted Label", y="Actual Ground Truth", color="Count"),
                    x=["Retained (0)", "Churned (1)"],
                    y=["Retained (0)", "Churned (1)"],
                    text_auto=True,
                    color_continuous_scale="Blues",
                )
                fig_cm.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=280,
                )
                st.plotly_chart(fig_cm, use_container_width=True)
                st.caption(f"Evaluated on {c_metrics.get('test_samples', 1000)} holdout test customer records.")

            with col_imp:
                st.markdown("#### 🔍 Feature Importances")
                imp_df = pd.DataFrame(c_metrics["feature_importances"])
                fig_imp = px.bar(
                    imp_df.head(7),
                    x="importance",
                    y="feature",
                    orientation="h",
                    color="importance",
                    color_continuous_scale="Viridis",
                    labels={"importance": "Gini Importance", "feature": "Predictive Feature"}
                )
                fig_imp.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    coloraxis_showscale=False,
                    height=280,
                )
                st.plotly_chart(fig_imp, use_container_width=True)

        else:
            st.warning("Churn model is not trained yet. Visit Settings to trigger model training.")

        st.markdown("---")

        # Other models summary
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.subheader("📈 Revenue Forecasting Model (Gradient Boosting)")
            r_metrics = rev_pred.metrics
            if r_metrics:
                rm1, rm2, rm3 = st.columns(3)
                with rm1:
                    st.metric("R² Score", f"{r_metrics['r2_score']:.4f}")
                with rm2:
                    st.metric("MAE", f"${r_metrics['mae']:,.2f}")
                with rm3:
                    st.metric("RMSE", f"${r_metrics['rmse']:,.2f}")
                st.caption(f"Trained on {r_metrics.get('n_samples', 5000)} historical customer accounts.")
            else:
                st.info("Revenue model metrics not available.")

        with col_m2:
            st.subheader("🎯 Lead Qualification Model (Gradient Boosting)")
            l_metrics = lead_scorer.metrics
            if l_metrics:
                lm1, lm2, lm3 = st.columns(3)
                with lm1:
                    st.metric("ROC-AUC", f"{l_metrics['roc_auc']:.4f}")
                with lm2:
                    st.metric("Accuracy", f"{l_metrics['accuracy']*100:.1f}%")
                with lm3:
                    st.metric("F1 Score", f"{l_metrics['f1_score']*100:.1f}%")
                st.caption(f"Trained on {l_metrics.get('n_samples', 2000)} qualified lead records.")
            else:
                st.info("Lead scoring metrics not available.")

    # --- TAB 2: INTERACTIVE WHAT-IF RISK SIMULATOR ---
    with tab_sim:
        st.subheader("🧪 Real-Time Customer Churn & Health Simulator")
        st.markdown(
            "Tweak behavioral parameters below to observe real-time predictions from the trained Random Forest and Health scoring engine."
        )

        sim_c1, sim_c2 = st.columns(2)
        with sim_c1:
            s_recency = st.slider("Recency (Days since last purchase)", min_value=1, max_value=365, value=45)
            s_freq = st.slider("Historical Order Frequency", min_value=0, max_value=30, value=6)
            s_monetary = st.slider("Total Historical Spend ($)", min_value=0, max_value=50000, value=7500, step=500)
            s_aov = s_monetary / max(1, s_freq)
            s_tenure = st.slider("Customer Tenure (Days)", min_value=30, max_value=1800, value=450)

        with sim_c2:
            s_interactions = st.slider("Total Customer Interactions", min_value=0, max_value=20, value=4)
            s_support = st.slider("Support Requests Logged", min_value=0, max_value=10, value=1)
            s_negative = st.slider("Negative / Escalated Outcomes", min_value=0, max_value=6, value=0)
            s_days_int = st.slider("Days Since Last Interaction", min_value=1, max_value=180, value=20)
            s_avg_qty = 2.0

        # Construct feature dict
        test_features = {
            "recency_days": float(s_recency),
            "frequency": float(s_freq),
            "monetary_total": float(s_monetary),
            "monetary_avg": float(s_aov),
            "tenure_days": float(s_tenure),
            "interaction_count": float(s_interactions),
            "support_tickets": float(s_support),
            "negative_interactions": float(s_negative),
            "days_since_interaction": float(s_days_int),
            "avg_quantity": float(s_avg_qty),
        }

        # Predict Live
        if churn_pred.model is not None:
            pred_prob, risk_category = churn_pred.predict_single(test_features)
        else:
            pred_prob, risk_category = 0.25, "Low"

        # Calculate Health Live
        health_score, h_breakdown = calculate_health_score(
            recency_days=s_recency,
            frequency=s_freq,
            monetary_total=s_monetary,
            interaction_count=s_interactions,
            support_tickets=s_support,
            negative_interactions=s_negative
        )

        # Determine NBA Live
        sim_customer_data = test_features.copy()
        sim_customer_data["churn_probability"] = pred_prob
        sim_customer_data["health_score"] = health_score
        sim_customer_data["customer_segment"] = "High-Value Loyal" if s_monetary > 12000 else "Growth Opportunity"
        nba = determine_next_best_action(sim_customer_data)

        st.markdown("---")
        st.markdown("### 📊 Live Simulation Predictions")

        p1, p2, p3 = st.columns(3)
        risk_label, risk_color, risk_bg = get_risk_badge(pred_prob)
        health_label, health_color, health_bg = get_health_badge(health_score)

        with p1:
            st.markdown(
                f"""
                <div style="background-color: {risk_bg}; border: 1px solid {risk_color}; border-radius: 8px; padding: 15px; text-align: center;">
                    <div style="color: {risk_color}; font-size: 0.85rem; font-weight: 700;">PREDICTED CHURN RISK</div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: {risk_color};">{pred_prob*100:.1f}%</div>
                    <div style="color: #E2E8F0; font-size: 0.85rem;">Category: <strong>{risk_label}</strong></div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with p2:
            st.markdown(
                f"""
                <div style="background-color: {health_bg}; border: 1px solid {health_color}; border-radius: 8px; padding: 15px; text-align: center;">
                    <div style="color: {health_color}; font-size: 0.85rem; font-weight: 700;">CUSTOMER HEALTH SCORE</div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: {health_color};">{health_score:.0f} / 100</div>
                    <div style="color: #E2E8F0; font-size: 0.85rem;">Status: <strong>{health_label}</strong></div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with p3:
            st.markdown(
                f"""
                <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; padding: 15px;">
                    <div style="color: #A5B4FC; font-size: 0.85rem; font-weight: 700;">NEXT-BEST-ACTION</div>
                    <div style="color: #FFFFFF; font-size: 1.05rem; font-weight: 600; margin: 4px 0;">⚡ {nba['title']}</div>
                    <div style="color: #94A3B8; font-size: 0.8rem;">Priority: <strong style="color: #38BDF8;">{nba['priority']}</strong> &bull; Channel: {nba['channel']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # --- TAB 3: PORTFOLIO REVENUE FORECASTING ---
    with tab_rev:
        st.subheader("💰 Customer & Portfolio Revenue Projections")
        rev_df = query_df("""
            SELECT
                c.customer_id,
                c.company,
                c.industry,
                COALESCE(SUM(t.amount), 0.0) as current_revenue,
                p.predicted_revenue,
                p.churn_probability,
                ROUND(p.predicted_revenue * p.churn_probability, 2) as revenue_at_risk,
                ROUND(p.predicted_revenue * 1.18, 2) as potential_expansion_revenue
            FROM customers c
            LEFT JOIN transactions t ON c.customer_id = t.customer_id
            JOIN predictions p ON c.customer_id = p.customer_id
            GROUP BY c.customer_id
            ORDER BY p.predicted_revenue DESC
            LIMIT 500;
        """)

        if not rev_df.empty:
            tot_curr = rev_df["current_revenue"].sum()
            tot_pred = rev_df["predicted_revenue"].sum()
            tot_risk = rev_df["revenue_at_risk"].sum()
            tot_exp = rev_df["potential_expansion_revenue"].sum()

            rf1, rf2, rf3, rf4 = st.columns(4)
            with rf1:
                st.metric("Historical Revenue", format_currency(tot_curr, compact=True))
            with rf2:
                st.metric("Predicted 12M Revenue", format_currency(tot_pred, compact=True))
            with rf3:
                st.metric("Total Revenue at Risk", format_currency(tot_risk, compact=True), delta_color="inverse")
            with rf4:
                st.metric("Potential Expansion", format_currency(tot_exp, compact=True))

            st.markdown("---")

            display_rev = rev_df.copy()
            display_rev["current_revenue"] = display_rev["current_revenue"].map(lambda v: f"${v:,.2f}")
            display_rev["predicted_revenue"] = display_rev["predicted_revenue"].map(lambda v: f"${v:,.2f}")
            display_rev["revenue_at_risk"] = display_rev["revenue_at_risk"].map(lambda v: f"${v:,.2f}")
            display_rev["potential_expansion_revenue"] = display_rev["potential_expansion_revenue"].map(lambda v: f"${v:,.2f}")
            display_rev["churn_probability"] = display_rev["churn_probability"].map(lambda p: f"{p*100:.1f}%")

            st.dataframe(
                display_rev,
                use_container_width=True,
                column_config={
                    "customer_id": "Customer ID",
                    "company": "Company",
                    "industry": "Industry",
                    "current_revenue": "Historical Spend",
                    "predicted_revenue": "Predicted Future Spend",
                    "churn_probability": "Churn Risk",
                    "revenue_at_risk": "Revenue at Risk",
                    "potential_expansion_revenue": "Expansion Potential"
                },
                hide_index=True
            )
        else:
            st.info("No revenue predictions stored in database.")


if __name__ == "__main__":
    render_predictions()
