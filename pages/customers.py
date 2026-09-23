"""
pages/customers.py
==================
Customer Management and 360-Degree Intelligence View.
Supports search, multi-criteria filtering, CRUD operations, transaction history,
interaction timelines, next-best-action guidance, and cross-sell product recommendations.
"""

import streamlit as st
import pandas as pd
from crm.customers import (
    get_all_customers,
    get_customer_profile,
    create_customer,
    update_customer,
    delete_customer,
)
from crm.interactions import log_interaction
from recommendations.next_best_action import recommend_products_for_customer
from utils.helpers import (
    format_currency,
    format_percent,
    format_date,
    get_risk_badge,
    get_health_badge,
)
from database.database import query_df


def render_customers():
    """Render customer directory and 360 profile page."""
    st.markdown("## 👥 Customer 360 & Directory Management")
    st.markdown("Search, inspect, update, and manage full customer profiles with predictive intelligence.")

    # --- TOP ACTIONS & FILTERS ---
    tab_dir, tab_add = st.tabs(["📋 Customer Directory & 360 Profile", "➕ Add New Customer"])

    with tab_dir:
        # Filter controls
        f1, f2, f3 = st.columns([2, 1, 1])
        with f1:
            search_query = st.text_input("🔍 Search Customers", placeholder="Search by name, company, email, or ID...")
        with f2:
            ind_options = ["All Industries"] + query_df("SELECT DISTINCT industry FROM customers ORDER BY industry;")["industry"].tolist()
            selected_ind = st.selectbox("Industry", ind_options)
        with f3:
            status_options = ["All Statuses", "Active", "At-Risk", "Inactive", "Churned"]
            selected_status = st.selectbox("Status", status_options)

        # Load filtered customers
        cust_df = get_all_customers(
            search_query=search_query,
            industry=selected_ind,
            status=selected_status,
            limit=200
        )

        st.caption(f"Showing {len(cust_df)} customer accounts (filtered from database).")

        if cust_df.empty:
            st.warning("No customers matched your filter criteria.")
            return

        # Selection box for deep 360 inspection
        customer_options = [
            f"{row['customer_id']} - {row['company']} ({row['name']})"
            for _, row in cust_df.iterrows()
        ]
        
        selected_option = st.selectbox("🎯 Select Account to Inspect 360 Profile:", customer_options, index=0)
        selected_cid = selected_option.split(" - ")[0]

        # Fetch and render deep customer 360 profile
        profile = get_customer_profile(selected_cid)

        if profile:
            cust = profile["customer"]
            fin = profile["financials"]
            intel = profile["intelligence"]
            txs = profile["transactions"]
            itxs = profile["interactions"]

            st.markdown("---")
            # Profile Header
            h1, h2, h3 = st.columns([2, 1, 1])
            with h1:
                st.markdown(f"### {cust['company']}")
                st.markdown(f"**Contact:** {cust['name']} | **Email:** `{cust['email']}` | **Phone:** `{cust['phone']}`")
                st.caption(f"**Industry:** {cust['industry']} | **Location:** {cust['location']} | **Customer Since:** {format_date(cust['customer_since'])}")

            risk_label, risk_color, risk_bg = get_risk_badge(intel.get("churn_probability", 0.2))
            health_label, health_color, health_bg = get_health_badge(intel.get("health_score", 70.0))

            with h2:
                st.markdown(
                    f"""
                    <div style="background-color: {health_bg}; border: 1px solid {health_color}; border-radius: 8px; padding: 12px; text-align: center;">
                        <span style="color: {health_color}; font-size: 0.85rem; font-weight: 600;">CUSTOMER HEALTH SCORE</span>
                        <div style="font-size: 1.8rem; font-weight: 700; color: {health_color};">{intel.get('health_score', 0):.0f} / 100</div>
                        <span style="font-size: 0.8rem; color: #E2E8F0;">Tier: {health_label}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with h3:
                st.markdown(
                    f"""
                    <div style="background-color: {risk_bg}; border: 1px solid {risk_color}; border-radius: 8px; padding: 12px; text-align: center;">
                        <span style="color: {risk_color}; font-size: 0.85rem; font-weight: 600;">CHURN PROBABILITY</span>
                        <div style="font-size: 1.8rem; font-weight: 700; color: {risk_color};">{intel.get('churn_probability', 0)*100:.1f}%</div>
                        <span style="font-size: 0.8rem; color: #E2E8F0;">Risk: {risk_label}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.write("")

            # Financial KPI Cards
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Revenue", format_currency(fin["total_revenue"]))
            with m2:
                st.metric("Total Purchases", f"{fin['total_purchases']} orders")
            with m3:
                st.metric("Average Order Value", format_currency(fin["avg_order_value"]))
            with m4:
                st.metric("Segment", intel.get("customer_segment", "N/A"))

            # Next Best Action Alert Banner
            st.markdown(
                f"""
                <div style="background: linear-gradient(90deg, #1E1B4B 0%, #312E81 100%); border-left: 5px solid #6366F1; padding: 14px 18px; border-radius: 6px; margin: 15px 0;">
                    <div style="color: #A5B4FC; font-size: 0.8rem; font-weight: 700; letter-spacing: 0.05em;">RECOMMENDED NEXT-BEST-ACTION</div>
                    <div style="color: #FFFFFF; font-size: 1.1rem; font-weight: 600; margin-top: 4px;">⚡ {intel.get('recommended_action', 'Conduct Account Review')}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Profile Tabs: Transactions, Interactions, Recommendations, Edit Account
            p_tab1, p_tab2, p_tab3, p_tab4 = st.tabs([
                "💳 Transaction History",
                "📞 Interaction Timeline",
                "🎁 Product Recommendations",
                "✏️ Edit / Delete Account"
            ])

            with p_tab1:
                if not txs.empty:
                    tx_display = txs.copy()
                    tx_display["amount"] = tx_display["amount"].map(lambda a: f"${a:,.2f}")
                    st.dataframe(
                        tx_display,
                        use_container_width=True,
                        column_config={
                            "transaction_id": "Txn ID",
                            "transaction_date": "Date",
                            "product_name": "Product",
                            "category": "Category",
                            "quantity": "Qty",
                            "amount": "Amount"
                        },
                        hide_index=True
                    )
                else:
                    st.info("No recorded transactions for this account.")

            with p_tab2:
                # Log New Interaction Form
                with st.expander("➕ Log New Touchpoint / Interaction"):
                    with st.form(f"log_int_form_{selected_cid}"):
                        i_col1, i_col2 = st.columns(2)
                        with i_col1:
                            int_type = st.selectbox("Interaction Type", ['Phone Call', 'Email', 'Meeting', 'Demo', 'Support Request', 'Follow-up'])
                        with i_col2:
                            int_outcome = st.selectbox("Outcome", ['Positive', 'Neutral', 'Negative', 'Resolved', 'Pending', 'Action Required'])
                        int_notes = st.text_area("Notes & Follow-up Actions", placeholder="Record key topics discussed, objections, or commitments made...")
                        if st.form_submit_button("Save Touchpoint"):
                            if int_notes.strip():
                                ok, msg, iid = log_interaction(selected_cid, int_type, int_outcome, int_notes)
                                if ok:
                                    st.success(f"{msg}")
                                    st.rerun()
                                else:
                                    st.error(msg)
                            else:
                                st.error("Please add interaction notes before submitting.")

                # Timeline Display
                if not itxs.empty:
                    for _, irow in itxs.iterrows():
                        outcome_color = "#10B981" if irow["outcome"] in ("Positive", "Resolved") else ("#EF4444" if irow["outcome"] == "Negative" else "#F59E0B")
                        st.markdown(
                            f"""
                            <div style="border-left: 3px solid {outcome_color}; padding-left: 12px; margin-bottom: 12px;">
                                <div style="font-size: 0.85rem; color: #94A3B8;">{irow['interaction_date']} &bull; <strong>{irow['interaction_type']}</strong> &bull; <span style="color: {outcome_color};">{irow['outcome']}</span></div>
                                <div style="font-size: 0.95rem; color: #F1F5F9; margin-top: 2px;">{irow['notes']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No recorded interactions yet. Use the form above to log the first touchpoint.")

            with p_tab3:
                st.subheader("🛍️ Complementary Product Cross-Sell Opportunities")
                prods = recommend_products_for_customer(selected_cid, limit=3)
                if prods:
                    r_cols = st.columns(len(prods))
                    for idx, prod in enumerate(prods):
                        with r_cols[idx]:
                            st.markdown(
                                f"""
                                <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; padding: 14px; min-height: 180px;">
                                    <div style="font-size: 0.75rem; color: #38BDF8; font-weight: 600;">{prod['category']}</div>
                                    <div style="font-size: 1.05rem; font-weight: 700; color: #FFFFFF; margin: 4px 0;">{prod['product_name']}</div>
                                    <div style="font-size: 1.25rem; font-weight: 700; color: #10B981;">{format_currency(prod['price'])}</div>
                                    <p style="font-size: 0.85rem; color: #CBD5E1; margin-top: 8px;">{prod['reason']}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                else:
                    st.info("No additional product recommendations available.")

            with p_tab4:
                st.subheader("✏️ Update Customer Information")
                with st.form(f"update_cust_{selected_cid}"):
                    u_col1, u_col2 = st.columns(2)
                    with u_col1:
                        u_name = st.text_input("Contact Name", value=cust["name"])
                        u_comp = st.text_input("Company", value=cust["company"])
                        u_ind = st.selectbox("Industry", [
                            "SaaS & Cloud Software", "Financial Services & FinTech",
                            "Healthcare & Life Sciences", "Manufacturing & Industrial",
                            "Retail & E-Commerce", "Professional Services & Consulting",
                            "Telecommunications", "Media & Entertainment"
                        ], index=0)
                    with u_col2:
                        u_email = st.text_input("Email", value=cust["email"])
                        u_phone = st.text_input("Phone", value=cust["phone"])
                        u_loc = st.text_input("Location", value=cust["location"])

                    u_status = st.selectbox("Customer Status", ["Active", "At-Risk", "Inactive", "Churned"], index=["Active", "At-Risk", "Inactive", "Churned"].index(cust["customer_status"]))

                    if st.form_submit_button("Save Changes"):
                        ok, msg = update_customer(selected_cid, u_name, u_comp, u_ind, u_email, u_phone, u_loc, u_status)
                        if ok:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)

                st.markdown("---")
                st.markdown("#### 🗑️ Danger Zone: Delete Account")
                if st.button("Delete Account Permanently", type="primary", key=f"del_{selected_cid}"):
                    del_ok, del_msg = delete_customer(selected_cid)
                    if del_ok:
                        st.success(del_msg)
                        st.rerun()
                    else:
                        st.error(del_msg)

    # --- ADD NEW CUSTOMER TAB ---
    with tab_add:
        st.subheader("➕ Register New Customer Account")
        with st.form("new_customer_form"):
            c1, c2 = st.columns(2)
            with c1:
                n_name = st.text_input("Contact Full Name*", placeholder="e.g. Elena Rostova")
                n_comp = st.text_input("Company Name*", placeholder="e.g. Apex Dynamics Corp")
                n_ind = st.selectbox("Industry*", [
                    "SaaS & Cloud Software", "Financial Services & FinTech",
                    "Healthcare & Life Sciences", "Manufacturing & Industrial",
                    "Retail & E-Commerce", "Professional Services & Consulting",
                    "Telecommunications", "Media & Entertainment"
                ])
            with c2:
                n_email = st.text_input("Email Address*", placeholder="elena.r@apexdynamics.com")
                n_phone = st.text_input("Phone Number*", placeholder="+1-555-234-5678")
                n_loc = st.text_input("Location*", placeholder="e.g. San Francisco, CA")

            n_status = st.selectbox("Initial Status", ["Active", "At-Risk", "Inactive"], index=0)

            submitted = st.form_submit_button("Register Customer Account")
            if submitted:
                ok, msg, new_id = create_customer(
                    name=n_name,
                    company=n_comp,
                    industry=n_ind,
                    email=n_email,
                    phone=n_phone,
                    location=n_loc,
                    customer_status=n_status
                )
                if ok:
                    st.success(f"{msg} Auto-assigned ID: **{new_id}**")
                else:
                    st.error(msg)


if __name__ == "__main__":
    render_customers()
