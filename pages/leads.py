"""
pages/leads.py
==============
AI Lead Scoring, Qualification, and Sales Prioritization.
Allows sales representatives to prioritize high-intent leads, track pipeline velocity,
and convert qualified leads directly into active customer accounts.
"""

import streamlit as st
import pandas as pd
from crm.leads import (
    get_all_leads,
    create_lead,
    update_lead_status,
    convert_lead_to_customer,
)
from utils.helpers import format_currency, get_lead_priority_badge
from database.database import query_df


def render_leads():
    """Render the AI Lead Scoring & Pipeline view."""
    st.markdown("## 🎯 AI Lead Scoring & Sales Qualification")
    st.markdown(
        "Machine-learning prioritized sales pipeline ranking leads by conversion likelihood (0–100) with prescribed next-steps."
    )

    tab_list, tab_add, tab_convert = st.tabs([
        "📊 Scored Leads Leaderboard",
        "➕ Add New Lead",
        "🚀 Convert Lead to Customer"
    ])

    with tab_list:
        # Filter controls
        f1, f2, f3 = st.columns(3)
        with f1:
            rep_options = ["All Salespeople"] + query_df("SELECT DISTINCT salesperson FROM leads ORDER BY salesperson;")["salesperson"].tolist()
            selected_rep = st.selectbox("Assigned Salesperson", rep_options)
        with f2:
            status_options = ["All Statuses", "New", "Contacted", "Qualified", "Proposal Sent", "Negotiation", "Won", "Lost"]
            selected_status = st.selectbox("Stage / Status", status_options)
        with f3:
            src_options = ["All Sources"] + query_df("SELECT DISTINCT source FROM leads ORDER BY source;")["source"].tolist()
            selected_src = st.selectbox("Acquisition Source", src_options)

        leads_df = get_all_leads(
            salesperson=selected_rep,
            status=selected_status,
            source=selected_src,
            scored=True,
            limit=500
        )

        # Summary KPIs
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Total Filtered Leads", f"{len(leads_df):,}")
        with k2:
            high_count = len(leads_df[leads_df["priority"] == "High Priority"]) if not leads_df.empty else 0
            st.metric("🔥 High Priority Deals", high_count)
        with k3:
            total_budget = leads_df["budget"].sum() if not leads_df.empty else 0.0
            st.metric("Total Pipeline Value", format_currency(total_budget, compact=True))
        with k4:
            avg_score = leads_df["lead_score"].mean() if not leads_df.empty else 0.0
            st.metric("Average Lead Score", f"{avg_score:.1f} / 100")

        st.markdown("---")

        if not leads_df.empty:
            display_df = leads_df[[
                "lead_id", "company", "industry", "budget",
                "lead_score", "priority", "status", "salesperson", "recommended_action"
            ]].copy()

            display_df["budget"] = display_df["budget"].map(lambda b: f"${b:,.2f}")

            st.dataframe(
                display_df,
                use_container_width=True,
                column_config={
                    "lead_id": "Lead ID",
                    "company": "Company",
                    "industry": "Industry",
                    "budget": "Deal Budget",
                    "lead_score": st.column_config.ProgressColumn(
                        "AI Score (0-100)", min_value=0, max_value=100, format="%.1f"
                    ),
                    "priority": "Priority",
                    "status": "Pipeline Stage",
                    "salesperson": "Sales Rep",
                    "recommended_action": "Actionable Next Step"
                },
                hide_index=True
            )

            # Quick Stage Update Section
            with st.expander("⚡ Quick Update Lead Pipeline Stage"):
                u1, u2, u3 = st.columns([2, 2, 1])
                with u1:
                    target_lead_id = st.selectbox(
                        "Select Lead ID",
                        leads_df["lead_id"].tolist()
                    )
                with u2:
                    new_stage = st.selectbox(
                        "New Stage",
                        ['New', 'Contacted', 'Qualified', 'Proposal Sent', 'Negotiation', 'Won', 'Lost'],
                        key="quick_stage_select"
                    )
                with u3:
                    st.write("")
                    st.write("")
                    if st.button("Update Stage"):
                        ok, msg = update_lead_status(target_lead_id, new_stage)
                        if ok:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
        else:
            st.info("No leads found matching current criteria.")

    with tab_add:
        st.subheader("➕ Create New Sales Lead")
        with st.form("add_lead_form"):
            a1, a2 = st.columns(2)
            with a1:
                l_comp = st.text_input("Company Name*", placeholder="e.g. Stratosphere Systems")
                l_ind = st.selectbox("Industry*", [
                    "SaaS & Cloud Software", "Financial Services & FinTech",
                    "Healthcare & Life Sciences", "Manufacturing & Industrial",
                    "Retail & E-Commerce", "Professional Services & Consulting",
                    "Telecommunications", "Media & Entertainment"
                ])
                l_src = st.selectbox("Lead Source*", [
                    "Referral", "Webinar", "Organic Search", "LinkedIn Campaign",
                    "Partner Channel", "Inbound Demo Request", "Cold Outreach"
                ])
            with a2:
                l_budget = st.number_input("Estimated Budget ($)*", min_value=0.0, value=25000.0, step=1000.0)
                l_eng = st.slider("Initial Engagement Score (0-100)", min_value=0, max_value=100, value=65)
                l_rep = st.selectbox("Assign Salesperson*", [
                    "Sarah Jenkins", "Alex Rivera", "Michael Chang",
                    "Emily Watson", "David Kim", "Priya Patel"
                ])

            l_status = st.selectbox("Initial Pipeline Status", ['New', 'Contacted', 'Qualified'], index=0)

            if st.form_submit_button("Create Lead"):
                ok, msg, lid = create_lead(l_comp, l_ind, l_src, l_budget, float(l_eng), l_status, l_rep)
                if ok:
                    st.success(f"{msg} Auto-assigned ID: **{lid}**")
                else:
                    st.error(msg)

    with tab_convert:
        st.subheader("🚀 Convert Qualified Lead into Customer Account")
        st.markdown("Promote a won deal directly into a permanent active customer account.")

        open_leads = query_df(
            "SELECT lead_id, company, budget, salesperson FROM leads WHERE status NOT IN ('Lost') ORDER BY budget DESC LIMIT 100;"
        )

        if not open_leads.empty:
            lead_choices = [
                f"{r['lead_id']} - {r['company']} (Budget: ${r['budget']:,.2f} | Rep: {r['salesperson']})"
                for _, r in open_leads.iterrows()
            ]
            selected_choice = st.selectbox("Select Lead to Convert:", lead_choices)
            chosen_lid = selected_choice.split(" - ")[0]

            with st.form("convert_lead_form"):
                cv1, cv2 = st.columns(2)
                with cv1:
                    c_name = st.text_input("Primary Contact Full Name*", placeholder="e.g. John Sterling")
                    c_email = st.text_input("Contact Email*", placeholder="john@company.com")
                with cv2:
                    c_phone = st.text_input("Contact Phone*", placeholder="+1-555-888-9999")
                    c_loc = st.text_input("Office Location*", placeholder="Austin, TX")

                if st.form_submit_button("Confirm Conversion to Customer"):
                    ok, msg, cid = convert_lead_to_customer(chosen_lid, c_name, c_email, c_phone, c_loc)
                    if ok:
                        st.success(f"🎉 {msg}")
                    else:
                        st.error(msg)
        else:
            st.info("No eligible open leads found for conversion.")


if __name__ == "__main__":
    render_leads()
