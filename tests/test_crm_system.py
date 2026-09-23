"""
tests/test_crm_system.py
========================
Automated end-to-end integration and unit tests for CustomerPulse CRM.
Tests database tables, data generation, CRUD operations, machine learning models,
health scoring, and next-best-action recommendations.
"""

import os
import sys
import unittest

# Insert customerpulse-crm root into path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from database.database import get_db_path, init_db, query_df, execute_query
from utils.data_generator import load_or_generate_data
from utils.helpers import validate_email, validate_phone, format_currency, get_risk_badge, get_health_badge
from crm.customers import create_customer, get_customer_profile, update_customer, delete_customer
from crm.leads import create_lead, update_lead_status, convert_lead_to_customer
from crm.interactions import log_interaction, get_customer_timeline
from analytics.customer_health import calculate_health_score
from analytics.kpis import get_executive_kpis
from recommendations.next_best_action import determine_next_best_action, recommend_products_for_customer
from models.churn_model import ChurnPredictor
from models.lead_scoring import LeadScorer
from models.revenue_prediction import RevenuePredictor
from models.segmentation import CustomerSegmentation
from models.pipeline import run_full_ml_pipeline


class TestCustomerPulseCRM(unittest.TestCase):
    """Test suite covering the complete CustomerPulse CRM system."""

    @classmethod
    def setUpClass(cls):
        """Ensure database and baseline data are available."""
        cls.db_path = get_db_path()
        init_db(cls.db_path)
        # Ensure data is populated
        load_or_generate_data(db_path=cls.db_path)

    def test_01_database_tables_exist(self):
        """Verify all 6 core tables exist in SQLite."""
        tables_df = query_df("SELECT name FROM sqlite_master WHERE type='table';")
        tables = set(tables_df["name"].tolist())
        expected = {"customers", "transactions", "leads", "interactions", "products", "predictions"}
        for t in expected:
            self.assertIn(t, tables, f"Expected table '{t}' was not found in SQLite.")

    def test_02_data_volume(self):
        """Verify database contains expected scale of synthetic data."""
        cust_cnt = query_df("SELECT COUNT(*) as cnt FROM customers;")["cnt"].iloc[0]
        tx_cnt = query_df("SELECT COUNT(*) as cnt FROM transactions;")["cnt"].iloc[0]
        lead_cnt = query_df("SELECT COUNT(*) as cnt FROM leads;")["cnt"].iloc[0]
        int_cnt = query_df("SELECT COUNT(*) as cnt FROM interactions;")["cnt"].iloc[0]
        prod_cnt = query_df("SELECT COUNT(*) as cnt FROM products;")["cnt"].iloc[0]

        self.assertGreaterEqual(cust_cnt, 500, "Expected at least 500 customers.")
        self.assertGreaterEqual(tx_cnt, 1000, "Expected at least 1,000 transactions.")
        self.assertGreaterEqual(lead_cnt, 200, "Expected at least 200 leads.")
        self.assertGreaterEqual(int_cnt, 500, "Expected at least 500 interactions.")
        self.assertEqual(prod_cnt, 30, "Expected exactly 30 catalog products.")

    def test_03_helpers_validation(self):
        """Verify helper validation and formatting functions."""
        self.assertTrue(validate_email("user.test@example.com"))
        self.assertFalse(validate_email("invalid-email"))
        self.assertTrue(validate_phone("+1-555-234-5678"))
        self.assertFalse(validate_phone("123"))

        self.assertEqual(format_currency(1250.5), "$1,250.50")
        self.assertEqual(get_risk_badge(0.15)[0], "Low Risk")
        self.assertEqual(get_risk_badge(0.45)[0], "Medium Risk")
        self.assertEqual(get_risk_badge(0.85)[0], "High Risk")
        self.assertEqual(get_health_badge(95)[0], "Excellent")
        self.assertEqual(get_health_badge(40)[0], "Critical")

    def test_04_customer_crud_lifecycle(self):
        """Test full customer lifecycle: Create -> Read Profile -> Update -> Delete."""
        # 1. Create
        success, msg, new_cid = create_customer(
            name="Test User",
            company="Test Horizon Labs",
            industry="SaaS & Cloud Software",
            email="test.user.horizon@example.com",
            phone="+1-555-909-1234",
            location="Seattle, WA",
            customer_status="Active"
        )
        self.assertTrue(success, f"Customer creation failed: {msg}")
        self.assertIsNotNone(new_cid)

        # 2. Read Profile
        profile = get_customer_profile(new_cid)
        self.assertIsNotNone(profile)
        self.assertEqual(profile["customer"]["company"], "Test Horizon Labs")

        # 3. Update
        up_ok, up_msg = update_customer(
            customer_id=new_cid,
            name="Test User Updated",
            company="Test Horizon Labs Global",
            industry="Financial Services & FinTech",
            email="test.user.horizon@example.com",
            phone="+1-555-909-5678",
            location="San Francisco, CA",
            customer_status="At-Risk"
        )
        self.assertTrue(up_ok, f"Customer update failed: {up_msg}")

        # Verify update persisted
        updated_profile = get_customer_profile(new_cid)
        self.assertEqual(updated_profile["customer"]["customer_status"], "At-Risk")
        self.assertEqual(updated_profile["customer"]["name"], "Test User Updated")

        # 4. Delete
        del_ok, del_msg = delete_customer(new_cid)
        self.assertTrue(del_ok, f"Customer delete failed: {del_msg}")
        self.assertIsNone(get_customer_profile(new_cid))

    def test_05_interaction_logging_and_timeline(self):
        """Test interaction logging and timeline retrieval."""
        cust_id = query_df("SELECT customer_id FROM customers LIMIT 1;")["customer_id"].iloc[0]
        ok, msg, iid = log_interaction(
            customer_id=cust_id,
            interaction_type="Meeting",
            outcome="Positive",
            notes="Comprehensive QBR with executive sponsor."
        )
        self.assertTrue(ok)
        self.assertIsNotNone(iid)

        timeline = get_customer_timeline(cust_id)
        self.assertFalse(timeline.empty)
        self.assertIn("Comprehensive QBR", timeline["notes"].iloc[0])

    def test_06_lead_workflow_and_conversion(self):
        """Test lead creation, status updates, and lead-to-customer conversion."""
        ok, msg, lid = create_lead(
            company="Quantum Venture Capital",
            industry="Financial Services & FinTech",
            source="Referral",
            budget=75000.0,
            engagement_score=88.0,
            status="Qualified",
            salesperson="Sarah Jenkins"
        )
        self.assertTrue(ok)
        self.assertIsNotNone(lid)

        # Update status
        up_ok, _ = update_lead_status(lid, "Proposal Sent")
        self.assertTrue(up_ok)

        # Convert to customer
        conv_ok, conv_msg, new_cid = convert_lead_to_customer(
            lead_id=lid,
            contact_name="Victoria Sterling",
            email="v.sterling@quantumvc.test",
            phone="+1-555-432-1111",
            location="New York, NY"
        )
        self.assertTrue(conv_ok, f"Conversion failed: {conv_msg}")
        self.assertIsNotNone(new_cid)

        # Clean up test customer
        delete_customer(new_cid)

    def test_07_customer_health_score(self):
        """Test Customer Health Score calculation and boundary clipping."""
        score_high, breakdown_high = calculate_health_score(
            recency_days=10, frequency=12, monetary_total=25000, interaction_count=6,
            support_tickets=0, negative_interactions=0
        )
        self.assertGreaterEqual(score_high, 90.0)
        self.assertEqual(breakdown_high["health_category"], "Excellent")

        score_low, breakdown_low = calculate_health_score(
            recency_days=250, frequency=1, monetary_total=50, interaction_count=0,
            support_tickets=4, negative_interactions=3
        )
        self.assertLessEqual(score_low, 49.0)
        self.assertEqual(breakdown_low["health_category"], "Critical")

    def test_08_next_best_action_logic(self):
        """Test rule-based + ML Next-Best-Action recommendations."""
        # High churn + high revenue customer
        action_high_churn = determine_next_best_action({
            "churn_probability": 0.75,
            "monetary_total": 8000.0,
            "health_score": 45.0,
            "negative_interactions": 0
        })
        self.assertIn("Schedule Executive Retention Call", action_high_churn["title"])
        self.assertEqual(action_high_churn["priority"], "Urgent")

        # Loyal customer
        action_loyal = determine_next_best_action({
            "churn_probability": 0.05,
            "monetary_total": 15000.0,
            "health_score": 95.0,
            "customer_segment": "High-Value Loyal"
        })
        self.assertIn("VIP Client Advisory Board", action_loyal["title"])

    def test_09_product_recommendations(self):
        """Test complementary product recommendations for customer."""
        cust_id = query_df("SELECT customer_id FROM customers LIMIT 1;")["customer_id"].iloc[0]
        recs = recommend_products_for_customer(cust_id, limit=3)
        self.assertIsInstance(recs, list)
        self.assertGreater(len(recs), 0)
        self.assertIn("product_name", recs[0])
        self.assertIn("potential_value", recs[0])

    def test_10_full_ml_pipeline_execution(self):
        """Verify full ML pipeline training and predictions storage."""
        results = run_full_ml_pipeline(force_retrain=False)
        self.assertIn("churn_metrics", results)
        self.assertIn("revenue_metrics", results)
        self.assertIn("lead_metrics", results)
        self.assertGreater(results["total_predictions_stored"], 0)

        # Check predictions table in DB
        pred_cnt = query_df("SELECT COUNT(*) as cnt FROM predictions;")["cnt"].iloc[0]
        self.assertGreaterEqual(pred_cnt, 500)


if __name__ == "__main__":
    unittest.main()
