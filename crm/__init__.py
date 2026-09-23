"""CRM Core Operations package for CustomerPulse CRM."""
from crm.customers import (
    get_all_customers,
    get_customer_profile,
    create_customer,
    update_customer,
    delete_customer,
)
from crm.leads import (
    get_all_leads,
    create_lead,
    update_lead_status,
    convert_lead_to_customer,
)
from crm.sales import get_pipeline_stages, get_sales_velocity
from crm.interactions import log_interaction, get_customer_timeline

__all__ = [
    "get_all_customers",
    "get_customer_profile",
    "create_customer",
    "update_customer",
    "delete_customer",
    "get_all_leads",
    "create_lead",
    "update_lead_status",
    "convert_lead_to_customer",
    "get_pipeline_stages",
    "get_sales_velocity",
    "log_interaction",
    "get_customer_timeline",
]
