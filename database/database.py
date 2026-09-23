"""
database/database.py
====================
SQLite Database management module for CustomerPulse CRM.
Provides schema initialization, parameterized queries, and safe execution contexts.
"""

import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterable, List, Optional, Tuple, Union
import pandas as pd

DEFAULT_DB_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(DEFAULT_DB_DIR, "crm.db")


def get_db_path() -> str:
    """Return the absolute path to the SQLite CRM database."""
    return os.environ.get("CRM_DB_PATH", DEFAULT_DB_PATH)


@contextmanager
def get_connection(db_path: Optional[str] = None):
    """
    Context manager for safe SQLite database connection handling.
    Enables foreign keys and ensures proper transaction commit/rollback.
    """
    path = db_path or get_db_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")  # High-concurrency write-ahead logging
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Optional[str] = None) -> None:
    """
    Initialize SQLite database schema with indexes and constraints.
    Creates tables: customers, transactions, leads, interactions, products, predictions.
    """
    path = db_path or get_db_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with get_connection(path) as conn:
        cursor = conn.cursor()

        # 1. Products Table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS products (
                product_id TEXT PRIMARY KEY,
                product_name TEXT NOT NULL,
                category TEXT NOT NULL,
                price REAL NOT NULL CHECK(price >= 0)
            );
            """
        )

        # 2. Customers Table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS customers (
                customer_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                company TEXT NOT NULL,
                industry TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                phone TEXT NOT NULL,
                location TEXT NOT NULL,
                customer_since TEXT NOT NULL,
                customer_status TEXT NOT NULL DEFAULT 'Active'
                    CHECK(customer_status IN ('Active', 'At-Risk', 'Inactive', 'Churned')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        # 3. Transactions Table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                product_id TEXT NOT NULL,
                quantity INTEGER NOT NULL CHECK(quantity > 0),
                amount REAL NOT NULL CHECK(amount >= 0),
                transaction_date TEXT NOT NULL,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE,
                FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE RESTRICT
            );
            """
        )

        # 4. Leads Table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS leads (
                lead_id TEXT PRIMARY KEY,
                company TEXT NOT NULL,
                industry TEXT NOT NULL,
                source TEXT NOT NULL,
                budget REAL NOT NULL CHECK(budget >= 0),
                engagement_score REAL NOT NULL CHECK(engagement_score >= 0 AND engagement_score <= 100),
                status TEXT NOT NULL DEFAULT 'New'
                    CHECK(status IN ('New', 'Contacted', 'Qualified', 'Proposal Sent', 'Negotiation', 'Won', 'Lost')),
                created_date TEXT NOT NULL,
                salesperson TEXT NOT NULL
            );
            """
        )

        # 5. Interactions Table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                interaction_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                interaction_type TEXT NOT NULL
                    CHECK(interaction_type IN ('Phone Call', 'Email', 'Meeting', 'Demo', 'Support Request', 'Follow-up')),
                interaction_date TEXT NOT NULL,
                notes TEXT,
                outcome TEXT NOT NULL
                    CHECK(outcome IN ('Positive', 'Neutral', 'Negative', 'Resolved', 'Pending', 'Action Required')),
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
            );
            """
        )

        # 6. Predictions Table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                customer_id TEXT PRIMARY KEY,
                churn_probability REAL NOT NULL CHECK(churn_probability >= 0.0 AND churn_probability <= 1.0),
                customer_segment TEXT NOT NULL,
                health_score REAL NOT NULL CHECK(health_score >= 0.0 AND health_score <= 100.0),
                predicted_revenue REAL NOT NULL,
                recommended_action TEXT NOT NULL,
                prediction_date TEXT NOT NULL,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
            );
            """
        )

        # Create performance indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_customers_status ON customers(customer_status);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_customers_industry ON customers(industry);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tx_customer ON transactions(customer_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tx_date ON transactions(transaction_date);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_leads_status ON leads(status);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_leads_salesperson ON leads(salesperson);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_interact_customer ON interactions(customer_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_interact_date ON interactions(interaction_date);")


def query_df(sql: str, params: Union[Tuple[Any, ...], List[Any], dict] = (), db_path: Optional[str] = None) -> pd.DataFrame:
    """Execute a parameterized SQL SELECT query and return results as a Pandas DataFrame."""
    with get_connection(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def execute_query(sql: str, params: Union[Tuple[Any, ...], List[Any], dict] = (), db_path: Optional[str] = None) -> int:
    """Execute a single parameterized INSERT, UPDATE, or DELETE statement. Returns affected row count."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        return cursor.rowcount


def execute_many(sql: str, param_list: Iterable[Union[Tuple[Any, ...], List[Any]]], db_path: Optional[str] = None) -> int:
    """Execute a parameterized query against an iterable of parameter tuples. Returns affected rows."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany(sql, param_list)
        return cursor.rowcount


def insert_df(table_name: str, df: pd.DataFrame, if_exists: str = "append", db_path: Optional[str] = None) -> None:
    """
    Insert a Pandas DataFrame into a database table safely.
    If if_exists='replace', deletes existing rows while keeping table schema intact.
    """
    path = db_path or get_db_path()
    with get_connection(path) as conn:
        if if_exists == "replace":
            conn.execute(f"DELETE FROM {table_name};")
        df.to_sql(table_name, conn, if_exists="append", index=False)

