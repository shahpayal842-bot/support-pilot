import os
import sqlite3
from typing import Any, Optional


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "tickets.db")


def init_database() -> None:
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT NOT NULL,
                customer_name TEXT,
                department TEXT,
                description TEXT NOT NULL,
                email TEXT,
                category TEXT NOT NULL,
                priority TEXT,
                confidence REAL NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        columns = {row[1] for row in connection.execute("PRAGMA table_info(tickets)")}
        if "customer_name" not in columns:
            connection.execute("ALTER TABLE tickets ADD COLUMN customer_name TEXT")
        if "department" not in columns:
            connection.execute("ALTER TABLE tickets ADD COLUMN department TEXT")
        if "priority" not in columns:
            connection.execute("ALTER TABLE tickets ADD COLUMN priority TEXT")
        connection.commit()


def save_processed_ticket(
    ticket_id: str,
    customer_name: Optional[str],
    department: Optional[str],
    description: str,
    email: Optional[str],
    category: str,
    priority: Optional[str],
    confidence: float,
    status: str,
) -> None:
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            INSERT INTO tickets
                (ticket_id, customer_name, department, description, email, category, priority, confidence, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ticket_id, customer_name, department, description, email, category, priority, confidence, status),
        )
        connection.commit()


def list_processed_tickets(limit: int = 50):
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT ticket_id, customer_name, department, description, email,
                   category, priority, confidence, status, created_at
            FROM tickets
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


init_database()
