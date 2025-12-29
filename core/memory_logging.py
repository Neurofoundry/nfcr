"""
memory_logging
==============

This module provides helper functions for logging events to a SQLite
database and querying those logs.  It uses the Neuroforge system's
main database located at ``neuroforge_system_data/database.sqlite3`` by
default.  The log table stores rich metadata (session_id, module,
event_type, content, tags, author, summary) along with a timestamp.
Clients can log events via ``log_event_to_sql`` and search stored
events via ``query_logs``.

The table definition and indexes are created on demand by
``ensure_logs_table()``.  Queries search for a case‑insensitive
substring across several columns (content, tags, summary, module,
event_type, author, session_id) and may limit the number of results.
"""

from __future__ import annotations

import os
import sqlite3
from typing import List, Dict, Optional, Any


# Determine the path to the main Neuroforge database.  Use the
# environment variable NEUROFORGE_DB_PATH if set, otherwise fall back
# to the default location ``neuroforge_system_data/database.sqlite3``.
DEFAULT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neuroforge_system_data")
DB_PATH = os.environ.get("NEUROFORGE_DB_PATH") or os.path.join(DEFAULT_BASE, "database.sqlite3")


def ensure_logs_table(conn: sqlite3.Connection) -> None:
    """Ensure that the logs table and its indexes exist.

    If the table doesn't exist, create it along with indexes on
    ``tags`` and ``session_id``.  This function is idempotent.
    """
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            session_id TEXT,
            module TEXT,
            event_type TEXT,
            content TEXT,
            tags TEXT,
            author TEXT,
            summary TEXT
        )
        """
    )
    c.execute("CREATE INDEX IF NOT EXISTS idx_logs_tags ON logs(tags)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_logs_session ON logs(session_id)")
    conn.commit()


def log_event_to_sql(
    session_id: str,
    module: str,
    event_type: str,
    content: str,
    tags: str,
    author: str = "Architect",
    summary: Optional[str] = None,
    db_path: Optional[str] = None,
) -> None:
    """Log a single event to the SQLite database.

    Parameters
    ----------
    session_id : str
        Identifier linking this event to a particular session summary.
    module : str
        The module or file where the event occurred (e.g. crawler_worker.py).
    event_type : str
        A short label categorising the event (e.g. 'keyword_match').
    content : str
        The full event content or message.
    tags : str
        A comma‑separated list of tags describing the event.
    author : str, optional
        The person or persona responsible for the event; defaults to
        'Architect'.
    summary : str, optional
        A short summary of the event; optional but recommended for
        quick recall.
    db_path : str, optional
        Override the default database path; if None, use ``DB_PATH``.
    """
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    ensure_logs_table(conn)
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO logs (session_id, module, event_type, content, tags, author, summary)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (session_id, module, event_type, content, tags, author, summary),
    )
    conn.commit()
    conn.close()


def query_logs(
    query: str,
    max_results: int = 10,
    search_fields: Optional[List[str]] = None,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search the logs table for entries matching a substring.

    The function performs a case‑insensitive substring search across
    selected fields (default: session_id, module, event_type, content,
    tags, author, summary).  Results are ordered by timestamp DESC and
    limited by ``max_results``.

    Parameters
    ----------
    query : str
        Substring to search for (case‑insensitive).  If empty, returns
        an empty list.
    max_results : int, optional
        Maximum number of results to return (default 10).  Non‑positive
        values return all matches.
    search_fields : List[str], optional
        List of column names to search.  If None, all text fields are
        searched.  Unknown fields are ignored.
    db_path : str, optional
        Override the default database path.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries representing matching log rows.
    """
    if not query:
        return []
    fields = search_fields or [
        "session_id",
        "module",
        "event_type",
        "content",
        "tags",
        "author",
        "summary",
    ]
    # Sanitize field names to avoid SQL injection; only allow known columns
    valid_fields = {
        "session_id", "module", "event_type", "content", "tags", "author", "summary"
    }
    selected_fields = [f for f in fields if f in valid_fields]
    if not selected_fields:
        selected_fields = ["content"]

    like_pattern = f"%{query.lower()}%"
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    ensure_logs_table(conn)
    c = conn.cursor()
    where_clauses = [f"LOWER({field}) LIKE ?" for field in selected_fields]
    sql = "SELECT * FROM logs WHERE " + " OR ".join(where_clauses) + " ORDER BY timestamp DESC"
    params = [like_pattern] * len(selected_fields)
    if max_results and max_results > 0:
        sql += " LIMIT ?"
        params.append(max_results)
    try:
        rows = c.execute(sql, params).fetchall()
    except Exception:
        rows = []
    conn.close()
    return [dict(row) for row in rows]