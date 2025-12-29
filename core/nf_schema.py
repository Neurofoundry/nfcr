"""
Neuroforge Database Schema Migration
===================================

This module provides idempotent database schema migrations for Neuroforge.
All migrations are safe and additive - they only add missing tables, columns, and indexes.

Schema Overview:
- knowledge: Documents with embeddings for semantic search
- memory_facts: Textual facts with FTS5 for full-text search  
- turnlog: Conversation turns with feedback and reactions
- operational_logs: Events across all modules with rich metadata

Usage:
    from nf_schema import migrate_database, get_schema_version
    
    # Run all pending migrations
    migrate_database()
    
    # Check current schema version
    version = get_schema_version()
"""

import sqlite3
import json
import logging
from typing import Dict, Any, Optional
from nf_paths import DB_PATH


# Schema version for tracking migrations
CURRENT_SCHEMA_VERSION = 1


def get_db_connection() -> sqlite3.Connection:
    """Get a database connection with proper configuration."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row  # Enable column access by name
    
    # Enable WAL mode for better concurrency
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    
    # Enable JSON functions (available in SQLite 3.45+)
    try:
        conn.execute("SELECT json_extract('{}', '$')")
    except sqlite3.OperationalError:
        logging.warning("JSON functions not available in this SQLite version")
    
    return conn


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Check if a table exists in the database."""
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    )
    return cursor.fetchone() is not None


def column_exists(conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    try:
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
        return column_name in columns
    except sqlite3.OperationalError:
        return False


def index_exists(conn: sqlite3.Connection, index_name: str) -> bool:
    """Check if an index exists in the database."""
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
        (index_name,)
    )
    return cursor.fetchone() is not None


def get_schema_version(conn: Optional[sqlite3.Connection] = None) -> int:
    """Get the current schema version from the database."""
    if conn is None:
        conn = get_db_connection()
        should_close = True
    else:
        should_close = False
    
    try:
        # Check if schema_version table exists
        if not table_exists(conn, 'schema_version'):
            return 0
        
        cursor = conn.execute("SELECT version FROM schema_version ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        return row[0] if row else 0
    except sqlite3.OperationalError:
        return 0
    finally:
        if should_close:
            conn.close()


def set_schema_version(conn: sqlite3.Connection, version: int) -> None:
    """Set the schema version in the database."""
    # Create schema_version table if it doesn't exist
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version INTEGER NOT NULL,
            applied_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            description TEXT
        )
    """)
    
    # Insert new version record
    conn.execute(
        "INSERT INTO schema_version (version, description) VALUES (?, ?)",
        (version, f"Applied schema version {version}")
    )
    conn.commit()


def create_knowledge_table(conn: sqlite3.Connection) -> None:
    """Create the knowledge table for documents with embeddings."""
    if table_exists(conn, 'knowledge'):
        logging.info("Knowledge table already exists")
        return
    
    conn.execute("""
        CREATE TABLE knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            source_url TEXT,
            created_ts DATETIME DEFAULT CURRENT_TIMESTAMP,
            tags TEXT DEFAULT '[]',  -- JSON array of strings
            embedding_model TEXT,
            vector TEXT,  -- JSON array of floats for now
            content_hash TEXT,  -- For deduplication
            
            -- Generated column for fast tag filtering (SQLite 3.31+)
            tags_flat TEXT GENERATED ALWAYS AS (
                replace(replace(replace(tags, '["', ''), '"]', ''), '","', ' ')
            ) STORED
        )
    """)
    
    # Create indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_created_ts ON knowledge(created_ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_source_url ON knowledge(source_url)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_content_hash ON knowledge(content_hash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_tags_flat ON knowledge(tags_flat)")
    
    logging.info("Created knowledge table with indexes")


def create_memory_facts_table(conn: sqlite3.Connection) -> None:
    """Create the memory_facts virtual table for full-text search."""
    if table_exists(conn, 'memory_facts'):
        logging.info("Memory facts table already exists")
        return
    
    # Create FTS5 virtual table
    conn.execute("""
        CREATE VIRTUAL TABLE memory_facts USING fts5(
            keyphrase,
            answer,
            source_url,
            tags,
            created_ts UNINDEXED,
            content_hash UNINDEXED,
            content = 'memory_facts_content'
        )
    """)
    
    # Create content table for FTS5
    conn.execute("""
        CREATE TABLE memory_facts_content (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            keyphrase TEXT NOT NULL,
            answer TEXT NOT NULL,
            source_url TEXT,
            tags TEXT DEFAULT '[]',  -- JSON array of strings
            created_ts DATETIME DEFAULT CURRENT_TIMESTAMP,
            content_hash TEXT  -- For deduplication
        )
    """)
    
    # Create indexes on content table
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_facts_created_ts ON memory_facts_content(created_ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_facts_content_hash ON memory_facts_content(content_hash)")
    
    logging.info("Created memory_facts FTS5 table")


def create_turnlog_table(conn: sqlite3.Connection) -> None:
    """Create the turnlog table for conversation tracking."""
    if table_exists(conn, 'turnlog'):
        logging.info("Turnlog table already exists")
        return
    
    conn.execute("""
        CREATE TABLE turnlog (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            turn_index INTEGER NOT NULL,
            role TEXT NOT NULL,  -- 'user', 'assistant', 'system'
            content TEXT,
            reaction TEXT DEFAULT 'none',  -- 'up', 'down', 'none'
            tags TEXT DEFAULT '[]',  -- JSON array of strings
            notes TEXT,
            created_ts DATETIME DEFAULT CURRENT_TIMESTAMP,
            
            -- Ensure unique turns per session
            UNIQUE(session_id, turn_index)
        )
    """)
    
    # Create indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_turnlog_session_id ON turnlog(session_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_turnlog_created_ts ON turnlog(created_ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_turnlog_reaction ON turnlog(reaction)")
    
    logging.info("Created turnlog table with indexes")


def create_operational_logs_table(conn: sqlite3.Connection) -> None:
    """Create the operational_logs table for system events."""
    if table_exists(conn, 'operational_logs'):
        logging.info("Operational logs table already exists")
        return
    
    conn.execute("""
        CREATE TABLE operational_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            module_name TEXT NOT NULL,
            event_type TEXT NOT NULL,
            message TEXT,
            json_payload TEXT DEFAULT '{}',  -- JSON object
            tags TEXT DEFAULT '[]',  -- JSON array of strings
            session_id TEXT,
            created_ts DATETIME DEFAULT CURRENT_TIMESTAMP,
            
            -- Generated column for fast tag filtering
            tags_flat TEXT GENERATED ALWAYS AS (
                replace(replace(replace(tags, '["', ''), '"]', ''), '","', ' ')
            ) STORED
        )
    """)
    
    # Create indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_operational_logs_created_ts ON operational_logs(created_ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_operational_logs_module_name ON operational_logs(module_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_operational_logs_event_type ON operational_logs(event_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_operational_logs_session_id ON operational_logs(session_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_operational_logs_tags_flat ON operational_logs(tags_flat)")
    
    logging.info("Created operational_logs table with indexes")


def migrate_legacy_logs_table(conn: sqlite3.Connection) -> None:
    """Migrate data from legacy logs table to operational_logs if needed."""
    if not table_exists(conn, 'logs'):
        return
        
    if table_exists(conn, 'operational_logs'):
        # Check if migration already done
        cursor = conn.execute("SELECT COUNT(*) FROM operational_logs WHERE module_name = 'legacy_migration'")
        if cursor.fetchone()[0] > 0:
            logging.info("Legacy logs already migrated")
            return
    
    # Create operational_logs if it doesn't exist
    create_operational_logs_table(conn)
    
    # Migrate data
    cursor = conn.execute("""
        SELECT timestamp, session_id, module, event_type, content, tags, author, summary
        FROM logs
    """)
    
    migrated_count = 0
    for row in cursor.fetchall():
        timestamp, session_id, module, event_type, content, tags, author, summary = row
        
        # Normalize legacy tags to JSON array
        if tags:
            try:
                # Try parsing as JSON first
                json.loads(tags)
                normalized_tags = tags
            except (json.JSONDecodeError, ValueError):
                # Treat as comma-separated
                tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
                normalized_tags = json.dumps(tag_list)
        else:
            normalized_tags = '[]'
        
        # Create JSON payload from legacy fields
        payload = {
            'content': content,
            'author': author,
            'summary': summary,
            'legacy_migration': True
        }
        
        conn.execute("""
            INSERT INTO operational_logs 
            (module_name, event_type, message, json_payload, tags, session_id, created_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            module or 'legacy',
            event_type or 'unknown',
            summary or content,
            json.dumps(payload),
            normalized_tags,
            session_id,
            timestamp
        ))
        
        migrated_count += 1
    
    # Add migration marker
    conn.execute("""
        INSERT INTO operational_logs 
        (module_name, event_type, message, json_payload, tags, session_id)
        VALUES ('legacy_migration', 'completed', ?, '{}', '["status:completed"]', NULL)
    """, (f"Migrated {migrated_count} records from legacy logs table",))
    
    conn.commit()
    logging.info(f"Migrated {migrated_count} records from legacy logs table")


def add_missing_columns(conn: sqlite3.Connection) -> None:
    """Add any missing columns to existing tables."""
    # Add tags_flat to knowledge table if missing (for older SQLite versions)
    if table_exists(conn, 'knowledge') and not column_exists(conn, 'knowledge', 'tags_flat'):
        try:
            conn.execute("""
                ALTER TABLE knowledge ADD COLUMN tags_flat TEXT 
                GENERATED ALWAYS AS (
                    replace(replace(replace(tags, '["', ''), '"]', ''), '","', ' ')
                ) STORED
            """)
            logging.info("Added tags_flat column to knowledge table")
        except sqlite3.OperationalError as e:
            # Fallback for older SQLite - create regular column and trigger
            logging.warning(f"Could not create generated column: {e}")
            try:
                conn.execute("ALTER TABLE knowledge ADD COLUMN tags_flat TEXT")
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS update_knowledge_tags_flat 
                    AFTER INSERT ON knowledge BEGIN
                        UPDATE knowledge SET tags_flat = 
                            replace(replace(replace(NEW.tags, '["', ''), '"]', ''), '","', ' ')
                        WHERE id = NEW.id;
                    END
                """)
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS update_knowledge_tags_flat_update
                    AFTER UPDATE OF tags ON knowledge BEGIN
                        UPDATE knowledge SET tags_flat = 
                            replace(replace(replace(NEW.tags, '["', ''), '"]', ''), '","', ' ')
                        WHERE id = NEW.id;
                    END
                """)
                logging.info("Added tags_flat column with triggers for knowledge table")
            except sqlite3.OperationalError:
                logging.warning("Could not add tags_flat column to knowledge table")


def migrate_database(conn: Optional[sqlite3.Connection] = None) -> bool:
    """
    Run all database migrations. Returns True if any changes were made.
    
    Args:
        conn: Optional database connection. If None, creates a new one.
        
    Returns:
        True if migrations were applied, False if already up to date
    """
    if conn is None:
        conn = get_db_connection()
        should_close = True
    else:
        should_close = False
    
    try:
        current_version = get_schema_version(conn)
        logging.info(f"Current schema version: {current_version}")
        
        if current_version >= CURRENT_SCHEMA_VERSION:
            logging.info("Database schema is up to date")
            return False
        
        # Apply migrations based on current version
        if current_version < 1:
            logging.info("Applying schema version 1...")
            
            # Create all tables
            create_knowledge_table(conn)
            create_memory_facts_table(conn)
            create_turnlog_table(conn)
            create_operational_logs_table(conn)
            
            # Migrate legacy data
            migrate_legacy_logs_table(conn)
            
            # Add missing columns
            add_missing_columns(conn)
            
            # Update schema version
            set_schema_version(conn, 1)
            
            logging.info("Applied schema version 1")
        
        logging.info(f"Database migration completed. New version: {CURRENT_SCHEMA_VERSION}")
        return True
        
    except Exception as e:
        logging.error(f"Database migration failed: {e}")
        conn.rollback()
        raise
    finally:
        if should_close:
            conn.close()


def validate_schema(conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
    """
    Validate the current database schema.
    
    Returns:
        Dictionary with validation results
    """
    if conn is None:
        conn = get_db_connection()
        should_close = True
    else:
        should_close = False
    
    try:
        results = {
            'schema_version': get_schema_version(conn),
            'tables': {},
            'errors': []
        }
        
        expected_tables = ['knowledge', 'memory_facts', 'turnlog', 'operational_logs']
        
        for table in expected_tables:
            exists = table_exists(conn, table)
            results['tables'][table] = {
                'exists': exists,
                'row_count': 0
            }
            
            if exists:
                try:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    results['tables'][table]['row_count'] = cursor.fetchone()[0]
                except sqlite3.OperationalError as e:
                    results['errors'].append(f"Error counting rows in {table}: {e}")
            else:
                results['errors'].append(f"Missing table: {table}")
        
        return results
        
    finally:
        if should_close:
            conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Neuroforge Database Schema Migration")
    print("=" * 40)
    
    # Validate current state
    print("\nCurrent schema state:")
    validation = validate_schema()
    print(f"Schema version: {validation['schema_version']}")
    print(f"Tables: {list(validation['tables'].keys())}")
    
    if validation['errors']:
        print(f"Issues found: {len(validation['errors'])}")
        for error in validation['errors']:
            print(f"  - {error}")
    
    # Run migration
    print("\nRunning migration...")
    changes_made = migrate_database()
    
    if changes_made:
        print("✅ Migration completed successfully")
        
        # Validate after migration
        print("\nPost-migration validation:")
        validation = validate_schema()
        for table, info in validation['tables'].items():
            status = "✅" if info['exists'] else "❌"
            print(f"  {status} {table}: {info['row_count']} rows")
    else:
        print("✅ Database schema was already up to date")