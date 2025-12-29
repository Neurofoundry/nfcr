#!/usr/bin/env python3
"""
Database Unification Script
===========================
Safely migrates all data to a single unified database (Master).
Preserves all existing data while creating a single source of truth.
"""

import sqlite3
import shutil
import json
from pathlib import Path
from datetime import datetime

# Database paths
MASTER_DB = "./neuroforge_system_data/database_master.sqlite3"
SYSTEM_DB = "./neuroforge_system_data/database.sqlite3" 
DATA_DB = "./Data/System Data/database.sqlite3"

def backup_databases():
    """Create backups before migration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(f"./database_backups_{timestamp}")
    backup_dir.mkdir(exist_ok=True)
    
    print(f"📦 Creating backups in {backup_dir}/")
    
    databases = [MASTER_DB, SYSTEM_DB, DATA_DB]
    for db_path in databases:
        if Path(db_path).exists():
            backup_path = backup_dir / Path(db_path).name
            shutil.copy2(db_path, backup_path)
            print(f"  ✅ Backed up {db_path}")
    
    return backup_dir

def create_unified_schema(master_conn):
    """Ensure Master database has all required tables."""
    print("\n🔧 Creating unified schema...")
    
    cursor = master_conn.cursor()
    
    # Create memory_facts table if missing (from System DB schema)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory_facts (
            keyphrase TEXT PRIMARY KEY,
            answer TEXT NOT NULL,
            source_url TEXT,
            tags TEXT,
            created_ts TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create FTS5 virtual table for memory_facts
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_facts_fts USING fts5(
            keyphrase, answer, source_url, tags, created_ts,
            content='memory_facts',
            content_rowid='rowid'
        )
    """)
    
    # Create operational_logs table if missing
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS operational_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            module_name TEXT,
            event_type TEXT,
            message TEXT,
            json_payload TEXT,
            tags TEXT,
            session_id TEXT,
            created_ts TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Ensure turnlog has proper structure
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS turnlog_unified (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conv_id TEXT,
            role TEXT,
            text TEXT,
            rating INTEGER,
            tags TEXT,
            ts TEXT
        )
    """)
    
    # Ensure conversations table is consistent
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations_unified (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            turn INTEGER,
            user_prompt TEXT,
            ai_response TEXT,
            timestamp TEXT
        )
    """)
    
    master_conn.commit()
    print("  ✅ Schema updated")

def migrate_conversations(master_conn):
    """Migrate all conversations to Master."""
    print("\n💬 Migrating conversations...")
    
    cursor = master_conn.cursor()
    
    # Get existing conversation IDs in Master to avoid duplicates
    cursor.execute("SELECT DISTINCT conversation_id FROM conversations")
    existing_convs = {row[0] for row in cursor.fetchall()}
    print(f"  📊 Master has {len(existing_convs)} existing conversations")
    
    total_migrated = 0
    
    # Migrate from System DB
    for source_db, db_name in [(SYSTEM_DB, "System"), (DATA_DB, "Data")]:
        if not Path(source_db).exists():
            continue
            
        source_conn = sqlite3.connect(source_db)
        source_cursor = source_conn.cursor()
        
        try:
            source_cursor.execute("SELECT conversation_id, turn, user_prompt, ai_response, timestamp FROM conversations")
            rows = source_cursor.fetchall()
            
            new_conversations = []
            for row in rows:
                conv_id = row[0]
                if conv_id not in existing_convs:
                    new_conversations.append(row)
                    existing_convs.add(conv_id)
            
            if new_conversations:
                cursor.executemany(
                    "INSERT INTO unified_conversations (conversation_id, turn, role, text, timestamp) VALUES (?, ?, ?, ?, ?), (?, ?, ?, ?, ?)",
                    new_conversations
                )
                total_migrated += len(new_conversations)
                print(f"  ✅ Migrated {len(new_conversations)} new conversations from {db_name}")
            else:
                print(f"  ⏭️  No new conversations in {db_name}")
                
        except sqlite3.Error as e:
            print(f"  ⚠️  Error reading {db_name}: {e}")
        finally:
            source_conn.close()
    
    master_conn.commit()
    print(f"  🎯 Total conversations migrated: {total_migrated}")

def migrate_memory_facts(master_conn):
    """Migrate memory facts from System DB."""
    print("\n🧠 Migrating memory facts...")
    
    if not Path(SYSTEM_DB).exists():
        print("  ⏭️  System DB not found, skipping")
        return
    
    source_conn = sqlite3.connect(SYSTEM_DB)
    source_cursor = source_conn.cursor()
    master_cursor = master_conn.cursor()
    
    try:
        source_cursor.execute("SELECT keyphrase, answer, source_url, tags, created_ts FROM memory_facts")
        facts = source_cursor.fetchall()
        
        if facts:
            # Get existing keyphrases to avoid duplicates
            master_cursor.execute("SELECT keyphrase FROM memory_facts")
            existing_keys = {row[0] for row in master_cursor.fetchall()}
            
            new_facts = [fact for fact in facts if fact[0] not in existing_keys]
            
            if new_facts:
                master_cursor.executemany(
                    "INSERT INTO memory_facts (keyphrase, answer, source_url, tags, created_ts) VALUES (?, ?, ?, ?, ?)",
                    new_facts
                )
                master_conn.commit()
                print(f"  ✅ Migrated {len(new_facts)} new memory facts")
            else:
                print("  ⏭️  No new memory facts to migrate")
        else:
            print("  ⏭️  No memory facts found in System DB")
            
    except sqlite3.Error as e:
        print(f"  ⚠️  Error migrating memory facts: {e}")
    finally:
        source_conn.close()

def migrate_knowledge(master_conn):
    """Migrate knowledge from System DB."""
    print("\n📚 Migrating knowledge...")
    
    if not Path(SYSTEM_DB).exists():
        print("  ⏭️  System DB not found, skipping")
        return
    
    source_conn = sqlite3.connect(SYSTEM_DB)
    source_cursor = source_conn.cursor()
    master_cursor = master_conn.cursor()
    
    try:
        # Check if System DB has knowledge with different schema
        source_cursor.execute("PRAGMA table_info(knowledge)")
        sys_cols = [col[1] for col in source_cursor.fetchall()]
        
        master_cursor.execute("PRAGMA table_info(knowledge)")
        master_cols = [col[1] for col in master_cursor.fetchall()]
        
        print(f"  📋 System columns: {sys_cols}")
        print(f"  📋 Master columns: {master_cols}")
        
        # Get existing doc_ids to avoid duplicates
        if 'doc_id' in master_cols:
            master_cursor.execute("SELECT doc_id FROM knowledge")
            existing_docs = {row[0] for row in master_cursor.fetchall()}
        else:
            existing_docs = set()
        
        # Migrate compatible data
        if 'doc_id' in sys_cols and sys_cols == master_cols:
            # Schemas match, direct migration
            source_cursor.execute("SELECT * FROM knowledge")
            knowledge_rows = source_cursor.fetchall()
            
            new_knowledge = [row for row in knowledge_rows if row[0] not in existing_docs]
            
            if new_knowledge:
                placeholders = ', '.join(['?'] * len(sys_cols))
                master_cursor.executemany(f"INSERT INTO knowledge VALUES ({placeholders})", new_knowledge)
                print(f"  ✅ Migrated {len(new_knowledge)} knowledge documents")
            else:
                print("  ⏭️  No new knowledge to migrate")
        else:
            print("  ⚠️  Schema mismatch, manual migration needed")
            
    except sqlite3.Error as e:
        print(f"  ⚠️  Error migrating knowledge: {e}")
    finally:
        source_conn.close()
    
    master_conn.commit()

def update_configuration():
    """Update nf_paths.py to point to unified database."""
    print("\n⚙️  Updating configuration...")
    
    # Read current nf_paths.py
    nf_paths_file = Path("./nf_paths.py")
    if not nf_paths_file.exists():
        print("  ⚠️  nf_paths.py not found")
        return
    
    content = nf_paths_file.read_text()
    
    # Update DB_PATH to point to Master
    if "database_master.sqlite3" in content:
        print("  ✅ nf_paths.py already points to Master database")
    else:
        # Create backup
        backup_path = Path("./nf_paths.py.backup")
        shutil.copy2(nf_paths_file, backup_path)
        print(f"  📦 Backed up nf_paths.py to {backup_path}")
        
        # Update path (this would need actual implementation)
        print("  ⚠️  Manual update of nf_paths.py needed:")
        print("     Change DB_PATH to point to neuroforge_system_data/database_master.sqlite3")

def verify_migration(master_conn):
    """Verify the migration was successful."""
    print("\n✅ Verifying migration...")
    
    cursor = master_conn.cursor()
    
    # Count final data
    tables_to_check = [
        ("conversations", "SELECT COUNT(*) FROM unified_conversations WHERE role IS NOT NULL"),
        ("knowledge", "SELECT COUNT(*) FROM knowledge"), 
        ("memory_facts", "SELECT COUNT(*) FROM memory_facts"),
        ("turnlog", "SELECT COUNT(*) FROM unified_conversations WHERE source_table = 'turnlog'"),
        ("operational_logs", "SELECT COUNT(*) FROM operational_logs")
    ]
    
    for table_name, query in tables_to_check:
        try:
            cursor.execute(query)
            count = cursor.fetchone()[0]
            print(f"  📊 {table_name}: {count:,} rows")
        except sqlite3.Error as e:
            print(f"  ⚠️  {table_name}: Error - {e}")

def main():
    """Main migration process."""
    print("🔄 DATABASE UNIFICATION STARTING")
    print("=" * 50)
    
    # Step 1: Backup
    backup_dir = backup_databases()
    
    # Step 2: Open Master database
    master_conn = sqlite3.connect(MASTER_DB)
    
    try:
        # Step 3: Create unified schema
        create_unified_schema(master_conn)
        
        # Step 4: Migrate data
        migrate_conversations(master_conn)
        migrate_memory_facts(master_conn)
        migrate_knowledge(master_conn)
        
        # Step 5: Update configuration
        update_configuration()
        
        # Step 6: Verify
        verify_migration(master_conn)
        
        print(f"\n🎉 UNIFICATION COMPLETE!")
        print(f"📦 Backups stored in: {backup_dir}")
        print(f"🗄️  Unified database: {MASTER_DB}")
        print(f"⚠️  Next: Update nf_paths.py to use Master database")
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        print(f"📦 Restore from backups in: {backup_dir}")
        raise
    finally:
        master_conn.close()

if __name__ == "__main__":
    main()