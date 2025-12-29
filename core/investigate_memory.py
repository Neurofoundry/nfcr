#!/usr/bin/env python3
"""
Investigate conversational memory storage and retrieval system
"""

import sqlite3
import json

def investigate_conversation_memory():
    """Check how personal conversation details are stored and retrieved."""
    print("🧠 INVESTIGATING CONVERSATIONAL MEMORY SYSTEM")
    print("=" * 60)
    
    db_path = "neuroforge_system_data/database.sqlite3"
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 1. Check memory_facts table (this should store personal details)
        print("📋 MEMORY_FACTS TABLE:")
        cursor.execute("SELECT COUNT(*) FROM memory_facts")
        facts_count = cursor.fetchone()[0]
        print(f"  Total memory facts: {facts_count}")
        
        if facts_count > 0:
            cursor.execute("SELECT fact_text, context, confidence FROM memory_facts LIMIT 5")
            facts = cursor.fetchall()
            print("  Sample memory facts:")
            for i, (fact, context, confidence) in enumerate(facts, 1):
                print(f"    {i}. {fact} (confidence: {confidence})")
                print(f"       Context: {context}")
        else:
            print("  ❌ NO MEMORY FACTS FOUND!")
        
        # 2. Check if conversations are being processed for memory extraction
        print(f"\n💭 CONVERSATION PROCESSING:")
        cursor.execute("""
            SELECT conversation_id, user_prompt, ai_response 
            FROM unified_conversations WHERE role IN ('user', 'assistant') AND user_prompt LIKE '%coffee%' OR user_prompt LIKE '%cup%' OR user_prompt LIKE '%green%'
            LIMIT 3
        """)
        
        relevant_convs = cursor.fetchall()
        print(f"  Conversations about coffee/cups/green: {len(relevant_convs)}")
        
        for i, (conv_id, user_prompt, ai_response) in enumerate(relevant_convs, 1):
            print(f"    {i}. {conv_id[:8]}...")
            print(f"       User: {user_prompt[:80]}...")
            print(f"       AI: {ai_response[:80]}...")
        
        # 3. Check turnlog for detailed conversation history
        print(f"\n📝 TURNLOG ANALYSIS:")
        cursor.execute("SELECT COUNT(*) FROM unified_conversations WHERE source_table = 'turnlog'")
        turnlog_count = cursor.fetchone()[0]
        print(f"  Total turns: {turnlog_count}")
        
        # Look for personal details in recent conversations
        cursor.execute("""
            SELECT conv_id, role, text 
            FROM unified_conversations WHERE text LIKE '%coffee%' OR text LIKE '%cup%' OR text LIKE '%green%'
            ORDER BY id DESC
            LIMIT 5
        """)
        
        personal_turns = cursor.fetchall()
        print(f"  Turns with personal details: {len(personal_turns)}")
        
        for i, (conv_id, role, text) in enumerate(personal_turns, 1):
            print(f"    {i}. {conv_id[:8]}: {role}")
            print(f"       {text[:100]}...")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

def check_memory_extraction_system():
    """Check if there's a system to extract personal facts from conversations."""
    print(f"\n🔍 CHECKING MEMORY EXTRACTION SYSTEM")
    print("=" * 60)
    
    # Look for memory-related files
    import os
    memory_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if any(keyword in file.lower() for keyword in ['memory', 'recall', 'fact', 'extract']):
                memory_files.append(os.path.join(root, file))
    
    print(f"📁 Memory-related files:")
    for file in memory_files:
        print(f"  {file}")
    
    # Check for specific memory processing functions
    print(f"\n🔧 CHECKING MEMORY PROCESSING:")
    
    # Check memory_recall.py
    if os.path.exists("memory_recall.py"):
        print("✅ Found memory_recall.py")
        # Read first few lines to understand its purpose
        with open("memory_recall.py", 'r') as f:
            lines = f.readlines()[:10]
        print("  Purpose:")
        for line in lines:
            if line.strip().startswith('"""') or line.strip().startswith('#'):
                print(f"    {line.strip()}")
    
    # Check memory_logging.py
    if os.path.exists("memory_logging.py"):
        print("✅ Found memory_logging.py")
        with open("memory_logging.py", 'r') as f:
            lines = f.readlines()[:10]
        print("  Purpose:")
        for line in lines:
            if line.strip().startswith('"""') or line.strip().startswith('#'):
                print(f"    {line.strip()}")

def check_memory_integration():
    """Check if memory system is integrated into main chat flow."""
    print(f"\n🔗 CHECKING MEMORY INTEGRATION")
    print("=" * 60)
    
    # Check aegis_unified_core.py for memory calls
    import os
    if os.path.exists("aegis_unified_core.py"):
        print("📄 Checking aegis_unified_core.py for memory integration...")
        
        with open("aegis_unified_core.py", 'r') as f:
            content = f.read()
        
        memory_keywords = ['memory_facts', 'memory_recall', 'extract_facts', 'store_memory', 'recall_memory']
        
        for keyword in memory_keywords:
            if keyword in content:
                print(f"  ✅ Found: {keyword}")
                # Find the line for context
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if keyword in line:
                        print(f"    Line {i+1}: {line.strip()}")
                        break
            else:
                print(f"  ❌ Missing: {keyword}")

if __name__ == "__main__":
    investigate_conversation_memory()
    check_memory_extraction_system()
    check_memory_integration()