#!/usr/bin/env python3
"""
Memory Fact Extraction System
==============================

This module extracts personal facts and preferences from conversation history
and stores them in the memory_facts table for future recall.

The system looks for patterns like:
- "I use a green cup for coffee"
- "My favorite color is blue" 
- "I work at Microsoft"
- "I have two cats named Fluffy and Mittens"

These get stored as searchable facts that can be recalled later.
"""

import sqlite3
import re
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import uuid

from nf_paths import DB_PATH

# Patterns that indicate personal facts
PERSONAL_PATTERNS = [
    # Possessions and preferences
    (r"I (use|have|own|like|prefer|love|hate|drive|drink|eat|wear) (\w.*)", "possession"),
    (r"My (favorite|preferred|usual|regular|typical) (\w+) is (\w.*)", "preference"),
    (r"My (\w+) is (\w.*)", "attribute"),
    
    # Work and location
    (r"I (work at|work for|am employed by|job at) (\w.*)", "employment"),
    (r"I live in (\w.*)", "location"),
    (r"I am from (\w.*)", "origin"),
    
    # Family and relationships
    (r"My (\w+) is named (\w.*)", "relationship"),
    (r"I have (\d+|\w+) (\w+)", "quantity"),
    
    # Activities and habits
    (r"I (usually|typically|always|often|never) (\w.*)", "habit"),
    (r"I (go to|visit|attend) (\w.*)", "activity"),
]

def extract_personal_facts(text: str, conversation_id: str) -> List[Dict]:
    """Extract personal facts from conversation text."""
    facts = []
    
    for pattern, fact_type in PERSONAL_PATTERNS:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        for match in matches:
            if fact_type == "possession":
                action, item = match.groups()
                fact = f"User {action} {item}"
                keyphrase = f"{action} {item.split()[0] if item.split() else item}"
                
            elif fact_type == "preference":
                category, subcategory, value = match.groups()
                fact = f"User's {category} {subcategory} is {value}"
                keyphrase = f"{category} {subcategory}"
                
            elif fact_type == "attribute":
                attribute, value = match.groups()
                fact = f"User's {attribute} is {value}"
                keyphrase = attribute
                
            elif fact_type == "employment":
                action, company = match.groups()
                fact = f"User works at {company}"
                keyphrase = "employment"
                
            elif fact_type == "location":
                location = match.groups()[0]
                fact = f"User lives in {location}"
                keyphrase = "location"
                
            elif fact_type == "origin":
                origin = match.groups()[0]
                fact = f"User is from {origin}"
                keyphrase = "origin"
                
            elif fact_type == "relationship":
                relation, name = match.groups()
                fact = f"User's {relation} is named {name}"
                keyphrase = f"{relation} name"
                
            elif fact_type == "quantity":
                amount, item = match.groups()
                fact = f"User has {amount} {item}"
                keyphrase = item
                
            elif fact_type == "habit":
                frequency, activity = match.groups()
                fact = f"User {frequency} {activity}"
                keyphrase = f"habit {activity.split()[0] if activity.split() else activity}"
                
            elif fact_type == "activity":
                action, place = match.groups()
                fact = f"User goes to {place}"
                keyphrase = f"goes to {place.split()[0] if place.split() else place}"
            
            else:
                continue
                
            facts.append({
                "keyphrase": keyphrase[:100],  # Limit length
                "answer": fact[:500],           # Limit length  
                "source_url": f"conversation:{conversation_id}",
                "confidence": 0.8,              # Medium confidence for pattern matching
                "fact_type": fact_type,
                "original_text": match.group(0)
            })
    
    return facts

def store_memory_facts(facts: List[Dict], conversation_id: str) -> int:
    """Store extracted facts in the memory_facts table."""
    if not facts:
        return 0
    
    db_path = str(DB_PATH)
    stored_count = 0
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        for fact in facts:
            # Check if similar fact already exists
            cursor.execute("""
                SELECT COUNT(*) FROM memory_facts 
                WHERE keyphrase = ? AND answer = ?
            """, (fact["keyphrase"], fact["answer"]))
            
            if cursor.fetchone()[0] == 0:  # Not a duplicate
                # Insert into memory_facts table using existing schema
                cursor.execute("""
                    INSERT OR REPLACE INTO memory_facts_fts (keyphrase, answer, source_url, tags, created_ts)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    fact["keyphrase"],
                    fact["answer"], 
                    fact["source_url"],
                    json.dumps({
                        "confidence": fact["confidence"],
                        "fact_type": fact["fact_type"],
                        "original_text": fact["original_text"],
                        "conversation_id": conversation_id
                    }),
                    datetime.utcnow().isoformat()
                ))
                stored_count += 1
        
        conn.commit()
        conn.close()
        
        print(f"📝 Stored {stored_count} new memory facts from conversation {conversation_id[:8]}...")
        
    except Exception as e:
        print(f"❌ Error storing memory facts: {e}")
    
    return stored_count

def process_conversation_for_memory(conversation_id: str, user_text: str, ai_text: str) -> int:
    """Process a single conversation for memory extraction."""
    # Extract facts from user messages (not AI responses)
    facts = extract_personal_facts(user_text, conversation_id)
    
    if facts:
        print(f"🧠 Found {len(facts)} potential facts in conversation {conversation_id[:8]}...")
        for fact in facts:
            print(f"   - {fact['keyphrase']}: {fact['answer']}")
        
        return store_memory_facts(facts, conversation_id)
    
    return 0

def process_all_conversations() -> int:
    """Process all existing conversations for memory extraction."""
    print("🔄 PROCESSING ALL CONVERSATIONS FOR MEMORY EXTRACTION")
    print("=" * 60)
    
    db_path = str(DB_PATH)
    total_facts = 0
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all conversations
        cursor.execute("""
            SELECT conversation_id, user_prompt, ai_response 
            FROM unified_conversations ORDER BY timestamp DESC
        """)
        
        conversations = cursor.fetchall()
        print(f"📋 Processing {len(conversations)} conversations...")
        
        for conv_id, user_prompt, ai_response in conversations:
            facts_count = process_conversation_for_memory(conv_id, user_prompt or "", ai_response or "")
            total_facts += facts_count
        
        conn.close()
        
        print(f"✅ Extraction complete! Total facts stored: {total_facts}")
        
        # Show sample of what was stored
        show_stored_facts()
        
    except Exception as e:
        print(f"❌ Error processing conversations: {e}")
    
    return total_facts

def show_stored_facts():
    """Show sample of stored memory facts."""
    db_path = str(DB_PATH)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM memory_facts")
        total = cursor.fetchone()[0]
        
        print(f"\n📊 MEMORY FACTS SUMMARY")
        print(f"Total stored facts: {total}")
        
        if total > 0:
            cursor.execute("""
                SELECT keyphrase, answer, source_url 
                FROM memory_facts 
                ORDER BY created_ts DESC 
                LIMIT 5
            """)
            
            facts = cursor.fetchall()
            print("Recent facts:")
            for i, (keyphrase, answer, source) in enumerate(facts, 1):
                print(f"  {i}. {keyphrase}: {answer}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error showing facts: {e}")

if __name__ == "__main__":
    # Test the extraction system
    print("🧠 MEMORY FACT EXTRACTION SYSTEM")
    print("=" * 60)
    
    # Test with sample text
    test_text = "I use a green cup to drink coffee every morning. My favorite color is blue and I work at Microsoft."
    test_conv_id = "test-conv-123"
    
    print(f"\n🧪 Testing extraction with: '{test_text}'")
    facts = extract_personal_facts(test_text, test_conv_id)
    
    print(f"Extracted {len(facts)} facts:")
    for fact in facts:
        print(f"  - {fact['keyphrase']}: {fact['answer']}")
    
    # Process all existing conversations
    process_all_conversations()