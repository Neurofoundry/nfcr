"""
Unified Memory System for Neuroforge
====================================

This module provides a single interface for all memory operations in Neuroforge.
It integrates:
- Personal facts (memory_facts FTS5 table)
- Knowledge base (knowledge table with embeddings)
- Session memories (JSON files)
- Automatic fact extraction from conversations
- Natural language memory queries

The goal is to make memory access seamless and natural, without explicit commands.
"""

from __future__ import annotations

import sqlite3
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import re
from datetime import datetime, timezone

# Import existing systems
from nf_recall import search_facts, search_knowledge, search_logs, get_db_connection
from memory_recall import recall_memory
from memory_extraction import extract_personal_facts, store_memory_facts
from nf_personality import _phrase_rotator
from nf_tags import has_conversation_tags, extract_role_text


@dataclass
class MemoryResult:
    """Unified memory search result."""
    content: str
    source: str  # 'facts', 'knowledge', 'sessions', 'logs'
    relevance_score: float
    metadata: Dict[str, Any]


class UnifiedMemorySystem:
    """
    Unified interface for all memory operations in Neuroforge.
    
    This class combines all existing memory systems into a single,
    natural interface that automatically handles:
    - Fact extraction from conversations
    - Multi-source memory search
    - Contextual memory retrieval
    - Phrase rotation for responses
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def search_all_memory(
        self, 
        query: str, 
        conversation_id: Optional[str] = None,
        limit: int = 5
    ) -> Tuple[List[MemoryResult], str]:
        """
        Search across all memory sources and return unified results.
        
        Args:
            query: The search query
            conversation_id: Current conversation ID for context
            limit: Maximum results to return
            
        Returns:
            Tuple of (results, intro_phrase)
        """
        all_results = []
        
        try:
            conn = get_db_connection()
            
            # 1. Search personal facts (FTS5)
            facts_results = search_facts(conn, query, limit)
            for fact in facts_results:
                all_results.append(MemoryResult(
                    content=f"Q: {fact.get('keyphrase', '')}\nA: {fact.get('answer', '')}",
                    source='facts',
                    relevance_score=fact.get('relevance', 0.0),
                    metadata=fact
                ))
            
            # 2. Search knowledge base (documents/crawled content)
            knowledge_results = search_knowledge(conn, query, limit)
            for knowledge in knowledge_results:
                all_results.append(MemoryResult(
                    content=knowledge.get('content', '')[:300] + "...",
                    source='knowledge',
                    relevance_score=knowledge.get('relevance', 0.0),
                    metadata=knowledge
                ))
            
            # 3. Search operational logs
            logs_results = search_logs(conn, query, limit)
            for log in logs_results:
                all_results.append(MemoryResult(
                    content=log.get('message', log.get('content', ''))[:200] + "...",
                    source='logs',
                    relevance_score=log.get('relevance', 0.0),
                    metadata=log
                ))
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error searching database memory: {e}")
        
        # 4. Search session memories (JSON files)
        try:
            session_results = recall_memory(query, max_results=limit//2)
            for session in session_results:
                all_results.append(MemoryResult(
                    content=session.get('content', session.get('summary', ''))[:250] + "...",
                    source='sessions',
                    relevance_score=0.5,  # Default relevance for sessions
                    metadata=session
                ))
        except Exception as e:
            self.logger.error(f"Error searching session memory: {e}")
        
        # Sort by relevance score
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Limit total results
        all_results = all_results[:limit]
        
        # Generate appropriate intro phrase
        intro_phrase = self._get_memory_intro_phrase(all_results)
        
        return all_results, intro_phrase
    
    def _get_memory_intro_phrase(self, results: List[MemoryResult]) -> str:
        """Generate contextual intro phrase based on search results."""
        # Don't inject phrases - let the AI respond naturally
        # The enriched_prompt already includes context, no need for meta-commentary
        return ""
    
    def extract_and_store_facts(
        self, 
        user_message: str, 
        conversation_id: str,
        auto_extract: bool = True
    ) -> int:
        """
        Automatically extract and store personal facts from user messages.
        
        Args:
            user_message: The user's message
            conversation_id: Current conversation ID
            auto_extract: Whether to automatically extract facts
            
        Returns:
            Number of facts extracted and stored
        """
        if not auto_extract:
            return 0
            
        try:
            # If the incoming message appears to be a conversation blob containing
            # role markers (User:/Assistant: or tokenized markers), extract only
            # the user's segments so we don't accidentally feed assistant text to
            # the personal-fact extractor.
            if has_conversation_tags(user_message).get('has_user'):
                user_text = extract_role_text(user_message, role='user')
            else:
                user_text = user_message

            # Extract facts using existing system
            facts = extract_personal_facts(user_text, conversation_id)
            
            if facts:
                # Store facts using existing system
                stored_count = store_memory_facts(facts, conversation_id)
                self.logger.info(f"Extracted and stored {stored_count} personal facts from conversation {conversation_id}")
                return stored_count
                
        except Exception as e:
            self.logger.error(f"Error extracting/storing facts: {e}")
            
        return 0
    
    def is_memory_query(self, user_message: str) -> bool:
        """
        Detect if user message is asking for memory/recall without explicit commands.
        
        Natural phrases that indicate memory queries:
        - "what do you know about..."
        - "do you remember..."
        - "tell me about my..."
        - "what did we discuss..."
        """
        memory_patterns = [
            r"what\s+do\s+you\s+know\s+about",
            r"do\s+you\s+remember",
            r"tell\s+me\s+about\s+my",
            r"what\s+did\s+we\s+discuss",
            r"remind\s+me\s+about",
            r"what\s+have\s+I\s+told\s+you",
            r"do\s+you\s+recall",
            r"what's\s+my\s+(?:favorite|usual|typical)",
            r"how\s+do\s+I\s+usually",
            r"what\s+are\s+my\s+preferences"
        ]
        
        message_lower = user_message.lower()
        for pattern in memory_patterns:
            if re.search(pattern, message_lower):
                return True
        
        return False
    
    def extract_memory_query(self, user_message: str) -> str:
        """
        Extract the actual query from a memory request.
        
        "What do you know about my coffee habits?" -> "coffee habits"
        """
        # Remove common memory trigger phrases
        patterns_to_remove = [
            r"what\s+do\s+you\s+know\s+about\s+",
            r"do\s+you\s+remember\s+",
            r"tell\s+me\s+about\s+",
            r"what\s+did\s+we\s+discuss\s+about\s+",
            r"remind\s+me\s+about\s+",
            r"what\s+have\s+I\s+told\s+you\s+about\s+",
            r"do\s+you\s+recall\s+",
        ]
        
        cleaned_query = user_message.lower()
        for pattern in patterns_to_remove:
            cleaned_query = re.sub(pattern, "", cleaned_query, flags=re.IGNORECASE)
        
        # Remove question marks and clean up
        cleaned_query = cleaned_query.strip("?.,!").strip()
        
        return cleaned_query or user_message
    
    def format_memory_response(
        self, 
        results: List[MemoryResult], 
        intro_phrase: str
    ) -> str:
        """
        Format memory search results into a natural response.
        """
        if not results:
            return intro_phrase
        
        response_parts = [intro_phrase]
        
        # Group results by source for better organization
        by_source = {}
        for result in results:
            if result.source not in by_source:
                by_source[result.source] = []
            by_source[result.source].append(result)
        
        # Format each source group
        for source, source_results in by_source.items():
            if source == 'facts' and source_results:
                response_parts.append("\nPersonal details:")
                for result in source_results[:2]:  # Limit to avoid overwhelming
                    response_parts.append(f"• {result.content}")
            
            elif source == 'sessions' and source_results:
                response_parts.append("\nFrom our conversations:")
                for result in source_results[:2]:
                    response_parts.append(f"• {result.content}")
            
            elif source == 'knowledge' and source_results:
                response_parts.append("\nFrom my knowledge base:")
                for result in source_results[:1]:  # Knowledge can be lengthy
                    response_parts.append(f"• {result.content}")
        
        return "\n".join(response_parts)


# Global unified memory instance
unified_memory = UnifiedMemorySystem()


# ==============================================================================
# Convenience Functions (for backward compatibility)
# ==============================================================================

def unified_memory_lookup(
    query: str, 
    conversation_id: Optional[str] = None,
    limit: int = 5
) -> Tuple[List[str], str]:
    """
    Unified memory lookup that combines all memory sources.
    
    This replaces the broken memory_lookup() function with a unified approach.
    
    Returns:
        Tuple of (formatted_results, intro_phrase)
    """
    results, intro_phrase = unified_memory.search_all_memory(query, conversation_id, limit)
    
    # Format results for backward compatibility
    formatted_results = []
    for result in results:
        formatted_results.append(result.content)
    
    return formatted_results, intro_phrase


def process_conversation_for_memory(
    user_message: str,
    conversation_id: str,
    auto_extract: bool = True
) -> int:
    """
    Process a user message for automatic memory extraction.
    
    Returns:
        Number of facts extracted and stored
    """
    return unified_memory.extract_and_store_facts(user_message, conversation_id, auto_extract)


def detect_and_handle_memory_query(
    user_message: str,
    conversation_id: Optional[str] = None
) -> Optional[str]:
    """
    Detect natural memory queries and return formatted response.
    
    Args:
        user_message: The user's message
        conversation_id: Current conversation ID
        
    Returns:
        Formatted memory response if it's a memory query, None otherwise
    """
    if not unified_memory.is_memory_query(user_message):
        return None
    
    # Extract the actual query
    memory_query = unified_memory.extract_memory_query(user_message)
    
    # Search memory
    results, intro_phrase = unified_memory.search_all_memory(memory_query, conversation_id)
    
    # Format response
    return unified_memory.format_memory_response(results, intro_phrase)


# ==============================================================================
# Testing / Standalone Execution
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Unified Memory System Test")
    print("=" * 70)
    
    # Test 1: Memory query detection
    print("\n🔍 Test 1: Memory Query Detection")
    print("-" * 40)
    test_queries = [
        "What do you know about my coffee habits?",
        "Do you remember what we discussed about Python?",
        "Tell me about my work preferences",
        "How's the weather today?",  # Not a memory query
        "What's my favorite programming language?",
    ]
    
    for query in test_queries:
        is_memory = unified_memory.is_memory_query(query)
        print(f"'{query}' -> Memory query: {is_memory}")
    
    # Test 2: Unified search
    print("\n🔍 Test 2: Unified Memory Search")
    print("-" * 40)
    results, intro = unified_memory.search_all_memory("coffee", limit=3)
    print(f"Query: 'coffee'")
    print(f"Intro: {intro}")
    print(f"Results: {len(results)}")
    for i, result in enumerate(results, 1):
        print(f"  {i}. [{result.source}] {result.content[:100]}...")
    
    print("\n" + "=" * 70)
    print("✅ Unified Memory System test complete!")