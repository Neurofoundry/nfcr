"""
nf_personality.py
==================

Dynamic personality and context management for Neuroforge.

This module provides intelligent system prompt generation that adapts based on:
- Conversation history (recent turns for context continuity)
- User feedback sentiment (thumbs up/down tracking)
- Recalled memories (relevant past interactions)
- Phrase rotation (prevents repetitive responses)

The goal is to make Neuroforge feel like a thoughtful companion rather than
a robotic command processor.
"""

from __future__ import annotations

import os
import json
import random
import sqlite3
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone, timedelta

from nf_paths import DB_PATH


# ==============================================================================
# Phrase Rotation System
# ==============================================================================

class PhraseRotator:
    """
    Manages phrase variety to prevent repetitive responses.
    
    Tracks usage frequency of common phrases and provides rotation
    strategies to keep conversations fresh.
    """
    
    def __init__(self):
        self.usage_counts: Dict[str, int] = {}
        self.phrase_pools: Dict[str, List[str]] = {
            "closing_status": [
                "System integrity is nominal. Neuroforge remains vigilant.",
                "All systems operational. Standing by for your next move.",
                "Everything's running smooth—ready when you are.",
                "Fortress is secure. What's next, Architect?",
                "Status: Green across the board. I'm here.",
            ],
            "retrieval_miss": [
                "Didn't pull a hit—want me to look deeper?",
                "Nothing on record for that—yet.",
                "Index is dry on that one.",
                "That's a blank—nothing indexed.",
                "No matches found. Should I expand the search?",
            ],
            "acknowledgment": [
                "Got it.",
                "Understood.",
                "Noted.",
                "On it.",
                "Copy that.",
            ]
        }
    
    def get_phrase(self, category: str, usage_threshold: float = 0.20) -> Optional[str]:
        """
        Get a phrase from the specified category using weighted rotation.
        
        Args:
            category: The phrase category (e.g., 'closing_status')
            usage_threshold: Maximum usage frequency before forcing rotation (0.0-1.0)
        
        Returns:
            A phrase from the pool, or None if category doesn't exist
        """
        if category not in self.phrase_pools:
            return None
        
        pool = self.phrase_pools[category]
        if not pool:
            return None
        
        # Calculate usage frequency for each phrase
        total_uses = sum(self.usage_counts.get(f"{category}:{p}", 0) for p in pool)
        
        # Build weighted selection based on inverse usage
        weights = []
        for phrase in pool:
            key = f"{category}:{phrase}"
            uses = self.usage_counts.get(key, 0)
            
            # If any phrase is over-used (above threshold), strongly prefer others
            if total_uses > 0:
                frequency = uses / total_uses
                if frequency > usage_threshold:
                    weight = 0.1  # Heavily discourage
                else:
                    weight = 1.0 / (uses + 1)  # Favor less-used phrases
            else:
                weight = 1.0
            
            weights.append(weight)
        
        # Weighted random selection
        selected = random.choices(pool, weights=weights, k=1)[0]
        
        # Track usage
        key = f"{category}:{selected}"
        self.usage_counts[key] = self.usage_counts.get(key, 0) + 1
        
        return selected
    
    def should_use_phrase(self, category: str, probability: float = 0.20) -> bool:
        """
        Decide whether to use a phrase from this category at all.
        
        Args:
            category: The phrase category
            probability: Chance of using the phrase (0.0-1.0)
        
        Returns:
            True if the phrase should be used this time
        """
        return random.random() < probability


# Global phrase rotator instance
_phrase_rotator = PhraseRotator()


# ==============================================================================
# Sentiment Analysis
# ==============================================================================

def get_feedback_sentiment(conversation_id: Optional[str] = None, days: int = 7) -> Dict[str, Any]:
    """
    Analyze user feedback sentiment from operational_logs.
    
    Args:
        conversation_id: Specific conversation to analyze (None = all recent)
        days: Number of days to look back
    
    Returns:
        Dictionary with sentiment metrics:
        {
            "thumbs_up": int,
            "thumbs_down": int,
            "ratio": float,  # positive ratio (0.0-1.0)
            "total_feedback": int,
            "recent_tags": List[str]  # most common feedback tags
        }
    """
    # Note: operational_logs table was dropped in Phase 1 migration (was empty)
    # Return default values until sentiment logging is re-implemented
    return {
        "thumbs_up": 0,
        "thumbs_down": 0,
        "ratio": 0.5,
        "total_feedback": 0,
        "recent_tags": []
    }


# ==============================================================================
# Conversation Context
# ==============================================================================

def get_recent_turns(conversation_id: str, limit: int = 5) -> List[Dict[str, str]]:
    """
    Retrieve recent conversation turns for context.
    
    Args:
        conversation_id: The conversation to fetch
        limit: Maximum number of recent turns to return
    
    Returns:
        List of turns, each with 'role' and 'text' keys
    """
    db_path = str(DB_PATH)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT role, text
            FROM unified_conversations
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (conversation_id, limit * 2))  # Get more since user+assistant are separate rows
        
        rows = cursor.fetchall()
        conn.close()
        
        # Reverse to get chronological order
        turns = [{"role": r[0], "text": r[1]} for r in reversed(rows)]
        
        return turns[:limit]
    
    except Exception as e:
        print(f"Error getting recent turns: {e}")
        return []


# ==============================================================================
# Enhanced System Prompt Builder
# ==============================================================================

def build_contextual_prompt(
    base_prompt: str,
    conversation_id: Optional[str] = None,
    user_message: Optional[str] = None,
    include_sentiment: bool = True,
    include_context: bool = True
) -> str:
    """
    Build an enhanced system prompt with dynamic context.
    
    Args:
        base_prompt: The base system prompt from response_profile.yaml
        conversation_id: Current conversation ID for context retrieval
        user_message: The current user message (for relevance filtering)
        include_sentiment: Whether to include feedback sentiment
        include_context: Whether to include recent conversation context
    
    Returns:
        Enhanced system prompt string
    """
    parts = [base_prompt]
    
    # Add feedback sentiment if available (but prioritize user preferences for compliance)
    if include_sentiment:
        sentiment = get_feedback_sentiment(conversation_id, days=7)
        
        if sentiment["total_feedback"] > 0:
            # Only add minimal feedback context to avoid conflicting with compliance goals
            if sentiment["ratio"] < 0.3:  # Negative feedback
                parts.append("\n[Context: Focus on being helpful and compliant with user instructions.]")
            
            # Limit tag-specific guidance to essentials only
            if "hallucination" in sentiment["recent_tags"]:
                parts.append("[Important: Use provided context and memory accurately.]")
            # Skip other feedback tags that might encourage verbosity or questioning
    
    # Add recent conversation context
    if include_context and conversation_id:
        turns = get_recent_turns(conversation_id, limit=3)
        
        if turns:
            context_lines = []
            for turn in turns[-3:]:  # Last 3 turns for brevity
                role = "User" if turn["role"] == "user" else "You"
                text = turn["text"][:100] + "..." if len(turn["text"]) > 100 else turn["text"]
                context_lines.append(f"{role}: {text}")
            
            if context_lines:
                parts.append("\n[Recent conversation context:]")
                parts.extend(context_lines)
    
    # Skip phrase variation guidance to maintain compliance focus
    # parts.append("\n[Note: Skip the status report closing unless specifically relevant to security/system topics]")
    
    return "\n".join(parts)


# ==============================================================================
# Phrase Suggestion Helper
# ==============================================================================

def suggest_phrase_alternatives(category: str) -> List[str]:
    """
    Get all available phrases in a category for variety.
    
    Args:
        category: The phrase category
    
    Returns:
        List of phrases in that category
    """
    return _phrase_rotator.phrase_pools.get(category, [])


# ==============================================================================
# Testing / Standalone Execution
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("nf_personality.py - Dynamic Personality System Test")
    print("=" * 70)
    
    # Test 1: Phrase rotation
    print("\n📝 Test 1: Phrase Rotation")
    print("-" * 70)
    for i in range(10):
        phrase = _phrase_rotator.get_phrase("closing_status")
        print(f"  {i+1}. {phrase}")
    
    # Test 2: Feedback sentiment
    print("\n👍 Test 2: Feedback Sentiment Analysis")
    print("-" * 70)
    sentiment = get_feedback_sentiment(days=30)
    print(f"  Thumbs Up: {sentiment['thumbs_up']}")
    print(f"  Thumbs Down: {sentiment['thumbs_down']}")
    print(f"  Positive Ratio: {sentiment['ratio']:.2f}")
    print(f"  Recent Tags: {', '.join(sentiment['recent_tags']) if sentiment['recent_tags'] else 'None'}")
    
    # Test 3: Contextual prompt
    print("\n🧠 Test 3: Contextual Prompt Generation")
    print("-" * 70)
    base = "You are Neuroforge, a helpful AI assistant."
    enhanced = build_contextual_prompt(base, include_sentiment=True, include_context=False)
    print(enhanced)
    
    print("\n" + "=" * 70)
    print("✅ All tests complete!")
