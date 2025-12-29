"""
nf_intent.py
=============

Intent classification for natural language understanding.

This module analyzes user messages to determine whether they're requesting
tool execution or just having a conversation. This prevents false-positive
triggers like "the baby is crawling" activating the web crawler.

Key Features:
- Context-aware intent detection
- Confidence scoring
- Tool disambiguation (which tool, if any)
- Conversational vs. command classification
"""

from __future__ import annotations

import re
import json
from typing import Optional, Tuple, List, Dict, Any


# ==============================================================================
# Intent Patterns
# ==============================================================================

# Each tool has multiple trigger patterns with confidence weights
TOOL_PATTERNS = {
    "crawl": [
        # High confidence patterns (explicit requests)
        (r"(?:please\s+)?(?:crawl|scrape|fetch)\s+(?:this\s+)?(?:url|website|site|page)?.*(?:https?://\S+)", 0.95),
        (r"(?:ingest|index)\s+(?:this\s+)?(?:url|website|site).*(?:https?://\S+)", 0.90),
        (r"(?:go\s+)?(?:crawl|scrape)\s+(?:https?://\S+)", 0.95),
        (r"crawl.*(?:for\s+me|please).*https?://", 0.90),  # "Crawl this site for me"
        
        # Medium confidence (contextual)
        (r"(?:can\s+you\s+)?crawl\s+\S+\.(com|org|net|io)", 0.75),
        (r"(?:please\s+)?(?:pull|grab|get)\s+(?:data|content)\s+from\s+(?:https?://|www\.)", 0.80),
        
        # Low confidence (ambiguous)
        (r"\bcrawl\b", 0.15),  # Just the word "crawl" alone
    ],
    "recall": [
        # High confidence
        (r"(?:please\s+)?recall\s+(?:what|when|where|how|everything)", 0.95),
        (r"(?:do\s+you\s+)?remember\s+(?:when|what|that time|our|the)", 0.85),
        (r"(?:do\s+you\s+)?remember.*(?:conversation|discussion|talk)", 0.85),
        (r"(?:please\s+)?(?:search|find)\s+(?:in\s+)?(?:your\s+)?(?:memory|logs|history)", 0.90),
        
        # Medium confidence
        (r"what\s+(?:did\s+)?(?:we|I)\s+(?:talk|discuss|say)\s+about", 0.70),
        (r"(?:look\s+up|find)\s+\S+\s+(?:in\s+)?(?:your\s+)?(?:records|database)", 0.75),
    ],
    "ingest": [
        # High confidence
        (r"(?:please\s+)?ingest\s+(?:this\s+)?(?:the\s+)?(?:file|document|text)", 0.95),
        (r"(?:please\s+)?(?:add|import|load)\s+(?:this\s+)?(?:the\s+)?(?:file|document)\s+(?:to|into)\s+(?:the\s+)?(?:database|knowledge)", 0.90),
        (r"ingest.*file.*\.(txt|pdf|md|doc)", 0.90),  # "ingest the file notes.txt"
        
        # Medium confidence
        (r"(?:can\s+you\s+)?(?:read|parse|process)\s+(?:this\s+)?(?:the\s+)?(?:file|document)(?:\s+for\s+me)?", 0.70),
    ],
    "render": [
        # High confidence
        (r"(?:please\s+)?(?:render|generate|create|make)\s+(?:an?\s+)?image", 0.95),
        (r"(?:please\s+)?(?:draw|visualize|show\s+me)\s+", 0.85),
        (r"/feature\s+4", 0.99),  # Explicit feature command
    ],
    "backup": [
        # High confidence
        (r"(?:please\s+)?(?:create|make|do)\s+(?:a\s+)?backup", 0.95),
        (r"(?:please\s+)?backup\s+(?:the\s+)?(?:system|database|everything)", 0.95),
    ],
    "feedback": [
        # High confidence - explicit corrections
        (r"(?:please\s+)?(?:don't|do not|dont)\s+(?:say|use|include)\s+[\"']?(.+?)[\"']?(?:\s+|$)", 0.95),
        (r"[\"']?(.+?)[\"']?\s+(?:is|was)\s+(?:an?\s+)?(?:error|mistake|wrong|incorrect)", 0.90),
        (r"(?:that|the)\s+[\"']?(.+?)[\"']?\s+(?:thing|part|word|phrase)\s+(?:you\s+said\s+)?(?:is|was)\s+(?:wrong|incorrect|bad)", 0.90),
        (r"(?:please\s+)?(?:stop|avoid)\s+(?:saying|using)\s+[\"']?(.+?)[\"']?", 0.85),
        
        # Medium confidence - general corrections
        (r"(?:that|this)\s+(?:was|is)\s+(?:wrong|incorrect|bad|not right)", 0.70),
        (r"(?:please\s+)?(?:fix|correct)\s+(?:that|this)", 0.65),
    ]
}


# Negative context patterns - if these match, reduce confidence significantly
NEGATIVE_CONTEXTS = [
    # "crawl" in baby context
    (r"\b(?:baby|infant|child|toddler)\b.*\bcrawl", "crawl", -0.80),
    (r"\bcrawl.*\b(?:baby|infant|child|toddler)\b", "crawl", -0.80),
    
    # "crawl" in animal/creature context
    (r"\b(?:spider|bug|ant|creature|snake)\b.*\bcrawl", "crawl", -0.70),
    
    # "recall" in memory/emotion context (not data recall)
    (r"(?:can't|cannot|don't)\s+recall", "recall", -0.60),
    (r"(?:trying|attempt)\s+to\s+recall", "recall", -0.50),
    
    # General conversational context
    (r"\b(?:what|how|why)\s+(?:is|are|does)\b.*\?$", None, -0.30),  # Questions about concepts
    (r"\b(?:tell|explain|describe)\s+(?:me\s+)?(?:about|how)\b", None, -0.40),  # Educational requests
]


# ==============================================================================
# Intent Classifier
# ==============================================================================

def classify_intent(user_message: str, conversation_context: Optional[List[str]] = None) -> Tuple[Optional[str], float]:
    """
    Classify user intent to determine if a tool should be executed.
    
    Args:
        user_message: The user's current message
        conversation_context: List of recent messages for context (optional)
    
    Returns:
        Tuple of (tool_name, confidence) where:
        - tool_name is None if this is conversational (no tool needed)
        - confidence is a float from 0.0 (definitely not) to 1.0 (definitely yes)
    
    Examples:
        >>> classify_intent("Please crawl https://example.com")
        ('crawl', 0.95)
        
        >>> classify_intent("The baby is crawling now!")
        (None, 0.0)
        
        >>> classify_intent("Can you help me understand how crawling works?")
        (None, 0.15)
    """
    msg_lower = user_message.lower()
    
    best_match: Optional[str] = None
    best_confidence: float = 0.0
    
    # Check each tool's patterns
    for tool_name, patterns in TOOL_PATTERNS.items():
        for pattern, base_confidence in patterns:
            if re.search(pattern, msg_lower, re.IGNORECASE):
                # Found a potential match
                confidence = base_confidence
                
                # Apply negative context adjustments
                for neg_pattern, neg_tool, penalty in NEGATIVE_CONTEXTS:
                    if neg_tool is None or neg_tool == tool_name:
                        if re.search(neg_pattern, msg_lower, re.IGNORECASE):
                            confidence += penalty
                
                # Ensure confidence stays in bounds
                confidence = max(0.0, min(1.0, confidence))
                
                # Track best match
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = tool_name
    
    # If confidence is below threshold, treat as conversational
    if best_confidence < 0.50:
        return (None, 0.0)
    
    return (best_match, best_confidence)


def is_conversational(user_message: str) -> bool:
    """
    Quick check if message is purely conversational (no tool needed).
    
    Args:
        user_message: The user's message
    
    Returns:
        True if message is conversational, False if tool execution suspected
    """
    tool, confidence = classify_intent(user_message)
    return tool is None or confidence < 0.50


def extract_url(user_message: str) -> Optional[str]:
    """
    Extract a URL from user message if present.
    
    Args:
        user_message: The user's message
    
    Returns:
        The first URL found, or None
    """
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    match = re.search(url_pattern, user_message)
    return match.group(0) if match else None


def extract_feedback_pattern(user_message: str) -> Optional[str]:
    """
    Extract the specific phrase/pattern the user is flagging as problematic.
    
    Args:
        user_message: The user's feedback message
    
    Returns:
        The extracted pattern to avoid, or None if not found
    
    Examples:
        "please don't say thefuck" -> "thefuck"
        "thefuck is an error" -> "thefuck"
        'that "here's what I found" phrase was wrong' -> "here's what I found"
    """
    # Try patterns with capture groups (most specific first)
    patterns = [
        r"(?:please\s+)?(?:don't|do not|dont)\s+(?:say|use|include)\s+[\"'](.+?)[\"']",  # quoted
        r"(?:please\s+)?(?:don't|do not|dont)\s+(?:say|use|include)\s+(\S+)",  # single word
        r"[\"'](.+?)[\"']\s+(?:is|was)\s+(?:an?\s+)?(?:error|mistake|wrong|incorrect)",  # quoted error
        r"(\S+)\s+(?:is|was)\s+(?:an?\s+)?(?:error|mistake|wrong|incorrect)",  # word error
        r"(?:that|the)\s+[\"'](.+?)[\"']\s+(?:thing|part|word|phrase)\s+(?:you\s+said\s+)?(?:is|was)\s+(?:wrong|incorrect|bad)",  # quoted critique
        r"(?:please\s+)?(?:stop|avoid)\s+(?:saying|using)\s+[\"'](.+?)[\"']",  # quoted stop
        r"(?:please\s+)?(?:stop|avoid)\s+(?:saying|using)\s+(\S+)",  # word stop
    ]
    
    for pattern in patterns:
        match = re.search(pattern, user_message, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            if extracted:
                return extracted
    
    return None


def should_execute_tool(user_message: str, min_confidence: float = 0.65) -> Tuple[bool, Optional[str], float]:
    """
    Determine if a tool should be executed based on intent analysis.
    
    Args:
        user_message: The user's message
        min_confidence: Minimum confidence threshold for execution (default 0.65)
    
    Returns:
        Tuple of (should_execute, tool_name, confidence)
    
    Examples:
        >>> should_execute_tool("Please crawl https://example.com")
        (True, 'crawl', 0.95)
        
        >>> should_execute_tool("Tell me about web crawling")
        (False, None, 0.0)
    """
    tool, confidence = classify_intent(user_message)
    
    if tool and confidence >= min_confidence:
        return (True, tool, confidence)
    else:
        return (False, None, confidence)


# ==============================================================================
# Intent Explanation (for debugging/logging)
# ==============================================================================

def explain_intent(user_message: str) -> Dict[str, Any]:
    """
    Provide detailed explanation of intent classification.
    
    Useful for debugging and understanding why a particular classification was made.
    
    Args:
        user_message: The user's message
    
    Returns:
        Dictionary with classification details
    """
    tool, confidence = classify_intent(user_message)
    msg_lower = user_message.lower()
    
    matched_patterns = []
    
    # Find all matching patterns
    for tool_name, patterns in TOOL_PATTERNS.items():
        for pattern, base_conf in patterns:
            if re.search(pattern, msg_lower, re.IGNORECASE):
                matched_patterns.append({
                    "tool": tool_name,
                    "pattern": pattern,
                    "base_confidence": base_conf
                })
    
    # Find applied penalties
    applied_penalties = []
    for neg_pattern, neg_tool, penalty in NEGATIVE_CONTEXTS:
        if re.search(neg_pattern, msg_lower, re.IGNORECASE):
            applied_penalties.append({
                "pattern": neg_pattern,
                "affects_tool": neg_tool or "all",
                "penalty": penalty
            })
    
    return {
        "message": user_message,
        "classified_as": tool or "conversational",
        "confidence": confidence,
        "matched_patterns": matched_patterns,
        "applied_penalties": applied_penalties,
        "is_conversational": tool is None or confidence < 0.50,
        "should_execute": confidence >= 0.65
    }


# ==============================================================================
# Testing / Standalone Execution
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("nf_intent.py - Intent Classification Test Suite")
    print("=" * 70)
    
    test_cases = [
        # Crawl tests
        ("Please crawl https://example.com", "crawl", True),
        ("The baby is crawling now!", None, False),
        ("Can you explain how web crawling works?", None, False),
        ("Crawl this site for me: https://news.ycombinator.com", "crawl", True),
        ("I saw a spider crawling on the wall", None, False),
        
        # Recall tests
        ("Please recall what we discussed about embeddings", "recall", True),
        ("Do you remember our conversation yesterday?", "recall", True),
        ("I can't recall where I put my keys", None, False),
        ("What did we talk about last week?", "recall", True),
        
        # Ingest tests
        ("Please ingest the file notes.txt", "ingest", True),
        ("Can you read this document for me?", "ingest", True),
        
        # Conversational tests
        ("Hello, how are you?", None, False),
        ("What's the weather like?", None, False),
        ("Tell me about yourself", None, False),
    ]
    
    print("\n🧪 Running Test Cases:")
    print("-" * 70)
    
    passed = 0
    failed = 0
    
    for message, expected_tool, expected_execute in test_cases:
        tool, confidence = classify_intent(message)
        should_exec, _, _ = should_execute_tool(message)
        
        # Determine if test passed
        tool_match = (tool == expected_tool)
        exec_match = (should_exec == expected_execute)
        test_passed = tool_match and exec_match
        
        if test_passed:
            passed += 1
            status = "✅ PASS"
        else:
            failed += 1
            status = "❌ FAIL"
        
        print(f"\n{status}")
        print(f"  Message: \"{message}\"")
        print(f"  Expected: {expected_tool or 'conversational'} (execute={expected_execute})")
        print(f"  Got: {tool or 'conversational'} (confidence={confidence:.2f}, execute={should_exec})")
    
    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 70)
    
    # Detailed explanation example
    print("\n🔍 Detailed Intent Analysis Example:")
    print("-" * 70)
    explanation = explain_intent("The baby is crawling now, should I crawl that website?")
    print(json.dumps(explanation, indent=2))
