"""
Neuroforge Tag Normalization
============================

This module provides consistent tag handling across all Neuroforge components.
Tags are normalized to JSON arrays of strings with namespace support.

Usage:
    from nf_tags import normalize_tags, format_tags_for_db
    
    # Normalize various input formats to List[str]
    tags = normalize_tags("tag1,tag2")  # -> ["tag1", "tag2"]
    tags = normalize_tags(["tag1", "tag2"])  # -> ["tag1", "tag2"]
    tags = normalize_tags('["tag1", "tag2"]')  # -> ["tag1", "tag2"]
    
    # Format for database storage
    db_tags = format_tags_for_db(tags)  # -> JSON string
    
Namespace conventions:
    src:web-crawl    - Source: web crawler
    src:session      - Source: session files
    src:persona      - Source: persona patches
    agent:executor   - Agent: executor worker
    agent:crawler    - Agent: crawler worker
    audit:thumbs_up  - Audit: positive feedback
    audit:thumbs_down- Audit: negative feedback
    kw:embedding     - Keyword: embedding-related
    type:invoice     - Type: invoice document
    type:summary     - Type: summary content
"""

import json
import re
from typing import List, Union, Any
from typing import Dict, Tuple


def normalize_tags(value: Union[str, List[str], None]) -> List[str]:
    """
    Normalize tag input to a consistent List[str] format.
    
    Args:
        value: Tags in various formats:
               - None -> []
               - str (comma-separated) -> ["tag1", "tag2"]
               - str (JSON array) -> parsed list
               - List[str] -> as-is (cleaned)
               - Other -> [str(value)]
    
    Returns:
        List of cleaned tag strings
    """
    if value is None:
        return []
    
    if isinstance(value, list):
        # Already a list, clean up each item
        return [str(tag).strip() for tag in value if tag and str(tag).strip()]
    
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        
        # Try to parse as JSON first
        if value.startswith('[') and value.endswith(']'):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(tag).strip() for tag in parsed if tag and str(tag).strip()]
            except (json.JSONDecodeError, ValueError):
                pass
        
        # Split by comma
        tags = [tag.strip() for tag in value.split(',') if tag.strip()]
        return tags
    
    # Fallback: convert to string
    return [str(value).strip()] if str(value).strip() else []


def format_tags_for_db(tags: List[str]) -> str:
    """
    Format normalized tags for database storage as JSON.
    
    Args:
        tags: List of tag strings
        
    Returns:
        JSON string representation
    """
    return json.dumps(tags)


def parse_tags_from_db(db_value: Union[str, None]) -> List[str]:
    """
    Parse tags from database JSON string back to List[str].
    
    Args:
        db_value: JSON string from database or None
        
    Returns:
        List of tag strings
    """
    if not db_value:
        return []
    
    try:
        parsed = json.loads(db_value)
        if isinstance(parsed, list):
            return [str(tag) for tag in parsed if tag]
        return [str(parsed)] if parsed else []
    except (json.JSONDecodeError, ValueError):
        # Fallback: treat as comma-separated
        return normalize_tags(db_value)


def validate_namespace(tag: str) -> bool:
    """
    Validate that a tag follows namespace conventions.
    
    Args:
        tag: Tag string to validate
        
    Returns:
        True if tag has valid namespace format (prefix:value)
    """
    if ':' not in tag:
        return False
    
    prefix, value = tag.split(':', 1)
    
    # Known prefixes
    valid_prefixes = {
        'src', 'agent', 'audit', 'kw', 'type', 'status', 'priority',
        'module', 'session', 'user', 'model', 'version'
    }
    
    return prefix in valid_prefixes and bool(value.strip())


def suggest_namespace(tag: str) -> str:
    """
    Suggest a namespace for an unnamespaced tag.
    
    Args:
        tag: Tag without namespace
        
    Returns:
        Suggested namespaced tag
    """
    tag_lower = tag.lower()
    
    # Common mappings
    if tag_lower in ['web-crawl', 'crawler', 'crawling']:
        return f'src:{tag}'
    elif tag_lower in ['executor', 'worker']:
        return f'agent:{tag}'
    elif tag_lower in ['thumbs_up', 'thumbs_down', 'positive', 'negative']:
        return f'audit:{tag}'
    elif tag_lower in ['embedding', 'vector', 'similarity']:
        return f'kw:{tag}'
    elif tag_lower in ['invoice', 'summary', 'document', 'report']:
        return f'type:{tag}'
    else:
        return f'misc:{tag}'


def clean_and_namespace_tags(tags: List[str], auto_namespace: bool = False) -> List[str]:
    """
    Clean tags and optionally add namespaces.
    
    Args:
        tags: List of tag strings
        auto_namespace: Whether to automatically add namespaces to unnamespaced tags
        
    Returns:
        List of cleaned and optionally namespaced tags
    """
    cleaned = []
    
    for tag in tags:
        tag = tag.strip().lower()
        if not tag:
            continue
            
        # Remove invalid characters
        tag = re.sub(r'[^\w\-:.]', '_', tag)
        
        if auto_namespace and not validate_namespace(tag):
            tag = suggest_namespace(tag)
            
        cleaned.append(tag)
    
    return list(set(cleaned))  # Remove duplicates


# Conversation marker detection utilities
# These patterns cover common conversation formats used across many models
# and chat datasets (plain "User:", "Assistant:", OpenAI style tokens, etc.).
CONVERSATION_MARKER_PATTERNS = {
    # Use robust "\bROLE\b\s*:" style patterns which reliably match e.g. "User:", "user :", "Assistant:"
    'user': [
        r"\buser\b\s*:",
        r"<\|im_start\|>user",
        r"\bhuman\b\s*:",
        r"\bhuman\b\s*:",
    ],
    'assistant': [
        r"\bassistant\b\s*:",
        r"<\|im_start\|>assistant",
        r"\bai\b\s*:",
        r"\bbot\b\s*:",
    ]
}


def _compile_marker_regexes(patterns: Dict[str, List[str]]):
    """Compile the pattern lists into case-insensitive regex objects."""
    import re

    compiled = {}
    for role, pats in patterns.items():
        combined = '(' + ')|('.join(pats) + ')'
        compiled[role] = re.compile(combined, re.IGNORECASE | re.MULTILINE)
    return compiled


_COMPILED_MARKERS = _compile_marker_regexes(CONVERSATION_MARKER_PATTERNS)


def has_conversation_tags(text: str) -> Dict[str, bool]:
    """Return whether the provided text contains user/assistant conversation markers.

    Args:
        text: The text to inspect.

    Returns:
        Dict with boolean keys 'has_user' and 'has_assistant'.
    """
    if not text:
        return {'has_user': False, 'has_assistant': False}

    return {
        'has_user': bool(_COMPILED_MARKERS['user'].search(text)),
        'has_assistant': bool(_COMPILED_MARKERS['assistant'].search(text)),
    }


def find_conversation_markers(text: str, max_matches: int = 50) -> List[Tuple[str, int, str]]:
    """Find conversation marker matches and return a list of tuples
    (role, position, snippet).

    Args:
        text: The text to search.
        max_matches: Max number of matches to return.

    Returns:
        List of (role, pos, snippet) where role is 'user' or 'assistant', pos is start index,
        and snippet is a short context string around the match.
    """
    results: List[Tuple[str, int, str]] = []
    if not text:
        return results

    for role, regex in _COMPILED_MARKERS.items():
        for m in regex.finditer(text):
            if len(results) >= max_matches:
                break
            start = m.start()
            # Grab 40 chars of context after marker to help identify usage
            snippet = text[start:start + 80].replace('\n', ' ')
            results.append((role, start, snippet))

    # Sort by position
    results.sort(key=lambda x: x[1])
    return results


def extract_role_text(text: str, role: str = 'user') -> str:
    """Extract and return the concatenated text segments for the given role.

    This helps when a single stored document contains alternating User/Assistant
    sections: we commonly want only the user's messages for fact extraction or
    indexing.

    Args:
        text: Full conversation text.
        role: 'user' or 'assistant'.

    Returns:
        Concatenated text segments for the requested role (empty string if none).
    """
    import re
    if not text or role not in ('user', 'assistant'):
        return ''

    role = role.lower()

    segments: List[str] = []

    # Pattern set 1: ROLE: ... (until next ROLE2: or end)
    role_label = re.escape(role)
    other_role = 'assistant' if role == 'user' else 'user'
    pattern1 = re.compile(rf"(?i){role_label}\s*:\s*(.*?)(?=(?:{other_role}\s*:|$))", re.DOTALL)

    for m in pattern1.finditer(text):
        seg = m.group(1).strip()
        if seg:
            segments.append(seg)

    # Pattern set 2: tokenized markers like <|im_start|>user ... <|im_start|>assistant
    token_pat = re.compile(rf"<\|im_start\|>{role}\s*(.*?)(?=<\|im_start\|>{other_role}|$)", re.DOTALL | re.IGNORECASE)
    for m in token_pat.finditer(text):
        seg = m.group(1).strip()
        if seg:
            segments.append(seg)

    # If we found segments, join them; otherwise return empty string
    if segments:
        return "\n\n".join(segments)

    return ''


# Predefined tag constants for common use cases
class Tags:
    """Common tag constants with proper namespacing."""
    
    # Sources
    SRC_WEB_CRAWL = "src:web-crawl"
    SRC_SESSION = "src:session"
    SRC_PERSONA = "src:persona"
    SRC_USER_UPLOAD = "src:user-upload"
    SRC_USER_INPUT = "src:user-input"
    SRC_MANUAL = "src:manual"
    
    # Agents
    AGENT_EXECUTOR = "agent:executor"
    AGENT_CRAWLER = "agent:crawler"
    AGENT_CORE = "agent:core"
    AGENT_BRIDGE = "agent:bridge"
    AGENT_UI = "agent:ui"
    
    # Audit/Feedback
    AUDIT_THUMBS_UP = "audit:thumbs_up"
    AUDIT_THUMBS_DOWN = "audit:thumbs_down"
    AUDIT_PRECISE = "audit:precise"
    AUDIT_HALLUCINATION = "audit:hallucination"
    
    # Keywords/Content
    KW_EMBEDDING = "kw:embedding"
    KW_SIMILARITY = "kw:similarity"
    KW_VECTOR = "kw:vector"
    
    # Types
    TYPE_INVOICE = "type:invoice"
    TYPE_SUMMARY = "type:summary"
    TYPE_DOCUMENT = "type:document"
    TYPE_REPORT = "type:report"
    TYPE_PATCH = "type:patch"
    TYPE_CONVERSATION = "type:conversation"
    TYPE_FEEDBACK = "type:feedback"
    
    # Status
    STATUS_PROCESSED = "status:processed"
    STATUS_PENDING = "status:pending"
    STATUS_ERROR = "status:error"
    STATUS_COMPLETED = "status:completed"


if __name__ == "__main__":
    # Test the normalization functions
    test_cases = [
        "tag1,tag2,tag3",
        ["tag1", "tag2", "tag3"],
        '["tag1", "tag2", "tag3"]',
        None,
        "",
        "single-tag",
        ["src:web-crawl", "type:document", "status:processed"]
    ]
    
    print("Tag Normalization Tests")
    print("=" * 40)
    
    for case in test_cases:
        normalized = normalize_tags(case)
        db_format = format_tags_for_db(normalized)
        parsed_back = parse_tags_from_db(db_format)
        
        print(f"Input:    {case!r}")
        print(f"Normal:   {normalized}")
        print(f"DB:       {db_format}")
        print(f"Parsed:   {parsed_back}")
        print(f"Clean:    {clean_and_namespace_tags(normalized, auto_namespace=True)}")
        print()