import re

def sanitize_fts5_query(term: str) -> str:
    """
    Sanitize a user query for safe FTS5 MATCH usage.
    - Removes non-word, non-space chars
    - Collapses multiple spaces
    - Wraps in double quotes for phrase search
    """
    safe_term = re.sub(r"[^\w\s]", " ", term).strip()
    safe_term = re.sub(r"\s+", " ", safe_term)
    return f'"{safe_term}"'
