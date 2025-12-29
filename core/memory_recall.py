"""
memory_recall
==============

This module provides helper functions for recalling past session summaries and
persona patches.  It searches the existing ``./Memory/Sessions`` and
``./Memory/Persona`` directories for JSON files matching a query string and
returns structured results instead of printing to stdout.  It does not
create any new directories and will silently ignore missing paths.

Functions
---------
recall_memory(query: str, search_sessions: bool = True,
              search_patches: bool = True, max_results: int = 5) -> List[dict]
    Search session summaries and/or persona patches for a query string and
    return up to ``max_results`` hits as a list of dictionaries.
"""

from __future__ import annotations

import os
import json
from typing import List, Dict, Any


SESSIONS_DIR = os.path.join(".", "Memory", "Sessions")
PERSONA_FILE = os.path.join(".", "Memory", "Persona", "persona_patches.json")


def recall_memory(query: str, search_sessions: bool = True, search_patches: bool = True,
                  max_results: int = 5) -> List[Dict[str, Any]]:
    """Search stored sessions and persona patches for a query string.

    Args:
        query: The case‑insensitive substring to search for in stored JSON data.
        search_sessions: Whether to search session summary files under
            ``./Memory/Sessions``.
        search_patches: Whether to search persona patch entries in
            ``./Memory/Persona/persona_patches.json``.
        max_results: Maximum number of results to return.  If 0 or negative,
            all matches are returned.

    Returns:
        A list of result dictionaries.  Each dictionary has keys:
            ``type``: either ``"session"`` or ``"patch"``.
            ``file``: the filename from which the data was loaded.
            ``data``: the parsed JSON object.
    """
    if not query:
        return []
    q = query.lower()
    results: List[Dict[str, Any]] = []

    # Search session summaries
    if search_sessions and os.path.isdir(SESSIONS_DIR):
        try:
            for fname in sorted(os.listdir(SESSIONS_DIR)):
                if not fname.endswith(".json"):
                    continue
                fpath = os.path.join(SESSIONS_DIR, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                except Exception:
                    continue
                if q in json.dumps(data).lower():
                    results.append({"type": "session", "file": fname, "data": data})
                    if 0 < max_results <= len(results):
                        return results
        except Exception:
            # Ignore directory listing errors
            pass

    # Search persona patches
    if search_patches and os.path.isfile(PERSONA_FILE):
        try:
            with open(PERSONA_FILE, "r", encoding="utf-8") as fh:
                patches = json.load(fh)
            if isinstance(patches, list):
                for patch in patches:
                    if isinstance(patch, dict) and q in json.dumps(patch).lower():
                        results.append({"type": "patch", "file": os.path.basename(PERSONA_FILE), "data": patch})
                        if 0 < max_results <= len(results):
                            return results
        except Exception:
            pass

    return results