"""
Neuroforge Unified Recall API
=============================

This module provides a unified /recall endpoint that searches across:
- Facts: Full-text search on memory_facts using FTS5
- Knowledge: Semantic search on knowledge using vector similarity
- Logs: Text search on operational_logs with tag filtering

The API returns structured results with separate sections for each type.

Usage:
    POST /recall
    {
        "query": "search term",
        "top_k": 5,
        "modes": ["facts", "knowledge", "logs"],
        "filters": {
            "tags": ["src:web-crawl"],
            "session_id": "optional-session-id"
        }
    }

Response:
    {
        "ok": true,
        "query": "search term",
        "results": {
            "facts": [...],
            "knowledge": [...],
            "logs": [...]
        },
        "total_results": 15,
        "execution_time_ms": 45
    }
"""

import json
import time
import sqlite3
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel

from nf_paths import DB_PATH
from nf_schema import get_db_connection
from nf_tags import normalize_tags, parse_tags_from_db

# Import for semantic search (Phase 3)
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
    _MODEL_CACHE = None  # Cache the model to avoid reloading
except ImportError:
    HAS_EMBEDDINGS = False
    print("⚠️ sentence-transformers not available - semantic search disabled")


class RecallRequest(BaseModel):
    query: str
    top_k: int = 5
    modes: List[str] = ["facts", "knowledge", "logs"]
    filters: Optional[Dict[str, Any]] = None


class RecallResponse(BaseModel):
    ok: bool
    query: str
    results: Dict[str, List[Dict[str, Any]]]
    total_results: int
    execution_time_ms: float


@dataclass
class SearchResult:
    """Base class for search results."""
    id: int
    content: str
    relevance_score: float
    tags: List[str]
    created_ts: str
    source: str  # 'facts', 'knowledge', or 'logs'


# ============================================================================
# PHASE 3: Semantic Search Functions
# ============================================================================

def get_embedding_model():
    """Get or load the embedding model (cached)."""
    global _MODEL_CACHE
    if not HAS_EMBEDDINGS:
        return None
    if _MODEL_CACHE is None:
        _MODEL_CACHE = SentenceTransformer('all-MiniLM-L6-v2')
    return _MODEL_CACHE


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def from_blob(blob: bytes) -> np.ndarray:
    """Convert SQLite blob to numpy array."""
    return np.frombuffer(blob, dtype=np.float32)


def semantic_search_knowledge(
    conn: sqlite3.Connection,
    query: str,
    top_k: int = 10,
    alpha: float = 0.7,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Hybrid search combining lexical and semantic embeddings.
    
    Args:
        conn: Database connection
        query: Search query
        top_k: Number of results to return
        alpha: Weight for lexical vs semantic (0.7 = 70% lexical, 30% semantic)
        filters: Optional tag filters
    
    Returns:
        List of search results with hybrid scores
    """
    if not HAS_EMBEDDINGS:
        # Fallback to regular search if embeddings not available
        return search_knowledge(conn, query, top_k, filters)
    
    model = get_embedding_model()
    if not model:
        return search_knowledge(conn, query, top_k, filters)
    
    # Generate query embedding
    q_vec = model.encode(query, normalize_embeddings=True).astype(np.float32)
    
    # Build SQL query
    sql = """
        SELECT 
            k.rowid,
            k.doc_id,
            k.source_type,
            k.source_location,
            k.content,
            k.tags,
            k.ingested_at,
            k.topic,
            k.summary,
            e.hybrid_embedding,
            GROUP_CONCAT(kt.tag, ', ') as normalized_tags
        FROM knowledge k
        LEFT JOIN knowledge_embeddings e ON k.rowid = e.knowledge_id
        LEFT JOIN knowledge_tags kt ON k.rowid = kt.knowledge_id
    """
    
    where_clauses = []
    params = []
    
    # Add tag filtering if specified
    if filters and "tags" in filters and normalize_tags(filters["tags"]):
        tag_conditions = []
        for tag in normalize_tags(filters["tags"]):
            tag_conditions.append("kt.tag = ?")
            params.append(tag)
        where_clauses.append("(" + " OR ".join(tag_conditions) + ")")
    
    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)
    
    sql += " GROUP BY k.rowid"
    
    cursor = conn.execute(sql, params)
    results = []
    
    for row in cursor.fetchall():
        (rowid, doc_id, source_type, source_location, content, tags_json, 
         ingested_at, topic, summary, hybrid_emb, normalized_tags) = row
        
        # Calculate lexical score (simple keyword matching)
        query_lower = query.lower()
        content_lower = (content or "").lower()
        
        if query_lower in content_lower:
            lexical_score = 1.0
        elif any(word in content_lower for word in query_lower.split()):
            lexical_score = 0.7
        else:
            lexical_score = 0.3
        
        # Calculate semantic score
        if hybrid_emb:
            emb_vec = from_blob(hybrid_emb)
            semantic_score = cosine_similarity(q_vec, emb_vec)
        else:
            semantic_score = 0.0
        
        # Hybrid scoring
        final_score = alpha * lexical_score + (1 - alpha) * semantic_score
        
        # Truncate content for display
        display_content = (content[:200] + "...") if content and len(content) > 200 else (content or "")
        
        # Use normalized tags if available
        tag_list = normalized_tags.split(', ') if normalized_tags else parse_tags_from_db(tags_json)
        
        results.append({
            "id": rowid,
            "type": "knowledge",
            "doc_id": doc_id,
            "source_type": source_type or "",
            "source_location": source_location or "",
            "content": display_content,
            "full_content": content or "",
            "tags": tag_list,
            "topic": topic,
            "summary": summary,
            "ingested_at": ingested_at,
            "relevance_score": float(final_score),
            "lexical_score": float(lexical_score),
            "semantic_score": float(semantic_score)
        })
    
    # Sort by final score and return top_k
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return results[:top_k]


# ============================================================================
# Original Search Functions (Enhanced for Phase 2)
# ============================================================================

def create_recall_router() -> APIRouter:
    """Create the recall API router."""
    router = APIRouter(prefix="/recall", tags=["recall"])
    
    @router.post("/", response_model=RecallResponse)
    async def recall_search(request: RecallRequest = Body(...)) -> RecallResponse:
        """
        Unified search across facts, knowledge, and logs.
        Phase 3: Automatically uses semantic search when available.
        """
        start_time = time.time()
        
        try:
            conn = get_db_connection()
            results = {}
            total_results = 0
            
            # Search facts if requested
            if "facts" in request.modes:
                facts_results = search_facts(
                    conn, request.query, request.top_k, request.filters
                )
                results["facts"] = facts_results
                total_results += len(facts_results)
            
            # Search knowledge if requested - PHASE 3: Use semantic search
            if "knowledge" in request.modes:
                # Try semantic search first (hybrid), fallback to lexical
                if HAS_EMBEDDINGS:
                    knowledge_results = semantic_search_knowledge(
                        conn, request.query, request.top_k, 
                        alpha=0.7,  # 70% lexical, 30% semantic
                        filters=request.filters
                    )
                else:
                    # Fallback to enhanced lexical search
                    knowledge_results = search_knowledge(
                        conn, request.query, request.top_k, request.filters
                    )
                results["knowledge"] = knowledge_results
                total_results += len(knowledge_results)
            
            # Search logs if requested
            if "logs" in request.modes:
                logs_results = search_logs(
                    conn, request.query, request.top_k, request.filters
                )
                results["logs"] = logs_results
                total_results += len(logs_results)
            
            conn.close()
            
            execution_time = (time.time() - start_time) * 1000
            
            return RecallResponse(
                ok=True,
                query=request.query,
                results=results,
                total_results=total_results,
                execution_time_ms=round(execution_time, 2)
            )
            
        except Exception as e:
            logging.error(f"Recall search failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/stats")
    async def recall_stats() -> Dict[str, Any]:
        """Get statistics about the recall database."""
        try:
            conn = get_db_connection()
            
            stats = {}
            
            # Count facts
            cursor = conn.execute("SELECT COUNT(*) FROM memory_facts_content")
            stats["facts_count"] = cursor.fetchone()[0]
            
            # Count knowledge
            cursor = conn.execute("SELECT COUNT(*) FROM knowledge")
            stats["knowledge_count"] = cursor.fetchone()[0]
            
            # Count logs
            cursor = conn.execute("SELECT COUNT(*) FROM operational_logs")
            stats["logs_count"] = cursor.fetchone()[0]
            
            # Count turnlog
            cursor = conn.execute("SELECT COUNT(*) FROM unified_conversations WHERE source_table = 'turnlog'")
            stats["turnlog_count"] = cursor.fetchone()[0]
            
            # Get recent activity (last 7 days)
            cursor = conn.execute("""
                SELECT COUNT(*) FROM operational_logs 
                WHERE created_ts > datetime('now', '-7 days')
            """)
            stats["recent_logs"] = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "ok": True,
                "stats": stats,
                "database_path": str(DB_PATH)
            }
            
        except Exception as e:
            logging.error(f"Stats retrieval failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return router


def search_facts(
    conn: sqlite3.Connection, 
    query: str, 
    top_k: int, 
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Search memory_facts using FTS5 full-text search.
    Existing FTS5 schema: keyphrase, answer, source_url, tags, created_ts
    """
    try:
        from fts5_utils import sanitize_fts5_query
        fts_query = sanitize_fts5_query(query)

        sql = """
            SELECT 
                keyphrase,
                answer,
                source_url,
                tags,
                created_ts,
                bm25(memory_facts_fts) as relevance
            FROM memory_facts_fts
            WHERE memory_facts_fts MATCH ?
        """

        params = [fts_query]

        # Add tag filtering if specified
        if filters and "tags" in filters:
            tag_conditions = []
            for tag in normalize_tags(filters["tags"]):
                tag_conditions.append("tags LIKE ?")
                params.append(f'%"{tag}"%')
            if tag_conditions:
                sql += " AND (" + " OR ".join(tag_conditions) + ")"

        sql += " ORDER BY relevance LIMIT ?"
        params.append(top_k)

        cursor = conn.execute(sql, params)
        results = []

        for row in cursor.fetchall():
            keyphrase, answer, source_url, tags_json, created_ts, relevance = row
            results.append({
                "type": "fact",
                "keyphrase": keyphrase,
                "answer": answer,
                "source_url": source_url or "",
                "tags": parse_tags_from_db(tags_json),
                "created_ts": created_ts,
                "relevance_score": abs(float(relevance)) if relevance else 0.0  # BM25 returns negative scores
            })
        
        return results
        
    except sqlite3.OperationalError as e:
        if "no such table" in str(e):
            logging.warning("Memory facts table not found")
            return []
        raise


def search_knowledge(
    conn: sqlite3.Connection, 
    query: str, 
    top_k: int, 
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Search knowledge using enhanced text similarity with normalized tag support.
    Enhanced for Phase 2+: Unified search across content, tags, and AI metadata.
    """
    try:
        # Enhanced unified search with LEFT JOIN for comprehensive coverage
        sql = """
            SELECT DISTINCT
                k.rowid,
                k.doc_id,
                k.source_type,
                k.source_location,
                k.content,
                k.tags,
                k.ingested_at,
                k.topic,
                k.summary,
                GROUP_CONCAT(kt.tag, ', ') as normalized_tags,
                CASE 
                    WHEN k.content LIKE ? THEN 2.0
                    WHEN k.content LIKE ? THEN 1.5
                    WHEN kt.tag LIKE ? THEN 1.2
                    WHEN k.topic LIKE ? THEN 1.0
                    WHEN k.summary LIKE ? THEN 0.8
                    ELSE 0.5
                END as relevance_score
            FROM knowledge k
            LEFT JOIN knowledge_tags kt ON k.rowid = kt.knowledge_id
            WHERE (k.content LIKE ? OR k.content LIKE ? OR kt.tag LIKE ? 
                   OR k.topic LIKE ? OR k.summary LIKE ?)
        """
        
        # Create search patterns for unified search
        exact_pattern = f"%{query}%"
        word_pattern = f"%{' '.join(query.split())}%"
        tag_pattern = f"%{query.lower()}%"
        topic_pattern = f"%{query}%"
        summary_pattern = f"%{query}%"
        
        # Parameters for relevance scoring and WHERE clause
        params = [
            exact_pattern, word_pattern, tag_pattern, topic_pattern, summary_pattern,  # For relevance scoring
            exact_pattern, word_pattern, tag_pattern, topic_pattern, summary_pattern   # For WHERE clause
        ]
        
        # Add specific tag filtering if specified
        if filters and "tags" in filters and normalize_tags(filters["tags"]):
            tag_conditions = []
            for tag in normalize_tags(filters["tags"]):
                tag_conditions.append("kt.tag = ?")
                params.append(tag)
            
            sql += " AND (" + " OR ".join(tag_conditions) + ")"
        
        sql += " GROUP BY k.rowid ORDER BY relevance_score DESC LIMIT ?"
        params.append(top_k)
        
        cursor = conn.execute(sql, params)
        results = []
        
        for row in cursor.fetchall():
            rowid, doc_id, source_type, source_location, content, tags_json, ingested_at, topic, summary, normalized_tags, relevance = row
            
            # Truncate content for display
            display_content = (content[:200] + "...") if content and len(content) > 200 else (content or "")
            
            # Use normalized tags if available, fallback to JSON tags
            tag_list = normalized_tags.split(', ') if normalized_tags else parse_tags_from_db(tags_json)
            
            results.append({
                "id": rowid,  # Use rowid as id
                "type": "knowledge",
                "doc_id": doc_id,
                "source_type": source_type or "",
                "source_location": source_location or "",
                "content": display_content,
                "full_content": content or "",
                "tags": tag_list,
                "topic": topic,
                "summary": summary,
                "ingested_at": ingested_at,
                "relevance_score": float(relevance)
            })
        
        return results
        
    except sqlite3.OperationalError as e:
        if "no such table" in str(e):
            logging.warning("Knowledge table not found")
            return []
        raise
        raise


def search_logs(
    conn: sqlite3.Connection, 
    query: str, 
    top_k: int, 
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Search operational_logs using text search with tag filtering.
    """
    try:
        sql = """
            SELECT 
                id,
                module_name,
                event_type,
                message,
                json_payload,
                tags,
                session_id,
                created_ts,
                CASE 
                    WHEN message LIKE ? THEN 1.0
                    WHEN json_payload LIKE ? THEN 0.8
                    WHEN event_type LIKE ? THEN 0.6
                    ELSE 0.4
                END as relevance_score
            FROM operational_logs
            WHERE message LIKE ? 
               OR json_payload LIKE ? 
               OR event_type LIKE ?
               OR module_name LIKE ?
        """
        
        # Create search patterns
        search_pattern = f"%{query}%"
        params = [search_pattern] * 7  # For relevance calculation and WHERE clause
        
        # Add session filter if specified
        if filters and "session_id" in filters:
            sql += " AND session_id = ?"
            params.append(filters["session_id"])
        
        # Add tag filtering if specified
        if filters and "tags" in filters:
            tag_conditions = []
            for tag in normalize_tags(filters["tags"]):
                tag_conditions.append("tags LIKE ?")
                params.append(f'%"{tag}"%')
            
            if tag_conditions:
                sql += " AND (" + " OR ".join(tag_conditions) + ")"
        
        sql += " ORDER BY relevance_score DESC, created_ts DESC LIMIT ?"
        params.append(top_k)
        
        cursor = conn.execute(sql, params)
        results = []
        
        for row in cursor.fetchall():
            (id_, module_name, event_type, message, json_payload, 
             tags_json, session_id, created_ts, relevance) = row
            
            # Parse JSON payload safely
            try:
                payload = json.loads(json_payload) if json_payload else {}
            except (json.JSONDecodeError, ValueError):
                payload = {}
            
            results.append({
                "id": id_,
                "type": "log",
                "module_name": module_name,
                "event_type": event_type,
                "message": message,
                "payload": payload,
                "tags": parse_tags_from_db(tags_json),
                "session_id": session_id,
                "created_ts": created_ts,
                "relevance_score": float(relevance)
            })
        
        return results
        
    except sqlite3.OperationalError as e:
        if "no such table" in str(e):
            logging.warning("Operational logs table not found")
            return []
        raise


# For testing purposes
def test_recall_search(query: str = "test", top_k: int = 3) -> Dict[str, Any]:
    """Test function for the recall search."""
    request = RecallRequest(
        query=query,
        top_k=top_k,
        modes=["facts", "knowledge", "logs"]
    )
    
    start_time = time.time()
    conn = get_db_connection()
    
    results = {}
    
    try:
        results["facts"] = search_facts(conn, request.query, request.top_k)
        results["knowledge"] = search_knowledge(conn, request.query, request.top_k)
        results["logs"] = search_logs(conn, request.query, request.top_k)
    finally:
        conn.close()
    
    execution_time = (time.time() - start_time) * 1000
    
    return {
        "ok": True,
        "query": query,
        "results": results,
        "total_results": sum(len(r) for r in results.values()),
        "execution_time_ms": round(execution_time, 2)
    }


if __name__ == "__main__":
    # Test the recall functionality
    print("Testing Neuroforge Unified Recall")
    print("=" * 40)
    
    result = test_recall_search("test")
    
    print(f"Query: {result['query']}")
    print(f"Total results: {result['total_results']}")
    print(f"Execution time: {result['execution_time_ms']} ms")
    print()
    
    for section, items in result['results'].items():
        print(f"{section.title()}: {len(items)} results")
        for item in items[:2]:  # Show first 2 results
            print(f"  - {item.get('type', 'unknown')}: {item.get('content', item.get('message', item.get('keyphrase', 'N/A')))[:60]}...")
        print()
    
    # Test Phase 3 semantic search
    if HAS_EMBEDDINGS:
        print("\n" + "=" * 40)
        print("Testing Phase 3 Semantic Search")
        print("=" * 40)
        
        conn = get_db_connection()
        semantic_results = semantic_search_knowledge(conn, "artificial intelligence", top_k=3)
        
        print(f"\nSemantic search for 'artificial intelligence':")
        for i, result in enumerate(semantic_results):
            print(f"\n{i+1}. {result['doc_id']}")
            print(f"   Topic: {result.get('topic', 'N/A')}")
            print(f"   Hybrid Score: {result['relevance_score']:.3f}")
            print(f"   Lexical: {result['lexical_score']:.3f}, Semantic: {result['semantic_score']:.3f}")
            print(f"   Content: {result['content'][:80]}...")
        
        conn.close()