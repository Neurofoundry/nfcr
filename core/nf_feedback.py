"""
Neuroforge UI Feedback Logging
===============================

This module provides endpoints and utilities for logging user feedback
from the chat UI to the database with proper audit tags.

Features:
- Thumbs up/down feedback on AI responses
- Conversation turn logging
- User feedback metadata tracking
- Audit trail with normalized tags

Usage:
    from nf_feedback import create_feedback_router
    
    # In your FastAPI app:
    app.include_router(create_feedback_router())
"""

import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field

from nf_paths import DB_PATH
from nf_schema import get_db_connection
from nf_tags import Tags, normalize_tags, format_tags_for_db


# ============================================================================
# Pydantic Models for API Requests/Responses
# ============================================================================

class FeedbackRequest(BaseModel):
    """Request model for thumbs up/down feedback."""
    turn_id: Optional[int] = Field(None, description="Turn log ID if available")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    feedback_type: str = Field(..., description="'thumbs_up' or 'thumbs_down'")
    message_role: str = Field(default="assistant", description="'user' or 'assistant'")
    message_content: Optional[str] = Field(None, description="Message content (optional)")
    user_comment: Optional[str] = Field(None, description="Optional user comment")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class TurnLogRequest(BaseModel):
    """Request model for logging conversation turns."""
    conversation_id: str = Field(..., description="Unique conversation identifier")
    turn_number: int = Field(..., description="Turn number in conversation")
    user_message: str = Field(..., description="User's input message")
    assistant_message: str = Field(..., description="Assistant's response")
    model_name: Optional[str] = Field(None, description="Model used for response")
    tokens_used: Optional[int] = Field(None, description="Tokens consumed")
    latency_ms: Optional[float] = Field(None, description="Response latency in ms")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class FeedbackResponse(BaseModel):
    """Response model for feedback operations."""
    ok: bool
    message: str
    feedback_id: Optional[int] = None
    turn_id: Optional[int] = None


# ============================================================================
# Database Operations
# ============================================================================

def log_feedback_to_db(
    feedback_type: str,
    turn_id: Optional[int] = None,
    conversation_id: Optional[str] = None,
    message_role: str = "assistant",
    message_content: Optional[str] = None,
    user_comment: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> int:
    """
    Log user feedback to the operational_logs table.
    
    Args:
        feedback_type: 'thumbs_up' or 'thumbs_down'
        turn_id: Optional turn log ID
        conversation_id: Optional conversation ID
        message_role: 'user' or 'assistant'
        message_content: Optional message content
        user_comment: Optional user comment
        metadata: Optional metadata dict
    
    Returns:
        The ID of the inserted log entry
    """
    conn = get_db_connection()
    
    # Determine audit tag based on feedback type
    if feedback_type == "thumbs_up":
        audit_tag = Tags.AUDIT_THUMBS_UP
    elif feedback_type == "thumbs_down":
        audit_tag = Tags.AUDIT_THUMBS_DOWN
    else:
        audit_tag = "audit:feedback"
    
    # Build tags list
    tags = [
        Tags.AGENT_UI,
        Tags.SRC_USER_INPUT,
        audit_tag,
        f"type:feedback",
        f"role:{message_role}"
    ]
    
    if conversation_id:
        tags.append(f"conv:{conversation_id}")
    
    tags_normalized = normalize_tags(tags)
    tags_json = format_tags_for_db(tags_normalized)
    
    # Build JSON payload
    payload = {
        "feedback_type": feedback_type,
        "message_role": message_role
    }
    
    if turn_id is not None:
        payload["turn_id"] = turn_id
    
    if conversation_id:
        payload["conversation_id"] = conversation_id
    
    if message_content:
        payload["message_content"] = message_content[:500]  # Truncate if too long
    
    if user_comment:
        payload["user_comment"] = user_comment
    
    if metadata:
        payload["metadata"] = metadata
    
    payload_json = json.dumps(payload)
    
    # Build message
    if user_comment:
        message = f"User feedback: {feedback_type} - {user_comment[:100]}"
    else:
        message = f"User feedback: {feedback_type}"
    
    from datetime import timezone
    
    # Insert into operational_logs
    cursor = conn.execute("""
        INSERT INTO operational_logs (
            module_name,
            event_type,
            message,
            tags,
            json_payload,
            created_ts
        ) VALUES (?, ?, ?, ?, ?, ?)
    """, (
        "ui_feedback",
        feedback_type,
        message,
        tags_json,
        payload_json,
        datetime.now(timezone.utc).isoformat()
    ))
    
    feedback_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return feedback_id


def log_conversation_turn(
    conversation_id: str,
    turn_number: int,
    user_message: str,
    assistant_message: str,
    model_name: Optional[str] = None,
    tokens_used: Optional[int] = None,
    latency_ms: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> int:
    """
    Log a conversation turn to the turnlog table.
    
    The turnlog table schema uses: conv_id, role, text, rating, tags, ts
    We'll log both user and assistant messages as separate entries.
    
    Args:
        conversation_id: Unique conversation identifier
        turn_number: Turn number in conversation
        user_message: User's input message
        assistant_message: Assistant's response
        model_name: Optional model name
        tokens_used: Optional token count
        latency_ms: Optional response latency
        metadata: Optional metadata dict
    
    Returns:
        The ID of the last inserted turn log entry
    """
    conn = get_db_connection()
    
    # Build tags for the turn
    tags = [
        Tags.AGENT_CORE,
        Tags.SRC_USER_INPUT,
        Tags.TYPE_CONVERSATION,
        f"conv:{conversation_id}",
        f"turn:{turn_number}"
    ]
    
    if model_name:
        tags.append(f"model:{model_name}")
    
    tags_normalized = normalize_tags(tags)
    tags_json = format_tags_for_db(tags_normalized)
    
    import time
    timestamp = int(time.time())
    
    # Insert user message
    cursor = conn.execute("""
        INSERT INTO turnlog (
            conv_id,
            role,
            text,
            tags,
            ts
        ) VALUES (?, ?, ?, ?, ?)
    """, (
        conversation_id,
        "user",
        user_message,
        tags_json,
        timestamp
    ))
    
    # Insert assistant message
    cursor = conn.execute("""
        INSERT INTO turnlog (
            conv_id,
            role,
            text,
            tags,
            ts
        ) VALUES (?, ?, ?, ?, ?)
    """, (
        conversation_id,
        "assistant",
        assistant_message,
        tags_json,
        timestamp + 1  # Slightly different timestamp
    ))
    
    turn_id = cursor.lastrowid
    
    # Also log to operational_logs for metadata tracking
    if metadata or tokens_used or latency_ms:
        meta_payload = metadata or {}
        if tokens_used is not None:
            meta_payload["tokens_used"] = tokens_used
        if latency_ms is not None:
            meta_payload["latency_ms"] = latency_ms
        if model_name:
            meta_payload["model_name"] = model_name
        
        meta_payload["turn_number"] = turn_number
        meta_payload["user_message_length"] = len(user_message)
        meta_payload["assistant_message_length"] = len(assistant_message)
        
        from datetime import datetime, timezone
        conn.execute("""
            INSERT INTO operational_logs (
                module_name,
                event_type,
                message,
                tags,
                json_payload,
                created_ts
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            "conversation",
            "turn_logged",
            f"Turn {turn_number} in {conversation_id}",
            tags_json,
            json.dumps(meta_payload),
            datetime.now(timezone.utc).isoformat()
        ))
    
    conn.commit()
    conn.close()
    
    return turn_id


def get_feedback_stats(
    conversation_id: Optional[str] = None,
    days: int = 7
) -> Dict[str, Any]:
    """
    Get feedback statistics.
    
    Args:
        conversation_id: Optional conversation ID to filter by
        days: Number of days to look back
    
    Returns:
        Dictionary with feedback statistics
    """
    conn = get_db_connection()
    
    # Build query
    where_clauses = [
        "module_name = 'ui_feedback'",
        f"created_ts > datetime('now', '-{days} days')"
    ]
    
    if conversation_id:
        where_clauses.append(f"json_payload LIKE '%\"conversation_id\": \"{conversation_id}\"%'")
    
    where_sql = " AND ".join(where_clauses)
    
    # Get counts by feedback type
    cursor = conn.execute(f"""
        SELECT event_type, COUNT(*) as count
        FROM operational_logs
        WHERE {where_sql}
        GROUP BY event_type
    """)
    
    feedback_counts = {}
    for row in cursor.fetchall():
        feedback_counts[row[0]] = row[1]
    
    # Get total feedback
    cursor = conn.execute(f"""
        SELECT COUNT(*) FROM operational_logs WHERE {where_sql}
    """)
    total_feedback = cursor.fetchone()[0]
    
    # Get recent feedback
    cursor = conn.execute(f"""
        SELECT event_type, message, created_ts
        FROM operational_logs
        WHERE {where_sql}
        ORDER BY created_ts DESC
        LIMIT 10
    """)
    
    recent_feedback = []
    for row in cursor.fetchall():
        recent_feedback.append({
            "type": row[0],
            "message": row[1],
            "timestamp": row[2]
        })
    
    conn.close()
    
    return {
        "total_feedback": total_feedback,
        "feedback_counts": feedback_counts,
        "thumbs_up": feedback_counts.get("thumbs_up", 0),
        "thumbs_down": feedback_counts.get("thumbs_down", 0),
        "recent_feedback": recent_feedback,
        "period_days": days
    }


# ============================================================================
# FastAPI Router
# ============================================================================

def create_feedback_router() -> APIRouter:
    """Create the feedback API router."""
    router = APIRouter(prefix="/feedback", tags=["feedback"])
    
    @router.post("/thumbs", response_model=FeedbackResponse)
    async def log_thumbs_feedback(request: FeedbackRequest = Body(...)) -> FeedbackResponse:
        """
        Log thumbs up/down feedback from the UI.
        
        Example:
            POST /feedback/thumbs
            {
                "feedback_type": "thumbs_up",
                "conversation_id": "conv-123",
                "message_role": "assistant",
                "user_comment": "Great answer!"
            }
        """
        try:
            # Validate feedback type
            if request.feedback_type not in ["thumbs_up", "thumbs_down"]:
                raise HTTPException(
                    status_code=400,
                    detail="feedback_type must be 'thumbs_up' or 'thumbs_down'"
                )
            
            # Log feedback
            feedback_id = log_feedback_to_db(
                feedback_type=request.feedback_type,
                turn_id=request.turn_id,
                conversation_id=request.conversation_id,
                message_role=request.message_role,
                message_content=request.message_content,
                user_comment=request.user_comment,
                metadata=request.metadata
            )
            
            return FeedbackResponse(
                ok=True,
                message=f"Feedback logged successfully: {request.feedback_type}",
                feedback_id=feedback_id
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/turn", response_model=FeedbackResponse)
    async def log_turn(request: TurnLogRequest = Body(...)) -> FeedbackResponse:
        """
        Log a conversation turn.
        
        Example:
            POST /feedback/turn
            {
                "conversation_id": "conv-123",
                "turn_number": 1,
                "user_message": "What is Python?",
                "assistant_message": "Python is a programming language...",
                "model_name": "qwen2.5:3b-instruct",
                "tokens_used": 150,
                "latency_ms": 234.5
            }
        """
        try:
            turn_id = log_conversation_turn(
                conversation_id=request.conversation_id,
                turn_number=request.turn_number,
                user_message=request.user_message,
                assistant_message=request.assistant_message,
                model_name=request.model_name,
                tokens_used=request.tokens_used,
                latency_ms=request.latency_ms,
                metadata=request.metadata
            )
            
            return FeedbackResponse(
                ok=True,
                message="Conversation turn logged successfully",
                turn_id=turn_id
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/stats")
    async def feedback_statistics(
        conversation_id: Optional[str] = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get feedback statistics.
        
        Query Parameters:
            conversation_id: Optional conversation ID to filter by
            days: Number of days to look back (default: 7)
        """
        try:
            stats = get_feedback_stats(
                conversation_id=conversation_id,
                days=days
            )
            
            return {
                "ok": True,
                "stats": stats
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return router


# ============================================================================
# Standalone Test Functions
# ============================================================================

if __name__ == "__main__":
    """Test the feedback logging functionality."""
    import sys
    
    print("🧪 Testing UI Feedback Logging")
    print("=" * 70)
    
    # Test 1: Log thumbs up feedback
    print("\n📊 Test 1: Log thumbs up feedback")
    try:
        feedback_id = log_feedback_to_db(
            feedback_type="thumbs_up",
            conversation_id="test-conv-001",
            message_role="assistant",
            message_content="Python is a high-level programming language...",
            user_comment="Very helpful answer!"
        )
        print(f"✅ Logged thumbs up feedback (ID: {feedback_id})")
    except Exception as e:
        print(f"❌ Failed: {e}")
        sys.exit(1)
    
    # Test 2: Log thumbs down feedback
    print("\n📊 Test 2: Log thumbs down feedback")
    try:
        feedback_id = log_feedback_to_db(
            feedback_type="thumbs_down",
            conversation_id="test-conv-001",
            message_role="assistant",
            user_comment="Answer was too technical"
        )
        print(f"✅ Logged thumbs down feedback (ID: {feedback_id})")
    except Exception as e:
        print(f"❌ Failed: {e}")
        sys.exit(1)
    
    # Test 3: Log conversation turn
    print("\n📊 Test 3: Log conversation turn")
    try:
        turn_id = log_conversation_turn(
            conversation_id="test-conv-002",
            turn_number=1,
            user_message="What is FastAPI?",
            assistant_message="FastAPI is a modern, fast web framework for building APIs with Python 3.7+",
            model_name="qwen2.5:3b-instruct",
            tokens_used=120,
            latency_ms=245.3
        )
        print(f"✅ Logged conversation turn (ID: {turn_id})")
    except Exception as e:
        print(f"❌ Failed: {e}")
        sys.exit(1)
    
    # Test 4: Get feedback stats
    print("\n📊 Test 4: Get feedback statistics")
    try:
        stats = get_feedback_stats(days=30)
        print(f"✅ Feedback stats:")
        print(f"   Total feedback: {stats['total_feedback']}")
        print(f"   Thumbs up: {stats['thumbs_up']}")
        print(f"   Thumbs down: {stats['thumbs_down']}")
        print(f"   Recent feedback: {len(stats['recent_feedback'])} items")
    except Exception as e:
        print(f"❌ Failed: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("✅ All tests passed!")
