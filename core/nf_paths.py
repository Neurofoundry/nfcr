"""
Neuroforge Centralized Path Management
=====================================

This module provides consistent path resolution for all Neuroforge components.
All paths are resolved once at import time and can be overridden via environment variables.

Usage:
    from nf_paths import DB_PATH, DATA_DIR, CONFIG_DIR
    
Environment Variables:
    NEUROFORGE_ROOT - Project root directory
    NEUROFORGE_DATA_DIR - Main data directory (default: neuroforge_system_data)
    NEUROFORGE_MEMORY_DIR - Alternative name for data directory (backward compatibility)
    NEUROFORGE_DB_PATH - Full path to SQLite database
    NEUROFORGE_ROUTING_PATH - Path to routing configuration
"""

import os
from pathlib import Path
from typing import Optional

# Load environment if available (optional dependency)
try:
    from dotenv import load_dotenv  # type: ignore
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
except ImportError:
    pass  # dotenv not available, continue with system env vars only


def _resolve_project_root() -> Path:
    """Resolve the project root directory."""
    if root := os.environ.get("NEUROFORGE_ROOT"):
        return Path(root).resolve()
    
    # Default: directory containing this file
    return Path(__file__).parent.resolve()


def _resolve_data_dir() -> Path:
    """Resolve the main data directory."""
    # Check for preferred NEUROFORGE_DATA_DIR first
    if data_dir := os.environ.get("NEUROFORGE_DATA_DIR"):
        return Path(data_dir).resolve()
    
    # Check for backward compatibility alias
    if memory_dir := os.environ.get("NEUROFORGE_MEMORY_DIR"):
        return Path(memory_dir).resolve()
    
    # Default: neuroforge_system_data relative to project root
    return PROJECT_ROOT / "neuroforge_system_data"


def _resolve_db_path() -> Path:
    """Resolve the SQLite database path."""
    if db_path := os.environ.get("NEUROFORGE_DB_PATH"):
        return Path(db_path).resolve()
    
    # Default: database_master.sqlite3 in data directory (unified database)
    return DATA_DIR / "database_master.sqlite3"


def _resolve_routing_path() -> Path:
    """Resolve the routing configuration path."""
    if routing_path := os.environ.get("NEUROFORGE_ROUTING_PATH"):
        return Path(routing_path).resolve()
    
    # Default: routing.json in project root
    return PROJECT_ROOT / "routing.json"


def _ensure_dir_exists(path: Path) -> Path:
    """Ensure directory exists, create if needed."""
    if not path.suffix:  # It's a directory
        path.mkdir(parents=True, exist_ok=True)
    else:  # It's a file, ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ======================== Public Path Constants ========================

# Core directories
PROJECT_ROOT = _resolve_project_root()
DATA_DIR = _ensure_dir_exists(_resolve_data_dir())
CONFIG_DIR = _ensure_dir_exists(DATA_DIR / "config")

# Database and configuration files
DB_PATH = _resolve_db_path()
ROUTING_PATH = _resolve_routing_path()

# Memory and logging directories
MEMORY_DIR = _ensure_dir_exists(DATA_DIR / ".." / "Memory")  # For backward compatibility
MEMORY_LOGS_DIR = _ensure_dir_exists(MEMORY_DIR / "Logs")
MEMORY_INGEST_DIR = _ensure_dir_exists(MEMORY_DIR / "Ingest")
MEMORY_COMPLETED_DIR = _ensure_dir_exists(MEMORY_INGEST_DIR / "Completed")
MEMORY_PENDING_DIR = _ensure_dir_exists(MEMORY_INGEST_DIR / "Pending")

# Business action directories
BUSINESS_ACTIONS_DIR = _ensure_dir_exists(DATA_DIR / "business_actions")
INVOICES_DIR = _ensure_dir_exists(BUSINESS_ACTIONS_DIR / "invoices")
LEADS_DIR = _ensure_dir_exists(BUSINESS_ACTIONS_DIR / "leads")
CONTRACTS_DIR = _ensure_dir_exists(BUSINESS_ACTIONS_DIR / "contracts")
EMAILS_DIR = _ensure_dir_exists(BUSINESS_ACTIONS_DIR / "emails")
REPORTS_DIR = _ensure_dir_exists(BUSINESS_ACTIONS_DIR / "reports")

# Configuration file paths
RESPONSE_PROFILE_PATH = CONFIG_DIR / "response_profile.yaml"
RESPONSE_SNIPPETS_PATH = CONFIG_DIR / "response_snippets.yaml"
CAPABILITIES_PATH = CONFIG_DIR / "capabilities.yaml"
BRIDGE_ROUTING_PATH = CONFIG_DIR / "bridge_routing.json"

# Features and uploads
FEATURES_DIR = _ensure_dir_exists(PROJECT_ROOT / "features")
UPLOADS_DIR = _ensure_dir_exists(FEATURES_DIR / "uploads")
IMAGES_DIR = _ensure_dir_exists(FEATURES_DIR / "images")
OCR_DIR = _ensure_dir_exists(FEATURES_DIR / "ocr")

# Misc and UI assets
MISC_DIR = _ensure_dir_exists(PROJECT_ROOT / "Misc")


def get_path_info() -> dict:
    """Return dictionary of all resolved paths for debugging."""
    return {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "DATA_DIR": str(DATA_DIR),
        "CONFIG_DIR": str(CONFIG_DIR),
        "DB_PATH": str(DB_PATH),
        "ROUTING_PATH": str(ROUTING_PATH),
        "MEMORY_DIR": str(MEMORY_DIR),
        "BUSINESS_ACTIONS_DIR": str(BUSINESS_ACTIONS_DIR),
        "FEATURES_DIR": str(FEATURES_DIR),
    }


def validate_paths() -> bool:
    """Validate that all critical paths are accessible."""
    try:
        # Check that data directory is writable
        test_file = DATA_DIR / ".write_test"
        test_file.touch()
        test_file.unlink()
        
        # Check that project root contains expected files
        if not (PROJECT_ROOT / "aegis_unified_core.py").exists():
            print(f"Warning: aegis_unified_core.py not found in {PROJECT_ROOT}")
            return False
            
        return True
    except Exception as e:
        print(f"Path validation failed: {e}")
        return False


if __name__ == "__main__":
    print("Neuroforge Path Configuration")
    print("=" * 40)
    for key, value in get_path_info().items():
        print(f"{key:20}: {value}")
    print()
    print(f"Paths valid: {validate_paths()}")