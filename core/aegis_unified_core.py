import os
import json
import uvicorn
import logging
import uuid
import requests
import sqlite3
import time
import numpy as np
from datetime import datetime
import re
from fastapi import FastAPI, HTTPException, BackgroundTasks, APIRouter, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response
from pydantic import BaseModel
from typing import Optional, List, Dict
from bs4 import BeautifulSoup
import importlib
import pkgutil
import pathlib

# Additional imports for profile, snippets, capabilities, and dynamic execution
import yaml
import random
import re
import shutil
import threading

# Personality system imports
from nf_intent import should_execute_tool, classify_intent, extract_url
from nf_personality import build_contextual_prompt, PhraseRotator

# ---------------------------------------------------------------------------
# Environment setup
#
# To allow the Aegis core to be configured via .env files, load any variables
# present in the file specified by DOTENV_CONFIG_PATH.  If the environment
# variable is not set, fall back to a .env file located alongside this script.
#
# We wrap the import in a try/except to make python-dotenv an optional
# dependency.  If it's not installed, the core will still run using its
# hard‑coded defaults.
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None  # noqa: F401

def _load_environment() -> None:
    """Load environment variables from the configured .env file."""
    if load_dotenv is None:
        return
    # Determine which .env file to load: the path set via DOTENV_CONFIG_PATH
    # takes precedence, otherwise fall back to a .env file in the same
    # directory as this script.
    env_path = os.environ.get(
        "DOTENV_CONFIG_PATH",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
    )
    if os.path.exists(env_path):
        load_dotenv(env_path)

# Load environment variables once at module import time
_load_environment()

# Import centralized paths
from nf_paths import (
    DATA_DIR, DB_PATH, ROUTING_PATH, CONFIG_DIR,
    RESPONSE_PROFILE_PATH, RESPONSE_SNIPPETS_PATH, CAPABILITIES_PATH,
    BUSINESS_ACTIONS_DIR, FEATURES_DIR
)
from nf_recall import search_facts, search_knowledge, search_logs
from nf_schema import get_db_connection
from nf_schema import migrate_database
from nf_recall import create_recall_router
from nf_feedback import create_feedback_router
from unified_memory import (
    unified_memory_lookup, process_conversation_for_memory,
    detect_and_handle_memory_query
)

# Import phrase rotator from nf_personality (don't create duplicate)
from nf_personality import _phrase_rotator
from nf_tags import has_conversation_tags, extract_role_text

# ======================== 1. Configuration & Setup ========================
def setup_logging() -> None:
    """Configure the root logger for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

# Convert Path objects to strings for backward compatibility
DATA_BASE_DIR = str(DATA_DIR)
DATABASE_PATH = str(DB_PATH)
ROUTING_CONFIG_PATH = str(ROUTING_PATH)

# Ensure the chosen data directory exists
os.makedirs(DATA_BASE_DIR, exist_ok=True)

# Initialise logging and FastAPI
setup_logging()
# Advertise the new Neuroforge branding in the API metadata.
app = FastAPI(title="Neuroforge Unified AI Core", version="8.2.1 CORS Hotfix")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ======================== Feature Plugin System ========================
# Define a registry to store information about loaded feature modules.
FEATURE_REGISTRY: Dict[int, Dict] = {}

def load_feature_modules(app: FastAPI, config: Dict):
    """
    Dynamically load feature modules from the 'features' package and register any
    routers they expose.  Each feature module must be named 'feature_NN_xxx.py'
    where NN is the slot number.  A module may define a 'register' function which
    will be called with (app, feature_cfg, ROUTING_CONFIG_PATH).  Modules are only
    loaded if they contain a register() function.  Enabled flags are stored in
    FEATURE_REGISTRY but each feature checks its own 'enabled' flag at request time.
    """
    features_cfg = config.get("features", {}) if config else {}
    features_pkg = "features"
    pkg_path = pathlib.Path(__file__).with_name(features_pkg)
    if not pkg_path.exists():
        return
    for mod_info in sorted(pkgutil.iter_modules([str(pkg_path)]), key=lambda m: m.name):
        mod_name = mod_info.name
        if not mod_name.startswith("feature_"):
            continue
        parts = mod_name.split("_")
        try:
            slot = int(parts[1])
        except Exception:
            continue
        enabled = bool(features_cfg.get(str(slot), {}).get("enabled", False))
        try:
            module = importlib.import_module(f"{features_pkg}.{mod_name}")
            if hasattr(module, "register"):
                info = module.register(app, features_cfg.get(str(slot), {}), ROUTING_CONFIG_PATH)
                FEATURE_REGISTRY[slot] = {"enabled": enabled, "info": info or {}}
        except Exception as e:
            logging.error(f"Feature load failed for slot {slot}: {e}")

# ======================== 2. Pydantic Models ========================
class ChatQuery(BaseModel):
    """
    Represents a single chat query.  The ``model`` field is optional; if not provided
    by the client the server will fall back to the default model defined in routing.json.
    """
    prompt: str
    model: Optional[str] = None
    conversation_id: Optional[str] = None

class IngestURL(BaseModel):
    url: str
    tags: Optional[List[str]] = []

class IngestText(BaseModel):
    content: str
    source: Optional[str] = "manual"
    tags: Optional[List[str]] = []

class RenderRequest(BaseModel):
    """Request model for image rendering via feature 4.

    The prompt field contains a free-form description of the desired image. The
    server will generate a simple placeholder graphic embedding the prompt
    text. In a full deployment this endpoint could be wired into an actual
    image generation model or external API.
    """
    prompt: str


# ---------------------------------------------------------------------------
# Response Profile, Snippets, and Capabilities
#
# The following global state objects store configuration loaded from YAML
# files.  They enable dynamic adjustment of the AI's system prompt, tone,
# reusable phrases, and actions triggered by natural language commands.

# state holds the loaded response profile and runtime weight adjustments.
state: Dict[str, Dict] = {"profile": None, "weights": {}}

# snippets_state holds the parsed snippet configuration including sections
# and global defaults.
snippets_state: Dict[str, Dict] = {
    "cfg": {
        "sections": {},
        "defaults": {"pick": "weighted_random", "missing_policy": "fallback"},
        "helpers": {}
    }
}

# ---------------------------------------------------------------------------
# Personality Configuration
#
# A separate personality section is defined in routing.json to allow users to
# customise the behaviour and tone of the AI assistant beyond the more formal
# response profile.  The fields in this section include a core_trait (e.g.
# "Guardian"), a communication_style (e.g. "Concise & Analytical") and a
# list of quirks that describe idiosyncratic behaviours (e.g. always referring
# to the user as "Architect").  These values are loaded on startup and
# incorporated into the system prompt dynamically.  An empty dict indicates
# that no personality overrides are defined.
personality_cfg: Dict[str, Dict] = {}

# _ops holds the loaded capabilities specification and alias map.  It is
# populated by load_caps() on startup and reload.
_ops: Dict[str, Dict] = {"caps": {}, "aliases": {}}

# Ensure the config directory exists for storing YAML files
CONFIG_DIR = os.path.join(DATA_BASE_DIR, "config")
os.makedirs(CONFIG_DIR, exist_ok=True)

# ======================== 3. Core Services ========================

# The legacy setup_database() function has been removed. All database setup
# and migrations are now handled via migrate_database() in nf_schema.py. If
# other parts of the code previously referenced setup_database(), they should
# instead call migrate_database(). The placeholder below prevents NameError
# if any residual references remain but performs no action.
def setup_database():
    logging.info("setup_database() is deprecated; please use migrate_database() instead.")
    return

def db_execute(query: str, params: tuple = ()):
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA journal_mode=WAL;')
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        return cursor

# Base URL for the Ollama backend.  Can be overridden via the OLLAMA_URL environment variable.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

def get_embedding(text: str) -> List[float]:
    """Generate an embedding vector for the given text using the nomic-embed-text model."""
    url = f"{OLLAMA_URL}/api/embeddings"
    model = "nomic-embed-text"
    try:
        if not text.strip():
            return []
        response = requests.post(url, json={"model": model, "prompt": text}, timeout=90)
        response.raise_for_status()
        return response.json().get("embedding", [])
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to get embedding. Error: {e}")
        return []

def query_ollama(prompt: str, system_prompt: str, model: str, json_mode: bool = False) -> str:
    """
    Send a completion request to the Ollama backend.  If the backend returns an error
    status code, log the response text for easier debugging.  When json_mode is True,
    request a JSON formatted response from the LLM.
    """
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": model, 
        "prompt": prompt, 
        "system": system_prompt, 
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "num_predict": 512
        }
    }
    if json_mode:
        payload["format"] = "json"
    try:
        response = requests.post(url, json=payload, timeout=120)
        if response.status_code >= 400:
            logging.error("Ollama error %s: %s", response.status_code, response.text)
            response.raise_for_status()
        return response.json().get("response", "{}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Ollama query error: {e}")
        return f"Error: Could not connect to Ollama. ({e})"

def cosine_similarity(v1, v2):
    vec1 = np.array(v1)
    vec2 = np.array(v2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def save_knowledge_document(data: Dict) -> str:
    doc_id = str(uuid.uuid4())
    content = data.get('content', '')
    vector = get_embedding(content)
    if not vector:
        logging.warning(f"Could not generate vector for {data.get('source_location')}.")
    
    query = "INSERT INTO knowledge (doc_id, source_type, source_location, content, tags, vector, ingested_at) VALUES (?, ?, ?, ?, ?, ?, ?)"
    params = (doc_id, data.get('source_type'), data.get('source_location'), content,
              json.dumps(data.get("tags", [])), json.dumps(vector), datetime.utcnow().isoformat())
    db_execute(query, params)
    logging.info(f"Ingested: {doc_id} from {data.get('source_location')}")
    # If the content appears to contain conversation markers, store extracted user text
    try:
        content = data.get('content', '') or ''
        if content and has_conversation_tags(content).get('has_user'):
            user_text = extract_role_text(content, role='user')
            if user_text and user_text.strip():
                db_execute("UPDATE knowledge SET extracted_user_text = ? WHERE doc_id = ?", (user_text, doc_id))
    except Exception:
        # Non-fatal: continue even if this update fails
        pass
    
    # Phase 3: Generate semantic embeddings for new knowledge entry
    try:
        from semantic_intelligence import on_knowledge_insert
        # Get the rowid of the just-inserted entry
        conn = get_db_connection()
        rowid = conn.execute("SELECT rowid FROM knowledge WHERE doc_id = ?", (doc_id,)).fetchone()
        if rowid:
            on_knowledge_insert(rowid[0])
        conn.close()
    except Exception as e:
        logging.warning(f"Could not generate semantic embeddings: {e}")
    
    return doc_id

def ingest_text_content(content: str, source: str, tags: List[str] = None):
    if tags is None:
        tags = []
    save_knowledge_document({
        "source_type": "web" if source.startswith("http") else "file",
        "source_location": source,
        "content": content,
        "tags": tags
    })

# ---------------------------------------------------------------------------
# Profile and Snippets Loading
#
# The following functions load the response profile, response snippets, and
# capabilities configuration from YAML files.  They are invoked on startup
# and via the /config/reload endpoint.  These helpers populate the global
# state objects defined above.

def load_profile() -> None:
    """Load the response profile from the configured path.

    This function populates state['profile'] with the parsed YAML content and
    initialises state['weights'] for each characteristic found.  If the file
    cannot be loaded, the state remains unchanged.
    """
    try:
        # Determine the profile path from routing.json or fallback
        with open(ROUTING_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        path_str = cfg.get("response_profile_path") or str(RESPONSE_PROFILE_PATH)
    except Exception:
        path_str = str(RESPONSE_PROFILE_PATH)
    path = pathlib.Path(path_str)
    if not path.exists():
        logging.warning(f"Response profile file {path} not found; keeping existing profile state")
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        state["profile"] = data
        # Initialise weights keys if not already present
        for k in data.get("characteristics", {}).keys():
            state["weights"].setdefault(k, 0.0)
        logging.info(f"Loaded response profile from {path}")
    except Exception as e:
        logging.error(f"Failed to load response profile: {e}")


def load_snippets() -> None:
    """Load the response snippets from the configured path.

    The snippets file defines sections of reusable phrases.  If not found,
    snippets_state is reset to an empty configuration with defaults.
    """
    try:
        with open(ROUTING_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        path_str = cfg.get("response_snippets_path") or str(RESPONSE_SNIPPETS_PATH)
    except Exception:
        path_str = str(RESPONSE_SNIPPETS_PATH)
    path = pathlib.Path(path_str)
    if not path.exists():
        logging.warning(f"Response snippets file {path} not found; using empty snippets")
        snippets_state["cfg"] = {"sections": {}, "defaults": {"pick": "weighted_random", "missing_policy": "fallback"}, "helpers": {}}
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        data.setdefault("sections", {})
        data.setdefault("defaults", {"pick": "weighted_random", "missing_policy": "fallback"})
        data.setdefault("helpers", {})
        snippets_state["cfg"] = data
        logging.info(f"Loaded response snippets from {path}")
    except Exception as e:
        logging.error(f"Failed to load response snippets: {e}")


def load_caps() -> None:
    """Load the capabilities specification from the configured path.

    The capabilities YAML defines natural language aliases and the actions
    available for each capability.  It populates the _ops dict with two
    sub-keys: 'caps' (capability definitions) and 'aliases' (phrase mappings).
    """
    try:
        # Determine capabilities path from routing.json or fallback
        with open(ROUTING_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        path_str = cfg.get("capabilities_path") or str(CAPABILITIES_PATH)
    except Exception:
        path_str = str(CAPABILITIES_PATH)
    path = pathlib.Path(path_str)
    if not path.exists():
        logging.warning(f"Capabilities file {path} not found; disabling natural language actions")
        _ops["caps"], _ops["aliases"] = {}, {}
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _ops["caps"] = data.get("capabilities", {})
        _ops["aliases"] = {k.strip().lower(): v for k, v in data.get("aliases", {}).items()}
        logging.info(f"Loaded capabilities from {path}")
    except Exception as e:
        logging.error(f"Failed to load capabilities: {e}")
        _ops["caps"], _ops["aliases"] = {}, {}


def load_personality() -> None:
    """Load the personality configuration from routing.json.

    The personality section allows end users to specify a high‑level core trait,
    preferred communication style and a list of quirks.  These settings are
    reflected in the system prompt via ``build_system_prompt``.  If the file or
    section cannot be read, ``personality_cfg`` remains unchanged.
    """
    global personality_cfg
    try:
        with open(ROUTING_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        persona = cfg.get("personality", {}) or {}
        # Ensure persona is a dict with expected keys
        if isinstance(persona, dict):
            # Reset the existing config and update with new values
            personality_cfg.clear()
            # Copy keys individually to avoid mutating the original dict
            for k in ["core_trait", "communication_style", "quirks"]:
                if k in persona:
                    personality_cfg[k] = persona[k]
            logging.info(f"Loaded personality configuration from {ROUTING_CONFIG_PATH}")
    except Exception as e:
        # Do not raise on error; just log and leave defaults
        logging.error(f"Failed to load personality configuration: {e}")


def resolve_alias(phrase: str) -> Optional[str]:
    """Resolve a human phrase to a capability identifier using the alias map."""
    key = phrase.strip().lower()
    return _ops.get("aliases", {}).get(key)


def parse_action(nl: str) -> Optional[Dict]:
    """Parse a natural language command into a structured action dict.

    This parser handles simple expressions such as "change function 4 to function 9",
    "enable function 4", "disable function 4", and commands like "crawl https://..."
    or "reindex".  It returns a dict with keys: capability, action, params.
    Returns None if no pattern matches.
    """
    s = nl.strip().lower()
    # Pattern: change or map one function to another
    m = re.search(r"(change|set|map)\s+(function\s+\d+)\s+(to|=>)\s+(function\s+\d+)", s)
    if m:
        src_alias = resolve_alias(m.group(2))
        dst_alias = resolve_alias(m.group(4))
        if src_alias and dst_alias:
            return {"capability": src_alias, "action": "bind", "params": {"module_name": dst_alias}}
    # Pattern: enable/disable/bind by specifying function alias
    for act, verbs in {
        "enable": ["enable", "turn on", "switch on"],
        "disable": ["disable", "turn off", "switch off"],
        "bind": ["bind", "assign", "map", "change to", "set to"],
        "start": ["start", "begin", "run", "launch", "kick off"]
    }.items():
        for verb in verbs:
            if s.startswith(verb + " "):
                tail = s[len(verb):].strip()
                cap = resolve_alias(tail) or resolve_alias("function " + tail)
                if cap:
                    return {"capability": cap, "action": act, "params": {}}
    # Crawl commands
    if s.startswith("crawl ") and "http" in s:
        url = s.split(" ", 1)[1].strip()
        return {"capability": "crawl_url", "action": "start", "params": {"url": url}}
    # Reindex commands
    if s.startswith("reindex"):
        return {"capability": "reindex_corpus", "action": "start", "params": {"source": "all"}}
    return None


def _match_when(meta: Dict[str, str], cond: Dict) -> bool:
    """Return True if all key-value pairs in cond match meta."""
    if not cond:
        return True
    for k, v in cond.items():
        if str(meta.get(k)) != str(v):
            return False
    return True

def _expand_vars(text: str) -> str:
    """Expand helper variables such as {time_greeting} in snippet text."""
    helpers = snippets_state.get("cfg", {}).get("helpers", {})
    # Determine time of day
    hour = datetime.now().hour
    time_key = "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"
    time_list = helpers.get("time_of_day", {}).get(time_key, [])
    if time_list:
        greeting = random.choice(time_list)
        text = text.replace("{time_greeting}", greeting)
    return text

def choose_snippet(section: str, meta: Optional[Dict[str, str]] = None, default: str = "") -> str:
    """Return a phrase from a snippet section based on metadata and weights."""
    cfg = snippets_state.get("cfg", {})
    sec_cfg = cfg.get("sections", {}).get(section)
    meta = meta or {}
    if not sec_cfg:
        return default if cfg.get("defaults", {}).get("missing_policy", "fallback") == "fallback" else ""
    variants = sec_cfg.get("variants", [])
    candidates: List[Dict[str, float]] = []
    for item in variants:
        if isinstance(item, str):
            candidates.append({"text": item, "weight": 1.0})
        else:
            if _match_when(meta, item.get("when")):
                candidates.append({"text": item.get("text", ""), "weight": float(item.get("weight", 1.0))})
    if not candidates:
        return default if cfg.get("defaults", {}).get("missing_policy", "fallback") == "fallback" else ""
    mode = sec_cfg.get("pick", cfg.get("defaults", {}).get("pick", "weighted_random")).lower()
    if mode == "first":
        chosen = candidates[0]["text"]
    elif mode == "random":
        chosen = random.choice(candidates)["text"]
    else:  # weighted_random
        weights = [max(0.0, c.get("weight", 1.0)) for c in candidates]
        chosen = random.choices([c["text"] for c in candidates], weights=weights, k=1)[0]
    return _expand_vars(chosen)


def active_characteristics(meta: Optional[Dict[str, str]] = None) -> Dict[str, float]:
    """Compute current style characteristic values based on the profile, weights and sentiment."""
    meta = meta or {}
    prof = state.get("profile") or {}
    base = {k: float(v) for k, v in prof.get("characteristics", {}).items()}
    # Apply learned weights adjustments
    for k, w in state.get("weights", {}).items():
        base[k] = base.get(k, 0.0) + float(w)
    # Apply sentiment bias
    sentiment = meta.get("sentiment") or prof.get("tagging", {}).get("default_sentiment", "neutral")
    bias = prof.get("tagging", {}).get("sentiment_bias", {}).get(sentiment, {})
    for k, dv in bias.items():
        base[k] = base.get(k, 0.0) + float(dv)
    # Clamp values between 0 and 1.5
    for k in list(base.keys()):
        base[k] = max(0.0, min(1.5, base[k]))
    return base


def build_system_prompt(meta: Optional[Dict[str, str]] = None) -> str:
    """Construct a dynamic system prompt from the profile and runtime state."""
    meta = meta or {}
    prof = state.get("profile") or {}
    preamble = prof.get("system_preamble", "")
    protocols = prof.get("protocols", {})
    banned = prof.get("banned_phrases", [])
    chars = active_characteristics(meta)
    parts = []
    if preamble:
        parts.append(preamble.strip())
    # Inject personality information if available.  The personality
    # configuration comes from routing.json (see load_personality).  It
    # supplements the core system prompt with a high‑level trait, a preferred
    # communication style and a set of quirks.  Each component is optional,
    # allowing partial definitions.  These lines are added after the
    # preamble and before other protocol/style details.
    if personality_cfg:
        persona_lines = []
        trait = personality_cfg.get("core_trait")
        style = personality_cfg.get("communication_style")
        quirks = personality_cfg.get("quirks")
        if trait:
            persona_lines.append(f"You are characterised as a {trait}.")
        if style:
            persona_lines.append(f"Your communication style is {style}.")
        if quirks:
            # Join quirks into a semicolon‑separated description
            if isinstance(quirks, list):
                q_desc = "; ".join(str(q).strip() for q in quirks if q)
            else:
                q_desc = str(quirks)
            persona_lines.append(f"Quirks: {q_desc}.")
        if persona_lines:
            parts.append("\n".join(persona_lines))
    # Protocols
    if protocols:
        proto_lines = []
        for key in ["on_retrieval_hit", "on_retrieval_miss", "on_uncertainty_high", "on_hallucination_risk"]:
            if key in protocols:
                name = key.replace("_", " ").capitalize()
                proto_lines.append(f"- {name}: {protocols[key].strip()}")
        if proto_lines:
            parts.append("Enforce protocols:\n" + "\n".join(proto_lines))
    # Style weights
    if chars:
        parts.append("Style weights: " + ", ".join([f"{k}={v:.2f}" for k, v in chars.items()]))
    # Banned phrases
    if banned:
        parts.append("Banned phrases: " + ", ".join(banned))
    return "\n\n".join(parts)


def _apply_feature_action(slot: int, action: str, params: Dict) -> Dict:
    """Modify a feature slot in routing.json and reload feature modules."""
    # Read routing configuration
    with open(ROUTING_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    features = cfg.get("features", {})
    slot_key = str(slot)
    feat = features.get(slot_key, {})
    if action == "enable":
        feat["enabled"] = True
    elif action == "disable":
        feat["enabled"] = False
    elif action == "bind":
        module_name = params.get("module_name")
        if not module_name or not re.match(r"^feature_.*", module_name):
            raise ValueError("module_name missing or invalid")
        feat["name"] = module_name
        feat["enabled"] = True
    else:
        raise ValueError(f"Unsupported action {action}")
    features[slot_key] = feat
    cfg["features"] = features
    # Persist routing changes
    with open(ROUTING_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    # Reload feature modules
    load_feature_modules(app, cfg)
    return {"slot": slot, "feature": feat}

def search_knowledge_by_vector(search_term: str, limit: int = 3, threshold: float = 0.5) -> List[str]:
    """
    Vector search for knowledge - UPDATED for Phase 3.
    Now uses semantic_search_knowledge from nf_recall.py.
    """
    try:
        from nf_recall import semantic_search_knowledge, get_db_connection
        
        conn = get_db_connection()
        results = semantic_search_knowledge(conn, search_term, top_k=limit)
        conn.close()
        
        # Filter by threshold and return content
        filtered = [r['full_content'] for r in results if r.get('relevance_score', 0) >= threshold]
        logging.info(f"Found {len(filtered)} relevant knowledge chunks (semantic search).")
        return filtered
        
    except Exception as e:
        logging.error(f"Error in semantic search: {e}")
        # Fallback to basic text search
        try:
            cursor = db_execute("SELECT content FROM knowledge WHERE content LIKE ? LIMIT ?", 
                              (f"%{search_term}%", limit))
            results = [row['content'] for row in cursor.fetchall()]
            logging.info(f"Found {len(results)} knowledge chunks (fallback search).")
            return results
        except:
            return []

def save_conversation_turn(conv_id: str, user_prompt: str, ai_response: str):
    cursor = db_execute("SELECT MAX(turn) FROM unified_conversations WHERE conversation_id = ?", (conv_id,))
    max_turn = (cursor.fetchone()[0] or 0) + 1
    timestamp = datetime.utcnow().isoformat()
    
    # Insert user message
    db_execute(
        "INSERT INTO unified_conversations (conversation_id, turn, role, text, timestamp, source_table) VALUES (?, ?, ?, ?, ?, ?)",
        (conv_id, max_turn, 'user', user_prompt, timestamp, 'aegis_core')
    )
    
    # Insert assistant response  
    db_execute(
        "INSERT INTO unified_conversations (conversation_id, turn, role, text, timestamp, source_table) VALUES (?, ?, ?, ?, ?, ?)",
        (conv_id, max_turn, 'assistant', ai_response, timestamp, 'aegis_core')
    )
    
    # Extract and store personal facts from user message using unified system
    try:
        # Also populate extracted_user_text column for this new user row when possible
        try:
            if has_conversation_tags(user_prompt).get('has_user'):
                user_text = extract_role_text(user_prompt, role='user')
            else:
                user_text = user_prompt

            if user_text and user_text.strip():
                # Update the newly-inserted user row's extracted_user_text
                db_execute(
                    "UPDATE unified_conversations SET extracted_user_text = ? WHERE conversation_id = ? AND turn = ? AND role = 'user' AND (extracted_user_text IS NULL OR extracted_user_text = '')",
                    (user_text, conv_id, max_turn)
                )
        except Exception:
            # Non-fatal: if nf_tags isn't available or update fails, continue
            pass

        extracted_facts = process_conversation_for_memory(user_prompt, conv_id)
        if extracted_facts > 0:
            logging.info(f"Stored {extracted_facts} personal facts from conversation {conv_id}")
    except Exception as e:
        logging.warning(f"Unified memory extraction failed for conversation {conv_id}: {e}")

def get_conversation_history(conv_id: str, turns: int = 5) -> List[Dict]:
    """
    Retrieves the last few turns of a conversation in chronological order.
    """
    cursor = db_execute("""
        SELECT role, text, extracted_user_text, timestamp 
        FROM unified_conversations 
        WHERE conversation_id = ? 
        ORDER BY id DESC 
        LIMIT ?
    """, (conv_id, turns * 2))  # Get more since user+assistant are separate rows
    rows = cursor.fetchall()
    
    # The rows are fetched in reverse chronological order (newest first).
    # We must reverse them to get chronological order (oldest first) for the LLM context.
    history = []
    for row in reversed(rows):
        # Prefer extracted_user_text for user role if available (normalized user-only text)
        # sqlite3.Row supports mapping access via row['colname'] and has row.keys()
        content = row["text"] if "text" in row.keys() else None
        if row["role"] == 'user':
            if "extracted_user_text" in row.keys():
                ext = row["extracted_user_text"]
                if ext and str(ext).strip():
                    content = ext

        history.append({"role": row["role"], "content": content})
        
    return history

def execute_crawl_url(url: str):
    logging.info(f"BACKGROUND CRAWL: Starting for {url}")
    try:
        headers = {'User-Agent': 'NeuroforgeCrawler/1.0'}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
        text_content = soup.get_text(separator='\n', strip=True)
        if text_content:
            ingest_text_content(text_content, source=url, tags=["web-crawl", "automated"])
            logging.info(f"BACKGROUND CRAWL: Success for {url}")
        else:
            logging.warning(f"BACKGROUND CRAWL: No text content found at {url}")
    except requests.exceptions.RequestException as e:
        logging.error(f"BACKGROUND CRAWL: Failed for {url}. Error: {e}")

def execute_crawl_batch(filename: str, source_folder: str):
    logging.info(f"BACKGROUND BATCH CRAWL: Starting for file '{filename}' in '{source_folder}'")
    try:
        with open(ROUTING_CONFIG_PATH, 'r') as f:
            config = json.load(f)
        approved_folders = config.get("folders", {})
        if source_folder not in approved_folders:
            logging.error(f"SECURITY ALERT: Denying access to unapproved folder '{source_folder}' for batch crawl.")
            return
        folder_path = approved_folders[source_folder]
        file_path = os.path.join(folder_path, filename)
        if not os.path.exists(file_path):
            logging.error(f"BACKGROUND BATCH CRAWL: File not found at '{file_path}'")
            return
        with open(file_path, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip() and line.startswith('http')]
        logging.info(f"Found {len(urls)} URLs in batch file. Crawling...")
        for url in urls:
            execute_crawl_url(url)
            time.sleep(5)
        logging.info(f"BACKGROUND BATCH CRAWL: Finished '{filename}'.")
    except Exception as e:
        logging.error(f"Error during batch crawl: {e}")

def execute_ingest_file(filename: str, source_folder: str) -> str:
    logging.info(f"FILE INGESTION: Starting for '{filename}' in '{source_folder}'")
    try:
        with open(ROUTING_CONFIG_PATH, 'r') as f:
            config = json.load(f)
        
        approved_folders = config.get("folders", {})
        if source_folder not in approved_folders:
            logging.error(f"SECURITY ALERT: Denying access to unapproved folder '{source_folder}'.")
            return f"Access Denied: '{source_folder}' is not a whitelisted directory."

        folder_path = approved_folders[source_folder]
        file_path = os.path.join(folder_path, filename)

        if not os.path.exists(file_path):
            logging.error(f"FILE INGESTION: Not found at '{file_path}'")
            return f"File not found: '{filename}' in '{source_folder}'."

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        ingest_text_content(content, source=file_path, tags=["file-ingest", "manual-chat"])
        return f"Successfully ingested '{filename}'. It is now part of my knowledge base."

    except Exception as e:
        logging.error(f"Error during file ingestion: {e}")
        return f"An error occurred while ingesting the file: {e}"


def parse_intent_with_llm(prompt: str, model: str) -> Dict:
    system_prompt = """
    You are a highly intelligent NLU model. Analyze the user's prompt and respond with a JSON object.
    The possible intents are: "crawl_url", "crawl_batch", "ingest_file", "chat".
    
    - "crawl_url": User wants to crawl a single website. Extract the "url".
    - "crawl_batch": User wants to crawl a list of websites from a file. Extract "filename" and "source_folder".
    - "ingest_file": User wants to ingest the content of a local file. Extract "filename" and "source_folder".
    - "chat": For all other general conversation.
    
    Examples:
    - User: "Aegis, please crawl https://www.technologyreview.com/"
      AI: {"intent": "crawl_url", "entities": {"url": "https://www.technologyreview.com/"}}
    - User: "Hey Aegis, start a batch crawl using my urls.txt file from the projects folder."
      AI: {"intent": "crawl_batch", "entities": {"filename": "urls.txt", "source_folder": "projects"}}
    - User: "ingest the file notes___.txt from my projects folder"
      AI: {"intent": "ingest_file", "entities": {"filename": "notes___.txt", "source_folder": "projects"}}
    
    Respond ONLY with the JSON object.
    """
    try:
        response_json_str = query_ollama(prompt, system_prompt, model, json_mode=True)
        return json.loads(response_json_str)
    except (json.JSONDecodeError, TypeError):
        return {"intent": "chat", "entities": {}}


# ======================== 4. API Endpoints ========================
@app.on_event("startup")
def on_startup():
    """
    Initialise the database and load any feature plugins defined in routing.json.
    """
    # Initialise database and load configuration files
    logging.info("Running database migrations from nf_schema.py...")
    migrate_database()
    logging.info("Database migration check complete.")
    # Load response profile, snippets and capabilities
    load_profile()
    load_snippets()
    load_caps()
    load_personality()
    # Load feature modules from routing.json
    try:
        with open(ROUTING_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        cfg = {}
    load_feature_modules(app, cfg)

    # Register unified recall and feedback routers from nf_recall.py and nf_feedback.py
    logging.info("Registering unified recall API from nf_recall.py...")
    app.include_router(create_recall_router())

    logging.info("Registering feedback API from nf_feedback.py...")
    app.include_router(create_feedback_router())

# ---------------------------------------------------------------------------
# Explicit feature endpoints
#
# To improve reliability and bypass LLM-based intent parsing, we expose
# dedicated API endpoints for ingestion and crawling.  These endpoints
# correspond to common actions (file ingestion and URL crawling) and
# directly invoke the underlying helper functions.  They accept
# parameters via JSON bodies or query strings, perform basic validation
# and return a simple status message.  Clients may use these instead
# of relying on chat-based commands.

@app.post("/features/3/ingest")
async def ingest_file_endpoint(background_tasks: BackgroundTasks, filename: str = Body(...), folder: str = Body(...)):
    """
    Ingest a local file into the knowledge base.  This endpoint wraps
    ``execute_ingest_file`` and is equivalent to issuing an
    ``ingest_file`` intent via chat.  The call is synchronous: it
    returns the ingestion result as soon as the file has been processed.

    Parameters
    ----------
    filename : str
        The name of the file to ingest, relative to the selected folder.
    folder : str
        The source folder, one of "projects", "documents" or "downloads".
    """
    try:
        content = execute_ingest_file(filename, folder)
        return {"ok": True, "content": content}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/crawl")
async def crawl_endpoint(payload: Dict[str, object] = Body(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    """
    Start a URL crawl.  Accepts a JSON body with keys ``url`` and
    optional ``async_mode``.  If ``async_mode`` is true (default), the
    crawl is scheduled as a background task and the response is
    returned immediately.  If false, the crawl runs synchronously and
    returns its status when complete.  Batch crawling of URL lists is
    not supported via this endpoint.

    Example request body::

        {"url": "https://example.com", "async_mode": true}

    Parameters
    ----------
    payload : dict
        JSON body containing ``url`` (str) and ``async_mode`` (bool).
    """
    url = payload.get("url") if isinstance(payload, dict) else None
    async_mode = payload.get("async_mode", True) if isinstance(payload, dict) else True
    if not isinstance(url, str) or not url.lower().startswith("http"):
        raise HTTPException(status_code=400, detail="A full URL string is required.")
    if async_mode:
        background_tasks.add_task(execute_crawl_url, url)
        return {"ok": True, "content": f"Crawl started for {url} (background)."}
    try:
        result = execute_crawl_url(url)
        return {"ok": True, "content": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=FileResponse)
def serve_chat_ui():
    chat_html_path = "Neuroforge_CHATUI.html"
    if not os.path.exists(chat_html_path):
        raise HTTPException(status_code=404, detail="Chat UI file not found.")
    return FileResponse(chat_html_path)

@app.get("/health")
def health_check():
    """Simple health check endpoint used by the UI to determine service availability.

    Returns HTTP 200 with a JSON body indicating that the unified core is alive.
    You can extend this in the future to include more diagnostics or status details.
    """
    return {"ok": True}

# ---------------------------------------------------------------------------
# Aggregated service status endpoint
#
# This endpoint probes the health of the core (port 8010) and vault/unified core (port
# 8080) services from the server side.  It returns a JSON object indicating
# which services are up and whether the bridge (both services) is operational.
@app.get("/status/services")
def services_status():
    """Return status of core and vault services plus bridge (both up)."""
    core_ok = False
    vault_ok = False
    # Probe core on port 8010
    try:
        r_core = requests.get("http://127.0.0.1:8010/health", timeout=2)
        core_ok = r_core.ok
    except Exception:
        core_ok = False
    # Probe vault/unified core on port 8080 (self or other instance)
    try:
        r_vault = requests.get("http://127.0.0.1:8080/health", timeout=2)
        vault_ok = r_vault.ok
    except Exception:
        vault_ok = False
    bridge_ok = core_ok and vault_ok
    return {"core_ok": core_ok, "vault_ok": vault_ok, "bridge_ok": bridge_ok}

# ---------------------------- Log aggregation endpoint ---------------------------
# This endpoint reads the most recent lines from known log files and returns
# them as plain text.  Use the ``limit`` query parameter to specify how many
# lines to return (default: 100).  Clients (e.g. the web UI) can poll this
# endpoint periodically to display a unified view of server logs in the UI.
@app.get("/logs/combined")
def combined_logs(
    limit: int = 100,
    since_seconds: int = 10800,
    mask_ollama: bool = True,
) -> Response:
    """Return a cleaned slice of aggregated logs for the core and proxy.

    This endpoint reads only ``aegis_core.log`` and ``cloudflared.log`` from
    the ``logs`` directory.  It excludes stale lines older than ``since_seconds``
    (default three hours) when a timestamp can be parsed, and aggressively
    filters out messages that appear to originate from the Ollama/llama language
    model or embedding subsystem.  Set ``mask_ollama`` to ``False`` to disable
    this filtering.

    The ``limit`` parameter caps the number of returned lines (max 1000).
    """
    # Cap limit between 1 and 1000
    try:
        limit = int(limit)
    except Exception:
        limit = 100
    limit = max(1, min(limit, 1000))
    try:
        since_seconds = int(since_seconds)
    except Exception:
        since_seconds = 10800
    # Determine which log files to read
    # Allow the log directory to be overridden via routing.json (folders.logs).
    def _get_log_dir() -> str:
        try:
            with open(ROUTING_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            ld = cfg.get("folders", {}).get("logs")
            if ld:
                # Expand and normalise the path
                return os.path.abspath(os.path.expanduser(ld))
        except Exception:
            pass
        # Fallback to a local "logs" subfolder in the current working directory
        return os.path.join(os.getcwd(), "logs")

    log_dir = _get_log_dir()
    files_to_read = [
        os.path.join(log_dir, "aegis_core.log"),
        os.path.join(log_dir, "cloudflared.log"),
    ]
    lines: List[str] = []
    # Helper to decide if a line is recent based on its timestamp
    def is_recent(line: str) -> bool:
        if since_seconds <= 0:
            return True
        import datetime as dt
        import re as _re
        # Pattern 1: [YYYY-MM-DD HH:MM:SS]
        m1 = _re.search(r"\[(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})\]", line)
        # Pattern 2: time=2025-10-06T23:55:50.092-04:00
        m2 = _re.search(r"time=([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.+-]+)", line)
        try:
            if m1:
                ts = dt.datetime.fromisoformat(f"{m1.group(1)} {m1.group(2)}")
            elif m2:
                iso = m2.group(1)
                ts = dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
            else:
                return True
            now = dt.datetime.now(ts.tzinfo or dt.timezone.utc)
            return (now - ts) <= dt.timedelta(seconds=since_seconds)
        except Exception:
            return True
    # Read and filter lines
    for path in files_to_read:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    lines_in_file = fh.readlines()[-limit:]
                    src = os.path.basename(path)
                    for ln in lines_in_file:
                        if not is_recent(ln):
                            continue
                        lines.append(f"[{src}] {ln.rstrip()}")
            except Exception:
                continue
    # Optionally filter out Ollama/LLM-related noise using a regex
    if mask_ollama:
        # To avoid leaking logs from the Ollama/llama embedding subsystem or
        # unrelated server startup noise into the combined log feed, match
        # against a broad set of keywords.  This pattern catches common markers
        # such as the executable name (ollama), implementation identifiers
        # (llama_context, llama_model_loader), extended BERT and nomic variants,
        # and other noise like kv dumps or tokenizer metadata.  Additionally, it
        # filters out generic server startup lines produced by Uvicorn and
        # similar frameworks (e.g. "Application startup complete", "Started
        # server process", "Uvicorn running on").  You can extend this list
        # should you encounter further noisy prefixes.
        import re as _re
        noise_rx = _re.compile(
            r"(?:\bollama\b|llama_|gguf|nomic(?:-bert)?|bert\b|"
            r"llama_context|llama_model_loader|print_info:|EOG token|"
            r"\bembedding(?:s)?\b|output_reserve|server\.go:\d+|sched\.go:\d+|"
            r"runner started|tokenizer\.ggml|kv \d+:|general\.name|"
            r"uvicorn|application startup complete|started server process|"
            r"uvicorn running on|info: started)",
            _re.IGNORECASE,
        )
        lines = [ln for ln in lines if not noise_rx.search(ln)]
    # Limit lines to the requested amount
    if len(lines) > limit:
        lines = lines[-limit:]
    body = "\n".join(lines)
    return Response(
        content=body,
        media_type="text/plain",
        headers={"Cache-Control": "no-store"},
    )

# Endpoint to stream logs specific to the Ollama language model.  This reads
# only the ``ollama.log`` file located under the ``logs`` directory and
# returns the last ``limit`` lines.  Use this endpoint to populate Console 3
# in the UI.
@app.get("/logs/ollama")
def ollama_logs(limit: int = 100) -> Response:
    """Return the last ``limit`` lines from logs/ollama.log as plain text.

    This endpoint isolates logs produced by the local LLM (Ollama).  If
    ``ollama.log`` does not exist, an empty string is returned.  Cache is
    disabled for this response to ensure fresh content each poll.
    """
    try:
        limit_val = int(limit)
    except Exception:
        limit_val = 100
    limit_val = max(1, min(limit_val, 1000))
    # Determine log directory from configuration or fallback
    def _get_log_dir() -> str:
        try:
            with open(ROUTING_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            ld = cfg.get("folders", {}).get("logs")
            if ld:
                return os.path.abspath(os.path.expanduser(ld))
        except Exception:
            pass
        return os.path.join(os.getcwd(), "logs")

    log_dir = _get_log_dir()
    path = os.path.join(log_dir, "ollama.log")
    lines: List[str] = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                lines = fh.readlines()[-limit_val:]
        except Exception:
            lines = []
    body = "\n".join(ln.rstrip("\n") for ln in lines)
    return Response(
        content=body,
        media_type="text/plain",
        headers={"Cache-Control": "no-store"},
    )

@app.post("/chat")
def chat_endpoint(query: ChatQuery, background_tasks: BackgroundTasks):
    conv_id = query.conversation_id or str(uuid.uuid4())
    # Load base system prompt and default model.  If routing.json is
    # missing or does not specify a default, fall back to a generic
    # Neuroforge introduction.  This removes the old "Aegis" naming.
    try:
        with open(ROUTING_CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
        base_system_prompt = config.get("system_prompts", {}).get("default", "You are Neuroforge, a helpful AI.")
        default_model = config.get("ai_models", {}).get("default", "qwen2.5:3b-instruct")
    except FileNotFoundError:
        base_system_prompt = "You are Neuroforge, a helpful AI."
        default_model = "qwen2.5:3b-instruct"
    model = query.model or default_model
    
    # ==================== INTENT DETECTION (NEW) ====================
    # Use the new intent detection system to prevent false-positive tool triggers
    should_exec, detected_tool, confidence = should_execute_tool(query.prompt)
    
    response_content = ""
    
    # ==================== FEEDBACK DETECTION ====================
    # Check if user is giving feedback/correction
    if should_exec and detected_tool == "feedback":
        from nf_intent import extract_feedback_pattern
        from nf_feedback import log_feedback_to_db
        
        # Extract the specific pattern/phrase to avoid
        problematic_pattern = extract_feedback_pattern(query.prompt)
        
        # Log negative feedback
        user_comment = query.prompt
        if problematic_pattern:
            user_comment = f"Avoid using: '{problematic_pattern}'. Original feedback: {query.prompt}"
        
        try:
            # Get the last assistant message from conversation history
            cursor = db_execute(
                "SELECT text FROM unified_conversations WHERE conversation_id = ? AND role = 'assistant' ORDER BY id DESC LIMIT 1",
                (conv_id,)
            )
            last_response = cursor.fetchone()
            message_text = last_response['text'] if last_response else ""
            
            # Log the feedback
            log_feedback_to_db(
                feedback_type="thumbs_down",
                conversation_id=conv_id,
                message_role="assistant",
                message_text=message_text,
                user_comment=user_comment
            )
            
            if problematic_pattern:
                response_content = f"Got it - I'll avoid using '{problematic_pattern}' in the future. Thanks for the correction!"
            else:
                response_content = "Noted. I'll work on improving that. Thanks for the feedback!"
            
            logging.info(f"Logged negative feedback for pattern: {problematic_pattern}")
            
        except Exception as e:
            logging.error(f"Error logging feedback: {e}")
            response_content = "I understand you're giving feedback. I'll try to do better."
        
        # Save and return early
        save_conversation_turn(conv_id, query.prompt, response_content)
        return {"ok": True, "content": response_content, "conversation_id": conv_id}
    
    # If intent detection suggests a tool execution with high confidence, handle it
    if should_exec and detected_tool:
        if detected_tool == "crawl":
            url = extract_url(query.prompt)
            if url and url.startswith("http"):
                background_tasks.add_task(execute_crawl_url, url)
                response_content = f"Acknowledged, Architect. Initiating crawl for: {url}. This will run in the background."
            else:
                response_content = "Invalid URL. Please provide a full URL for the crawl command."
        elif detected_tool == "recall":
            # Use the unified recall API from nf_recall.py
            # Extract the query text (everything after common recall trigger phrases)
            recall_query = query.prompt
            for pattern in ["recall", "remember", "do you recall", "what did we discuss"]:
                recall_query = re.sub(rf"\b{pattern}\b", "", recall_query, flags=re.IGNORECASE).strip()
            
            # Call the unified recall API
            try:
                import requests
                recall_response = requests.post(
                    "http://127.0.0.1:8080/recall",
                    json={"query": recall_query, "top_k": 5},
                    timeout=10
                )
                if recall_response.ok:
                    recall_data = recall_response.json()
                    results = recall_data.get("results", [])
                    if results:
                        response_content = "Here's what I found:\n\n" + "\n\n".join([
                            f"• {r['content'][:200]}... (score: {r['score']:.2f})"
                            for r in results[:3]
                        ])
                    else:
                        response_content = "I couldn't find anything related to that in my memory."
                else:
                    response_content = "Memory recall system is currently unavailable."
            except Exception as e:
                logging.error(f"Recall API error: {e}")
                response_content = "I encountered an issue accessing my memory system."
        
        elif detected_tool == "ingest":
            # Handle file ingestion (keeping existing logic structure)
            # Parse the filename from the message
            filename_match = re.search(r'ingest.*?(["\']?)([^\s"\']+\.(txt|pdf|md|doc|docx))\1', query.prompt, re.IGNORECASE)
            if filename_match:
                filename = filename_match.group(2)
                source_folder = "Memory/Ingest/Pending"  # Default ingestion folder
                response_content = execute_ingest_file(filename, source_folder)
            else:
                response_content = "Please specify a filename to ingest."
        
        # If we handled a tool, save the conversation and return early
        if response_content:
            save_conversation_turn(conv_id, query.prompt, response_content)
            return {"ok": True, "content": response_content, "conversation_id": conv_id}
    
    # ==================== FALLBACK TO OLD INTENT SYSTEM ====================
    # If the new intent detection didn't trigger, fall back to the old NLU system
    # (This maintains backward compatibility during the transition period)
    parsed_command = parse_intent_with_llm(query.prompt, model)
    intent = parsed_command.get("intent", "chat")
    entities = parsed_command.get("entities", {})
    
    # Handle crawling and ingestion intents (old system)
    if intent == "crawl_url":
        url = entities.get("url")
        if url and url.startswith("http"):
            background_tasks.add_task(execute_crawl_url, url)
            response_content = f"Acknowledged, Architect. Initiating crawl for: {url}. This will run in the background."
        else:
            response_content = "Invalid URL. Please provide a full URL for the crawl command."
    elif intent == "crawl_batch":
        filename = entities.get("filename")
        source_folder = entities.get("source_folder")
        if filename and source_folder:
            background_tasks.add_task(execute_crawl_batch, filename, source_folder)
            response_content = f"Acknowledged. Batch crawl initiated from '{filename}' in '{source_folder}'."
        else:
            response_content = "A filename and source folder are required for a batch crawl."
    elif intent == "ingest_file":
        filename = entities.get("filename")
        source_folder = entities.get("source_folder")
        if filename and source_folder:
            response_content = execute_ingest_file(filename, source_folder)
        else:
            response_content = "A filename and source folder are required for file ingestion."
    else:
        # ==================== NATURAL MEMORY QUERY DETECTION ====================
        # Check if user is asking for memory without explicit commands
        memory_response = detect_and_handle_memory_query(query.prompt, conv_id)
        if memory_response:
            # Auto-extract any personal facts from the query before responding
            process_conversation_for_memory(query.prompt, conv_id)
            response_content = memory_response
        else:
            # RAG + Memory + Dynamic system prompt for general chat
            logging.info("Handling as general chat with RAG and unified memory.")
            
            # ==================== AUTOMATIC FACT EXTRACTION ====================
            # Extract personal facts from user message automatically
            extracted_facts = process_conversation_for_memory(query.prompt, conv_id)
            if extracted_facts > 0:
                logging.info(f"Extracted {extracted_facts} personal facts from user message")
            
            # Build metadata for snippet and style selection
            meta = {"sentiment": "neutral", "intent": intent}
            
            # ==================== UNIFIED MEMORY SEARCH ====================
            # Use unified memory system instead of broken memory_lookup
            mem_hits, memory_intro = unified_memory_lookup(query.prompt, conv_id, limit=5)
            vec_hits = search_knowledge_by_vector(query.prompt)
            
            # Combine hits, avoiding duplicates
            relevant_knowledge = mem_hits + [x for x in vec_hits if x not in mem_hits]
            
            # Conversation history context
            history = get_conversation_history(conv_id)
            history_context = "\n".join([f"{turn['role']}: {turn['content']}" for turn in history])
            knowledge_context = "\n\n".join(relevant_knowledge)
            
            # ==================== NO INTRO PHRASES ====================
            # Don't inject meta-commentary like "Here's what I found:"
            # Let the AI respond naturally with context already in the prompt
            intro = ""
        # ==================== DYNAMIC PERSONALITY SYSTEM (NEW) ====================
        # Build contextual system prompt using the new personality system
        # This includes: conversation context, feedback sentiment, phrase variety rules
        try:
            contextual_prompt = build_contextual_prompt(
                base_prompt=base_system_prompt,
                conversation_id=conv_id,
                user_message=query.prompt
            )
            full_system_prompt = contextual_prompt
            logging.info(f"Using dynamic contextual prompt for conversation {conv_id}")
        except Exception as e:
            # Fallback to base system prompt if new personality system fails
            logging.warning(f"Failed to build contextual prompt, using simple base prompt: {e}")
            full_system_prompt = base_system_prompt
        # Compose the user-visible context for the LLM
        # Keep it natural - don't over-instruct or the LLM will be too meta
        if relevant_knowledge or history:
            enriched_prompt = ""
            if relevant_knowledge:
                enriched_prompt += f"Relevant information from memory and knowledge base:\n{knowledge_context}\n\n"
            if history:
                enriched_prompt += f"Recent conversation history:\n{history_context}\n\n"
            # Append the current user message naturally (no 'User:' label)
            enriched_prompt += f"Current user message: {query.prompt}"
        else:
            # No context - just the user message
            enriched_prompt = query.prompt
        
        # Send to LLM
        model_response = query_ollama(enriched_prompt, full_system_prompt, model)
        
        # Only prepend intro if there actually is one
        if intro and intro.strip():
            response_content = f"{intro} {model_response}".strip()
        else:
            response_content = model_response.strip()
    # Save conversation turn and return response
    save_conversation_turn(conv_id, query.prompt, response_content)
    return {"ok": True, "content": response_content, "conversation_id": conv_id}

@app.post("/ingest/text")
def ingest_text_endpoint(data: IngestText):
    doc_id = save_knowledge_document({"source_type": "text", "source_location": data.source, "content": data.content, "tags": data.tags})
    return {"ok": True, "message": "Text ingested successfully.", "doc_id": doc_id}


# ======================== 5. Configuration & Feedback Endpoints ========================

@app.post("/config/reload")
def reload_configurations():
    """Reload the response profile, snippets and capabilities configuration.

    Invoking this endpoint forces the server to re-read the YAML files on disk
    without requiring a restart.  The response indicates which components
    were reloaded.
    """
    load_profile()
    load_snippets()
    load_caps()
    load_personality()
    return {"ok": True, "reloaded": ["profile", "snippets", "capabilities", "personality"]}

@app.post("/system/backup")
def system_backup():
    """
    Create a backup of the AEGIS_ROOT directory.
    The backup will be stored as <AEGIS_ROOT>_Backup.
    Existing backup folder will be overwritten.
    """
    root = os.environ.get("AEGIS_ROOT")
    if not root:
        return {"ok": False, "error": "AEGIS_ROOT not set"}
    backup_path = f"{root}_Backup"
    try:
        # remove old backup if exists
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
        shutil.copytree(root, backup_path)
        logging.info(f"Backup created at {backup_path}")
        return {"ok": True, "dest": backup_path}
    except Exception as e:
        logging.error(f"Backup failed: {e}")
        return {"ok": False, "error": str(e)}

@app.post("/system/shutdown")
def system_shutdown():
    """
    Shut down the running server. This will terminate the process.
    """
    def do_exit():
        time.sleep(0.5)
        os._exit(0)
    threading.Thread(target=do_exit, daemon=True).start()
    logging.info("Shutdown initiated via API")
    return {"ok": True, "message": "Server shutdown initiated."}


# The legacy /feedback and /export/turns.jsonl endpoints have been removed in favour of the new unified feedback API provided by nf_feedback.py.

# ---------------------------------------------------------------------------
# Conversation export
#
# Exports all conversation turns for a given conversation ID as structured
# JSON.  This endpoint can be used by the chat UI to reload past sessions or
# by administrators to archive conversations externally.  Each turn is
# returned with its turn number, user prompt, assistant response and
# timestamp.
@app.get("/conversations/{conv_id}/export")
def export_conversation(conv_id: str):
    """Export an entire conversation history as JSON."""
    try:
        cursor = db_execute(
            "SELECT turn, user_prompt, ai_response, timestamp FROM conversations WHERE conversation_id = ? ORDER BY turn",
            (conv_id,)
        )
        rows = cursor.fetchall()
        turns = [
            {
                "turn": r["turn"],
                "user_prompt": r["user_prompt"],
                "ai_response": r["ai_response"],
                "timestamp": r["timestamp"],
            }
            for r in rows
        ]
        return {"conversation_id": conv_id, "turns": turns}
    except Exception as e:
        logging.error(f"Conversation export failed: {e}")
        raise HTTPException(500, f"Conversation export failed: {e}")

# ---------------------------------------------------------------------------
# Memory notes API
#
# These endpoints allow clients to persist arbitrary snippets of text under a
# user-defined key (note_id) and later retrieve or list them.  Notes are
# stored in the ``notes`` table created in setup_database().

@app.post("/memory/notes")
async def create_note(
    note_id: str = Body(..., embed=True),
    content: str = Body(..., embed=True)
):
    """Create or update a memory note identified by ``note_id``."""
    ts = datetime.utcnow().isoformat()
    # Insert or replace: if the note_id already exists, update its content
    try:
        db_execute(
            "INSERT INTO notes(note_id, content, timestamp) VALUES (?,?,?) ON CONFLICT(note_id) DO UPDATE SET content=excluded.content, timestamp=excluded.timestamp",
            (note_id, content, ts),
        )
    except sqlite3.OperationalError as e:
        # If the notes table does not exist, initialise the database and retry.
        if "no such table: notes" in str(e).lower():
            # Run database migrations to ensure the notes table exists, then retry
            migrate_database()
            db_execute(
                "INSERT INTO notes(note_id, content, timestamp) VALUES (?,?,?) ON CONFLICT(note_id) DO UPDATE SET content=excluded.content, timestamp=excluded.timestamp",
                (note_id, content, ts),
            )
        else:
            raise
    return {"ok": True, "note_id": note_id, "timestamp": ts}

@app.get("/memory/notes/{note_id}")
def read_note(note_id: str):
    """Fetch a memory note by its identifier."""
    cursor = db_execute("SELECT note_id, content, timestamp FROM notes WHERE note_id = ?", (note_id,))
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(404, f"Note '{note_id}' not found")
    return {"note_id": row["note_id"], "content": row["content"], "timestamp": row["timestamp"]}

@app.get("/memory/notes")
def list_notes():
    """List all stored memory notes."""
    cursor = db_execute("SELECT note_id, content, timestamp FROM notes ORDER BY timestamp DESC")
    notes = [dict(note_id=r["note_id"], content=r["content"], timestamp=r["timestamp"]) for r in cursor.fetchall()]
    return {"notes": notes}

# ---------------------------------------------------------------------------
# Features listing API
#
# Return a summary of loaded features, including whether each slot is enabled
# and any metadata returned by the feature's register() function.  This
# endpoint allows clients to inspect the feature registry from the UI.

@app.get("/features")
def list_features():
    """Return a map of feature slots to their enabled state and info."""
    return {
        str(slot): {
            "enabled": data.get("enabled"),
            "info": data.get("info", {})
        }
        for slot, data in FEATURE_REGISTRY.items()
    }


# The legacy /recall and /recall/stats endpoints have been removed. Recall functionality is now provided by nf_recall via the router registered during startup.


# ======================== Feedback & Conversation Logging ========================
# Import feedback logging functionality
from nf_feedback import (
    log_feedback_to_db, log_conversation_turn, get_feedback_stats,
    FeedbackRequest, TurnLogRequest, FeedbackResponse
)


@app.post("/feedback/thumbs", response_model=FeedbackResponse)
def log_thumbs_feedback(request: FeedbackRequest):
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
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Feedback logging failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback/turn", response_model=FeedbackResponse)
def log_turn(request: TurnLogRequest):
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
        logging.error(f"Turn logging failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feedback/stats")
def feedback_statistics(
    conversation_id: Optional[str] = None,
    days: int = 7
):
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
        logging.error(f"Feedback stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/act")
def act(
    nl_command: str = Body(..., embed=True),
    dry_run: bool = Body(default=False, embed=True),
    actor: str = Body(default="user", embed=True)
):
    """Execute or dry-run a natural language operation defined in capabilities.yaml.

    - ``nl_command``: the natural language command to parse (e.g. "enable function 4").
    - ``dry_run``: if True, only simulate the operation and log it; do not modify state.
    - ``actor``: identifier of the user initiating the action; stored in audit log.
    Returns JSON with the result or an error message.
    """
    load_caps()  # ensure latest capabilities
    action_spec = parse_action(nl_command)
    if not action_spec:
        return PlainTextResponse(json.dumps({"ok": False, "reason": "unrecognized_command"}), status_code=400)
    capability = action_spec["capability"]
    action = action_spec["action"]
    params = action_spec.get("params", {})
    spec = _ops.get("caps", {}).get(capability)
    if not spec:
        return PlainTextResponse(json.dumps({"ok": False, "reason": "unknown_capability"}), status_code=400)
    if action not in spec.get("allowed_actions", []):
        return PlainTextResponse(json.dumps({"ok": False, "reason": "action_not_allowed"}), status_code=403)
    # Dry run only logs the intended operation
    if dry_run:
        db_execute(
            "INSERT INTO ops_audit(ts, action, capability, params, actor, mode, outcome) VALUES (?,?,?,?,?,?,?)",
            (int(time.time()), action, capability, json.dumps(params), actor, "dry_run", "ok")
        )
        return {"ok": True, "dry_run": True, "capability": capability, "action": action, "params": params}
    # Execute the action depending on its type
    try:
        outcome = "ok"
        result_data = None
        if spec.get("type") == "feature_slot":
            slot = spec.get("slot")
            result_data = _apply_feature_action(slot, action, params)
        elif spec.get("type") == "task":
            # Insert a new task into the queue
            db_execute(
                "INSERT INTO task_queue(ts_created, status, task_name, payload, requested_by, result) VALUES (?,?,?,?,?,?)",
                (int(time.time()), "queued", spec.get("task_name"), json.dumps(params), actor, json.dumps({}))
            )
            result_data = {"queued": spec.get("task_name"), "payload": params}
        else:
            return PlainTextResponse(json.dumps({"ok": False, "reason": "unknown_type"}), status_code=500)
        db_execute(
            "INSERT INTO ops_audit(ts, action, capability, params, actor, mode, outcome) VALUES (?,?,?,?,?,?,?)",
            (int(time.time()), action, capability, json.dumps(params), actor, "apply", outcome)
        )
        return {"ok": True, "result": result_data}
    except Exception as e:
        db_execute(
            "INSERT INTO ops_audit(ts, action, capability, params, actor, mode, outcome) VALUES (?,?,?,?,?,?,?)",
            (int(time.time()), action, capability, json.dumps(params), actor, "apply", "error")
        )
        return PlainTextResponse(json.dumps({"ok": False, "reason": "execution_error", "detail": str(e)}), status_code=500)

# ======================== 5. Server Runner ========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    logging.info(f"Neuroforge Unified AI Core launching on http://localhost:{port}")
    uvicorn.run("aegis_unified_core:app", host="0.0.0.0", port=port, reload=False)

