import os
import json
from typing import Any, Dict


def load_context_db() -> Dict[str, Any]:
    """
    Load shared context for contacts, databases, webhooks, topics, machines, etc.

    This is deliberately generic: each section is a dict keyed by a short name.
    """
    default = {
        "contacts": {},   # role/name -> { email, phone, ... }
        "databases": {},  # name -> { host, port, database, schema, credential_name, ... }
        "webhooks": {},   # name -> { url, method, ... }
        "topics": {},     # name -> { mqtt_topic, kafka_topic, ... }
        "machines": {},   # name -> { endpoint, topic, ... }
    }

    path = os.getenv("CONTEXT_PATH", "config/context.json")
    if not os.path.exists(path):
        return default

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        # If the file is corrupt, fail safe with an empty registry
        return default

    # Merge with defaults so missing sections don't cause KeyErrors
    for key, value in default.items():
        data.setdefault(key, value)
    return data


CONTEXT_DB: Dict[str, Any] = load_context_db()
