import os
import json
from typing import Any, Dict, Optional


DEFAULT_SETUP_CONTEXT: Dict[str, Any] = {
    "workspace": {
        "name": "Default Workspace",
        "description": "Generic automation workspace.",
        "domain_tags": []
    },
    "environment": {
        "default_language": "python",
        "secondary_languages": [],
        "runtime": "n8n_code_node",
        "python_version": "3.11",
        "preferred_libraries": ["pandas", "numpy"],
        "forbidden_capabilities": []
    },
    "human_output": {
        "node_id": "synthetic_human_review",
        "label": "Human Review Required",
        "role_description": "A human operator reviews this step and decides how to proceed.",
        "n8n_target_type": "manualReview",
        "default_queue": None
    },
    "tooling": {
        "databases": [],
        "file_stores": [],
        "analytics": []
    },
    "code_nodes": {
        "default_language": "python",
        "python": {
            "style": "black",
            "use_type_hints": True,
            "imports": [
                "import pandas as pd",
                "import numpy as np"
            ],
            "data_frame_name": "df",
            "io_contract": {
                "input": "items[0].json",
                "output": "return items"
            }
        },
        "sql": {
            "dialect": "postgres",
            "default_schema": "public"
        }
    },
    "llm_context": {
        "safety_notes": [],
        "preferred_units": [],
        "glossary": {}
    }
}


class SetupContext:
    """
    Loads optional setup-context (workplace) configuration from a JSON file,
    or falls back to DEFAULT_SETUP_CONTEXT.

    Provides helper accessors for:
      - human output node
      - environment / language
      - code-node generation context
      - LLM prompt context
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path or os.getenv("BUILDER_SETUP_FILE")
        self._config: Dict[str, Any] = {}
        self.reload()

    def reload(self) -> None:
        """Reload configuration from file (if available), else use defaults."""
        cfg = DEFAULT_SETUP_CONTEXT.copy()
        if self.path and os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    file_cfg = json.load(f)
                # shallow merge for now; can be made deep if needed
                for k, v in file_cfg.items():
                    cfg[k] = v
            except Exception as e:
                print(f"[SetupContext] Failed to load {self.path}: {e}")
        else:
            if self.path:
                print(
                    f"[SetupContext] No setup file found at {self.path}, using defaults.")
        self._config = cfg

    # ----- Raw access -----
    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    # ----- Human output spec -----
    def get_human_output_spec(self) -> Dict[str, Any]:
        return self._config.get("human_output", {})

    def get_human_output_node_id(self) -> str:
        return self.get_human_output_spec().get("node_id", "synthetic_human_review")

    # ----- Environment / language -----
    def get_default_language(self) -> str:
        return self._config.get("environment", {}).get("default_language", "python")

    def get_runtime(self) -> str:
        return self._config.get("environment", {}).get("runtime", "n8n_code_node")

    def get_preferred_libraries(self) -> Any:
        return self._config.get("environment", {}).get("preferred_libraries", [])

    def get_forbidden_capabilities(self) -> Any:
        return self._config.get("environment", {}).get("forbidden_capabilities", [])

    # ----- Code node context -----
    def get_code_node_config(self, language: Optional[str] = None) -> Dict[str, Any]:
        cn = self._config.get("code_nodes", {})
        lang = language or cn.get("default_language", "python")
        return cn.get(lang, {})

    def get_llm_code_context(self, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Build a compact payload you can drop into LLM prompts when asking it
        to generate code nodes (e.g. 'take mean of X data').
        """
        env = self._config.get("environment", {})
        node_cfg = self.get_code_node_config(language)
        return {
            "language": language or env.get("default_language", "python"),
            "runtime": env.get("runtime", "n8n_code_node"),
            "imports": node_cfg.get("imports", []),
            "io_contract": node_cfg.get("io_contract", {}),
            "data_frame_name": node_cfg.get("data_frame_name", "df"),
            "preferred_libraries": env.get("preferred_libraries", []),
            "forbidden_capabilities": env.get("forbidden_capabilities", []),
        }

    # ----- LLM global context -----
    def get_llm_global_context(self) -> Dict[str, Any]:
        return {
            "workspace": self._config.get("workspace", {}),
            "llm_context": self._config.get("llm_context", {}),
        }
