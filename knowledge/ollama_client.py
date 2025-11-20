# ollama_client.py

import os
import json
import logging
from typing import Any, Dict, List, Optional

import requests


logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Lightweight client for talking to a local Ollama server (e.g. llama3).

    It expects environment variables (which you already set in devcontainer.json):

        OLLAMA_URL   - e.g. "http://localhost:11434/api/chat"
        OLLAMA_MODEL - e.g. "llama3"

    Basic usage:

        from ollama_client import OllamaClient

        client = OllamaClient()
        resp = client.chat([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"}
        ])

        json_resp = client.chat_json(
            messages=[
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": "Give me a small JSON object."},
            ]
        )
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        default_timeout: int = 60,
    ) -> None:
        self.base_url = base_url or os.getenv(
            "OLLAMA_URL", "http://localhost:11434/api/chat")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3")
        self.default_timeout = default_timeout

    # ------------------------------------------------------------------ #
    # Low-level call
    # ------------------------------------------------------------------ #

    def _post(
        self,
        payload: Dict[str, Any],
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Low-level POST to Ollama. Raises requests.HTTPError on non-2xx.
        """
        t = timeout or self.default_timeout

        resp = requests.post(self.base_url, json=payload, timeout=t)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------ #
    # Chat helpers
    # ------------------------------------------------------------------ #

    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        temperature: float = 0.0,
        extra_params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generic chat call. Returns the raw Ollama response JSON.

        `messages` should be a list of dicts of the form:
            {"role": "system"|"user"|"assistant", "content": "..."}

        If stream=True, this returns whatever Ollama returns for streaming
        (you can extend this later to yield chunks).
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "stream": stream,
            "temperature": temperature,
            "messages": messages,
        }

        if extra_params:
            payload.update(extra_params)

        data = self._post(payload, timeout=timeout)
        return data

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        timeout: Optional[int] = None,
        strict: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Call Ollama and *parse* a JSON object out of the model's response.

        This sets `"format": "json"` in the payload. For safety, it will also
        scan the returned content for the first '{' and last '}' and attempt
        to parse that substring as JSON.

        Returns:
            dict on success, or None if parsing fails (unless strict=True, in
            which case it will raise ValueError).
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "temperature": temperature,
            "format": "json",  # ask Ollama to enforce JSON if supported
            "messages": messages,
        }

        try:
            data = self._post(payload, timeout=timeout)
        except requests.HTTPError as e:
            logger.error("Ollama HTTP error: %s", e)
            if strict:
                raise
            return None
        except Exception as e:
            logger.error("Ollama request error: %s", e)
            if strict:
                raise
            return None

        # Typical Ollama chat format:
        # {
        #   "model": "llama3",
        #   "message": { "role": "assistant", "content": "..." },
        #   ...
        # }
        content = (
            data.get("message", {}).get("content")
            or data.get("content")
            or ""
        )

        if not isinstance(content, str) or not content.strip():
            logger.warning("Ollama returned empty content: %r", content)
            if strict:
                raise ValueError("Empty content from Ollama")
            return None

        # Robust extraction of JSON object
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            logger.warning(
                "Could not locate JSON object in content: %s", content[:200])
            if strict:
                raise ValueError(
                    f"Could not locate JSON in content: {content!r}")
            return None

        json_text = content[start: end + 1]

        try:
            parsed = json.loads(json_text)
            return parsed
        except Exception as e:
            logger.error(
                "Failed to parse JSON from Ollama content: %s :: %s", e, json_text[:200])
            if strict:
                raise ValueError(
                    f"Failed to parse JSON from content: {e}") from e
            return None
