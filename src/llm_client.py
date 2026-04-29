"""Local LM Studio client for the music recommender.

LM Studio exposes an OpenAI-compatible API at localhost:1234 by default.
This module wraps that endpoint so the rest of the project can call a
local LLM without depending on any cloud provider.
"""

import json
import urllib.request
import urllib.error
from typing import Optional


DEFAULT_BASE_URL = "http://localhost:1234"


class LMStudioClient:
    """Thin client for an LM Studio local inference server."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = "", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    # -- public API ---------------------------------------------------------

    def chat(self, messages: list, **kwargs) -> Optional[str]:
        """Send a chat-completion request and return the assistant text.

        Parameters
            messages : OpenAI-style list of {"role", "content"} dicts.
            **kwargs  : Extra body keys (temperature, max_tokens, etc.).

        Returns
            The assistant message string, or None on failure.
        """
        body = {"messages": messages}
        if self.model:
            body["model"] = self.model
        body.update(kwargs)

        try:
            raw = self._post("/v1/chat/completions", body)
        except LMStudioError as e:
            print(f"[llm_client] {e}")
            return None

        content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        finish_reason = raw.get("choices", [{}])[0].get("finish_reason", "")

        if finish_reason == "length":
            print(f"[llm_client] Warning: Response truncated (max_tokens reached).")
        
        if not content:
            # Check for reasoning_content (common in CoT models) to provide better debugging
            reasoning = raw.get("choices", [{}])[0].get("message", {}).get("reasoning_content", "")
            if reasoning:
                print(f"[llm_client] Info: Model used reasoning but produced no content. "
                      f"Reasoning length: {len(reasoning)} chars.")
            else:
                print(f"[llm_client] Warning: Empty content in response. Raw: {raw}")
        
        return content.strip() if content else None

    def available_models(self) -> list[str]:
        """List models loaded by the LM Studio server."""
        try:
            data = self._get("/v1/models")
        except LMStudioError as e:
            print(f"[llm_client] {e}")
            return []
        models_list = data.get("data", []) if isinstance(data, dict) else data
        return [m["id"] for m in models_list] if isinstance(models_list, list) else []

    # -- internals ----------------------------------------------------------

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        url = f"{self.base_url}{path}"
        headers = {"Content-Type": "application/json"}
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            err_body = e.read().decode() if e.fp else ""
            raise LMStudioError(f"HTTP {e.code} from {url}: {err_body}")
        except urllib.error.URLError as e:
            raise LMStudioError(f"Cannot connect to {url}. Is LM Studio running? ({e.reason})")

    def _post(self, path: str, body: dict) -> dict:
        return self._request("POST", path, body)

    def _get(self, path: str) -> dict:
        return self._request("GET", path)


class LMStudioError(Exception):
    """Raised when the LM Studio server is unreachable or returns an error."""
    pass
