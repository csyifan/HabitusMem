"""Minimal OpenAI-compatible chat client (standard library only).

Works with any server that implements ``POST {base_url}/chat/completions``:
vLLM, SGLang, TGI, OpenAI, DeepSeek, Qwen/DashScope compatible mode, etc.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional


class ChatAPIError(RuntimeError):
    pass


def chat(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout: int = 600,
    retries: int = 5,
    extra_body: Optional[Dict] = None,
) -> str:
    """Return the assistant message content, retrying transient failures."""
    if not base_url:
        raise ChatAPIError("base_url is empty")
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra_body:
        payload.update(extra_body)
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_error: Optional[Exception] = None
    for attempt in range(retries):
        request = urllib.request.Request(url, data=body, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"] or ""
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            last_error = ChatAPIError(f"HTTP {error.code}: {detail}")
            if error.code in (400, 401, 403, 404):
                break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as error:
            last_error = error
        if attempt + 1 < retries:
            time.sleep(min(2 ** attempt, 30))
    raise ChatAPIError(f"chat request failed: {last_error}")
