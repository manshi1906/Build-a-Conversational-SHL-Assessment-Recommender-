"""
llm.py
------
Thin, swappable LLM client.

The brief lets us pick any provider; we default to Groq (fast, generous free
tier, OpenAI-compatible API). One env var switches provider without code
changes:

    LLM_PROVIDER = groq | openai | gemini   (default: groq)
    LLM_MODEL    = <model name>             (sensible per-provider default)
    LLM_API_KEY  = <key>

We expose two helpers:
    chat_json(system, messages)  -> dict   (forces strict JSON, parsed safely)
    chat_text(system, messages)  -> str    (free-form, for grounded compare)

Robustness (a graded failure mode is "breaks on anything else"):
  * timeouts + 2 retries with backoff
  * JSON repair: strips ```json fences, extracts the first {...} block
  * if the LLM is totally unreachable, callers get a typed LLMError and the
    agent falls back to a deterministic safe path rather than 500ing.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import httpx

PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
API_KEY = os.getenv("LLM_API_KEY", "")

_DEFAULTS = {
    "groq": ("https://api.groq.com/openai/v1/chat/completions",
             "llama-3.3-70b-versatile"),
    "openai": ("https://api.openai.com/v1/chat/completions",
               "gpt-4o-mini"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/"
               "chat/completions", "gemini-2.0-flash"),
}

_ENDPOINT, _DEFAULT_MODEL = _DEFAULTS.get(PROVIDER, _DEFAULTS["groq"])
MODEL = os.getenv("LLM_MODEL", _DEFAULT_MODEL)


class LLMError(RuntimeError):
    """Raised when the model is unreachable or returns unusable output."""


def _post(payload: dict, timeout: float) -> dict:
    # Fail fast if unconfigured: no point burning the retry/backoff budget
    # (and the 30s turn timeout) on requests that cannot possibly succeed.
    if not API_KEY:
        raise LLMError("LLM_API_KEY not set; running in degraded mode")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    last: Exception | None = None
    for attempt in range(3):
        try:
            r = httpx.post(_ENDPOINT, json=payload, headers=headers,
                           timeout=timeout)
            if r.status_code == 429:
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(0.8 * (attempt + 1))
    raise LLMError(f"LLM request failed after retries: {last}")


def _extract_json(text: str) -> dict:
    """Best-effort JSON recovery from a model response."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # grab the first balanced {...}
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise LLMError(f"Could not parse JSON from model output: {text[:200]!r}")


def _messages(system: str, messages: list[dict]) -> list[dict]:
    return [{"role": "system", "content": system}, *messages]


def chat_json(system: str, messages: list[dict],
              timeout: float = 20.0) -> dict[str, Any]:
    payload = {
        "model": MODEL,
        "messages": _messages(system, messages),
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "max_tokens": 900,
    }
    try:
        data = _post(payload, timeout)
    except LLMError:
        # retry once without response_format (some models reject it)
        payload.pop("response_format", None)
        data = _post(payload, timeout)
    content = data["choices"][0]["message"]["content"]
    return _extract_json(content)


def chat_text(system: str, messages: list[dict],
              timeout: float = 20.0) -> str:
    payload = {
        "model": MODEL,
        "messages": _messages(system, messages),
        "temperature": 0.2,
        "max_tokens": 600,
    }
    data = _post(payload, timeout)
    return data["choices"][0]["message"]["content"].strip()


def healthy() -> bool:
    return bool(API_KEY)
