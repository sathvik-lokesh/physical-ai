"""
Thin, dependency-light client for Ollama's /api/generate endpoint.

Kept deliberately small: the runner owns all retry/resume policy, this module
only knows how to send one image-and-prompt request and report what came back,
including Ollama's own timing counters (which are more trustworthy than
wall-clock when the laptop is swapping).
"""

import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_HOST = "http://localhost:11434"


class OllamaError(RuntimeError):
    """Any failure talking to Ollama — transport, HTTP status, or bad JSON."""


def encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _post(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:500]
        raise OllamaError(f"HTTP {e.code} from {url}: {body}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise OllamaError(f"transport error to {url}: {e}") from e
    except json.JSONDecodeError as e:
        raise OllamaError(f"non-JSON reply from {url}: {e}") from e


def version(host: str = DEFAULT_HOST, timeout: float = 5.0) -> str:
    """Cheap liveness probe — call before starting a long batch."""
    req = urllib.request.Request(f"{host}/api/version")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8")).get("version", "?")
    except Exception as e:  # noqa: BLE001 — any failure means "not usable"
        raise OllamaError(f"Ollama not reachable at {host}: {e}") from e


def installed_models(host: str = DEFAULT_HOST, timeout: float = 10.0) -> list[str]:
    req = urllib.request.Request(f"{host}/api/tags")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        raise OllamaError(f"cannot list models on {host}: {e}") from e
    return [m["name"] for m in tags.get("models", [])]


def generate(
    model: str,
    prompt: str,
    images: list[str] | None = None,
    *,
    host: str = DEFAULT_HOST,
    json_mode: bool = True,
    timeout: float = 900.0,
    num_predict: int = 256,
    temperature: float = 0.0,
    keep_alive: str = "5m",
) -> dict:
    """
    Run one generation. Returns a dict with the raw text plus timing.

    `timeout` defaults to 15 minutes because a cold model load on a 15 W CPU
    can genuinely take over 5 minutes before the first token appears.
    """
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    if images:
        payload["images"] = images
    if json_mode:
        payload["format"] = "json"

    t0 = time.perf_counter()
    reply = _post(f"{host}/api/generate", payload, timeout)
    wall_ms = round((time.perf_counter() - t0) * 1000, 1)

    ns = reply.get("total_duration")
    return {
        "text": (reply.get("response") or "").strip(),
        "wall_ms": wall_ms,
        "ollama_ms": round(ns / 1e6, 1) if ns else None,
        "prompt_tokens": reply.get("prompt_eval_count"),
        "output_tokens": reply.get("eval_count"),
        "done_reason": reply.get("done_reason"),
    }
