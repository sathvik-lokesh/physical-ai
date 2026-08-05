"""
Task specs: what to ask the VLM, and how to turn its answer into something scorable.

A task is a small JSON file so that projects p1..p4 can add tasks without
touching harness code. Shape:

    {
      "name": "failure_detect",
      "json_mode": true,
      "num_predict": 200,
      "prompt": "The robot was asked to {goal}. Did it fail? ...",
      "fields": {
        "failed":       {"type": "bool"},
        "failure_mode": {"type": "enum",
                         "values": ["stuck", "oscillation", "lost", "none"]},
        "reason":       {"type": "str"}
      }
    }

`prompt` is rendered by filling `{placeholder}` spans from the manifest item's
`meta` block. Unknown placeholders are left as-is rather than raising, so a
half-populated manifest degrades instead of crashing a 3-hour run — and literal
JSON braces in the prompt are passed through untouched.
"""

import json
import re
from pathlib import Path

_TRUE = {"true", "yes", "y", "1", "failed", "fail"}
_FALSE = {"false", "no", "n", "0", "success", "succeeded", "ok"}


_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def render_prompt(template: str, meta: dict) -> str:
    """
    Fill {placeholders} from meta, leaving anything else untouched.

    Deliberately *not* `str.format`: prompts routinely contain literal JSON —
    `Respond as {"category": "..."}` — which str.format reads as a format spec
    and rejects. Only bare `{identifier}` spans are substituted, and unknown
    identifiers are left in place rather than raising mid-run.
    """
    return _PLACEHOLDER.sub(
        lambda m: str(meta[m.group(1)]) if m.group(1) in meta else m.group(0),
        template,
    )


class Task:
    def __init__(self, spec: dict, source: Path | None = None):
        missing = {"name", "prompt"} - set(spec)
        if missing:
            raise ValueError(f"task spec {source} missing keys: {sorted(missing)}")
        self.name: str = spec["name"]
        self.prompt: str = spec["prompt"]
        self.fields: dict = spec.get("fields", {})
        self.json_mode: bool = spec.get("json_mode", True)
        self.num_predict: int = spec.get("num_predict", 256)
        self.source = source

    @classmethod
    def load(cls, path: str | Path) -> "Task":
        path = Path(path)
        with open(path) as f:
            return cls(json.load(f), source=path)

    def prompt_for(self, item: dict) -> str:
        return render_prompt(self.prompt, item.get("meta", {}))

    def parse(self, raw: str) -> dict:
        """
        Turn the model's raw text into {field: value} per the spec.

        Never raises: a garbled answer yields `{}` plus whatever fields could be
        recovered, and scoring treats a missing field as wrong. Losing one item
        of a long batch to an exception is worse than scoring it as a miss.
        """
        blob = _extract_json(raw)
        out: dict = {}
        for name, fspec in self.fields.items():
            value = blob.get(name) if isinstance(blob, dict) else None
            if value is None:
                value = _scan_for_field(raw, name)
            coerced = _coerce(value, fspec, raw)
            if coerced is not None:
                out[name] = coerced
        return out


def _extract_json(raw: str) -> dict:
    """Parse JSON, tolerating prose or code fences around it."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|```$", "", text).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, ValueError):
        pass
    # Fall back to the first balanced-looking {...} span in the text
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


def _scan_for_field(raw: str, name: str):
    """Last resort for non-JSON answers: look for `name: value` in the prose."""
    m = re.search(rf'["\']?{re.escape(name)}["\']?\s*[:=]\s*(.+)', raw, re.IGNORECASE)
    return m.group(1).strip().strip('",') if m else None


def _coerce(value, fspec: dict, raw: str):
    kind = fspec.get("type", "str")

    if kind == "bool":
        if isinstance(value, bool):
            return value
        text = str(value).strip().strip('."\'').lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        return None

    if kind == "enum":
        values = [str(v) for v in fspec.get("values", [])]
        text = str(value).strip().strip('."\'').lower() if value is not None else ""
        match = _match_enum(text, values)
        if match is None and fspec.get("scan_raw", True):
            # The field may be absent but the word present somewhere in the answer
            match = _match_enum(raw.lower(), values)
        return match

    if kind in ("int", "float"):
        if value is None:
            return None
        m = re.search(r"-?\d+(?:\.\d+)?", str(value))
        if not m:
            return None
        return int(float(m.group())) if kind == "int" else float(m.group())

    if value is None:
        return None
    return str(value).strip()


def _match_enum(text: str, values: list[str]) -> str | None:
    if not text:
        return None
    # Longest label first, so "goal unreachable" wins over "goal"
    for v in sorted(values, key=len, reverse=True):
        if v.lower() in text:
            return v
    tokens = set(re.findall(r"[a-z0-9_]+", text))
    best, best_score = None, 0
    for v in values:
        score = len(tokens & set(re.findall(r"[a-z0-9_]+", v.lower())))
        if score > best_score:
            best, best_score = v, score
    return best
