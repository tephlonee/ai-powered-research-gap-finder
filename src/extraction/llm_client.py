"""
Local LLM client — Ollama, with schema-constrained structured output.

Replaces every OpenAI/GPT-4o call in the old codebase. Two things make this robust on a
small local model running on a 16 GB GPU:

  1. We pass the Pydantic model's JSON schema to Ollama's `format` parameter. Ollama then
     does *constrained decoding* — it can only emit tokens that keep the output valid against
     the schema. So even a 7B/14B model returns parseable, schema-correct JSON every time;
     the model's job shrinks to choosing good *values*, not remembering JSON syntax.
  2. temperature=0 for determinism (pairs with the cache: same inputs -> same output).

No API key, no network, no per-token cost. The only resource is GPU time.
"""
from __future__ import annotations
from typing import Type, TypeVar, Optional

from pydantic import BaseModel

from config import config

try:
    import ollama
except ImportError as e:  # pragma: no cover - guidance, not logic
    raise ImportError(
        "The 'ollama' Python package is required. Install it with:\n"
        "    pip install ollama\n"
        "and make sure the Ollama server is running (https://ollama.com)."
    ) from e

T = TypeVar("T", bound=BaseModel)

_client = ollama.Client(host=config.ollama_host)


def _options() -> dict:
    return {"temperature": config.llm_temperature, "num_ctx": config.llm_num_ctx}


def structured(system: str, prompt: str, schema: Type[T], model: Optional[str] = None) -> T:
    """Chat completion constrained to `schema`. Returns a validated model instance."""
    resp = _client.chat(
        model=model or config.llm_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        format=schema.model_json_schema(),
        options=_options(),
    )
    return schema.model_validate_json(resp["message"]["content"])


def text(system: str, prompt: str, model: Optional[str] = None) -> str:
    """Plain free-text completion (used where no structure is needed)."""
    resp = _client.chat(
        model=model or config.llm_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        options=_options(),
    )
    return resp["message"]["content"]


def health_check() -> tuple[bool, str]:
    """Return (ok, message). Verifies the server is up and the model is pulled."""
    try:
        models = _client.list().get("models", [])
        names = {m.get("model", m.get("name", "")) for m in models}
        want = config.llm_model
        # Ollama tags like "qwen2.5:14b-instruct" may be listed with or without a trailing hash.
        ok = any(n == want or n.startswith(want) for n in names)
        if not ok:
            return False, (
                f"Ollama is up but model '{want}' is not pulled. Run:\n"
                f"    ollama pull {want}\n"
                f"Currently available: {', '.join(sorted(names)) or '(none)'}"
            )
        return True, f"Ollama OK, model '{want}' available."
    except Exception as exc:  # noqa: BLE001
        return False, (
            f"Could not reach Ollama at {config.ollama_host}: {exc}\n"
            "Is the Ollama server running? Start it with `ollama serve`."
        )
