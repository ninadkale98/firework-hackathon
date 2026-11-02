"""Every model the hackathon compares, by short name, behind the engine's own
``SchemaCompleter`` boundary.

Open models and GPT go through ``OpenAISchemaCompleter`` with an endpoint address;
Claude goes through ``AnthropicSchemaCompleter``. Run with ``backend`` on
``PYTHONPATH`` and the keys in ``backend/.env``.

Fireworks model ids marked ``verify`` come from third-party listings: check them
with ``firectl model get`` once the key is in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

#: The hackathon's own keys (git-ignored), read before ``backend/.env`` so they win:
#: ``FIREWORKS_API_KEY`` and ``DISTILL_OPENAI_API_KEY`` (the account with the credits).
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

FIREWORKS_URL = "https://api.fireworks.ai/inference/v1"
OLLAMA_URL = "http://localhost:11434/v1"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str  # anthropic | openai | fireworks | ollama
    model: str
    #: Anthropic effort, or OpenAI-style ``reasoning_effort``. None sends nothing.
    effort: Optional[str] = None
    #: $ per 1M tokens (input, output), for the cost columns. None when unconfirmed.
    price: Optional[tuple] = None
    note: str = ""


MODELS: Dict[str, ModelSpec] = {spec.name: spec for spec in (
    # The comparison bar and the test-set interviewer. Never trained on.
    ModelSpec("opus-5", "anthropic", "claude-opus-5", "medium", (5.0, 25.0)),
    ModelSpec("haiku-5.5", "anthropic", "claude-haiku-5-5", "medium", (0.10, 0.50)),
    # GPT: candidate simulator, judge, comparison dot.
    ModelSpec("gpt-5.4", "openai", "gpt-5.4", None, (2.50, 15.0)),
    ModelSpec("gpt-5.4-mini", "openai", "gpt-5.4-mini", None, (0.75, 4.50)),
    ModelSpec("gpt-5.5", "openai", "gpt-5.5", None, None, "price not checked"),
    # Open teacher candidates on Fireworks serverless (checked on the account 2026-11-01).
    ModelSpec("kimi-k3", "fireworks", "accounts/fireworks/models/kimi-k3", None, (3.00, 15.00)),
    ModelSpec("glm-5.3", "fireworks", "accounts/fireworks/models/glm-5p3", None, (1.40, 4.40)),
    ModelSpec("qwen3.8-max", "fireworks", "accounts/fireworks/models/qwen3p8-max", None, (2.00, 6.00)),
    ModelSpec("deepseek-v4.1-flash", "fireworks", "accounts/fireworks/models/deepseek-v4p1-flash",
              None, (0.30, 1.20)),
    ModelSpec("gpt-oss-120b", "fireworks", "accounts/fireworks/models/gpt-oss-120b",
              None, (0.15, 0.60)),
    # The student. Not serverless: base runs on an on-demand deployment, set
    # DISTILL_STUDENT_DEPLOYMENT to its model id once it is up.
    ModelSpec("qwen3.5-9b", "fireworks",
              os.getenv("DISTILL_STUDENT_DEPLOYMENT", "accounts/fireworks/models/qwen3p5-9b"),
              None, None, "on-demand H100 ~$8/h"),
    ModelSpec("qwen3-8b-local", "ollama", "qwen3-8b-32k", None, (0.0, 0.0)),
)}


def completer(name: str):
    """The engine's completer for a model in ``MODELS``."""
    spec = MODELS[name]
    if spec.provider == "anthropic":
        from app.services.interview.ai.anthropic_client import AnthropicSchemaCompleter

        return AnthropicSchemaCompleter(
            model=spec.model, effort=spec.effort or "medium", effort_env=None,
            timeout_seconds=180.0,
        )

    from app.services.interview.ai.openai_schema_client import OpenAISchemaCompleter

    if spec.provider == "openai":
        key, url = os.getenv("DISTILL_OPENAI_API_KEY") or os.environ["OPENAI_API_KEY"], None
    elif spec.provider == "fireworks":
        key, url = os.environ["FIREWORKS_API_KEY"], FIREWORKS_URL
    else:
        key, url = "ollama", OLLAMA_URL
    return OpenAISchemaCompleter(
        api_key=key, model=spec.model, base_url=url,
        reasoning_effort=spec.effort, timeout_seconds=300.0,
    )


def cost(name: str, tokens_in: int, tokens_out: int) -> Optional[float]:
    spec = MODELS[name]
    if spec.price is None:
        return None
    return (tokens_in * spec.price[0] + tokens_out * spec.price[1]) / 1_000_000
