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
    # SFT LoRA (codepraxis-interviewer-sft-v1), live-merged on the base's B200 shape.
    ModelSpec("qwen3.5-9b-tuned", "fireworks",
              "accounts/ninadnar98-z0upz59iv/models/codepraxis-interviewer-v1"
              "#accounts/ninadnar98-z0upz59iv/deployments/codepraxis-interviewer-v1",
              None, None, "on-demand B200, same shape as the base"),
    # DPO on top of the SFT LoRA (codepraxis-interviewer-dpo-v1), same shape.
    ModelSpec("qwen3.5-9b-dpo", "fireworks",
              "accounts/ninadnar98-z0upz59iv/models/codepraxis-interviewer-dpo-v1"
              "#accounts/ninadnar98-z0upz59iv/deployments/codepraxis-interviewer-dpo-v1",
              None, None, "on-demand B200, same shape as the base"),
    ModelSpec("qwen3-8b-local", "ollama", "qwen3-8b-32k", None, (0.0, 0.0)),
    # The tuned student: Muse Glimmer 30B + LoRA (serverless SFT on data/export/train.jsonl,
    # fireworks-training-runs/codepraxis-muse-v1). On-demand deployment, 1x H200 BF16.
    ModelSpec("muse-glimmer-tuned", "fireworks",
              os.getenv("DISTILL_TUNED_DEPLOYMENT",
                        "accounts/vishakha-d933-0nmp64/models/muse-starter-sl-codepraxis-muse-v1"
                        "#accounts/vishakha-d933-0nmp64/deployments/eqb76xt9"),
              None, None, "on-demand H200 $8/h"),
    # Base Muse Glimmer 30B, the tuned model's starting point (no serverless inference).
    ModelSpec("muse-glimmer-base", "fireworks",
              os.getenv("DISTILL_MUSE_BASE_DEPLOYMENT",
                        "accounts/fireworks/models/muse-glimmer-30b"
                        "#accounts/vishakha-d933-0nmp64/deployments/muse-glimmer-base-eval"),
              None, None, "on-demand H200 $8/h"),
)}


class _Retrying:
    """Retries a completer on rate limits and transient server errors, with
    exponential backoff, so one throttled call does not fail a whole session."""

    def __init__(self, inner, attempts: int = 7):
        self._inner = inner
        self._attempts = attempts
        self.model = getattr(inner, "model", None)

    async def complete(self, **kw):
        import asyncio
        import random as _r

        delay = 2.0
        for n in range(self._attempts):
            try:
                return await self._inner.complete(**kw)
            except Exception as exc:  # the SDKs' rate-limit / 5xx / timeout errors
                text = f"{type(exc).__name__} {exc}"
                transient = any(k in text for k in ("429", "RateLimit", "rate limit", "503", "502",
                                                    "500", "Timeout", "timed out", "Overloaded", "529"))
                if not transient or n == self._attempts - 1:
                    raise
                await asyncio.sleep(delay + _r.random() * delay)
                delay = min(delay * 2, 60.0)


def completer(name: str):
    """The engine's completer for a model in ``MODELS``, retrying on rate limits."""
    return _Retrying(_completer(name))


def _completer(name: str):
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
    from openai import AsyncOpenAI

    # Built the way backend main builds it, then pointed at the right endpoint: works
    # whether or not the backend's completer takes ``base_url`` itself.
    comp = OpenAISchemaCompleter(api_key=key, model=spec.model, timeout_seconds=300.0)
    comp._client = AsyncOpenAI(api_key=key, base_url=url, timeout=300.0)
    return comp


def cost(name: str, tokens_in: int, tokens_out: int) -> Optional[float]:
    spec = MODELS[name]
    if spec.price is None:
        return None
    return (tokens_in * spec.price[0] + tokens_out * spec.price[1]) / 1_000_000
