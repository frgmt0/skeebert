"""Runtime settings, read from the environment (and a ``.env`` file if present).

Everything operational lives here so no other module reads ``os.environ``
directly. Tests build a ``Settings`` by hand (or with ``from_env`` and an
explicit mapping) pointing at a temp directory, never at the live ``data/``.

Pricing
-------
The brain is billed per token. The defaults below are Anthropic's published
first-party rates for Claude Haiku 5.5 as of its launch coverage (October 7,
2026) for prompts of 100K tokens or fewer: input $0.10/MTok, output
$0.50/MTok, cache read $0.01/MTok, 5-minute cache write $0.125/MTok. Prompts
over 100K tokens are billed at a higher card ($0.50 / $2.50); Skeebert keeps
every request far below that (see ``brain_max_input_tokens``), but the long
card is still configured so the spend ledger stays honest if one ever slips
through. VERIFY these on https://www.anthropic.com/pricing (or the Claude
platform pricing docs) before relying on the daily budget, and override them
with the ``SKEEBERT_PRICE_*`` variables if they changed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping

# Default brain model. UNVERIFIED against a live API call: no credentials were
# available when this was written. Override with SKEEBERT_BRAIN_MODEL.
DEFAULT_BRAIN_MODEL = "claude-haiku-5-5"
DEFAULT_SITE_URL = "https://skeebert.frgmt.xyz"

# Above this many prompt tokens Haiku 5.5 switches to its long-context rate card.
LONG_CONTEXT_THRESHOLD_TOKENS = 100_000


@dataclass(frozen=True)
class Prices:
    """USD per million tokens. ``long_*`` apply when a prompt exceeds 100K tokens."""

    input: float = 0.10
    output: float = 0.50
    cache_read: float = 0.01
    cache_write: float = 0.125
    long_input: float = 0.50
    long_output: float = 2.50
    long_cache_read: float = 0.05
    long_cache_write: float = 0.625


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    checkpoint_dir: Path = Path("checkpoints")
    renders_dir: Path = Path("renders")
    discord_token: str | None = None
    anthropic_api_key: str | None = None
    dev_guild_id: int | None = None  # sync slash commands to one guild instantly (dev only)
    brain_model: str = DEFAULT_BRAIN_MODEL
    brain_effort: str = "low"
    brain_max_tokens: int = 1024
    brain_timeout_seconds: float = 20.0
    brain_max_input_tokens: int = 20_000  # hard ceiling on our own estimate; far under 100K
    context_messages: int = 6
    max_message_chars: int = 1000
    daily_budget_usd: float = 6.00
    prices: Prices = field(default_factory=Prices)
    rate_limit_per_hour: int = 30
    site_url: str = DEFAULT_SITE_URL
    device: str = "cpu"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "skeebert.db"

    @property
    def salt_path(self) -> Path:
        return self.data_dir / ".salt"

    @property
    def privacy_url(self) -> str:
        return self.site_url.rstrip("/") + "/privacy"

    def with_overrides(self, **changes) -> "Settings":
        return replace(self, **changes)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None, env_file: str | os.PathLike | None = ".env") -> "Settings":
        """Build settings from ``environ`` (default ``os.environ``) layered over ``env_file``.

        Real environment variables win over the ``.env`` file. Pass
        ``env_file=None`` to ignore any ``.env`` (tests do).
        """
        values: dict[str, str] = {}
        if env_file is not None and Path(env_file).exists():
            from dotenv import dotenv_values

            values.update({k: v for k, v in dotenv_values(env_file).items() if v is not None})
        values.update(os.environ if environ is None else environ)

        def get(name: str, default: str | None = None) -> str | None:
            v = values.get(name)
            return v if v not in (None, "") else default

        def num(name: str, default: float) -> float:
            raw = get(name)
            if raw is None:
                return default
            try:
                return float(raw)
            except ValueError:
                raise ValueError(f"{name} must be a number, got {raw!r}") from None

        p = Prices()
        prices = Prices(
            input=num("SKEEBERT_PRICE_INPUT_PER_MTOK", p.input),
            output=num("SKEEBERT_PRICE_OUTPUT_PER_MTOK", p.output),
            cache_read=num("SKEEBERT_PRICE_CACHE_READ_PER_MTOK", p.cache_read),
            cache_write=num("SKEEBERT_PRICE_CACHE_WRITE_PER_MTOK", p.cache_write),
            long_input=num("SKEEBERT_PRICE_LONG_INPUT_PER_MTOK", p.long_input),
            long_output=num("SKEEBERT_PRICE_LONG_OUTPUT_PER_MTOK", p.long_output),
            long_cache_read=num("SKEEBERT_PRICE_LONG_CACHE_READ_PER_MTOK", p.long_cache_read),
            long_cache_write=num("SKEEBERT_PRICE_LONG_CACHE_WRITE_PER_MTOK", p.long_cache_write),
        )
        guild = get("SKEEBERT_DEV_GUILD_ID")
        effort = get("SKEEBERT_BRAIN_EFFORT", "low")
        if effort not in ("low", "medium", "high", "xhigh", "max"):
            raise ValueError(f"SKEEBERT_BRAIN_EFFORT must be low|medium|high|xhigh|max, got {effort!r}")
        settings = cls(
            data_dir=Path(get("SKEEBERT_DATA_DIR", "data")),
            checkpoint_dir=Path(get("SKEEBERT_CHECKPOINT_DIR", "checkpoints")),
            renders_dir=Path(get("SKEEBERT_RENDERS_DIR", "renders")),
            discord_token=get("SKEEBERT_DISCORD_TOKEN"),
            anthropic_api_key=get("ANTHROPIC_API_KEY"),
            dev_guild_id=int(guild) if guild else None,
            brain_model=get("SKEEBERT_BRAIN_MODEL", DEFAULT_BRAIN_MODEL),
            brain_effort=effort,
            brain_max_tokens=int(num("SKEEBERT_BRAIN_MAX_TOKENS", 1024)),
            brain_timeout_seconds=num("SKEEBERT_BRAIN_TIMEOUT_SECONDS", 20.0),
            daily_budget_usd=num("SKEEBERT_DAILY_BUDGET_USD", 6.00),
            prices=prices,
            rate_limit_per_hour=int(num("SKEEBERT_RATE_LIMIT_PER_HOUR", 30)),
            site_url=get("SKEEBERT_SITE_URL", DEFAULT_SITE_URL),
            device=get("SKEEBERT_DEVICE", "cpu"),
            embed_model=get("SKEEBERT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        )
        if settings.brain_max_input_tokens >= LONG_CONTEXT_THRESHOLD_TOKENS:
            raise ValueError("brain_max_input_tokens must stay far below 100K")
        return settings
