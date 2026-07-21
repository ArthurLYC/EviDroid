from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from evidroid.io_utils import read_json

DEFAULT_DEEPSEEK_CONFIG: dict[str, Any] = {
    "provider": "deepseek",
    "api_key": "",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-v4-flash",
    "max_evidence_per_view": 80,
    "max_tokens": 4096,
    "temperature": 0,
    "thinking": {"type": "disabled"},
    "reasoning_effort": None,
    "evidence_budget_mode": "legacy",
    "view_budgets": None,
    "max_value_chars": None,
    "compact_evidence": False,
}

API_KEY_PLACEHOLDERS = {
    "",
    "your api key",
    "your_api_key",
    "your-deepseek-api-key",
    "在这里填入你的 deepseek api key",
    "在这里填入你的 DeepSeek API Key",
}


def load_llm_config(config_path: str | Path | None = None) -> dict[str, Any]:
    config = dict(DEFAULT_DEEPSEEK_CONFIG)
    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"LLM config file not found: {path}")
        raw = read_json(path)
        config.update(raw.get("llm", raw))

    api_key = str(config.get("api_key") or "").strip()
    if _is_placeholder_api_key(api_key):
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    config["api_key"] = api_key
    return config


def _is_placeholder_api_key(value: str) -> bool:
    return value.strip().lower() in {item.lower() for item in API_KEY_PLACEHOLDERS}
