from __future__ import annotations

import os

#: Values that look like keys but are almost certainly placeholders.
_PLACEHOLDER_KEYS = frozenset(
    {
        "",
        "your-api-key",
        "your_api_key",
        "your-gemini-api-key",
        "your_grok_api_key",
        "sk-your-key",
        "api-key",
        "api_key",
        "change-me",
    }
)


def get_api_key(*environment_names: str) -> str | None:
    """Load an API key from the environment (populated from ``.env``).

    Keys are never hardcoded in the source code: they must be provided either
    through a ``.env`` file loaded via ``python-dotenv`` or through the shell
    environment. Escapes values that look like placeholder strings.
    """
    for name in environment_names:
        value = os.getenv(name)
        if value is None:
            continue

        stripped = value.strip()
        if stripped and stripped.lower() not in _PLACEHOLDER_KEYS:
            return stripped

    return None


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default