"""Verify the generic LLM layer against a live provider, independently.

Runs one short completion through the LLMClient for the requested provider(s)
and reports exactly which provider/model was used, whether the call succeeded,
and (on failure) the classified error with any retry hint.

Usage:
    python tools/test_llm_providers.py                       # Gemini + Grok
    python tools/test_llm_providers.py --provider gemini
    python tools/test_llm_providers.py --provider grok --model grok-4.6
    python tools/test_llm_providers.py --prompt "Say hello in French"
    python tools/test_llm_providers.py --max-attempts 1      # no retries

Exit codes:
    0  all requested providers succeeded
    1  at least one provider call failed
    2  configuration error (e.g. missing key)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

SUPPORTED = ("gemini", "qwen")


def _error_summary(exc: Exception) -> str:
    status = getattr(exc, "status_code", None)
    message = getattr(exc, "message", None) or str(exc)
    hint = getattr(exc, "hint", None)
    retry_after = getattr(exc, "retry_after_seconds", None)

    lines = [
        f"  error type : {type(exc).__module__}.{type(exc).__name__}",
        f"  status     : {status}" if status is not None else f"  status     : n/a",
        f"  message    : {message}",
    ]
    if retry_after is not None:
        lines.append(f"  retry after: {retry_after:.1f}s")
    if hint:
        lines.append(f"  hint       : {hint}")
    return "\n".join(lines)


def run_provider(provider: str, prompt: str, model_name: str | None, max_attempts: int) -> int:
    from src.llm.client import LLMClient

    print(f"=== Provider: {provider} ===")

    try:
        client = LLMClient(provider=provider, model_name=model_name)
    except ValueError as exc:
        print(_error_summary(exc))
        print(f"RESULT: configuration error ({provider})")
        return 2

    api_key_source = _key_source(provider)
    print(f"  model      : {client.model_name}")
    print(f"  key stored : {api_key_source}")
    print(f"  max for retry attempts : {max_attempts or 'provider default'}")
    print(f"  prompt     : {prompt[:80]!r}")

    if max_attempts:
        client.llm.max_attempts = max(1, int(max_attempts))

    try:
        response = client.generate(prompt)
    except Exception as exc:
        print(_error_summary(exc))
        print(f"RESULT: FAILED ({provider})")
        print()
        return 1

    preview = response.strip().replace("\n", " ")
    print(f"  response   : {preview[:200]!r}")
    print(f"RESULT: OK ({provider})")
    print()
    return 0


def _key_source(provider: str) -> str:
    import os

    if provider == "gemini":
        names = ("GEMINI_API_KEY",)
    else:
        names = ("QWEN_API_KEY", "DASHSCOPE_API_KEY")

    for name in names:
        if os.getenv(name):
            return f"{name} is set ({len(os.getenv(name))} chars)"
    return "NOT set"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-p", "--provider", choices=SUPPORTED, help="provider to test")
    parser.add_argument("-m", "--model", default=None, help="override the configured model name")
    parser.add_argument(
        "--prompt",
        default="Reply with the single word: pong",
        help="prompt sent to the provider",
    )
    parser.add_argument("--max-attempts", type=int, default=None, help="override retry attempts")
    args = parser.parse_args()

    providers = [args.provider] if args.provider else list(SUPPORTED)

    print("You are running a live provider test that uses real API capacity.")
    print(f"Providers to test: {', '.join(providers)}")
    print()

    failures = 0
    for provider in providers:
        result = run_provider(provider, args.prompt, args.model, args.max_attempts)
        failures += 1 if result not in (0,) else 0

    if failures:
        print(f"{failures} provider(s) failed.")
        return 1

    print("All requested providers responded successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())