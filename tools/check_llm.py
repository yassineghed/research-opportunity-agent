import os
import sys
from pathlib import Path

from dotenv import load_dotenv

def safe_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

safe_print("--- Diagnostic script: check_llm.py ---")
cwd = Path.cwd()
# Determine project root robustly (handle running from notebooks/)
project_root = cwd
if not (project_root / "src").exists():
    project_root = project_root.parent
safe_print("project_root:", project_root)
env_path = project_root / ".env"
safe_print(".env exists:", env_path.exists())

load_dotenv(env_path)

safe_print("LLM_PROVIDER:", os.getenv("LLM_PROVIDER"))
safe_print("LLM_MODEL:", os.getenv("LLM_MODEL"))
safe_print("GEMINI_API_KEY present:", bool(os.getenv("GEMINI_API_KEY")))
safe_print("GROK_API_KEY present:", bool(os.getenv("GROK_API_KEY") or os.getenv("GORK_API_KEY")))

safe_print("sys.path (head):", sys.path[:5])

try:
    import google.genai
    safe_print("google.genai import: OK")
except Exception as e:
    safe_print("google.genai import error:", type(e).__name__, e)

# Try to instantiate LLMClient if possible
try:
    # ensure project root is importable like notebook does
    sys.path.append(str(project_root))
    from src.llm.client import LLMClient
    try:
        llm = LLMClient()
        safe_print("LLMClient instantiated successfully")
        try:
            resp = llm.generate("Test prompt: semantic similarity in two sentences.")
            safe_print("LLM generate response type:", type(resp))
            # print short preview
            safe_print("Response preview:", str(resp)[:500])
        except Exception as e:
            safe_print("LLMClient.generate error:", type(e).__name__, e)
    except Exception as e:
        safe_print("LLMClient init error:", type(e).__name__, e)
except Exception as e:
    safe_print("Import LLMClient error:", type(e).__name__, e)

safe_print("--- End diagnostic ---")
