from pathlib import Path
from dotenv import load_dotenv
import os, traceback
import sys

# Load repository .env
repo_root = Path(__file__).resolve().parent.parent
env_path = repo_root / ".env"
load_dotenv(env_path)

# Make repo root importable (so `src` package can be imported)
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

# Ensure model is set for this session
desired = os.getenv("LLM_MODEL") or "models/gemini-flash-latest"
if os.getenv("LLM_MODEL") != desired:
    os.environ["LLM_MODEL"] = desired

print("Using LLM_PROVIDER:", os.getenv("LLM_PROVIDER"))
print("Using LLM_MODEL:", os.getenv("LLM_MODEL"))

try:
    from src.llm.client import LLMClient
    llm = LLMClient()
    resp = llm.generate("Explain semantic similarity in two sentences.")
    print("\nLLM response:\n", resp)
except Exception:
    traceback.print_exc()
