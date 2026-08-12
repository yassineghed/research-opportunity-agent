import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai


def main():
    # load .env from repo root (two levels up from tools/)
    repo_root = Path(__file__).resolve().parent.parent
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provider != "gemini":
        print(f"Model listing is only available for Gemini in this script. Current provider: {provider}")
        return

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set in .env or environment.")
        return

    client = genai.Client(api_key=api_key)
    try:
        models = client.models.list()
        for m in models:
            print(m.name)
    except Exception as e:
        print("Error listing models:", type(e).__name__, e)


if __name__ == '__main__':
    main()
