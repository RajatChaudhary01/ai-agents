"""Gradio web UI entry point: ``uv run python app.py`` (add ``--host`` for hosting mode)."""

import sys

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

from ai_agents.web import main

if __name__ == "__main__":
    main()
