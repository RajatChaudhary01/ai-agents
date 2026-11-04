"""Web UI entry point: ``uv run python app.py`` (add ``--host`` for hosting mode)."""

from ai_agents._console import use_utf8_output

use_utf8_output()

from dotenv import load_dotenv

load_dotenv()

from ai_agents.web import main

if __name__ == "__main__":
    main()
