import os
import sys

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

from smolagents import GradioUI

from ai_agents.agents import create_first_agent


def _env_flag(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    host = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    # smolagents' GradioUI.launch() defaults to share=True, which publishes the agent on a
    # public *.gradio.live URL. Keep it local unless explicitly opted in.
    share = _env_flag("GRADIO_SHARE")

    agent = create_first_agent()
    print(f"🚀 Launching Agent Web UI on http://{host}:{port} ...")
    if share:
        print("⚠️  GRADIO_SHARE is enabled: the agent is also reachable via a public link.")
    GradioUI(agent).launch(share=share, server_name=host, server_port=port)


if __name__ == "__main__":
    main()
