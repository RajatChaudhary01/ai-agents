import sys

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

from smolagents import GradioUI

from ai_agents.agents import create_first_agent

# 1. Assemble our agent
agent = create_first_agent()

# 2. Launch the Gradio Web Chat Interface
if __name__ == "__main__":
    print("🚀 Launching Agent Web UI on http://127.0.0.1:7860 ...")
    GradioUI(agent).launch(server_name="127.0.0.1", server_port=7860)
