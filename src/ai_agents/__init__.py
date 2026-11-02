import os
import sys

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

from ai_agents.agents import create_first_agent


def main() -> None:
    prompt = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "What is the current time in Tokyo, and what is the 10th Fibonacci number?"
    )
    model_name = os.getenv("MODEL_ID", "ollama_chat/qwen2.5-coder:7b")
    print("=" * 60)
    print(f"[AI Agent] smolagents + {model_name}")
    print(f"Prompt: {prompt}")
    print("=" * 60 + "\n")

    agent = create_first_agent()
    result = agent.run(prompt)

    print("\n" + "=" * 60)
    print("[Final Answer]:")
    print(result)
    print("=" * 60)


if __name__ == "__main__":
    main()
