import sys

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

from ai_agents.agents import create_first_agent
from ai_agents.models import describe_model


def main() -> None:
    prompt = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "What is the current time in Tokyo, and what is the 10th Fibonacci number?"
    )
    agent = create_first_agent()
    print("=" * 60)
    print(f"[AI Agent] smolagents + {describe_model(agent.model)}")
    print(f"Prompt: {prompt}")
    print("=" * 60 + "\n")

    result = agent.run(prompt)

    print("\n" + "=" * 60)
    print("[Final Answer]:")
    print(result)
    print("=" * 60)


if __name__ == "__main__":
    main()
