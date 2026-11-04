import sys

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

from smolagents.utils import AgentGenerationError

from ai_agents.agents import create_first_agent
from ai_agents.models import describe_model


def main() -> None:
    prompt = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "What is the current time in Tokyo, and what is the 10th Fibonacci number?"
    )
    try:
        agent = create_first_agent()
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(f"[Config error] {exc}")
    print("=" * 60)
    print(f"[AI Agent] smolagents + {describe_model(agent.model)}")
    print(f"Prompt: {prompt}")
    print("=" * 60 + "\n")

    try:
        result = agent.run(prompt)
    except AgentGenerationError as exc:
        sys.exit(
            f"\n[Model error] {describe_model(agent.model)} could not generate a response.\n"
            f"{exc}\n"
            "Check that the model server is running (e.g. `ollama serve`) or that your API "
            "key is set, or configure FALLBACK_MODEL_ID in .env."
        )
    except KeyboardInterrupt:
        sys.exit("\n[Interrupted]")

    print("\n" + "=" * 60)
    print("[Final Answer]:")
    print(result)
    print("=" * 60)


if __name__ == "__main__":
    main()
