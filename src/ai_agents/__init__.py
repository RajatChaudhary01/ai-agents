"""Command-line entry point: ``uv run ai-agents "your question"``."""

import sys

from ai_agents._console import use_utf8_output

use_utf8_output()

from dotenv import load_dotenv

load_dotenv()  # before the agent package reads MODEL_* / AGENT_* settings

from smolagents.utils import AgentGenerationError

from ai_agents.agents import create_first_agent
from ai_agents.models import describe_model

DEFAULT_QUESTION = "How many seconds are there in a leap year, and what time is it in Delhi?"
RULE = "-" * 64


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() or DEFAULT_QUESTION
    try:
        agent = create_first_agent()
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(f"[Config error] {exc}")

    print(RULE)
    print(f"Model : {describe_model(agent.model)}")
    print(f"Task  : {question}")
    print(RULE)

    try:
        answer = agent.run(question)
    except AgentGenerationError as exc:
        sys.exit(
            f"\n[Model error] {describe_model(agent.model)} could not generate a response.\n"
            f"{exc}\n"
            "Check that the model server is running (e.g. `ollama serve`) or that your API "
            "key is set, or configure FALLBACK_MODEL_ID in .env."
        )
    except KeyboardInterrupt:
        sys.exit("\n[Interrupted]")

    print(RULE)
    print(f"Answer: {answer}")
    print(RULE)


if __name__ == "__main__":
    main()
