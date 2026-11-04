# Smolagents Multi-Model AI Agent

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![smolagents](https://img.shields.io/badge/smolagents-1.26%2B-ffcc00.svg)](https://github.com/huggingface/smolagents)
[![uv](https://img.shields.io/badge/uv-package%20manager-purple.svg)](https://docs.astral.sh/uv/)
[![Gradio](https://img.shields.io/badge/Gradio-6.30-orange.svg)](https://gradio.app/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

A modular, production-ready AI Agent built with Hugging Face `smolagents`. Features dual-interface interaction (terminal CLI and Gradio Web UI), pluggable model inference (local Ollama or high-speed cloud Groq), AST-sandboxed math evaluation, and real-time web search.

Developed as an enhanced, fully autonomous implementation of Unit 1 of the Hugging Face AI Agents Course.

Originally forked from [lowkey999-netizen/ai-agents](https://github.com/lowkey999-netizen/ai-agents); this repository is now developed independently.

---

## System Architecture

```text
+-------------------------------------------------------------------------------+
|                       User Input (CLI / Gradio Web UI)                        |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                     ReAct Engine (smolagents CodeAgent)                       |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                           LiteLLM Inference Router                            |
+---------------------------------------+---------------------------------------+
                                        |
                  +---------------------+---------------------+
                  |                                           |
                  v                                           v
+-----------------------------------+       +-----------------------------------+
|      Local Ollama (Windows 11)    |       |          Cloud Groq LPUs          |
|    RTX 4050 6GB + 24GB System RAM |       |       High-Throughput Tier        |
|  * qwen2.5-coder:7b               |       |  * qwen/qwen3.8-27b               |
|  * magnum-12b                     |       +-----------------------------------+
|  * qwen-coder-14b                 |                         |
+-----------------+-----------------+                         |
                  |                                           |
                  +---------------------+---------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                 AST Execution Sandbox (Restricted Python Env)                 |
+---------------------------------------+---------------------------------------+
                                        |
          +-----------------------------+-----------------------------+
          |                             |                             |
          v                             v                             v
+-------------------+         +-------------------+         +-------------------+
|  Calculator Tool  |         | World Clock Tool  |         |  Web Search Tool  |
| (safe AST math)   |         |  (pytz timezone)  |         | (DuckDuckGo API)  |
+---------+---------+         +---------+---------+         +---------+---------+
          |                             |                             |
          +-----------------------------+-----------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                           Synthesized Final Answer                            |
+-------------------------------------------------------------------------------+
```

---

## Key Features

- **Universal Model Portability**: Fully decoupled from hardcoded endpoints. Switch between offline local execution (`Ollama` + `qwen2.5-coder` or `qwen-coder-14b`) and ultra-fast cloud LPUs (`Groq` + `qwen3.8-27b`) by updating a single variable in `.env`.
- **AST-Sandboxed Math Calculator**: Built-in `calculate` tool that parses with `ast.parse`, whitelists nodes and identifiers, and evaluates the tree itself (no `eval`). Integer results are capped at 4000 digits and `factorial`/`comb`/`perm` arguments at 10,000, so inputs like `9**9**9` are rejected instantly instead of freezing the agent.
- **Automatic Model Fallback**: Set `FALLBACK_MODEL_ID` and requests that fail on the primary model (e.g. Ollama host offline) are retried on the fallback through a LiteLLM Router.
- **World Time Awareness**: Custom `get_current_time_in_timezone` tool supporting any valid IANA timezone identifier via `pytz`.
- **Resilient Live Search**: `web_search` tool built on `ddgs` with input validation, timeouts, retries with exponential backoff, multi-backend fallback (DuckDuckGo → auto → Wikipedia) and a result cache, so network failures degrade gracefully instead of crashing the agent.
- **Tuned Prompt Engineering**: Streamlined `src/ai_agents/prompts.yaml` (88 lines) stripped of hallucinated ghost tools, fully compliant with modern `smolagents` (v1.26+).
- **Dual Interfaces**: Interactive terminal CLI (`uv run ai-agents`) and browser chat interface (`uv run python app.py`).
- **Reproducible Environment**: Built with `uv` for fast dependency resolution and strictly locked virtual environments.

---

## Performance Benchmarks & Hardware

### Test Hardware Environment

- **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)
- **RAM**: 24 GB System RAM
- **OS**: Windows 11

### Execution Benchmarks

| Backend    | Model              | Resource Allocation          | Speed                                            | Notes                                          |
| :--------- | :----------------- | :--------------------------- | :----------------------------------------------- | :--------------------------------------------- |
| **Ollama** | `qwen2.5-coder:7b` | 100% in 6 GB VRAM            | ~15-25s / step                                   | Fits fully into GPU memory; 100% offline       |
| **Ollama** | `magnum-12b`       | Split: 6 GB VRAM + 24 GB RAM | ~23.69s (Single-step completion)                 | Balanced 12B reasoning offloaded to System RAM |
| **Ollama** | `qwen-coder-14b`   | Split: 6 GB VRAM + 24 GB RAM | 37.55s (Step 1), 11.63s (Step 2) [Total: 49.18s] | Detailed multi-step code planning              |
| **Groq**   | `qwen/qwen3.8-27b` | Cloud LPUs                   | 0.73s (Step 1), 0.35s (Step 2) [Total: 1.08s]    | Blistering cloud inference on free tier        |

---

## Directory Structure

```text
ai-agents/
├── .env.example               # Configuration template for model and API keys
├── .github/workflows/ci.yml   # PR checks: ruff lint, ruff format, pytest
├── app.py                     # Gradio Web UI entry point
├── docs/
│   └── images/                # Execution screenshots
├── pyproject.toml             # Project metadata and dependencies
├── requirements.txt           # Generated dependencies for container deployment
├── uv.lock                    # Fully pinned lockfile
└── src/
    └── ai_agents/
        ├── __init__.py        # CLI entry point (uv run ai-agents)
        ├── models.py          # Model factory with optional fallback (LiteLLM Router)
        ├── prompts.yaml       # ReAct system prompt and few-shot examples
        ├── agents/
        │   ├── __init__.py    # Agent package exports
        │   └── first_agent.py # Agent factory wiring Model, Tools, and Prompts
        └── tools/
            ├── __init__.py    # Tool package exports
            ├── math_tools.py  # AST-guarded calculation tool (calculate)
            ├── search_tools.py # Resilient web search with fallbacks (web_search)
            └── time_tools.py  # Timezone fetching tool (get_current_time_in_timezone)
└── tests/                     # pytest suite (no network or LLM required)
```

---

## Installation and Setup

### 1. Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- Optional: [Ollama](https://ollama.com/) for local offline inference

### 2. Clone and Install

```bash
git clone https://github.com/RajatChaudhary01/ai-agents.git
cd ai-agents
uv sync
```

### 3. Environment Configuration

Copy the configuration template:

```bash
cp .env.example .env
```

Configure your `.env` file according to your preferred provider:

#### Local Ollama (Default)

```env
MODEL_ID=ollama_chat/qwen2.5-coder:7b
MODEL_API_BASE=http://127.0.0.1:11434
MODEL_NUM_CTX=8192
```

#### Groq Cloud (Ultra-Fast)

```env
MODEL_ID=groq/qwen/qwen3.8-27b
GROQ_API_KEY=gsk_your_groq_api_key_here
```

#### Local First, Cloud Fallback

Use Ollama normally and fall back to Groq automatically when the Ollama host is unreachable or errors:

```env
MODEL_ID=ollama_chat/qwen2.5-coder:7b
MODEL_API_BASE=http://127.0.0.1:11434
FALLBACK_MODEL_ID=groq/llama-3.3-70b-versatile
GROQ_API_KEY=gsk_your_groq_api_key_here
```

The fallback accepts `FALLBACK_MODEL_API_BASE`, `FALLBACK_MODEL_API_KEY` and `FALLBACK_MODEL_NUM_CTX`, mirroring the `MODEL_*` variables. The CLI banner shows the active chain, e.g. `ollama_chat/qwen2.5-coder:7b -> groq/llama-3.3-70b-versatile`.

#### Custom Prompts

The agent loads `prompts.yaml` from inside the package, so it works from any directory. To use your own templates, set `PROMPTS_PATH=/path/to/prompts.yaml`.

---

## Usage

### Terminal CLI

Run queries directly from the command line:

```bash
uv run ai-agents "What is 45 * 12 + 10, and what time is it in London?"
```

#### Execution Output

**Step 1: Planning and Tool Execution**

![Terminal Run Step 1](docs/images/cli_run_step1.png)

**Step 2: Synthesis and Final Answer**

![Terminal Run Step 2](docs/images/cli_run_step2.png)

---

### Gradio Web Interface

Launch the browser chat application:

```bash
uv run python app.py
```

Navigate to `http://127.0.0.1:7860` to access the chat UI.

---

## Built-in Tools

| Tool            | Signature                                            | Security / Behavior                                                                                                              |
| :-------------- | :--------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------- |
| **Calculator**  | `calculate(expression: str) -> str`                  | Parses input via `ast.parse`, validates node types and identifiers, and evaluates the tree with its own interpreter (no `eval`). Rejects oversized integer work (over 4000 digits) before computing it. |
| **World Clock** | `get_current_time_in_timezone(timezone: str) -> str` | Validates timezone string with `pytz` and returns formatted local timestamp.                                                     |
| **Web Search**  | `web_search(query: str) -> str`                      | Validates/normalises the query, then tries each backend with a timeout and backoff retries on transient errors (timeouts, rate limits). Falls back to cached results, and finally returns a "Search unavailable" message rather than raising. |

### Web Search Configuration

Optional `.env` settings:

| Variable             | Default                      | Meaning                                         |
| :------------------- | :--------------------------- | :---------------------------------------------- |
| `SEARCH_MAX_RESULTS` | `5`                          | Results per query (clamped to 1-25)             |
| `SEARCH_TIMEOUT`     | `10`                         | Per-request timeout in seconds                  |
| `SEARCH_BACKENDS`    | `duckduckgo,auto,wikipedia`  | `ddgs` backends tried in order until one works  |

---

## Development

Install the project with dev tools (pytest, pytest-cov, ruff):

```bash
uv sync --group dev
```

| Task              | Command                                     |
| :---------------- | :------------------------------------------ |
| Run tests         | `uv run pytest`                             |
| Tests + coverage  | `uv run pytest --cov=ai_agents`             |
| Lint              | `uv run ruff check .`                       |
| Auto-fix lint     | `uv run ruff check . --fix`                 |
| Format            | `uv run ruff format .`                      |

The test suite fakes the search client and never calls the network or an LLM.

### Continuous Integration

`.github/workflows/ci.yml` runs on every pull request and on pushes to `main`:

1. **Lint & format**: `ruff check` and `ruff format --check`
2. **Tests**: `pytest` with coverage on Python 3.12 and 3.13

Both jobs install from `uv.lock` with `--locked`, so CI fails if the lockfile is out of date. Run `uv lock` after changing dependencies.

---

## Security Model

Arbitrary code execution is strictly restricted. The agent runs within `smolagents`'s AST interpreter, ensuring:

- Only whitelisted imports and functions can be invoked.
- File system and shell access are blocked by default.
- Custom tools perform strict type and input validation before execution.

---

## License

This project is licensed under the Apache 2.0 License.
