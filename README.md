# AI Agents

[![CI](https://github.com/RajatChaudhary01/ai-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/RajatChaudhary01/ai-agents/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776ab.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-2ea44f.svg)](LICENSE)

A code-writing AI agent built on Hugging Face [smolagents](https://github.com/huggingface/smolagents).
Ask it a question in the terminal or the browser; it plans, writes small Python snippets that
call its tools (a safe calculator, a world clock and resilient web search), and answers.

It runs on a local model through Ollama or on any cloud model LiteLLM supports, can fall back
through a chain of free-tier providers when one is down, and has a hosting mode with accounts,
abuse limits and a CAPTCHA for putting it on the internet.

---

## How it works

```text
 you ──► CLI (ai-agents) or web UI (app.py)
              │
              ▼
        CodeAgent (smolagents) ── writes Python that calls tools ──┐
              │                                                    │
              ▼                                                    ▼
        LiteLLM model chain                         calculate · world clock · web_search
   Ollama ─► Groq ─► Gemini ─► ...                 (run by smolagents' restricted executor)
   (next one is tried if a model fails)
```

1. The agent sends your task plus the tool descriptions to the model.
2. The model replies with a *Thought* and a short code block.
3. smolagents runs the code; tool results come back as an *Observation*.
4. This repeats (up to `AGENT_MAX_STEPS`) until the code calls `final_answer(...)`.

---

## Features

- **Any model**: local Ollama models or any LiteLLM provider (Groq, Gemini, OpenAI,
  Anthropic, OpenRouter, ...), chosen in `.env`.
- **Fallback chain**: up to 9 backup models, tried in order when the current one fails.
- **Safe calculator**: evaluates maths without `eval` and refuses work that would hang
  the process (e.g. `9**9**9`).
- **World clock**: current time for any IANA timezone.
- **Resilient web search**: timeouts, retries with backoff, several backends and a cache;
  a network failure never crashes the agent.
- **Two interfaces**: terminal CLI and a Gradio web UI with an isolated agent per browser tab
  and a **Tools** tab for using the calculator and clock directly.
- **Hosting mode**: sign-up/login, rate limits per account *and* per IP, Turnstile CAPTCHA,
  security headers, and a Docker + Caddy deployment with automatic HTTPS.

---

## Project Layout

```text
ai-agents/
├── app.py                     # Web UI entry point (wraps ai_agents.web)
├── Dockerfile                 # Hosting-mode image (non-root)
├── deploy/
│   ├── docker-compose.yml     # App + Caddy (automatic HTTPS), hardened containers
│   ├── Caddyfile              # Reverse proxy config
│   └── .env.example           # DOMAIN for Caddy
├── .env.example               # All settings, with comments
├── .github/workflows/ci.yml   # Lint, tests, Docker build, deploy-config validation
├── pyproject.toml / uv.lock   # Dependencies (uv)
├── requirements.txt           # Same dependencies for pip (generated)
├── src/ai_agents/
│   ├── __init__.py            # CLI entry point (uv run ai-agents)
│   ├── models.py              # Model chain (LiteLLM Router)
│   ├── prompts.yaml           # Agent prompt templates
│   ├── web.py                 # Web UI: per-session agents, Tools tab, local/hosting modes
│   ├── agents/first_agent.py  # Builds the CodeAgent
│   ├── hosting/
│   │   ├── accounts.py        # SQLite users and sessions (scrypt password hashes)
│   │   ├── captcha.py         # Cloudflare Turnstile verification
│   │   ├── ratelimit.py       # Sliding-window limits, client IP and IPv6 /64 grouping
│   │   └── server.py          # Sign-up/login pages, limits, security headers, Gradio mount
│   └── tools/
│       ├── math_tools.py      # calculate
│       ├── search_tools.py    # web_search
│       └── time_tools.py      # get_current_time_in_timezone
└── tests/                     # pytest suite (no network or model needed)
```

---

## Getting Started

You need Python 3.12+ and [uv](https://docs.astral.sh/uv/). For local models, install
[Ollama](https://ollama.com/) and pull a model, e.g. `ollama pull qwen2.5-coder:7b`.

```bash
git clone https://github.com/RajatChaudhary01/ai-agents.git
```

```bash
cd ai-agents && uv sync
```

```bash
cp .env.example .env
```

### Choosing a Model

Set `MODEL_ID` in `.env` to any [LiteLLM model ID](https://docs.litellm.ai/docs/providers).
The default runs locally through Ollama:

```env
MODEL_ID=ollama_chat/qwen2.5-coder:7b
MODEL_API_BASE=http://127.0.0.1:11434
MODEL_NUM_CTX=8192
```

For a cloud model, set its ID and the provider's key instead, e.g. Groq's free tier:

```env
MODEL_ID=groq/llama-3.3-70b-versatile
GROQ_API_KEY=your_groq_api_key
```

#### Local First, Cloud Fallback Chain

Use Ollama normally and fall through a chain of free-tier cloud APIs when it is offline or a
provider is rate-limited. Models are tried in order: `FALLBACK_MODEL_*`, then
`FALLBACK_MODEL_2_*` up to `FALLBACK_MODEL_9_*` (gaps are allowed).

```env
MODEL_ID=ollama_chat/qwen2.5-coder:7b
MODEL_API_BASE=http://127.0.0.1:11434

FALLBACK_MODEL_ID=groq/llama-3.3-70b-versatile
FALLBACK_MODEL_2_ID=gemini/gemini-2.5-flash
FALLBACK_MODEL_3_ID=openrouter/meta-llama/llama-3.3-70b-instruct:free

GROQ_API_KEY=gsk_your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

Each slot accepts `_API_BASE`, `_API_KEY` and `_NUM_CTX`, mirroring the `MODEL_*` variables;
without `_API_KEY`, LiteLLM uses the provider's own variable. Duplicate models are skipped.
The CLI banner shows the active chain, e.g.
`ollama_chat/qwen2.5-coder:7b -> groq/llama-3.3-70b-versatile -> gemini/gemini-2.5-flash`.

| Provider   | Model prefix   | API key variable     | Free tier                         |
| :--------- | :------------- | :------------------- | :-------------------------------- |
| Groq       | `groq/`        | `GROQ_API_KEY`       | Yes, rate-limited                 |
| Gemini     | `gemini/`      | `GEMINI_API_KEY`     | Yes, via Google AI Studio         |
| Cerebras   | `cerebras/`    | `CEREBRAS_API_KEY`   | Yes, rate-limited                 |
| OpenRouter | `openrouter/`  | `OPENROUTER_API_KEY` | Models ending in `:free`          |
| Mistral    | `mistral/`     | `MISTRAL_API_KEY`    | Yes, on the free "Experiment" plan |

Model names and free-tier limits change often; check each provider's model list.

#### Agent Behaviour

| Variable          | Default | Meaning                                       |
| :---------------- | :------ | :-------------------------------------------- |
| `AGENT_MAX_STEPS` | `6`     | Maximum reasoning/tool steps per task         |
| `AGENT_VERBOSITY` | `2`     | Console log level: 0 = off, 1 = info, 2 = debug |

Invalid values stop the CLI with a `[Config error]` message naming the variable. If the model
cannot be reached, the CLI prints a short `[Model error]` hint instead of a traceback.

#### Custom Prompts

The agent loads `prompts.yaml` from inside the package, so it works from any directory. To use your own templates, set `PROMPTS_PATH=/path/to/prompts.yaml`.

---

## Usage

### Terminal

```bash
uv run ai-agents "How many seconds are in a leap year, and what time is it in Delhi?"
```

The agent prints each step (its thought, the code it ran and the tool output) and then the
final answer. Without a question it runs a built-in example.

### Web Interface

```bash
uv run python app.py
```

`uv run ai-agents-web` does the same. Open `http://127.0.0.1:7860`. Every browser tab gets
its own agent, so separate chats never share memory; **Clear** starts a fresh conversation.

#### Local mode (default)

For your own use: binds to `127.0.0.1`, no login, one request at a time.

You can also use the calculator and world clock directly in the **Tools** tab, without the
AI model.

#### Hosting mode

For serving other people. Turn it on explicitly with `--host` (or `APP_MODE=host`). For a
public deployment, use the Docker + Caddy setup in [Deploying to the Internet](#deploying-to-the-internet).

```bash
uv run python app.py --host
```

What hosting mode adds:

- **Accounts**: visitors sign up and log in at `/signup` and `/login`. Passwords are hashed
  with scrypt, sessions are random tokens stored only as hashes in SQLite (`APP_DB_PATH`),
  and cookies are `HttpOnly`, `SameSite=Lax` and `Secure` (`__Host-` prefixed).
- **Rate limits per account *and* per IP**: every chat message counts against the account
  and against the client's network, plus a global daily cap that bounds total model spend.
  Dummy accounts made from one IP share that IP's budget, so mass sign-ups gain nothing;
  one account hopping between IPs is still held to its account limit.
- **Sign-up abuse controls**: limited accounts per IP per day, and IPv6 clients are grouped
  by /64 so rotating addresses inside one allocation does not help.
- **CAPTCHA**: optional [Cloudflare Turnstile](https://developers.cloudflare.com/turnstile/)
  on sign-up and login stops scripted account creation and password guessing that rotates
  through many IPs (botnets, proxy services), which per-IP limits alone cannot catch. It is
  free and usually invisible to real people. Verification fails closed, and it runs before
  any username check, so bots cannot even probe which names exist. See
  [Turning on the CAPTCHA](#turning-on-the-captcha).
- **Brute-force protection**: failed logins are limited per IP and per account.
- **Hardening**: security headers, cross-site form posts rejected, every Gradio route
  (including its API) requires a session, uploads capped at 1 MB, model errors hidden from
  visitors (logged on the server), and no Gradio API docs in the footer.

Manage accounts from the command line (inside Docker: `docker compose -f deploy/docker-compose.yml exec app python app.py users list`):

```bash
uv run ai-agents-web users list
```

```bash
uv run ai-agents-web users add alice
```

`users passwd <name>`, `users disable <name>` and `users enable <name>` also exist; changing
a password or disabling an account logs that user out everywhere.

| Variable                              | Default       | Meaning                                                       |
| :------------------------------------ | :------------ | :------------------------------------------------------------ |
| `APP_SIGNUP`                          | `open`        | `closed` disables `/signup`; create accounts with `users add` |
| `TURNSTILE_SITE_KEY`                  | none          | Turnstile site key; set with the secret to enable the CAPTCHA |
| `TURNSTILE_SECRET_KEY`                | none          | Turnstile secret key (server side only)                       |
| `APP_CAPTCHA_ON_LOGIN`                | `true`        | Also require the CAPTCHA on login (sign-up always has it)     |
| `APP_DB_PATH`                         | `data/app.db` | SQLite file for accounts and sessions                         |
| `APP_TRUST_PROXY`                     | `false`       | Read the client IP from `X-Forwarded-For` (only behind a proxy) |
| `APP_COOKIE_SECURE`                   | `true`        | Secure cookies; set `false` only for plain-http local testing |
| `APP_MESSAGES_PER_ACCOUNT_PER_HOUR`   | `20`          | Chat messages per account per hour                            |
| `APP_MESSAGES_PER_ACCOUNT_PER_DAY`    | `100`         | Chat messages per account per day                             |
| `APP_MESSAGES_PER_IP_PER_HOUR`        | `40`          | Chat messages per IP (all accounts together) per hour         |
| `APP_MESSAGES_PER_IP_PER_DAY`         | `200`         | Chat messages per IP per day                                  |
| `APP_MESSAGES_GLOBAL_PER_DAY`         | `2000`        | Chat messages for the whole service per day                   |
| `APP_TOOL_CALLS_PER_MINUTE`           | `60`          | Tools-tab uses per account per minute (2x per IP)             |
| `APP_SIGNUPS_PER_IP_PER_DAY`          | `3`           | New accounts per IP per day                                   |
| `APP_SIGNUP_ATTEMPTS_PER_IP_PER_HOUR` | `10`          | Sign-up form submissions per IP per hour                      |
| `APP_LOGIN_FAILURES_PER_IP`           | `20`          | Failed logins per IP per 15 minutes                           |
| `APP_LOGIN_FAILURES_PER_ACCOUNT`      | `10`          | Failed logins per account per 15 minutes                      |

Web UI settings for both modes:

| Variable                  | Local default | Host default | Meaning                                                     |
| :------------------------ | :------------ | :----------- | :---------------------------------------------------------- |
| `APP_MODE`                | `local`       | `host`       | Mode when `--host` is not passed                            |
| `APP_CONCURRENCY`         | `1`           | `4`          | Requests processed in parallel                              |
| `APP_MAX_SESSIONS`        | `5`           | `50`         | Open conversations kept; the least recently used is dropped |
| `APP_SESSION_TTL_MINUTES` | `60`          | `60`         | Idle conversations are dropped after this long              |
| `APP_MAX_MESSAGE_CHARS`   | `20000`       | `4000`       | Longest message accepted                                    |
| `GRADIO_SERVER_NAME`      | `127.0.0.1`   | `0.0.0.0`    | Interface to bind                                           |
| `GRADIO_SERVER_PORT`      | `7860`        | `7860`       | Port to listen on                                           |
| `GRADIO_SHARE`            | `false`       | not allowed  | `true` also publishes a public `*.gradio.live` link         |

#### Turning on the CAPTCHA

1. In the [Cloudflare dashboard](https://dash.cloudflare.com/) open **Turnstile** and
   **Add widget** (free; your domain does not need to use Cloudflare DNS). Add your domain
   as a hostname, and `localhost` if you want to test locally. Choose the **Managed** mode.
2. Put the two keys in `.env`:
   ```env
   TURNSTILE_SITE_KEY=0x4AAAAAAA...
   TURNSTILE_SECRET_KEY=0x4AAAAAAA...
   ```
3. Restart. The startup banner says `CAPTCHA: Turnstile on sign-up and login`.

To try it locally without an account, use Cloudflare's
[test keys](https://developers.cloudflare.com/turnstile/troubleshooting/testing/):
site key `1x00000000000000000000AA` with secret `1x0000000000000000000000000000000AA`
always passes; secret `2x0000000000000000000000000000000AA` always fails. The server must be
able to reach `challenges.cloudflare.com`; if it cannot, sign-ups and logins are refused
rather than let through unchecked.

Rate-limit counters live in memory and reset when the server restarts. Tip: set
`AGENT_VERBOSITY=0` or `1` to keep the server console readable with several users.

---

## Deploying to the Internet

`deploy/` runs the app in hosting mode behind [Caddy](https://caddyserver.com/), which
obtains and renews HTTPS certificates from Let's Encrypt automatically.

1. Point a domain's DNS `A`/`AAAA` record at your server and open ports **80** and **443**
   (and nothing else except SSH).
2. Configure:
   ```bash
   cp .env.example .env
   ```
   ```bash
   cp deploy/.env.example deploy/.env
   ```
   Set `DOMAIN` in `deploy/.env`; set your model, API keys, any `APP_*` limits and the
   [Turnstile keys](#turning-on-the-captcha) in `.env`.
   To use Ollama on the same machine, set `MODEL_API_BASE=http://host.docker.internal:11434`
   and make Ollama listen beyond localhost (`OLLAMA_HOST=0.0.0.0`, firewalled from the internet).
3. Start it:
   ```bash
   docker compose -f deploy/docker-compose.yml up -d --build
   ```
4. Open `https://your-domain/`, sign up, and check the logs:
   ```bash
   docker compose -f deploy/docker-compose.yml logs -f
   ```

How the pieces fit:

- Only Caddy publishes ports. The app container has none, so it can only be reached through
  Caddy; that is what makes `APP_TRUST_PROXY=true` safe. Caddy *replaces* any client-sent
  `X-Forwarded-For`, so spoofing it cannot dodge the per-IP limits.
- Caddy redirects HTTP to HTTPS, adds HSTS and rejects request bodies over 2 MB.
- The app container runs as an unprivileged user on a read-only filesystem with all Linux
  capabilities dropped, `no-new-privileges`, and memory/process limits.
- Accounts live in the `app-data` volume and certificates in `caddy-data`. Back up
  `app-data`, for example:
  ```bash
  docker compose -f deploy/docker-compose.yml exec app python -c "import sqlite3; sqlite3.connect('/data/app.db').backup(sqlite3.connect('/data/backup.db'))"
  ```

Behind Cloudflare or another CDN, every request appears to come from the CDN, so per-IP
limits would lump all users together: configure Caddy's `trusted_proxies` and read the
CDN's client-IP header first.

---

## Built-in Tools

| Tool          | Call                                    | What it does                                                                   |
| :------------ | :-------------------------------------- | :----------------------------------------------------------------------------- |
| Calculator    | `calculate(expression)`                 | Parses the expression, allows only numbers, operators and `math` functions, evaluates it without `eval`, and refuses integer results over 4000 digits before computing them. |
| World clock   | `get_current_time_in_timezone(timezone)`| Current local time for an IANA timezone such as `Asia/Kolkata`.                |
| Web search    | `web_search(query)`                     | Cleans the query, tries each backend with timeouts and retries, falls back to cached results, and otherwise returns a "Search unavailable" message instead of failing. |

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
3. **Docker image & deploy config**: builds the image, smoke-tests hosting mode in a
   read-only container, and validates the Caddyfile and compose file

Dependencies install from `uv.lock` with `--locked`, so CI fails if the lockfile is out of
date. Run `uv lock` after changing dependencies.

---

## Security Model

- **Calculator**: parses with `ast.parse`, whitelists syntax and names, evaluates the tree
  itself (no `eval`) and caps result size, so it cannot run code or hang the process.
- **Agent code**: the model's code runs in smolagents' `LocalPythonExecutor`, which only
  allows whitelisted imports and has operation, loop and time limits. It is a best-effort
  guard, **not a security sandbox**; a determined user may find an escape. When hosting for
  strangers, the Docker setup limits the damage (unprivileged user, read-only filesystem, no
  capabilities, resource limits), but the container still holds your API keys. For stronger
  isolation, smolagents supports remote executors such as Docker or E2B (`executor_type`).
- **Hosting mode**: see [Hosting mode](#hosting-mode) for accounts, rate limits and headers.

---

## License

Apache License 2.0. See [LICENSE](LICENSE).
