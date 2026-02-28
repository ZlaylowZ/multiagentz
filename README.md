# multiagentz (maz)

Multi-agent orchestration framework. Define domain-expert AI agent teams in YAML and run them from your terminal.

## What It Does

You describe your agent hierarchy in a YAML file — which code repositories each agent knows about, what models they use, how they're grouped — and `multiagentz` builds the full system: intelligent routing, parallel queries, response synthesis, caching, and a terminal interface.

**One YAML file is the entire interface.** No Python coding required.

## Quick Start

```bash

# 1. Clone the repo via

# brew install gh (and install homebrew if necesary)
gh repo clone ZlaylowZ/multiagentz

# 2. Change directory location of unzipped repo directory

cd /Users/<username>/path/to/repo

# 3. Install Python (preferably via homebrew) [ask agent for assistance if unsure)
# Create a python venv and cd to its location

source .venv/bin/activate

# 4. Install
pip install -e ".[all]"

# 5. Add your API key(s) to multiagentz/providers.py
# (just paste your key — no .env files needed)

# 6. Run
# First, use preferred LLM to construct YAML stack, then save the .yaml to multiagentz/stacks/example.yaml and then run

maz --config stacks/example.yaml
```
> **New to this?** See [SETUP.md](SETUP.md) for a complete beginner walkthrough with Claude Code.
> **Want the full picture?** See [GUIDE.md](GUIDE.md) for the user guide covering all complexity levels and the maz → Claude Code workflow.

## Features

- **Declarative YAML** — Define agent teams without writing code
- **Multi-provider support** — Anthropic, OpenAI, xAI (Grok), Google (Gemini)
- **Per-agent models** — Use different AI models for different agents
- **Cost optimization** — Expensive models for complex routing, cheap models for simple lookups
- **Orchestration modes** — Standard, consensus, and perspective-based analysis
- **Cross-pollination** — A/B twin agents using different models for cognitive diversity
- **Auto provider detection** — Model name determines provider automatically

## How It Works

### 1. Define a Stack (YAML)

```yaml
name: my-project

lead:
  keywords:
    backend: [api, endpoint, database, server]
    frontend: [react, component, hook, ui]

  agents:
    backend:
      repo_path: /path/to/backend
      description: Backend API and database layer
      model: claude-sonnet-4-20250514
      key_files: [src/, README.md]

    frontend:
      repo_path: /path/to/frontend
      description: React frontend application
      model: gemini-2.0-flash    # Cheaper model for frontend
      key_files: [src/, package.json]
```

### 2. Run It

```bash
maz --config stacks/my-stack.yaml
```

### 3. Ask Questions

The lead agent automatically routes to the right sub-agent(s), queries them in parallel, and synthesizes responses.

## Architecture

```
LeadAgent (classify -> route -> query -> synthesize -> cache)
+-- SubAgent "backend"     (loads /path/to/backend)
+-- SubAgent "frontend"    (loads /path/to/frontend)
+-- CoordinatorAgent "infra" (nested sub-coordinator)
|   +-- SubAgent "devops"
|   +-- SubAgent "database"
+-- FileHandlerAgent       (watches arbitrary files)
```

### Core Components

| Component | Purpose |
|-----------|---------|
| **SubAgent** | Domain expert backed by a local file tree. Loads key files into LLM context. |
| **CoordinatorAgent** | Routes to child agents, queries in parallel, synthesizes. Nestable. |
| **LeadAgent** | Top-level orchestrator: keyword hints + LLM routing + parallel query + synthesis + cache. |
| **FileHandlerAgent** | Watches arbitrary files/directories. Persistent watch list. |
| **LLMClient** | Unified multi-provider client (Anthropic, OpenAI, xAI, Google). |

## Supported LLM Providers

| Provider | Models | Setup |
|----------|--------|-------|
| **Anthropic** | Claude Opus 4, Sonnet 4, Haiku 3.5 | `console.anthropic.com` |
| **OpenAI** | GPT-4o, o4-mini, o3 | `platform.openai.com` |
| **xAI** | Grok-4, Grok-4.1 | `console.x.ai` |
| **Google** | Gemini 2.0 Flash, Gemini 2.5 Pro | `aistudio.google.com` |

Provider is automatically detected from model name (`claude-*` -> Anthropic, `grok-*` -> xAI, etc.).

## Configuration

### API Keys (providers.py)

Edit `multiagentz/providers.py` and paste your API keys:

```python
PROVIDERS = {
    "anthropic": {
        "api_key": "sk-ant-...",  # Paste your key here
        "default_model": "claude-sonnet-4-20250514",
    },
    "openai": {
        "api_key": "sk-...",
        "default_model": "o4-mini",
    },
    # ... xai, google
}
```

You only need **one** provider to get started. For cross-pollination (twin agents), you need at least **two**.

### YAML Stack Reference

```yaml
name: stack-name

orchestration:
  mode: standard          # standard | consensus | perspective
  max_iterations: 3       # Refinement cycles
  cross_pollination: false # A/B twin agent loops

lead:
  model: claude-sonnet-4-20250514   # Model for routing
  routing_prompt_extra: |
    Domain-specific routing instructions...
  keywords:
    agent_name: [keyword1, keyword2]

  agents:
    my_agent:
      repo_path: /absolute/path
      description: What this agent knows
      model: claude-sonnet-4-20250514  # Per-agent model
      system_prompt: |
        You are an expert on ...
      key_files:
        - src/
        - README.md

    my_group:
      type: coordinator
      description: Groups related agents
      agents:
        child1:
          repo_path: /path
          key_files: [src/]
        child2:
          repo_path: /path
          key_files: [lib/]
```

## Orchestration Modes

### Standard (Default)
Basic routing and synthesis.

### Consensus
Detects conflicts between agents, iteratively refines until agreement.

### Perspective
Multi-perspective analysis: independent solutions, LEAD review, iterative refinement, consensus synthesis.

### Cross-Pollination
A/B twin agents (different models) exchange outputs and refine. Produces cognitively diverse results.

## Performance Tuning

### Routing Optimizations

maz uses two fast-path optimizations to eliminate unnecessary LLM routing calls:

**Keyword fast-routing (LeadAgent)** — When `keywords` in your YAML match the query and resolve to a single agent, the lead skips the LLM routing call entirely and routes directly. This saves one full LLM round-trip per query. To take advantage of this, define thorough keyword lists for your agents:

```yaml
lead:
  keywords:
    backend: [api, endpoint, database, server, query, migration, schema]
    frontend: [react, component, hook, ui, css, layout, render]
```

**Twin fast-routing (CoordinatorAgent)** — When all sub-agents in a coordinator are configured as cross-pollination twins, the coordinator skips its routing LLM call since cross-pollination always runs both agents regardless of the routing decision. No configuration needed — this activates automatically when `twin:` is set on all agents in a coordinator group.

### Bootstrap Q&A Parallelization

During the Bootstrap Q&A phase of perspective mode, the LEAD agent answers perspective questions **in parallel** rather than sequentially. Each question can trigger a full routing → sub-agent → synthesis chain, so parallelizing them is the single largest performance improvement for perspective mode stacks.

No configuration needed — this is automatic.

### Tunable Parameters

These parameters control the speed/depth/cost tradeoff. All are set in your YAML stack file.

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| `max_iterations` | `orchestration:` | `3` | Number of LEAD review + refinement cycles in perspective mode. Each iteration adds a full round of LLM calls per perspective. |
| `key_files` | per agent | — | Files loaded into each agent's system prompt. More files = richer context but higher token usage per call. |
| `max_context_chars` | per agent | `600,000` | Character cap for file loading (~170K tokens). Reduce to lower per-agent token usage. |
| `max_tokens` | per agent | `32,768` | Max generation tokens per LLM call. Lower values produce shorter responses but reduce cost and latency. |
| `model` | per agent | global default | Model used for each agent. Cheaper/faster models for simple agents dramatically reduce cost. |
| Bootstrap Q&A turns | hardcoded | `3` | Max Q&A turns per perspective before proceeding. |
| Questions per turn | hardcoded | `5` | Max questions extracted per Q&A turn. |

#### Recommended Configurations

**Fast/cheap (quick analysis, 5-15 min):**
```yaml
orchestration:
  mode: perspective
  max_iterations: 1          # Single review pass, no refinement loop
  cross_pollination: false    # Skip the twin swap-refine cycle

lead:
  model: claude-sonnet-4-20250514   # Sonnet for routing (cheaper than Opus)
  agents:
    my_agent:
      model: claude-sonnet-4-20250514
      key_files: [src/main.py, README.md]   # Minimal files
```

**Balanced (thorough analysis, 20-40 min):**
```yaml
orchestration:
  mode: perspective
  max_iterations: 2          # One review + one refinement
  cross_pollination: true

lead:
  model: claude-sonnet-4-20250514
  agents:
    planner:
      type: coordinator
      agents:
        A:
          model: claude-opus-4-6     # Opus for depth
          key_files: [src/, docs/]
          twin: B
        B:
          model: claude-opus-4-6
          key_files: [src/, docs/]
          twin: A
```

**Maximum depth (exhaustive reconciliation, 45-90 min):**
```yaml
orchestration:
  mode: perspective
  max_iterations: 3          # Full three-pass refinement
  cross_pollination: true

lead:
  model: claude-opus-4-6     # Opus everywhere
  agents:
    planner:
      type: coordinator
      agents:
        A:
          model: claude-opus-4-6
          key_files: [src/, lib/, docs/, tests/]  # Full repo
          twin: B
        B:
          model: claude-opus-4-6
          key_files: [src/, lib/, docs/, tests/]
          twin: A
```

### Token Limits and Prompt Budgeting

LLM APIs enforce hard input token limits (e.g. Anthropic: 200K tokens). In complex orchestration modes, prompts can grow very large as Q&A context, solution text, and file contents accumulate across phases.

**multiagentz handles this automatically at two layers:**

1. **Orchestration-level guards** — Smart truncation at each accumulation point: Q&A bootstrap history (300K chars), solution generation context (300K chars), LEAD review payloads (550K chars spread across perspectives), refinement prompts (300K chars), synthesis inputs (550K chars spread), and cross-pollination swap rounds (300K chars per side). Each section is budgeted independently so no single section starves the others.

2. **`LLMClient` safety net** — Every API call runs a pre-flight token estimate. If the prompt + system prompt would exceed 195K tokens, the prompt is automatically truncated with a warning log. This is the last-resort catch-all that prevents `400 prompt is too long` errors.

These character budgets are defined as constants in `orchestration.py` and `llm_client.py`:

```python
# orchestration.py
_MAX_SECTION_CHARS = 300_000   # ~85K tokens — max for one text section
_MAX_COMBINED_CHARS = 550_000  # ~157K tokens — max for multi-section prompts

# llm_client.py
MAX_INPUT_TOKENS = 195_000     # Pre-flight token limit (below 200K API max)
```

To adjust these limits, edit the constants directly. Raising them increases context fidelity but risks hitting the API limit; lowering them reduces cost and increases headroom.

### Timeout Resilience

If an API call times out during refinement, reconciliation, or synthesis, the framework gracefully degrades instead of crashing:

- **Refinement timeout** → carries forward the previous solution
- **Reconciliation timeout** → falls back to the longer agent output
- **Synthesis timeout** → concatenates all perspective outputs with section headers

The underlying HTTP timeout is configured in `llm_client.py` (default: 600s with 3 retries). These resilience behaviors are automatic and require no configuration.

## REPL Commands

| Command | Description |
|---------|-------------|
| `/help` | List commands |
| `/clear` | Clear conversation memory |
| `/cache` | Cache stats |
| `/file <path>` | Load file as question |
| `/paste` | Multi-line input mode |
| `/watch <path>` | Watch a file/directory |
| `/scan <path>` | Watch + auto-summarize |
| `/brief` | Toggle brief responses |
| `/export [html\|md\|txt]` | Export last response |
| `/perspective "<q>"` | Multi-perspective analysis |
| `/consensus <q>` | Force consensus mode |
| `/promote <agent>` | Promote to LEAD_SUB |
| `/status` | Show orchestration status |

## Example Stacks

| File | Description |
|------|-------------|
| `stacks/example.yaml` | Simple two-agent stack |
| `stacks/advanced_example.yaml` | Multi-provider with cross-pollination |
| `stacks/template.yaml` | Full reference with all options |
| `stacks/minimal.yaml` | Bare minimum starter |

## Features
**Built for Claude Code** — Run deep multi-agent analysis, get an auto-exported HTML artifact, and hand it directly to Claude Code for implementation. The hard thinking happens here; the building happens​​​​​​​​​​​​​​​​


## License

MIT
