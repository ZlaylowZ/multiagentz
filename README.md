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

# 3. Change directory location of unzipped repo directory

cd /Users/<username>/path/to/repo

# 4. Install Python (preferably via homebrew) [ask agent for assistance if unsure)
# Create a python venv and cd to its parent directory

source .venv/bin/activate

# 5. Install
pip install -e ".[all]"

# 6. Add your API key(s) to multiagentz/providers.py
# (just paste your key — no .env files needed)

# 7. Run
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
