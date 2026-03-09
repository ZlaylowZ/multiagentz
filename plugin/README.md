# multiagentz — Claude Code Plugin

Multi-agent orchestration plugin for Claude Code. Run cross-pollination, consensus, and perspective analysis across multiple LLM providers.

## Features

| Skill | Command | Description |
|-------|---------|-------------|
| **Cross-Pollinate** | `/cross-pollinate` | Run a question through multiple LLM providers, have them critique each other |
| **Consensus** | `/consensus` | Query multiple domain experts, detect conflicts, synthesize unified answer |
| **Perspective** | `/perspective` | Deep 4-phase analysis: alignment → independent solutions → refinement → synthesis |
| **Build** | `/build` | Architect-driven task decomposition into a dependency DAG with parallel execution |
| **Configure Stack** | `/configure-stack` | Generate YAML stack configs for the multiagentz orchestration engine |

### Lead Agent

The plugin includes a **lead-agent** subagent that acts as an orchestration router — it analyzes questions and selects the optimal orchestration mode automatically.

## Installation

### From a marketplace

```bash
claude plugin install multiagentz@<marketplace-name>
```

### Local development

```bash
claude --plugin-dir ./plugin
```

## MCP Server (Optional)

For full programmatic access to the multiagentz orchestration engine, deploy the MCP server to Cloud Run:

### 1. Deploy to Cloud Run

```bash
# From the multiagentz repo root
gcloud run deploy multiagentz-mcp \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "ANTHROPIC_API_KEY=sk-...,OPENAI_API_KEY=sk-..."
```

### 2. Update the MCP config

Edit `plugin/.mcp.json` and replace the placeholder URL with your Cloud Run service URL:

```json
{
  "mcpServers": {
    "multiagentz": {
      "type": "url",
      "url": "https://multiagentz-mcp-YOURHASH.a.run.app/mcp"
    }
  }
}
```

### 3. MCP Tools Exposed

When connected to the MCP server, Claude Code gains these tools:

- `run_analysis(stack_config, question)` — Run a full orchestration analysis
- `run_consensus(question, agents)` — Consensus mode
- `run_cross_pollinate(question, providers)` — Cross-pollination mode
- `run_perspective(question, perspectives)` — Perspective mode
- `list_stacks()` — List available stack configurations

## Plugin Structure

```
plugin/
├── .claude-plugin/
│   └── plugin.json           # Plugin manifest
├── .mcp.json                 # MCP server config (optional)
├── agents/
│   └── lead-agent.md         # Orchestration router subagent
├── skills/
│   ├── cross-pollinate/
│   │   └── SKILL.md          # /cross-pollinate slash command
│   ├── consensus/
│   │   └── SKILL.md          # /consensus slash command
│   ├── perspective/
│   │   └── SKILL.md          # /perspective slash command
│   ├── build/
│   │   └── SKILL.md          # /build slash command
│   └── configure-stack/
│       └── SKILL.md          # /configure-stack slash command
└── README.md
```

## How It Works

### Without MCP Server (Skills-Only Mode)

Claude Code reads the skill markdown files and follows the orchestration patterns directly using its own tool-calling capabilities. This gives you:

- Cross-pollination via different reasoning frameworks
- Consensus analysis across domain perspectives
- Structured perspective analysis with iterative refinement
- Build mode with dependency-aware task execution

### With MCP Server (Full Engine Mode)

The MCP server runs the actual multiagentz Python orchestration engine, giving you:

- True multi-provider LLM calls (Claude, GPT-4, Grok, Gemini, Mistral, etc.)
- 8-provider unified interface with automatic model detection
- Token budget management and graceful degradation
- Disk-backed caching and session memory
- Full YAML stack configuration support

## Supported LLM Providers

| Provider | Models | Environment Variable |
|----------|--------|---------------------|
| Anthropic | claude-opus-4-6, claude-sonnet-4-20250514 | `ANTHROPIC_API_KEY` |
| OpenAI | gpt-4o, o4-mini, o3 | `OPENAI_API_KEY` |
| xAI | grok-4-1-fast-reasoning | `XAI_API_KEY` |
| Google | gemini-2.5-pro, gemini-2.0-flash | `GOOGLE_API_KEY` |
| Mistral | mistral-large-latest | `MISTRAL_API_KEY` |
| Cohere | command-a-03-2025 | `COHERE_API_KEY` |
| NVIDIA | llama-3.1-nemotron-70b-instruct | `NVIDIA_API_KEY` |
| Ollama | Any local model | (none) |

## License

MIT
