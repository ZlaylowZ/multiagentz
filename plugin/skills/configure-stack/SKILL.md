---
name: configure-stack
description: Generate or modify a multiagentz YAML stack configuration for multi-agent orchestration
---

# Configure Stack

You are helping the user create or modify a **multiagentz stack configuration** — a YAML file that defines the agents, orchestration mode, and routing behavior for a multi-agent system.

## Stack Configuration Format

A stack YAML file defines:

```yaml
# Stack metadata
stack_name: my-project-stack

# Lead agent configuration
lead:
  model: claude-sonnet-4-20250514    # LLM for routing decisions
  brief: false                        # Brief response mode
  system_prompt: |                    # Optional custom system prompt
    You are the lead orchestrator...

  # Orchestration settings
  orchestration:
    mode: standard                    # standard | consensus | perspective | builder
    max_iterations: 3                 # Max refinement iterations (perspective/consensus)
    cross_pollination: true           # Enable cross-pollination between twins

  # Agent definitions
  agents:
    backend:
      repo_path: /path/to/backend     # Code repository for context
      model: claude-sonnet-4-20250514
      keywords: [api, database, server, endpoint, migration]
      hints: "Backend API and database expert"

    frontend:
      repo_path: /path/to/frontend
      model: gpt-4o
      keywords: [react, component, css, ui, ux]
      hints: "Frontend React and UI specialist"

    # Cross-pollination twin pair
    analyst_a:
      model: claude-sonnet-4-20250514
      twin: analyst_b               # Cross-pollination partner
      keywords: [architecture, design]

    analyst_b:
      model: grok-4-1-fast-reasoning
      twin: analyst_a
      keywords: [architecture, design]

    # Coordinator (nested agent group)
    infra:
      type: coordinator
      model: claude-sonnet-4-20250514
      agents:
        devops:
          repo_path: /path/to/infra
          model: claude-sonnet-4-20250514
          keywords: [deploy, ci, docker, k8s]
        database:
          repo_path: /path/to/db
          model: claude-sonnet-4-20250514
          keywords: [sql, migration, schema]

  # Perspective mode config
  perspectives:
    security_engineer:
      model: claude-sonnet-4-20250514
      prompt: "Analyze from a security perspective"
    performance_architect:
      model: gpt-4o
      prompt: "Analyze from a performance and scalability perspective"
    pragmatic_developer:
      model: grok-4-1-fast-reasoning
      prompt: "Analyze from a practical implementation perspective"

# Builder mode config (optional)
architect:
  model: claude-opus-4-6

builder_defaults:
  model: claude-sonnet-4-20250514
  max_retries: 3
  validation_commands:
    - "python -m py_compile {file}"
    - "ruff check {file} --select E,F"

workspace: /path/to/project
```

## Supported Providers

| Provider | Model Examples | Env Variable |
|----------|---------------|--------------|
| Anthropic | claude-opus-4-6, claude-sonnet-4-20250514 | ANTHROPIC_API_KEY |
| OpenAI | gpt-4o, o4-mini, o3 | OPENAI_API_KEY |
| xAI (Grok) | grok-4-1-fast-reasoning | XAI_API_KEY |
| Google | gemini-2.5-pro, gemini-2.0-flash | GOOGLE_API_KEY |
| Mistral | mistral-large-latest | MISTRAL_API_KEY |
| Cohere | command-a-03-2025 | COHERE_API_KEY |
| NVIDIA | nvidia/llama-3.1-nemotron-70b-instruct | NVIDIA_API_KEY |
| Ollama | llama3, mistral:latest (local) | (none needed) |

## Instructions

When the user invokes `/configure-stack`:

1. **Ask about their project**: What repos/codebases are involved? What domains (backend, frontend, etc.)?
2. **Ask about orchestration needs**: Do they want cross-pollination? Consensus? Perspective analysis?
3. **Ask about providers**: Which LLM providers do they have API keys for?
4. **Generate the YAML**: Create a complete stack configuration
5. **Save it**: Write to `stacks/` directory or user-specified path

$ARGUMENTS
