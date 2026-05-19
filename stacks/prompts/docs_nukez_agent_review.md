# NukezAgent — Comprehensive Documentation

## Objective

Produce thorough, self-contained reference documentation for the
nukezAgent repository. This documentation will be used as input by
future planning and buildout stacks — it is NOT a plan itself.

The output must be accurate and detailed enough that an agent reading
only this document can understand nukezAgent's architecture, API surface,
integration contracts, configuration, and operational behavior without
reading source code.

## Context: What NukezAgent Is

NukezAgent is the **simplest possible way to give any AI agent persistent,
verifiable storage**. It is the front door to the Nukez ecosystem — the
thing developers actually interact with. Everything else (the gateway,
pynukez SDK, the MCP server) is infrastructure behind it.

The design targets three deployment scenarios, in priority order:

1. **Multi-agent systems** — NukezAgent runs as a sub-agent within larger
   orchestrations (CrewAI, LangGraph, AutoGen, custom). Trivially
   composable: hand it a task, get back a receipt. Capable of spawning
   its own sub-agents for task decomposition.

2. **Single-agent deployments** — Standalone agent accessed via MCP or
   A2A protocol. Any MCP-compatible host (Claude Desktop, Cursor, etc.)
   adds NukezAgent as a tool provider for persist/recall/verify.

3. **Programmatic SDK usage** — The `python/nukez` package provides a
   clean 3-intent API (persist, recall, verify) for direct integration.

NukezAgent depends on **pynukez** for auth, signing, HTTP, and payment.
This is intentional — nukezAgent should never reimplement what pynukez
provides.

## Documentation Scope

Document each area as it actually exists in the codebase today.
Be precise. Flag incomplete or stub implementations as-is — do not
speculate about intended behavior beyond what the code demonstrates.

### Service Layer (`service/`)
- **Agent core** (`agent.py`, `inference.py`, `tools.py`) — Document the
  inference loop, tool definitions, input/output schemas, multi-turn
  reasoning flow, and how tool selection works.
- **Adapters** (`mcp_adapter.py`, `a2a_adapter.py`) — Document the MCP
  and A2A protocol implementations: what capabilities are advertised,
  how requests are routed, what's fully implemented vs. stubbed.
- **Signing & security** (`signing.py`) — Document the envelope signing
  model, the critical invariant (agent never possesses owner's keypair,
  never executes crypto transfers), and the sign_request flow.
- **State & config** (`state.py`, `config.py`) — Document state
  management model, session handling, configuration surface.
- **Operational infrastructure** (`rate_limiter.py`, `instance_manager.py`,
  `Dockerfile`) — Document deployment model, rate limiting, instance
  lifecycle.
- **FastAPI app** (`app.py`, `models.py`) — Document HTTP endpoints,
  request/response models, middleware.

### Python SDK (`python/`)
- **3-intent API** — Document persist, recall, verify: signatures,
  parameters, return types, exceptions, usage examples.
- **pynukez boundary** — Document what the SDK owns vs. what it delegates
  to pynukez. Map every SDK method to its pynukez dependency.
- **Packaging** — Document install process, dependencies, pyproject.toml.
- **Error handling** — Document how errors from pynukez and the gateway
  surface to the consumer.

### Root Repo (contracts, manifests, docs, examples)
- **AAAP contracts** — Document the agent capability contract schemas,
  Context Envelope format, Typed Artifacts, failure taxonomy.
- **Agent manifests** — Document agent.json / agent-card contents and
  what capabilities they advertise.
- **Prompts** — Document any system prompts or prompt templates.
- **Examples** — Document what examples exist and what they demonstrate.

## Ground Truth

Before starting, read `/Users/zhanson/Desktop/nukezAgent/README.md`
for the project's own description of its architecture and goals.

## Output Format

```
# NukezAgent Documentation

## Table of Contents

## 1. Architecture Overview
### 1.1 System Architecture
### 1.2 Component Map
### 1.3 Data Flows
### 1.4 Dependency Graph (pynukez, gateway, external)
### 1.5 Key Design Invariants

## 2. Service Layer
### 2.1 Agent Core (inference loop, tool execution)
### 2.2 Tool Definitions (schemas, behaviors)
### 2.3 MCP Adapter
### 2.4 A2A Adapter
### 2.5 Signing Model
### 2.6 State Management
### 2.7 Configuration Reference
### 2.8 HTTP API (FastAPI endpoints, models)
### 2.9 Operational Infrastructure

## 3. Python SDK
### 3.1 API Reference (persist, recall, verify)
### 3.2 pynukez Integration Boundary
### 3.3 Error Handling
### 3.4 Installation & Packaging

## 4. Contracts & Manifests
### 4.1 AAAP Contract Schemas
### 4.2 Agent Manifests
### 4.3 Capability Advertisement

## 5. Cross-Repo Integration Points
### 5.1 nukezAgent ↔ pynukez interfaces
### 5.2 nukezAgent ↔ gateway interfaces
### 5.3 nukezAgent ↔ MCP host interfaces

## 6. Operational Reference
### 6.1 Configuration (env vars, config keys, defaults)
### 6.2 Deployment (Docker, standalone)
### 6.3 Failure Modes & Debugging
### 6.4 Security & Trust Boundaries
```
