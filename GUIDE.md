# multiagentz User Guide

## The Big Picture

multiagentz (`maz`) is a multi-agent analysis engine. You describe your project in a YAML file, and `maz` builds a team of AI agents — each one an expert on a different part of your codebase. When you ask a question, the system routes to the right agent(s), queries them in parallel, and synthesizes a unified answer.

**The real power is using maz as a front-end for Claude Code.** The workflow:

1. Ask maz a complex question (architecture design, refactoring plan, security audit)
2. maz routes to multiple domain experts, optionally runs cross-pollination or consensus
3. You get a high-confidence, multi-perspective analysis as an HTML export
4. Hand that analysis to Claude Code — it builds the implementation with a solid plan already in place

This works because maz can throw multiple models at the same problem (Claude, Grok, Gemini) and force them to critique each other before producing a final answer. By the time you hand it to Claude Code, the hard thinking is done.

---

## Choosing Your Complexity Level

maz scales from dead simple to extremely sophisticated. Pick the level you need.

### Level 1: Simple (2 agents, no orchestration)

**Use when:** You have a straightforward project with clear domain separation (backend/frontend, server/client, etc.) and just want smarter code Q&A.

```yaml
name: my-app

lead:
  keywords:
    backend: [api, database, server, endpoint]
    frontend: [react, component, ui, css]

  agents:
    backend:
      repo_path: /path/to/backend
      description: Backend API and database
      key_files: [src/, README.md]

    frontend:
      repo_path: /path/to/frontend
      description: React frontend
      key_files: [src/, package.json]
```

You ask a question, the router figures out which agent knows the answer, done. This is the starting point for everyone.

### Level 2: Multi-model (cost optimization)

**Use when:** You want to use expensive models only where they matter and cheap/fast models for simpler domains.

Same as Level 1, but add `model:` per agent:

```yaml
  agents:
    backend:
      repo_path: /path/to/backend
      description: Complex backend with auth and business logic
      model: claude-sonnet-4-20250514    # Smart model for complex domain
      key_files: [src/]

    frontend:
      repo_path: /path/to/frontend
      description: Standard React UI
      model: gemini-2.0-flash            # Fast/cheap for simpler domain
      key_files: [src/]
```

### Level 3: Consensus mode

**Use when:** You're asking questions where accuracy matters and you want the system to self-check. Multiple agents answer independently, the system detects contradictions, iteratively resolves them, and produces a verified answer.

```yaml
orchestration:
  mode: consensus
  max_iterations: 3
```

Or use it on-demand from the REPL: `/consensus How should we handle auth tokens across services?`

### Level 4: Cross-pollination (twin agents)

**Use when:** You want genuine cognitive diversity. Two agents with *different AI models* independently solve the same problem, swap their outputs, critique each other, and a coordinator reconciles the result.

This is the most powerful feature in maz. It forces disagreement and resolution, which produces dramatically better results than any single model alone.

```yaml
orchestration:
  cross_pollination: true

lead:
  agents:
    app_core:
      type: coordinator
      description: Backend application
      agents:
        CoreA:
          repo_path: /path/to/backend
          description: Backend expert (Claude perspective)
          model: claude-sonnet-4-20250514
          key_files: [src/]
          twin: CoreB

        CoreB:
          repo_path: /path/to/backend
          description: Backend expert (Grok perspective)
          model: grok-4-1-fast-non-reasoning
          key_files: [src/]
          twin: CoreA
```

When the router sends a question to `CoreA`, the cross-pollination engine automatically:
1. Sends the same question to both `CoreA` and `CoreB` in parallel
2. Swaps their outputs — CoreA sees CoreB's answer, CoreB sees CoreA's
3. Each refines their answer after reviewing the other's work
4. The coordinator synthesizes a final answer from both refined outputs

### Level 5: Perspective mode (full orchestration)

**Use when:** You're doing serious design work — architecture decisions, migration plans, complex refactoring — and you want multiple independent solution proposals that get iteratively refined by a lead agent before synthesis.

```yaml
orchestration:
  mode: perspective
  max_iterations: 3
```

Or use it on-demand: `/perspective "Design the authentication flow" backend frontend`

The perspective pipeline:
1. **Bootstrap Q&A** — each perspective agent asks clarifying questions, the lead answers
2. **Independent solutions** — each agent produces a complete solution proposal in parallel
3. **Lead review** — the lead agent reviews all proposals, identifies strengths/weaknesses
4. **Iterative refinement** — agents refine based on lead feedback (runs in parallel)
5. **Consensus synthesis** — final unified answer combining the best of all perspectives

---

## The maz → Claude Code Workflow

This is the intended power use case.

### Step 1: Run your analysis in maz

```bash
maz --config stacks/my-project.yaml
```

Ask your complex question:
```
You: Design a complete authentication system with OAuth2, JWT refresh tokens,
     and role-based access control. Consider the existing database schema and
     API patterns in the codebase.
```

Or force consensus/perspective mode for even deeper analysis:
```
You: /perspective "Design a complete authentication system with OAuth2, JWT
     refresh tokens, and role-based access control"
```

### Step 2: Get the exported analysis

Every maz response automatically exports to `./outputs/` as an HTML file. You'll see:
```
Saved: outputs/20260227_143022_Design_a_complete_auth.html
```

You can also re-export in different formats:
```
You: /export md
```

### Step 3: Hand to Claude Code

Open Claude Code and give it the analysis:

```
Read the file outputs/20260227_143022_Design_a_complete_auth.html

This is a multi-agent analysis of the auth system we need to build.
It was generated by querying multiple AI models against our actual codebase.

Implement the recommended approach. The analysis has already:
- Reviewed our existing code patterns
- Proposed architecture
- Identified potential issues
- Resolved conflicts between different perspectives

Follow the implementation plan in the analysis.
```

Claude Code now has a high-confidence blueprint to work from instead of figuring everything out from scratch.

---

## REPL Commands Reference

### Core
| Command | What it does |
|---------|-------------|
| `/help` | Show all commands |
| `quit` | Exit |

### File Management
| Command | What it does |
|---------|-------------|
| `/watch <path>` | Add a file or directory for agents to read |
| `/unwatch <path>` | Stop watching a path |
| `/watched` | List all watched paths |
| `/clear-watched` | Remove all watched paths |
| `/context` | Show how much context the file agent is using |
| `/scan <path>` | Watch a path and auto-generate a summary |
| `/file <path>` | Load a file's contents as your question |

### Analysis
| Command | What it does |
|---------|-------------|
| `/consensus <question>` | Force consensus mode (multi-agent conflict resolution) |
| `/perspective "<question>" [agents]` | Multi-perspective analysis with independent solutions |
| `/promote <agent>` | Make an agent the primary implementation coordinator |
| `/demote` | Remove LEAD_SUB promotion |
| `/status` | Show current orchestration config |

### Output
| Command | What it does |
|---------|-------------|
| `/export [html\|md\|txt]` | Export last response to file |
| `/brief` | Toggle brief response mode |

### Session
| Command | What it does |
|---------|-------------|
| `/clear` | Clear conversation memory |
| `/cache` | Show cache stats |
| `/cache clear` | Clear the response cache |
| `/paste` | Multi-line input mode (type END to finish) |

---

## Tips

**Start simple, add complexity when you need it.** Begin with a basic two-agent stack. When you find yourself asking questions that touch multiple domains and wanting deeper analysis, add orchestration modes.

**Use `/watch` liberally.** The file agent can read anything on disk. If a question involves files outside your configured repos, `/watch` them in.

**Consensus vs Perspective:** Consensus is for *verifying accuracy* — it detects contradictions and resolves them. Perspective is for *generating solutions* — it produces independent proposals and synthesizes the best parts.

**Cross-pollination is most valuable when agents use different models.** The whole point is cognitive diversity. Two Claude agents will largely agree. Claude + Grok will disagree in productive ways.

**The auto-export is a feature.** Every response lands in `./outputs/` as HTML. This is your handoff artifact for Claude Code, your audit trail, and your documentation.
