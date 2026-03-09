---
name: consensus
description: Query multiple domain experts and synthesize their responses into a unified answer with conflict resolution
---

# Consensus Analysis

You are performing a **consensus analysis** — querying multiple domain perspectives and synthesizing their responses into a single, well-reasoned answer.

## How It Works

### Phase 1: Multi-Domain Query
Route the question to all relevant domain experts simultaneously. Each expert answers independently based on their area of expertise.

### Phase 2: Conflict Detection
Compare the responses to identify:
- **Agreements**: Where experts align (high confidence areas)
- **Conflicts**: Where experts disagree (needs resolution)
- **Unique insights**: Points raised by only one expert

### Phase 3: Iterative Resolution
For any conflicts:
1. Ask each conflicting expert to respond to the opposing viewpoint
2. Identify if the conflict is due to different assumptions, scope, or genuine disagreement
3. Resolve with clear reasoning about which approach is stronger and why

### Phase 4: Synthesis
Produce a unified answer that reflects the consensus, noting:
- Areas of strong agreement
- Resolved conflicts and the reasoning
- Remaining open questions

## Execution Instructions

When the user invokes `/consensus`:

1. **Parse the question** from `$ARGUMENTS`
2. **Identify relevant domains**: Analyze the question to determine which expertise areas are relevant (e.g., backend, frontend, security, infrastructure, performance, UX)
3. **Generate domain-specific analyses**: For each relevant domain, produce an analysis from that expert perspective
4. **Detect and resolve conflicts**: Compare analyses and resolve disagreements
5. **Synthesize**: Produce the final unified answer

## Output Format

```
## Consensus Analysis

**Question**: [user's question]
**Domains consulted**: [list of expert perspectives]

### Domain Perspectives

#### [Domain 1]
[analysis from this perspective]

#### [Domain 2]
[analysis from this perspective]

### Consensus Points
- [point where experts agree] ✓
- [point where experts agree] ✓

### Resolved Conflicts
- **[topic]**: [Domain 1] suggested X, [Domain 2] suggested Y.
  **Resolution**: [reasoning for chosen approach]

### Unified Recommendation
[final synthesized answer]

### Open Questions
[any remaining uncertainties or areas needing more investigation]
```

$ARGUMENTS
