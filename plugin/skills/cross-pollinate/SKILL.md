---
name: cross-pollinate
description: Run a question through multiple LLM providers and have them critique each other's responses to surface blind spots and produce stronger analysis
---

# Cross-Pollination Analysis

You are performing a **cross-pollination analysis** — the signature capability of multiagentz. This technique runs the same question through multiple LLM providers and has them critique each other to produce a stronger combined result.

## How It Works

Cross-pollination follows this process:

### Phase 1: Independent Analysis
Run the user's question through two or more different LLM providers independently. Each provider answers without seeing the others' responses.

### Phase 2: Mutual Critique
Share each provider's response with the others and ask them to:
- Identify strengths in the other's analysis
- Point out blind spots, errors, or missing considerations
- Note where they agree and disagree
- Suggest improvements based on the other's insights

### Phase 3: Synthesis
Produce a final unified answer that:
- Integrates the strongest elements from all providers
- Resolves disagreements with clear reasoning
- Flags remaining uncertainties
- Notes where different models had unique insights

## Execution Instructions

When the user invokes `/cross-pollinate`, follow these steps:

1. **Parse the question**: Extract the user's question from `$ARGUMENTS`
2. **Select providers**: Use at least 2 different approaches. You can simulate cross-pollination by:
   - Analyzing from different expert perspectives (e.g., systems architect vs. security engineer)
   - Using different reasoning frameworks (first-principles vs. pattern-matching vs. adversarial)
   - If an MCP server is configured, delegate to the multiagentz orchestration engine
3. **Run Phase 1**: Generate independent analyses
4. **Run Phase 2**: Cross-critique between the analyses
5. **Run Phase 3**: Synthesize into a final answer

## Output Format

```
## Cross-Pollination Analysis

**Question**: [user's question]
**Providers**: [list of perspectives/providers used]

### Independent Analyses

#### Perspective A: [name]
[analysis]

#### Perspective B: [name]
[analysis]

### Cross-Critique

**A's critique of B**: [key points]
**B's critique of A**: [key points]

### Synthesized Answer

[unified answer integrating the best of both, resolving disagreements]

### Divergence Notes
[where the perspectives fundamentally disagreed and why]
```

## When to Use

- Architecture decisions where different design philosophies matter
- Code review where different expertise areas catch different issues
- Debugging where the problem could have multiple root causes
- Any analysis where a single perspective might have blind spots

$ARGUMENTS
