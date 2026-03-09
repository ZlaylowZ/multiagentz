---
name: perspective
description: Deep multi-phase analysis with bootstrap alignment, independent solutions, iterative refinement, and final synthesis across perspectives
---

# Perspective Analysis

You are performing a **perspective analysis** — the most thorough orchestration mode in multiagentz. This runs a structured multi-phase process that aligns perspectives, generates independent solutions, refines through feedback, and synthesizes.

## Four-Phase Process

### Phase 1: Bootstrap Alignment (Q&A)
Each perspective independently identifies what clarifying information it needs. You gather this information before any analysis begins. This ensures all perspectives work from the same shared understanding.

For each perspective:
1. Present the question and ask what clarifying questions this perspective needs answered
2. Answer those questions (from your knowledge, the codebase, or by asking the user)
3. Repeat until the perspective signals it has enough context (max 3 rounds)

### Phase 2: Independent Solution Generation
Each perspective generates a complete solution independently, without seeing the others' work. This prevents groupthink and ensures diverse approaches.

### Phase 3: Iterative Refinement
Review all solutions and provide structured feedback to each perspective:
- **Strengths**: What this perspective got right
- **Weaknesses**: What it missed or got wrong
- **Missing insights**: What other perspectives added that this one didn't
- **Required improvements**: Specific areas to refine

Each perspective then refines its solution based on this feedback. Repeat for up to 2-3 iterations or until solutions converge.

### Phase 4: Final Synthesis
Produce a unified recommendation that:
1. Integrates the strongest elements from all perspectives
2. Resolves disagreements with clear reasoning
3. Notes significant divergences that couldn't be reconciled
4. Provides a clear, actionable recommendation

## Execution Instructions

When the user invokes `/perspective`:

1. **Parse the input** from `$ARGUMENTS` — extract the question and optionally specific perspectives to use
2. **Select perspectives**: Choose 2-4 relevant perspectives based on the question (e.g., "security engineer", "performance architect", "pragmatic developer", "UX designer")
3. **Phase 1**: Bootstrap each perspective with Q&A alignment
4. **Phase 2**: Generate independent solutions from each perspective
5. **Phase 3**: Review, provide feedback, and refine (2 iterations)
6. **Phase 4**: Synthesize into final unified answer

## Output Format

```
## Perspective Analysis

**Question**: [user's question]
**Perspectives**: [list of perspectives used]

---

### Phase 1: Alignment

#### [Perspective A] — Key Assumptions
[what this perspective established during Q&A]

#### [Perspective B] — Key Assumptions
[what this perspective established during Q&A]

---

### Phase 2: Independent Solutions

#### [Perspective A] Solution
[complete solution from this viewpoint]

#### [Perspective B] Solution
[complete solution from this viewpoint]

---

### Phase 3: Refinement

#### Iteration 1 Feedback
[summary of cross-perspective feedback and how each perspective refined]

---

### Phase 4: Synthesized Recommendation

[unified answer integrating the strongest elements]

### Key Trade-offs
[important trade-offs the user should be aware of]

### Implementation Priority
[ordered steps for implementation]
```

$ARGUMENTS
