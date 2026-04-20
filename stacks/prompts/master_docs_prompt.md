# Master Docs — Prompt Template

> **Purpose:** Round 2 reconciliation. The individual `docs_*.yaml` stacks
> have already generated per-repo documentation (round 1). This master run
> reads ALL of those outputs via `prior_docs`, cross-references across repos,
> and produces a single unified foundational document set for the FOCUS_REPO.
>
> **The output of this stack is not a plan.** It is the canonical reference
> documentation that future planning stacks, buildout agents, and feature
> evaluations will consume as input — so they never need to do deep repo
> reviews themselves.
>
> **Usage:**
> 1. Set `FOCUS_REPO` in `master_docs.yaml` to the target team
> 2. `maz --config stacks/master_docs.yaml`
> 3. Paste the prompt below
> 4. Save output → `outputs/docs/master_[repo_name].md`
>
> **Run order:** pynukez → mcp → agent → gateway

---

## Prompt

```
Reconcile and unify the existing per-repo documentation for the current
FOCUS_REPO into a single, authoritative foundational reference.

CONTEXT:
Round 1 is complete. Each individual docs stack has already generated
detailed documentation for its repo. Those outputs are available via
prior_docs. Your job is NOT to regenerate documentation from scratch —
it is to RECONCILE, CROSS-VALIDATE, and UNIFY what already exists.

This output will be the canonical foundational document that all future
work references: buildout plans, feature evaluations, integration
assessments, and agent-driven development. It must be accurate and
complete enough that an agent can understand the FOCUS_REPO's architecture,
API surface, integration contracts, and operational behavior without
reading source code.

PROCESS:

1. INGEST — Query prior_docs for ALL available documentation. Read
   every prior doc, not just the one matching FOCUS_REPO. You need
   the full ecosystem picture.

2. RECONCILE — For the FOCUS_REPO, cross-check its round 1 docs
   against the other repos' docs. Identify and resolve:
   - Type name mismatches (e.g., Receipt described differently across repos)
   - Contract disagreements (e.g., pynukez says endpoint X expects field A,
     gateway docs say it expects field B)
   - Missing integration seams (e.g., round 1 docs describe a method but
     never explain how another repo actually calls it)
   - Stale or contradictory claims about behavior
   Where conflicts exist, query the relevant team's sub-agents to
   resolve against actual code. Prior docs are the starting point,
   code is the tiebreaker.

3. UNIFY — Produce one coherent document with these sections:

   ARCHITECTURE OVERVIEW
   - Component map with ownership boundaries and responsibilities
   - Data flows: request lifecycle, auth chain, storage pipeline
   - Dependency graph: imports, exports, and assumptions
   - Ecosystem position: how this repo connects upstream/downstream
   - Key design decisions and invariants (e.g., signing boundary)

   API SURFACE
   - Every public class, method, function, constant, config option
   - Signature, types, return values, exceptions, side effects
   - Organized by module — one subsection per source file
   - Flag cross-repo interfaces: which methods/types are consumed
     by other Nukez repos, and how
   - Mark any APIs that are undocumented, inconsistently typed, or
     that differ from what other repos' docs claim

   INTEGRATION CONTRACTS
   - Every interface this repo exposes to or consumes from other repos
   - For each contract: the method/endpoint, expected input/output
     types, auth requirements, error responses, and which repo(s)
     are on each side
   - Validated against both sides' documentation (and code if needed)

   OPERATIONAL REFERENCE
   - Configuration: every env var, config key, CLI flag, with defaults
   - Deployment: how to run, containerize, and configure for dev/prod
   - Debugging: common failure modes, error codes, diagnostic steps
   - Security: auth setup, key management, trust boundaries

4. REFINE (iteration 2) — Focus exclusively on:
   - Factual errors caught by cross-referencing other repos' docs
   - Missing APIs discovered by comparing file listings to API section
   - Contract mismatches between what this repo claims and what
     consuming/providing repos expect
   - Code examples with wrong signatures or types
   Do NOT spend refinement on rewording or reformatting.

OUTPUT RULES:
- Markdown with hierarchical headers and a table of contents
- Mark unresolved discrepancies with [CONFLICT: description]
- Mark gaps with [TODO: what's missing and where to find it]
- End with a CROSS-REPO DEPENDENCY MATRIX: a table listing every
  external interface, which repo provides it, which consumes it,
  and whether the contract is validated or unverified
- This document must stand alone. An agent reading ONLY this file
  should understand enough to work on this repo without reading code.

Save output to: outputs/docs/master_[repo_name].md
```
