# multiagentz/generate.py
"""
Stack generator — automated YAML stack creation via phased pipeline.

Phase 1: Repo Survey     (no LLM)  — walk filesystem, build manifest
Phase 2: Topology        (LLM)     — propose agent boundaries from manifest
Phase 3: Prompt Grounding(LLM)     — read actual files, write system prompts
Phase 4: Routing Calibration (LLM) — simulate queries, tune keywords

Usage:
    maz generate /path/to/repo --goal "document the codebase"
    maz generate /repo1 /repo2 --goal "plan a feature" --output stacks/my.yaml
"""

from __future__ import annotations

import json
import os
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml

from multiagentz.agents.base import READABLE_EXTENSIONS, EXCLUDE_PATTERNS, MAX_FILE_BYTES
from multiagentz.llm_client import LLMClient
from multiagentz import log as _log


# ── Constants ──────────────────────────────────────────────────────────

DEFAULT_MAX_CHARS_PER_AGENT = 200_000  # Target context budget per agent
DEFAULT_MODEL = "claude-sonnet-4-20250514"
CHARS_PER_TOKEN_ESTIMATE = 3.5
MAX_WALK_DEPTH = 8

# Module boundary markers by language ecosystem
MODULE_MARKERS = {
    "__init__.py",     # Python package
    "package.json",    # Node.js
    "go.mod",          # Go module
    "Cargo.toml",      # Rust crate
    "pom.xml",         # Java/Maven
    "build.gradle",    # Java/Gradle
    "Gemfile",         # Ruby
    "setup.py",        # Python (legacy)
    "pyproject.toml",  # Python (modern)
    "composer.json",   # PHP
    "mix.exs",         # Elixir
}

SUMMARY_FILES = {
    "README.md", "README.rst", "README.txt", "README",
    "CONTRIBUTING.md", "ARCHITECTURE.md", "DESIGN.md",
    "CHANGELOG.md", "CLAUDE.md",
}

ENTRY_POINT_PATTERNS = {
    "main.py", "app.py", "index.ts", "index.js", "main.go",
    "main.rs", "App.tsx", "App.jsx", "server.py", "server.ts",
    "cli.py", "manage.py", "__main__.py",
}


# ── Phase 1: Data structures ──────────────────────────────────────────

@dataclass
class FileEntry:
    """A single readable file in the repo."""
    rel_path: str      # relative to repo root
    chars: int          # character count
    ext: str            # file extension


@dataclass
class ModuleBoundary:
    """A directory that looks like a logical module/package."""
    rel_path: str       # relative to repo root
    marker: str         # what file makes it a boundary
    total_chars: int    # sum of readable chars underneath
    file_count: int
    depth: int          # directory depth from repo root


@dataclass
class RepoManifest:
    """Complete survey of a single repository."""
    repo_path: str
    repo_name: str
    files: list[FileEntry] = field(default_factory=list)
    boundaries: list[ModuleBoundary] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)
    summary_files: list[str] = field(default_factory=list)
    total_chars: int = 0
    total_files: int = 0


# ── Phase 2: Topology structures ──────────────────────────────────────

@dataclass
class AgentSpec:
    """Proposed agent with file assignments."""
    name: str
    key_files: list[str]        # relative paths
    description: str = ""
    system_prompt: str = ""
    est_chars: int = 0
    est_files: int = 0
    model: str = ""
    provider: str = ""
    repo_path: str = ""         # which repo this agent belongs to
    twin: str = ""              # twin agent name for cross-pollination


@dataclass
class CoordinatorSpec:
    """Proposed coordinator grouping agents."""
    name: str
    description: str = ""
    agents: list[AgentSpec] = field(default_factory=list)
    model: str = ""
    provider: str = ""


@dataclass
class TopologyProposal:
    """Complete proposed stack topology."""
    stack_name: str
    coordinators: list[CoordinatorSpec] = field(default_factory=list)
    standalone_agents: list[AgentSpec] = field(default_factory=list)
    orchestration_mode: str = "standard"
    routing_prompt_extra: str = ""
    keywords: dict = field(default_factory=dict)
    lead_model: str = ""
    lead_provider: str = ""


# ── Phase 1: Repo Survey ──────────────────────────────────────────────

_EXTRA_EXCLUDES = [
    ".maz_cache", "outputs", ".maz_routing.jsonl",
    "*.lock", "yarn.lock",
]


def _should_exclude(name: str) -> bool:
    """Check if a file/directory should be excluded."""
    import fnmatch
    return (
        any(fnmatch.fnmatch(name, pat) for pat in EXCLUDE_PATTERNS)
        or any(fnmatch.fnmatch(name, pat) for pat in _EXTRA_EXCLUDES)
    )


def survey_repo(repo_path: str) -> RepoManifest:
    """
    Walk a repository and build a complete manifest.

    No LLM calls — pure filesystem inspection.
    """
    root = Path(repo_path).resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    manifest = RepoManifest(
        repo_path=str(root),
        repo_name=root.name,
    )

    # Track per-directory stats for boundary detection
    dir_chars: dict[str, int] = {}
    dir_files: dict[str, int] = {}

    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = str(Path(dirpath).relative_to(root))
        depth = len(Path(rel_dir).parts) if rel_dir != "." else 0

        if depth > MAX_WALK_DEPTH:
            dirnames.clear()
            continue

        # Filter excluded directories
        dirnames[:] = [d for d in sorted(dirnames) if not _should_exclude(d)]

        for fn in sorted(filenames):
            if _should_exclude(fn):
                continue

            fp = Path(dirpath) / fn
            ext = fp.suffix.lower()

            # Check for summary files and entry points
            if fn in SUMMARY_FILES:
                manifest.summary_files.append(
                    str(fp.relative_to(root))
                )
            if fn in ENTRY_POINT_PATTERNS:
                manifest.entry_points.append(
                    str(fp.relative_to(root))
                )

            # Only count readable files
            if ext not in READABLE_EXTENSIONS:
                continue

            try:
                size = fp.stat().st_size
                if size > MAX_FILE_BYTES:
                    continue
                chars = size  # approximate; close enough for survey
            except OSError:
                continue

            rel = str(fp.relative_to(root))
            manifest.files.append(FileEntry(rel_path=rel, chars=chars, ext=ext))
            manifest.total_chars += chars
            manifest.total_files += 1

            # Accumulate into parent directories
            parts = Path(rel).parent.parts
            for i in range(len(parts)):
                d = str(Path(*parts[:i + 1]))
                dir_chars[d] = dir_chars.get(d, 0) + chars
                dir_files[d] = dir_files.get(d, 0) + 1

            # Check for module boundary markers
            if fn in MODULE_MARKERS:
                manifest.boundaries.append(ModuleBoundary(
                    rel_path=rel_dir if rel_dir != "." else "",
                    marker=fn,
                    total_chars=0,  # filled in below
                    file_count=0,
                    depth=depth,
                ))

    # Fill in boundary stats
    for b in manifest.boundaries:
        key = b.rel_path or "."
        # Sum all directories that start with this prefix
        b.total_chars = sum(
            v for k, v in dir_chars.items()
            if k == key or k.startswith(key + "/") or key == ""
        )
        b.file_count = sum(
            v for k, v in dir_files.items()
            if k == key or k.startswith(key + "/") or key == ""
        )

    return manifest


def format_manifest_for_llm(manifests: list[RepoManifest],
                             max_chars_per_agent: int) -> str:
    """Format repo manifests into a concise summary for the LLM."""
    parts = []
    for m in manifests:
        parts.append(f"## Repository: {m.repo_name}")
        parts.append(f"Path: {m.repo_path}")
        parts.append(f"Total: {m.total_files} files, {m.total_chars:,} chars "
                      f"(~{int(m.total_chars / CHARS_PER_TOKEN_ESTIMATE):,} tokens)")

        if m.summary_files:
            parts.append(f"Summary files: {', '.join(m.summary_files)}")
        if m.entry_points:
            parts.append(f"Entry points: {', '.join(m.entry_points)}")

        # Directory tree with sizes
        parts.append("\n### Directory structure (with char counts):")

        # Build directory-level summary
        dir_stats: dict[str, tuple[int, int]] = {}  # dir -> (chars, files)
        for f in m.files:
            parent = str(Path(f.rel_path).parent)
            if parent == ".":
                parent = "(root)"
            c, n = dir_stats.get(parent, (0, 0))
            dir_stats[parent] = (c + f.chars, n + 1)

        for d in sorted(dir_stats.keys()):
            chars, count = dir_stats[d]
            agents_needed = max(1, math.ceil(chars / max_chars_per_agent))
            marker = ""
            if agents_needed > 1:
                marker = f"  ← needs {agents_needed} agents"
            parts.append(f"  {d}/  ({count} files, {chars:,} chars){marker}")

        # Module boundaries
        if m.boundaries:
            parts.append("\n### Module boundaries:")
            for b in sorted(m.boundaries, key=lambda x: x.depth):
                parts.append(
                    f"  {b.rel_path or '(root)'}  [{b.marker}]  "
                    f"({b.file_count} files, {b.total_chars:,} chars)"
                )

        parts.append("")

    parts.append(f"\n### Budget: {max_chars_per_agent:,} chars per agent "
                  f"(~{int(max_chars_per_agent / CHARS_PER_TOKEN_ESTIMATE):,} tokens)")
    parts.append(f"Target: maximum subdivision — create as many focused agents as "
                  f"the budget allows. Narrower scope = better output quality.")

    return "\n".join(parts)


# ── Phase 2: Topology Proposal ────────────────────────────────────────

TOPOLOGY_SYSTEM_PROMPT = """You are an expert at designing multi-agent AI systems.

Your task: given a repository manifest (file tree with sizes), propose the MAXIMUM
subdivision into focused sub-agents. Each agent should own a narrow, non-overlapping
slice of the codebase.

## Design principles

1. **Maximize agent count** — more agents with narrower scope = better quality.
   A sub-agent with 50 focused files will outperform one with 300 files every time.
2. **Respect module boundaries** — packages, modules, and natural directory groupings
   are the best agent boundaries.
3. **Stay within budget** — each agent's files must fit within the char budget.
   If a directory exceeds the budget, split it into multiple agents by subdirectory
   or by file type (e.g., models vs routes vs tests).
4. **Group into coordinators** — cluster 3-5 related agents into coordinator groups.
   Use hierarchy (coordinators of coordinators) if needed for large repos.
5. **Name for routing** — agent names should be short, descriptive, and mutually
   distinguishable. They are the primary routing signal.
6. **Assign model tiers** — use cheaper models (claude-sonnet-4-20250514) for narrow-scope
   agents, heavier models (claude-opus-4-6) for coordinators and synthesis.

## Output format

Return ONLY valid JSON matching this schema:

```json
{
    "stack_name": "descriptive-stack-name",
    "orchestration_mode": "standard",
    "lead_model": "claude-opus-4-6",
    "lead_provider": "anthropic",
    "coordinators": [
        {
            "name": "team_name",
            "description": "What this coordinator group covers",
            "model": "claude-opus-4-6",
            "provider": "anthropic",
            "agents": [
                {
                    "name": "agent_name",
                    "key_files": ["src/models/", "src/schemas/"],
                    "description": "One-line description for routing",
                    "est_chars": 150000,
                    "est_files": 12,
                    "model": "claude-sonnet-4-20250514",
                    "provider": "anthropic",
                    "repo_path": "/absolute/path/to/repo"
                }
            ]
        }
    ],
    "standalone_agents": []
}
```

Rules:
- key_files should be directories (ending with /) or specific files
- Every file in the repo should be assigned to exactly one agent
- est_chars must not exceed the per-agent budget
- Use absolute repo_path for each agent
- agent names must be valid YAML keys (lowercase, underscores, no spaces)
"""


def propose_topology(
    manifests: list[RepoManifest],
    goal: str,
    max_chars_per_agent: int = DEFAULT_MAX_CHARS_PER_AGENT,
    llm: Optional[LLMClient] = None,
) -> TopologyProposal:
    """
    Phase 2: Use LLM to propose agent topology from manifest.
    """
    llm = llm or LLMClient(model=DEFAULT_MODEL)

    manifest_text = format_manifest_for_llm(manifests, max_chars_per_agent)

    prompt = f"""## User's Goal
{goal}

## Repository Manifest
{manifest_text}

Propose the optimal agent topology for maximum subdivision. Remember:
- More agents = better (narrower scope = higher quality output)
- Each agent must stay within the {max_chars_per_agent:,} char budget
- Group related agents into coordinator teams of 3-5
- Every directory/file in the manifest must be covered by exactly one agent
"""

    _log.step("Phase 2: Proposing agent topology...")
    result = llm.complete(prompt=prompt, system=TOPOLOGY_SYSTEM_PROMPT, max_tokens=16384)

    # Parse JSON response
    from multiagentz.utils import extract_json
    data = extract_json(str(result))

    # Build TopologyProposal
    proposal = TopologyProposal(
        stack_name=data.get("stack_name", "generated-stack"),
        orchestration_mode=data.get("orchestration_mode", "standard"),
        lead_model=data.get("lead_model", "claude-opus-4-6"),
        lead_provider=data.get("lead_provider", "anthropic"),
    )

    for coord_data in data.get("coordinators", []):
        coord = CoordinatorSpec(
            name=coord_data["name"],
            description=coord_data.get("description", ""),
            model=coord_data.get("model", ""),
            provider=coord_data.get("provider", ""),
        )
        for agent_data in coord_data.get("agents", []):
            coord.agents.append(AgentSpec(
                name=agent_data["name"],
                key_files=agent_data.get("key_files", []),
                description=agent_data.get("description", ""),
                est_chars=agent_data.get("est_chars", 0),
                est_files=agent_data.get("est_files", 0),
                model=agent_data.get("model", ""),
                provider=agent_data.get("provider", ""),
                repo_path=agent_data.get("repo_path", ""),
            ))
        proposal.coordinators.append(coord)

    for agent_data in data.get("standalone_agents", []):
        proposal.standalone_agents.append(AgentSpec(
            name=agent_data["name"],
            key_files=agent_data.get("key_files", []),
            description=agent_data.get("description", ""),
            est_chars=agent_data.get("est_chars", 0),
            est_files=agent_data.get("est_files", 0),
            model=agent_data.get("model", ""),
            provider=agent_data.get("provider", ""),
            repo_path=agent_data.get("repo_path", ""),
        ))

    return proposal


# ── Phase 3: Prompt Grounding ─────────────────────────────────────────

GROUNDING_SYSTEM_PROMPT = """You are generating a system prompt for a specialized AI sub-agent.

You will receive:
1. The agent's name, description, and file scope
2. The ACTUAL CONTENTS of representative files from the agent's scope
3. The user's overall goal

Your task: write a system prompt that is GROUNDED in what the files actually contain.

## Rules
- Reference real class names, function names, patterns, and architecture from the files
- Do NOT use generic language like "You are an expert in..." — be specific
- Mention the key abstractions, design patterns, and conventions you see
- Note any important relationships to other parts of the system
- Keep it to 3-8 sentences — dense and specific, not verbose
- Also refine the agent's one-line description to be maximally routing-differentiable
  (i.e., a router LLM should be able to distinguish this agent from siblings)

## Output format
Return ONLY valid JSON:
```json
{
    "system_prompt": "You own the FastAPI route handlers in src/routes/...",
    "description": "Route handlers for /api/v1/* endpoints: auth, upload, download, pricing"
}
```
"""


def _read_representative_files(
    agent: AgentSpec,
    max_chars: int = 50_000,
) -> str:
    """Read a sample of the agent's files for grounding."""
    repo = Path(agent.repo_path)
    sections = []
    total = 0

    # Resolve key_files to actual file paths
    resolved = []
    for kf in agent.key_files:
        full = repo / kf
        if full.is_file():
            resolved.append(full)
        elif full.is_dir():
            for root, dirs, files in os.walk(full):
                depth = len(Path(root).relative_to(full).parts)
                if depth > 3:
                    dirs.clear()
                    continue
                dirs[:] = [d for d in sorted(dirs) if not _should_exclude(d)]
                for fn in sorted(files):
                    if _should_exclude(fn):
                        continue
                    fp = Path(root) / fn
                    if fp.suffix.lower() in READABLE_EXTENSIONS:
                        resolved.append(fp)

    # Prioritize: entry points > shorter files > alphabetical
    def priority(p: Path) -> tuple:
        is_entry = p.name in ENTRY_POINT_PATTERNS
        is_summary = p.name in SUMMARY_FILES
        try:
            size = p.stat().st_size
        except OSError:
            size = 999999
        return (not is_summary, not is_entry, size)

    resolved.sort(key=priority)

    for fp in resolved:
        if total >= max_chars:
            break
        try:
            content = fp.read_text(encoding="utf-8")
            if len(content) > 20_000:
                content = content[:20_000] + "\n... [truncated]"
            rel = str(fp.relative_to(repo))
            sections.append(f"=== {rel} ===\n{content}")
            total += len(content)
        except Exception:
            continue

    return "\n\n".join(sections)


def ground_agent(
    agent: AgentSpec,
    goal: str,
    llm: Optional[LLMClient] = None,
) -> AgentSpec:
    """
    Phase 3: Read actual files and generate grounded system prompt.
    """
    llm = llm or LLMClient(model=DEFAULT_MODEL)

    file_sample = _read_representative_files(agent)
    if not file_sample:
        _log.warn(f"No readable files for {agent.name} — skipping grounding")
        return agent

    prompt = f"""## Agent: {agent.name}
## Current description: {agent.description}
## File scope: {', '.join(agent.key_files)}
## User's goal: {goal}

## Representative file contents:
{file_sample}

Generate a grounded system_prompt and refined description for this agent.
"""

    result = llm.complete(prompt=prompt, system=GROUNDING_SYSTEM_PROMPT, max_tokens=2048)

    try:
        from multiagentz.utils import extract_json
        data = extract_json(str(result))
        agent.system_prompt = data.get("system_prompt", agent.system_prompt)
        agent.description = data.get("description", agent.description)
    except Exception as e:
        _log.warn(f"Grounding parse failed for {agent.name}: {e}")

    return agent


def ground_all_agents(
    proposal: TopologyProposal,
    goal: str,
    llm: Optional[LLMClient] = None,
    max_workers: int = 6,
) -> TopologyProposal:
    """
    Phase 3: Ground all agents in parallel.
    """
    llm = llm or LLMClient(model=DEFAULT_MODEL)
    _log.step(f"Phase 3: Grounding prompts for all agents...")

    # Collect all agents
    all_agents: list[AgentSpec] = []
    for coord in proposal.coordinators:
        all_agents.extend(coord.agents)
    all_agents.extend(proposal.standalone_agents)

    _log.step(f"  Grounding {len(all_agents)} agents in parallel (max {max_workers} workers)...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(ground_agent, agent, goal, llm): agent
            for agent in all_agents
        }
        for future in as_completed(futures):
            agent = futures[future]
            try:
                future.result()
                _log.ok(f"  Grounded: {agent.name}")
            except Exception as e:
                _log.warn(f"  Grounding failed for {agent.name}: {e}")

    return proposal


# ── Phase 4: Routing Calibration ──────────────────────────────────────

CALIBRATION_SYSTEM_PROMPT = """You are calibrating the routing layer of a multi-agent system.

Given the agent topology (names + descriptions), generate test queries and
routing keywords to ensure accurate query routing.

## Your tasks

1. Generate 20-30 realistic sample queries a user might ask given their goal
2. For each query, determine which agent(s) should handle it
3. Identify queries that are ambiguous (could route to multiple agents)
4. For each agent, generate a keyword list (10-20 keywords) that would
   uniquely identify queries belonging to that agent
5. Write routing_prompt_extra instructions that help the router LLM
   distinguish between similar agents

## Output format
Return ONLY valid JSON:
```json
{
    "routing_prompt_extra": "Multi-line routing instructions...",
    "keywords": {
        "agent_name": ["keyword1", "keyword2", "..."]
    },
    "sample_queries": [
        {
            "query": "How does the auth middleware work?",
            "expected_agents": ["middleware_expert"],
            "ambiguous": false
        }
    ],
    "warnings": ["List any agents with overlapping scope that may cause misroutes"]
}
```
"""


def calibrate_routing(
    proposal: TopologyProposal,
    goal: str,
    llm: Optional[LLMClient] = None,
) -> TopologyProposal:
    """
    Phase 4: Generate keywords and routing instructions.
    """
    llm = llm or LLMClient(model=DEFAULT_MODEL)
    _log.step("Phase 4: Calibrating routing...")

    # Build agent summary for the LLM
    agent_lines = []
    for coord in proposal.coordinators:
        agent_lines.append(f"\n### {coord.name} (coordinator): {coord.description}")
        for agent in coord.agents:
            agent_lines.append(f"  - {agent.name}: {agent.description}")
            agent_lines.append(f"    files: {', '.join(agent.key_files[:5])}")
    for agent in proposal.standalone_agents:
        agent_lines.append(f"  - {agent.name}: {agent.description}")

    prompt = f"""## User's Goal
{goal}

## Agent Topology
{chr(10).join(agent_lines)}

Generate routing calibration: keywords per agent, routing_prompt_extra, and sample queries.
"""

    result = llm.complete(prompt=prompt, system=CALIBRATION_SYSTEM_PROMPT, max_tokens=8192)

    try:
        from multiagentz.utils import extract_json
        data = extract_json(str(result))
        proposal.routing_prompt_extra = data.get("routing_prompt_extra", "")
        proposal.keywords = data.get("keywords", {})

        # Log warnings
        for w in data.get("warnings", []):
            _log.warn(f"  Routing warning: {w}")

        # Log sample query coverage
        samples = data.get("sample_queries", [])
        ambiguous = sum(1 for s in samples if s.get("ambiguous"))
        _log.ok(f"  Generated {len(samples)} test queries ({ambiguous} ambiguous)")

    except Exception as e:
        _log.warn(f"Calibration parse failed: {e}")

    return proposal


# ── YAML Emitter ──────────────────────────────────────────────────────

def proposal_to_yaml(proposal: TopologyProposal) -> str:
    """Convert a TopologyProposal to a valid stack YAML string."""
    stack: dict = {
        "name": proposal.stack_name,
        "orchestration": {
            "mode": proposal.orchestration_mode,
            "max_iterations": 2,
        },
        "lead": {},
    }

    lead = stack["lead"]

    if proposal.lead_model:
        lead["model"] = proposal.lead_model
    if proposal.lead_provider:
        lead["provider"] = proposal.lead_provider
    if proposal.routing_prompt_extra:
        lead["routing_prompt_extra"] = proposal.routing_prompt_extra
    if proposal.keywords:
        # Convert keyword lists (YAML needs plain lists)
        lead["keywords"] = {
            name: sorted(kws) if isinstance(kws, list) else sorted(list(kws))
            for name, kws in proposal.keywords.items()
        }

    agents: dict = {}

    for coord in proposal.coordinators:
        coord_dict: dict = {
            "type": "coordinator",
            "description": coord.description,
        }
        if coord.model:
            coord_dict["model"] = coord.model
        if coord.provider:
            coord_dict["provider"] = coord.provider

        child_agents: dict = {}
        for agent in coord.agents:
            agent_dict: dict = {
                "repo_path": agent.repo_path,
                "description": agent.description,
                "key_files": agent.key_files,
            }
            if agent.system_prompt:
                agent_dict["system_prompt"] = agent.system_prompt
            if agent.model:
                agent_dict["model"] = agent.model
            if agent.provider:
                agent_dict["provider"] = agent.provider
            if agent.twin:
                agent_dict["twin"] = agent.twin
            child_agents[agent.name] = agent_dict

        coord_dict["agents"] = child_agents
        agents[coord.name] = coord_dict

    for agent in proposal.standalone_agents:
        agent_dict = {
            "repo_path": agent.repo_path,
            "description": agent.description,
            "key_files": agent.key_files,
        }
        if agent.system_prompt:
            agent_dict["system_prompt"] = agent.system_prompt
        if agent.model:
            agent_dict["model"] = agent.model
        if agent.provider:
            agent_dict["provider"] = agent.provider
        agents[agent.name] = agent_dict

    lead["agents"] = agents

    # Generate YAML with nice formatting
    return yaml.dump(
        stack,
        default_flow_style=False,
        sort_keys=False,
        width=120,
        allow_unicode=True,
    )


# ── Display helpers ───────────────────────────────────────────────────

def display_topology(proposal: TopologyProposal) -> str:
    """Format topology for terminal display."""
    lines = [f"\nProposed topology for [{proposal.stack_name}]:\n"]

    total_agents = 0
    total_chars = 0

    for coord in proposal.coordinators:
        lines.append(f"  {coord.name} (coordinator)")
        for i, agent in enumerate(coord.agents):
            is_last = i == len(coord.agents) - 1
            prefix = "└──" if is_last else "├──"
            chars_str = f"{agent.est_chars:,}" if agent.est_chars else "?"
            files_str = f"{agent.est_files}" if agent.est_files else "?"
            model_str = f" [{agent.model}]" if agent.model else ""
            lines.append(
                f"    {prefix} {agent.name}  "
                f"({chars_str} chars, {files_str} files){model_str}"
            )
            total_agents += 1
            total_chars += agent.est_chars

    for agent in proposal.standalone_agents:
        chars_str = f"{agent.est_chars:,}" if agent.est_chars else "?"
        lines.append(f"  {agent.name}  ({chars_str} chars)")
        total_agents += 1
        total_chars += agent.est_chars

    lines.append(f"\nTotal: {total_agents} agents, "
                  f"{len(proposal.coordinators)} coordinators, "
                  f"{total_chars:,} chars")
    lines.append("")

    return "\n".join(lines)


# ── Main pipeline ─────────────────────────────────────────────────────

def generate_stack(
    repo_paths: list[str],
    goal: str,
    output_path: Optional[str] = None,
    max_chars_per_agent: int = DEFAULT_MAX_CHARS_PER_AGENT,
    model: str = DEFAULT_MODEL,
    orchestration_mode: Optional[str] = None,
    interactive: bool = True,
) -> str:
    """
    Full 4-phase stack generation pipeline.

    Parameters
    ----------
    repo_paths : list[str]
        One or more repository paths to survey.
    goal : str
        What the user wants to accomplish with this stack.
    output_path : str, optional
        Where to write the generated YAML.
    max_chars_per_agent : int
        Target char budget per agent.
    model : str
        LLM model to use for generation phases.
    orchestration_mode : str, optional
        Override orchestration mode.
    interactive : bool
        If True, prompt for approval after Phase 2.

    Returns
    -------
    str
        The generated YAML content.
    """
    _log.init()
    llm = LLMClient(model=model)

    # ── Phase 1: Survey ──
    _log.phase(1, 4, "Repo Survey")
    manifests = []
    for rp in repo_paths:
        _log.step(f"Surveying {rp}...")
        manifest = survey_repo(rp)
        manifests.append(manifest)
        _log.ok(f"  {manifest.repo_name}: {manifest.total_files} files, "
                f"{manifest.total_chars:,} chars, "
                f"{len(manifest.boundaries)} module boundaries")

    # ── Phase 2: Topology ──
    _log.phase(2, 4, "Topology Proposal")
    proposal = propose_topology(manifests, goal, max_chars_per_agent, llm)

    if orchestration_mode:
        proposal.orchestration_mode = orchestration_mode

    # Show topology for approval
    topo_display = display_topology(proposal)
    print(topo_display)

    if interactive:
        confirm = input("Proceed to grounding? [y/e/N] (e=edit in $EDITOR) ").strip().lower()
        if confirm == "e":
            # Dump proposal as YAML for editing
            import tempfile
            import subprocess
            draft_yaml = proposal_to_yaml(proposal)
            editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "vi"))
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".yaml", prefix="maz_topo_", delete=False
                ) as f:
                    f.write("# Edit the topology below, then save and close.\n")
                    f.write("# You can add/remove agents, change file scopes, etc.\n\n")
                    f.write(draft_yaml)
                    tmp_path = f.name
                subprocess.run([editor, tmp_path], check=True)
                edited = Path(tmp_path).read_text(encoding="utf-8")
                # Re-parse as the final YAML (skip Phase 3+4 since user edited)
                _log.ok("Using edited topology")
                if output_path:
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(output_path).write_text(edited)
                    _log.ok(f"Saved to {output_path}")
                return edited
            except Exception as e:
                _log.warn(f"Edit failed: {e}")
                return ""
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        elif confirm != "y":
            _log.warn("Cancelled.")
            return ""

    # ── Phase 3: Grounding ──
    _log.phase(3, 4, "Prompt Grounding")
    proposal = ground_all_agents(proposal, goal, llm)

    # ── Phase 4: Calibration ──
    _log.phase(4, 4, "Routing Calibration")
    proposal = calibrate_routing(proposal, goal, llm)

    # ── Emit YAML ──
    yaml_content = proposal_to_yaml(proposal)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(yaml_content)
        _log.ok(f"Saved to {output_path}")
    else:
        print("\n--- Generated YAML ---\n")
        print(yaml_content)

    _log.done(f"Generated stack with "
              f"{sum(len(c.agents) for c in proposal.coordinators) + len(proposal.standalone_agents)} agents")

    return yaml_content
