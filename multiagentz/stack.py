# multiagentz/stack.py
"""
Stack loader — builds an agent hierarchy from a declarative YAML config.

Now supports orchestration configuration AND per-agent model specification:
- mode: standard | consensus | perspective
- perspective agent definitions
- max_iterations for refinement cycles
- Per-agent model/provider overrides via YAML
- cross_pollination: true to enable A/B twin agent loops
- twin: <agent_name> to link A/B pairs
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

from multiagentz.agents.base import SubAgent
from multiagentz.agents.coordinator import CoordinatorAgent
from multiagentz.agents.files import FileHandlerAgent
from multiagentz.lead import LeadAgent
from multiagentz.llm_client import LLMClient


def _create_llm_client_from_spec(spec: dict) -> Optional[LLMClient]:
    """
    Create LLMClient from agent spec if model/provider specified.
    Returns None if no custom config (agent will use default).
    
    Configuration is via YAML only - NO hardcoded API keys!
    API keys come from environment variables via llm_config.
    """
    model = spec.get("model")
    provider = spec.get("provider")
    
    # No custom config → return None (agent creates default LLMClient)
    if not model and not provider:
        return None
    
    # Create custom LLMClient (API keys from environment)
    return LLMClient(provider=provider, model=model)


def load_stack(config_path: str) -> LeadAgent:
    """
    Load a YAML stack config and return a fully wired LeadAgent.

    Parameters
    ----------
    config_path : str
        Path to the YAML stack definition file.

    Returns
    -------
    LeadAgent
        Ready-to-query lead agent with all sub-agents wired up.
    """
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Stack config not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    stack_name = config.get("name", path.stem)
    lead_config = config.get("lead", {})

    # Parse orchestration configuration (needed before building agents)
    orchestration_config = config.get("orchestration", {})
    orchestration_mode = orchestration_config.get("mode", "standard")
    cross_pollination = orchestration_config.get("cross_pollination", False)

    # Validate mode
    if orchestration_mode not in ("standard", "consensus", "perspective"):
        print(f"Warning: Unknown orchestration mode '{orchestration_mode}', defaulting to 'standard'")
        orchestration_mode = "standard"

    # Build the agent tree with per-agent model support
    agents_config = lead_config.get("agents", {})
    agents, twin_map = _build_agents(agents_config, cross_pollination)

    # Always include a files agent unless explicitly disabled
    if "files" not in agents and not config.get("disable_files_agent", False):
        agents["files"] = FileHandlerAgent()

    # Build keyword sets for pre-route heuristic
    keywords = {}
    for agent_name, kw_list in lead_config.get("keywords", {}).items():
        keywords[agent_name] = set(kw_list)

    # Build orchestration config dict
    orch_settings = {
        "max_iterations": orchestration_config.get("max_iterations", 3),
        "perspectives": orchestration_config.get("perspectives", []),
        "cross_pollination": cross_pollination,
        "twin_map": twin_map,
    }

    # Create custom LLMClient for LeadAgent if specified
    lead_llm_client = _create_llm_client_from_spec(lead_config)

    # Build the lead agent
    lead = LeadAgent(
        name=stack_name,
        agents=agents,
        routing_prompt_extra=lead_config.get("routing_prompt_extra", ""),
        keywords=keywords,
        cache_ttl_hours=config.get("cache_ttl_hours", 72),
        orchestration_mode=orchestration_mode,
        orchestration_config=orch_settings,
        llm_client=lead_llm_client,  # ← Pass custom client or None
    )

    return lead


def _build_agents(
    agents_config: dict[str, Any],
    cross_pollination: bool = False,
) -> tuple[dict[str, SubAgent | CoordinatorAgent | FileHandlerAgent], dict[str, str]]:
    """
    Recursively build agents from config dict with per-agent model support.

    Returns
    -------
    tuple[dict, dict]
        (agents_dict, twin_map)
        twin_map maps agent_name -> twin_agent_name for cross-pollination pairs.
    """
    agents = {}
    twin_map: dict[str, str] = {}

    for name, spec in agents_config.items():
        if isinstance(spec, str):
            # Shorthand: just a repo path
            agents[name] = SubAgent(name=name, repo_path=spec)
            continue

        # Parse per-agent model configuration
        llm_client = _create_llm_client_from_spec(spec)

        # Capture twin relationship
        twin_ref = spec.get("twin")
        if twin_ref:
            twin_map[name] = twin_ref

        agent_type = spec.get("type", "sub_agent")

        if agent_type == "coordinator":
            # Recursive: build child agents first
            child_specs = spec.get("agents", {})
            children, child_twins = _build_agents(child_specs, cross_pollination)
            agents[name] = CoordinatorAgent(
                name=name,
                sub_agents=children,
                description=spec.get("description", ""),
                routing_prompt_extra=spec.get("routing_prompt_extra", ""),
                max_workers=spec.get("max_workers", 8),
                llm_client=llm_client,
                twin_map=child_twins if cross_pollination else {},
            )

        elif agent_type == "files":
            agents[name] = FileHandlerAgent(
                watched_paths=spec.get("watched_paths"),
                llm_client=llm_client,
            )

        else:
            # Standard sub-agent
            agents[name] = SubAgent(
                name=name,
                repo_path=spec.get("repo_path", "."),
                description=spec.get("description", ""),
                system_prompt=spec.get("system_prompt", ""),
                key_files=spec.get("key_files", []),
                max_tokens=spec.get("max_tokens", 32768),
                max_context_chars=spec.get("max_context_chars", 600_000),
                llm_client=llm_client,
            )

    return agents, twin_map