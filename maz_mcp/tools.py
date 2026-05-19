"""
MAZ MCP Thick Tools — 6 high-level tools for multi-agent orchestration.

Tools:
    maz_configure        — Load/configure an agent stack
    maz_query            — Standard routed query
    maz_consensus        — Consensus analysis (iterative conflict resolution)
    maz_perspective      — Multi-perspective deep analysis (4-phase)
    maz_cross_pollinate  — A/B cross-pollination analysis
    maz_status           — Server status and diagnostics
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional


def register_thick_tools(mcp):
    """Register all 6 thick MCP tools on the FastMCP server."""

    from .server import (
        _resolve_stack,
        _stack_info,
        _format_error,
        _get_default_stack,
        get_config,
    )
    from .ratelimit import check_rate_limit, get_metrics

    # ─── Tool 1: maz_configure ──────────────────────────────────────

    @mcp.tool()
    def maz_configure(
        stack_yaml: str = "",
        stack_name: str = "",
    ) -> str:
        """Load or configure a multi-agent stack.

        Provide stack_yaml (inline YAML config) to load a custom agent stack.
        If neither is provided, resets to the default built-in stack.

        Returns stack info: name, agents, orchestration mode.
        """
        try:
            if stack_yaml and stack_yaml.strip():
                lead = _resolve_stack(stack_yaml)
            else:
                lead = _get_default_stack()

            info = _stack_info(lead)
            info["status"] = "loaded"
            return json.dumps(info, indent=2)

        except Exception as e:
            return json.dumps(_format_error(e), indent=2)

    # ─── Tool 2: maz_query ──────────────────────────────────────────

    @mcp.tool()
    def maz_query(
        question: str,
        stack_yaml: str = "",
        context: str = "",
    ) -> str:
        """Query the multi-agent stack with automatic routing.

        The question is routed to the best expert agent(s) based on keywords
        and LLM classification, then responses are synthesized.

        Args:
            question: The question or task to analyze.
            stack_yaml: Optional inline YAML to use a custom agent stack.
            context: Optional additional context to include with the question.
        """
        try:
            # Rate limit check
            rl = check_rate_limit("maz_query")
            if rl:
                return json.dumps(rl, indent=2)

            lead = _resolve_stack(stack_yaml)

            # Prepend context if provided
            full_question = question
            if context and context.strip():
                full_question = f"Context:\n{context}\n\nQuestion: {question}"

            t0 = time.time()
            response, agents_used = lead.query(full_question, memory=None)
            elapsed = round(time.time() - t0, 2)

            return json.dumps({
                "answer": response,
                "agents_used": agents_used,
                "mode": "standard",
                "elapsed_seconds": elapsed,
                "stack": lead.name,
            }, indent=2)

        except Exception as e:
            return json.dumps(_format_error(e), indent=2)

    # ─── Tool 3: maz_consensus ──────────────────────────────────────

    @mcp.tool()
    def maz_consensus(
        question: str,
        max_iterations: int = 3,
        stack_yaml: str = "",
        context: str = "",
    ) -> str:
        """Run consensus analysis across multiple agents.

        Iterative conflict resolution: agents provide initial answers,
        conflicts are detected, refinement questions are generated,
        agents refine their answers, and a final synthesis resolves
        remaining disagreements.

        Args:
            question: The question to analyze.
            max_iterations: Max refinement iterations (default 3).
            stack_yaml: Optional inline YAML for custom stack.
            context: Optional additional context.
        """
        try:
            rl = check_rate_limit("maz_consensus")
            if rl:
                return json.dumps(rl, indent=2)

            lead = _resolve_stack(stack_yaml)

            full_question = question
            if context and context.strip():
                full_question = f"Context:\n{context}\n\nQuestion: {question}"

            # Ensure orchestration engine is available
            from multiagentz.orchestration import OrchestrationEngine
            engine = OrchestrationEngine(lead)

            t0 = time.time()
            response, metadata = engine.execute_consensus(
                full_question,
                memory=None,
                max_iterations=max_iterations,
            )
            elapsed = round(time.time() - t0, 2)

            result = {
                "answer": response,
                "mode": "consensus",
                "elapsed_seconds": elapsed,
                "stack": lead.name,
            }

            # Include metadata if available
            if isinstance(metadata, dict):
                result["iterations"] = metadata.get("iterations", 0)
                result["conflicts_found"] = metadata.get("conflicts_found", False)
                result["agents_used"] = metadata.get("agents_used", [])

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps(_format_error(e), indent=2)

    # ─── Tool 4: maz_perspective ────────────────────────────────────

    @mcp.tool()
    def maz_perspective(
        question: str,
        agent_names: str = "",
        max_iterations: int = 3,
        bootstrap_qa: bool = True,
        stack_yaml: str = "",
        context: str = "",
    ) -> str:
        """Run multi-perspective deep analysis (4-phase).

        Phase 1: Bootstrap Q&A — each perspective asks clarifying questions
        Phase 2: Independent solution generation (parallel)
        Phase 3: LEAD review + iterative refinement
        Phase 4: Consensus synthesis

        Args:
            question: The question to analyze in depth.
            agent_names: Comma-separated agent names to use as perspectives.
                         If empty, uses all agents in the stack.
            max_iterations: Max refinement iterations (default 3).
            bootstrap_qa: Whether to run bootstrap Q&A phase (default True).
            stack_yaml: Optional inline YAML for custom stack.
            context: Optional additional context.
        """
        try:
            rl = check_rate_limit("maz_perspective")
            if rl:
                return json.dumps(rl, indent=2)

            lead = _resolve_stack(stack_yaml)

            full_question = question
            if context and context.strip():
                full_question = f"Context:\n{context}\n\nQuestion: {question}"

            # Build perspective configs from agents
            selected_agents = list(lead.agents.keys())
            if agent_names and agent_names.strip():
                selected_agents = [n.strip() for n in agent_names.split(",") if n.strip()]
                # Validate agent names
                for name in selected_agents:
                    if name not in lead.agents:
                        return json.dumps({
                            "error": True,
                            "message": f"Agent '{name}' not found. Available: {list(lead.agents.keys())}",
                        }, indent=2)

            # Filter out non-queryable agents (like FileHandlerAgent)
            from multiagentz.agents.base import SubAgent
            from multiagentz.agents.coordinator import CoordinatorAgent
            perspective_agents = [
                name for name in selected_agents
                if isinstance(lead.agents.get(name), (SubAgent, CoordinatorAgent))
            ]

            if len(perspective_agents) < 2:
                return json.dumps({
                    "error": True,
                    "message": f"Need at least 2 agents for perspective analysis. Available: {perspective_agents}",
                }, indent=2)

            perspective_configs = []
            for name in perspective_agents:
                agent = lead.agents[name]
                perspective_configs.append({
                    "name": name,
                    "agent_ref": name,
                    "role": getattr(agent, "description", name),
                    "memory_access": "shared",
                })

            from multiagentz.orchestration import OrchestrationEngine
            engine = OrchestrationEngine(lead)

            t0 = time.time()
            response, metadata = engine.execute_perspective(
                full_question,
                perspective_configs=perspective_configs,
                memory=None,
                bootstrap_qa=bootstrap_qa,
                max_iterations=max_iterations,
            )
            elapsed = round(time.time() - t0, 2)

            result = {
                "answer": response,
                "mode": "perspective",
                "perspectives_used": perspective_agents,
                "elapsed_seconds": elapsed,
                "stack": lead.name,
            }

            if isinstance(metadata, dict):
                result["phases_completed"] = metadata.get("phases_completed", 0)
                result["iterations"] = metadata.get("iterations", 0)

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps(_format_error(e), indent=2)

    # ─── Tool 5: maz_cross_pollinate ────────────────────────────────

    @mcp.tool()
    def maz_cross_pollinate(
        question: str,
        agent_a: str = "",
        agent_b: str = "",
        stack_yaml: str = "",
        context: str = "",
    ) -> str:
        """Run A/B cross-pollination analysis between two agents.

        Two agents independently analyze the question, then swap outputs.
        Each refines their answer incorporating the other's perspective.
        A final reconciliation synthesizes the best of both.

        Args:
            question: The question to cross-pollinate.
            agent_a: Name of first agent (defaults to first available).
            agent_b: Name of second agent (defaults to second available).
            stack_yaml: Optional inline YAML for custom stack.
            context: Optional additional context.
        """
        try:
            rl = check_rate_limit("maz_cross_pollinate")
            if rl:
                return json.dumps(rl, indent=2)

            lead = _resolve_stack(stack_yaml)

            full_question = question
            if context and context.strip():
                full_question = f"Context:\n{context}\n\nQuestion: {question}"

            # Select agents
            from multiagentz.agents.base import SubAgent
            from multiagentz.agents.coordinator import CoordinatorAgent
            queryable = [
                name for name, agent in lead.agents.items()
                if isinstance(agent, (SubAgent, CoordinatorAgent))
            ]

            if len(queryable) < 2:
                return json.dumps({
                    "error": True,
                    "message": f"Need at least 2 agents. Available: {queryable}",
                }, indent=2)

            name_a = agent_a.strip() if agent_a.strip() else queryable[0]
            name_b = agent_b.strip() if agent_b.strip() else queryable[1]

            if name_a not in lead.agents:
                return json.dumps({"error": True, "message": f"Agent '{name_a}' not found."}, indent=2)
            if name_b not in lead.agents:
                return json.dumps({"error": True, "message": f"Agent '{name_b}' not found."}, indent=2)

            ag_a = lead.agents[name_a]
            ag_b = lead.agents[name_b]

            t0 = time.time()

            # Phase 1: Parallel initial queries
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_a = executor.submit(ag_a.query, full_question)
                future_b = executor.submit(ag_b.query, full_question)
                response_a = future_a.result()
                response_b = future_b.result()

            # Phase 2: Swap outputs and refine
            refine_prompt_a = (
                f"Original question: {full_question}\n\n"
                f"Your initial analysis:\n{response_a}\n\n"
                f"A different expert ({name_b}) provided this analysis:\n{response_b}\n\n"
                "Refine your analysis incorporating any valid points from the other perspective. "
                "Note where you agree, disagree, or see complementary insights."
            )
            refine_prompt_b = (
                f"Original question: {full_question}\n\n"
                f"Your initial analysis:\n{response_b}\n\n"
                f"A different expert ({name_a}) provided this analysis:\n{response_a}\n\n"
                "Refine your analysis incorporating any valid points from the other perspective. "
                "Note where you agree, disagree, or see complementary insights."
            )

            with ThreadPoolExecutor(max_workers=2) as executor:
                future_ra = executor.submit(ag_a.query, refine_prompt_a)
                future_rb = executor.submit(ag_b.query, refine_prompt_b)
                refined_a = future_ra.result()
                refined_b = future_rb.result()

            # Phase 3: Synthesize
            synthesis_prompt = (
                f"Question: {full_question}\n\n"
                f"Expert A ({name_a}) refined analysis:\n{refined_a}\n\n"
                f"Expert B ({name_b}) refined analysis:\n{refined_b}\n\n"
                "Provide a comprehensive synthesis that captures the best insights from both experts. "
                "Note areas of strong agreement and any remaining differences of opinion."
            )
            final = lead._llm.complete(synthesis_prompt, system="You are an expert synthesizer.")
            elapsed = round(time.time() - t0, 2)

            return json.dumps({
                "answer": str(final),
                "mode": "cross_pollinate",
                "agent_a": name_a,
                "agent_b": name_b,
                "phases": ["parallel_query", "swap_and_refine", "synthesis"],
                "elapsed_seconds": elapsed,
                "stack": lead.name,
            }, indent=2)

        except Exception as e:
            return json.dumps(_format_error(e), indent=2)

    # ─── Tool 6: maz_status ─────────────────────────────────────────

    @mcp.tool()
    def maz_status() -> str:
        """Check MAZ MCP server status, loaded stacks, and configuration.

        Returns server info, available agents, and provider status.
        """
        try:
            config = get_config()

            # Check loaded stacks
            from .server import _stack_cache, _default_lead
            stacks_loaded = len(_stack_cache)

            result: Dict[str, Any] = {
                "server": "maz-mcp",
                "version": "0.1.0",
                "status": "running",
                "config": {
                    "default_model": config["default_model"],
                    "providers": {
                        "anthropic": "configured" if config["anthropic_key"] else "not set",
                        "openai": "configured" if config["openai_key"] else "not set",
                        "xai": "configured" if config["xai_key"] else "not set",
                        "google": "configured" if config["google_key"] else "not set",
                    },
                },
                "stacks_cached": stacks_loaded,
            }

            # Include default stack info if loaded
            if _default_lead is not None:
                result["default_stack"] = _stack_info(_default_lead)

            # Rate limiter metrics
            result["rate_limiter"] = get_metrics()

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps(_format_error(e), indent=2)
