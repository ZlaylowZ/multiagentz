# multiagentz/agents/coordinator.py
"""
CoordinatorAgent — a sub-coordinator that owns a group of SubAgents.

A coordinator
routes incoming questions to one or more of its children, queries them
in parallel, and synthesizes multi-agent responses.

Can be used as a top-level lead agent or as a nested sub-coordinator.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from multiagentz.llm_client import LLMClient, CompletionResult
from multiagentz.agents.base import SubAgent
from multiagentz.utils import extract_json, extract_routing_task
from multiagentz import log as _log


MAX_CONTINUATIONS = 6


class CoordinatorAgent:
    """
    Routes questions to child agents and synthesizes responses.

    Parameters
    ----------
    name : str
        Identifier for this coordinator (e.g., "backend", "infra").
    sub_agents : dict[str, SubAgent | CoordinatorAgent]
        Named child agents. Values can be SubAgents or nested Coordinators.
    description : str
        Short description of this coordinator's domain (used by parent routers).
    routing_prompt_extra : str
        Domain-specific routing instructions appended to the base routing prompt.
    max_workers : int
        Thread pool size for parallel sub-agent queries.
    twin_map : dict[str, str]
        Maps agent name to its twin for cross-pollination (e.g. {"A": "B", "B": "A"}).
    """

    def __init__(
        self,
        name: str,
        sub_agents: dict[str, "SubAgent | CoordinatorAgent"],
        description: str = "",
        routing_prompt_extra: str = "",
        max_workers: int = 8,
        llm_client: Optional[LLMClient] = None,
        twin_map: Optional[dict[str, str]] = None,
    ):
        self.name = name
        self.sub_agents = sub_agents
        self._description = description or f"Coordinator for {name}"
        self._routing_extra = routing_prompt_extra
        self._llm = llm_client or LLMClient()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._twin_map = twin_map or {}
        self._cross_poll_engine = None  # Lazy init
        self._last_successful_route: Optional[list[str]] = None  # Cache for fallback

    @property
    def description(self) -> str:
        return self._description

    # ── Routing ─────────────────────────────────────────────────────────

    def _routing_prompt(self) -> str:
        agent_descriptions = "\n".join(
            f"- **{name}**: {agent.description}"
            for name, agent in self.sub_agents.items()
        )

        return f"""You are a routing classifier. Your ONLY job is to output a JSON routing decision.
Do NOT answer the question itself.

Sub-agents:
{agent_descriptions}

{self._routing_extra}

## CRITICAL RULES
1. Output ONLY valid JSON — no commentary, no markdown, no explanation outside the JSON.
2. For broad/comprehensive questions that touch multiple sub-agents, route to ALL relevant ones.
   Each should receive a FOCUSED sub-question tailored to their expertise.
3. Only use agent names from the list above.

## Required JSON Format
```json
{{
    "reasoning": "brief explanation",
    "queries": [
        {{"agent": "agent_name", "question": "focused question for this agent"}}
    ]
}}
```"""

    def _route(self, question: str) -> dict:
        # Summarize/extract for routing classification
        routing_q = extract_routing_task(question)

        system = self._routing_prompt()

        # Attempt 1
        response = self._llm.complete(
            prompt=f"Question: {routing_q}",
            system=system,
            max_tokens=2048,
        )
        try:
            result = extract_json(str(response))
            # Cache successful routing decision
            self._last_successful_route = [q["agent"] for q in result.get("queries", [])]
            return result
        except (json.JSONDecodeError, IndexError):
            pass

        # Attempt 2 — retry with explicit instruction
        _log.warn(f"{self.name}: routing retry (JSON parse failed)")
        response = self._llm.complete(
            prompt=(
                f"Question: {routing_q}\n\n"
                "IMPORTANT: Respond with ONLY a JSON object. No other text."
            ),
            system=system,
            max_tokens=2048,
        )
        try:
            result = extract_json(str(response))
            # Cache successful routing decision
            self._last_successful_route = [q["agent"] for q in result.get("queries", [])]
            return result
        except (json.JSONDecodeError, IndexError):
            pass

        # Final fallback: use LAST SUCCESSFUL routing decision if available
        if self._last_successful_route:
            agents = self._last_successful_route
            _log.warn(f"{self.name}: routing failed, reusing last route")
            return {
                "reasoning": f"Parse failed, reusing last successful route: {agents}",
                "queries": [
                    {"agent": name, "question": question}
                    for name in agents
                ],
            }

        # No cached route — fall back to ALL, but warn loudly
        _log.warn(f"{self.name}: routing failed — querying ALL sub-agents")
        return {
            "reasoning": "Parse failed after retry (no cache), querying all",
            "queries": [
                {"agent": name, "question": question}
                for name in self.sub_agents
            ],
        }

    # ── Synthesis ───────────────────────────────────────────────────────

    def _synthesize(self, question: str, responses: dict[str, str]) -> str:
        responses_text = "\n\n".join(
            f"=== {agent} ===\n{resp}" for agent, resp in responses.items()
        )
        prompt = f"""Synthesize these responses into a unified answer.

Question: {question}

Sub-agent responses:
{responses_text}

Provide a coherent, integrated answer. Do not simply concatenate."""

        result = self._llm.complete(prompt=prompt, max_tokens=32768)
        parts = [str(result)]

        for _ in range(MAX_CONTINUATIONS):
            if not getattr(result, "truncated", False):
                break
            _log.warn(f"{self.name}: synthesis truncated, continuing...")
            result = self._llm.complete(
                prompt=(
                    f"Continue EXACTLY where this left off. Do not repeat.\n\n"
                    f"--- PARTIAL ---\n{parts[-1][-2000:]}\n--- END ---\n\nContinue:"
                ),
                max_tokens=32768,
            )
            parts.append(str(result))

        full = "\n".join(parts)
        if getattr(result, "truncated", False):
            full += "\n\n---\n⚠️ **Response truncated.**"
        return full

    # ── Cross-pollination ───────────────────────────────────────────────

    def _get_cross_poll_engine(self):
        """Lazy-init cross-pollination engine."""
        if self._cross_poll_engine is None:
            from multiagentz.orchestration import CrossPollinationEngine
            self._cross_poll_engine = CrossPollinationEngine(
                coordinator_llm=self._llm,
                max_workers=self._executor._max_workers,
            )
        return self._cross_poll_engine

    def _find_twin_for_routed(self, routed_agents: list[str]) -> Optional[tuple[str, str]]:
        """
        Check if any routed agent has a twin that should trigger cross-pollination.

        Returns (agent_name, twin_name) if a twin pair is found, else None.
        Only triggers when routing hits ONE agent that has a twin — the twin
        is implicitly included for cross-pollination.
        """
        if not self._twin_map:
            return None

        for agent_name in routed_agents:
            twin_name = self._twin_map.get(agent_name)
            if twin_name and twin_name in self.sub_agents:
                return (agent_name, twin_name)

        return None

    # ── Main entry ──────────────────────────────────────────────────────

    def query(self, question: str) -> str:
        t0 = time.time()
        routing = self._route(question)

        queries = routing.get("queries", [])
        routed_agents = [q["agent"] for q in queries]
        _log.coord(self.name, f"→ {', '.join(routed_agents)}")

        # Check for cross-pollination twin pair
        twin_pair = self._find_twin_for_routed(routed_agents)
        if twin_pair:
            a_name, b_name = twin_pair
            _log.coord(self.name, f"Cross-poll: {a_name} ↔ {b_name}")
            engine = self._get_cross_poll_engine()
            result, _metadata = engine.execute(
                question=question,
                agent_a=self.sub_agents[a_name],
                agent_b=self.sub_agents[b_name],
                coordinator_name=self.name,
            )
            elapsed = time.time() - t0
            _log.coord(self.name, f"Done ({elapsed:.1f}s)")
            return result

        # Standard path: parallel query + synthesis
        responses: dict[str, str] = {}

        def _query_one(spec: dict):
            agent_name = spec["agent"]
            q = spec["question"]
            if agent_name in self.sub_agents:
                return agent_name, self.sub_agents[agent_name].query(q)
            _log.warn(f"{self.name}: unknown agent '{agent_name}'")
            return None, None

        if queries:
            futures = [self._executor.submit(_query_one, s) for s in queries]
            for f in futures:
                name, resp = f.result()
                if name:
                    responses[name] = resp

        if not responses:
            _log.warn(f"{self.name}: no agents responded")
            return "No sub-agents responded."

        if len(responses) == 1:
            elapsed = time.time() - t0
            _log.coord(self.name, f"Done ({elapsed:.1f}s)")
            return next(iter(responses.values()))

        _log.coord(self.name, "Synthesizing...")
        result = self._synthesize(question, responses)
        elapsed = time.time() - t0
        _log.coord(self.name, f"Done ({elapsed:.1f}s)")
        return result
