# multiagentz/lead.py
"""
LeadAgent — top-level orchestrator for a multi-agent stack.

Top-level orchestrator pattern:
  classify → route (with keyword hints + LLM + post-route guard) → parallel query → synthesize → cache

Supports advanced orchestration modes:
  - standard: existing behavior
  - consensus: iterative conflict resolution
  - perspective: multi-perspective solution generation
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from multiagentz.cache import PersistentCache
from multiagentz.llm_client import LLMClient, CompletionResult
from multiagentz.utils import extract_json, extract_routing_task
from multiagentz import log as _log

# Type-only import to avoid circular deps
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from multiagentz.memory import SessionMemory


_ROUTING_LOG = Path(".maz_routing.jsonl")
MAX_CONTINUATIONS = 6


class LeadAgent:
    """
    Top-level coordinator that classifies, routes, and synthesizes.

    Parameters
    ----------
    name : str
        Stack name (used in logs and prompts).
    agents : dict
        Named child agents (SubAgent, CoordinatorAgent, or FileHandlerAgent).
    routing_prompt_extra : str
        Domain-specific routing instructions.
    keywords : dict[str, set[str]]
        Per-agent keyword sets for pre-route heuristic.
    cache_ttl_hours : int
        Default cache TTL.
    orchestration_mode : str
        Orchestration mode: "standard", "consensus", or "perspective"
    orchestration_config : dict
        Configuration for orchestration modes
    """

    def __init__(
        self,
        name: str,
        agents: dict,
        routing_prompt_extra: str = "",
        keywords: Optional[dict[str, set[str]]] = None,
        cache_ttl_hours: int = 72,
        orchestration_mode: str = "standard",
        orchestration_config: Optional[dict] = None,
        llm_client: Optional[LLMClient] = None
    ):
        self.name = name
        self.agents = agents
        self._routing_extra = routing_prompt_extra
        self._keywords = keywords or {}
        self._llm = llm_client or LLMClient()
        self._executor = ThreadPoolExecutor(max_workers=8)
        self._cache = PersistentCache(default_ttl_hours=cache_ttl_hours)
        self._recent: list[tuple[str, str]] = []
        self._max_recent = 5
        self.brief_mode = False

        # Orchestration
        self.orchestration_mode = orchestration_mode
        self.orchestration_config = orchestration_config or {}
        self._orchestration_engine = None  # Lazy init
        self._lead_sub_manager = None  # Lazy init
        self._last_successful_route: Optional[list[str]] = None  # Cache for fallback

        _log.lead(self.name, f"Agents: {', '.join(self.agents.keys())}")
        _log.lead(self.name, f"LLM: {self._llm.provider}/{self._llm.model}")
        _log.lead(self.name, f"Mode: {orchestration_mode}")

    # ── Orchestration integration ──────────────────────────────────────

    @property
    def orchestration(self):
        """Lazy-load orchestration engine."""
        if self._orchestration_engine is None:
            from multiagentz.orchestration import OrchestrationEngine
            self._orchestration_engine = OrchestrationEngine(self)
        return self._orchestration_engine
    
    @property
    def lead_sub(self):
        """Lazy-load LEAD_SUB manager."""
        if self._lead_sub_manager is None:
            from multiagentz.orchestration import LEADSUBPromotion
            self._lead_sub_manager = LEADSUBPromotion(self)
        return self._lead_sub_manager

    # ── Pre-route heuristic ────────────────────────────────────────────

    def _pre_route_hint(self, question: str) -> Optional[str]:
        """Fast keyword match. Returns single agent name or None."""
        q = question.lower()
        hits = {
            name: any(kw in q for kw in kws)
            for name, kws in self._keywords.items()
        }
        matched = [n for n, hit in hits.items() if hit]
        if len(matched) == 1:
            return matched[0]
        return None

    # ── Routing audit ──────────────────────────────────────────────────

    @staticmethod
    def _log_routing(question: str, hint: Optional[str], routed_to: list[str],
                     reasoning: str):
        entry = {
            "ts": datetime.now().isoformat(),
            "question": question[:200],
            "hint": hint,
            "routed_to": routed_to,
            "reasoning": reasoning[:200],
        }
        try:
            with open(_ROUTING_LOG, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    # ── Similarity detection ───────────────────────────────────────────

    def _check_similar(self, question: str, threshold: float = 0.85) -> tuple[bool, str]:
        q = question.lower().strip()
        for prev_q, prev_r in self._recent:
            if SequenceMatcher(None, q, prev_q.lower()).ratio() > threshold:
                return True, prev_r
        return False, ""

    def _track_recent(self, question: str, response: str):
        self._recent.append((question, response))
        if len(self._recent) > self._max_recent:
            self._recent.pop(0)

    # ── Routing prompt ─────────────────────────────────────────────────

    def _build_routing_prompt(self, hint: Optional[str] = None) -> str:
        agent_list = "\n".join(
            f"- **{name}**: {agent.description}"
            for name, agent in self.agents.items()
        )

        hint_section = ""
        if hint:
            hint_section = f"""
## Pre-Route Hint
Keyword analysis strongly suggests this belongs to **{hint}**.
Override ONLY if you have a clear reason. State "override" explicitly."""

        brevity = ""
        if self.brief_mode:
            brevity = "\n## Note: User wants BRIEF responses (2-3 paragraphs max)."

        return f"""You are a routing classifier for the **{self.name}** multi-agent system.
Your ONLY job is to output a JSON routing decision. Do NOT answer the question itself.

## Available Agents
{agent_list}

{self._routing_extra}
{hint_section}{brevity}

## CRITICAL RULES
1. Output ONLY valid JSON — no commentary, no markdown, no explanation outside the JSON.
2. For broad/comprehensive questions that span multiple agent domains, route to ALL relevant agents.
   Each agent should receive a FOCUSED sub-question tailored to their expertise.
3. For simple questions, one agent usually suffices.
4. Only use agent names from the Available Agents list above.

## Required JSON Format
```json
{{
    "reasoning": "Brief explanation of routing decision",
    "queries": [
        {{"agent": "agent_name", "question": "specific focused question for this agent"}}
    ],
    "can_answer_directly": false,
    "direct_answer": ""
}}
```"""

    # ── Route ──────────────────────────────────────────────────────────

    def _route_query(self, question: str, memory_context: str = "") -> dict:
        hint = self._pre_route_hint(question)
        if hint:
            _log.detail(f"Pre-route hint: {hint}")

        # Extract routing-friendly version of the question
        routing_q = extract_routing_task(question)

        prompt = f"User question:\n{routing_q}"
        if memory_context:
            prompt = f"{memory_context}\n\n{prompt}"

        system = self._build_routing_prompt(hint=hint)

        # Attempt 1
        response = self._llm.complete(prompt=prompt, system=system, max_tokens=2048)
        parsed = None
        try:
            parsed = extract_json(str(response))
            self._last_successful_route = [q["agent"] for q in parsed.get("queries", [])]
        except (json.JSONDecodeError, IndexError):
            pass

        # Attempt 2 — retry with very explicit instruction
        if parsed is None:
            _log.warn("Routing retry (JSON parse failed)")
            retry_prompt = (
                f"{prompt}\n\n"
                "IMPORTANT: You MUST respond with ONLY a JSON object. "
                "No other text. The JSON must have keys: reasoning, queries, can_answer_directly."
            )
            response = self._llm.complete(prompt=retry_prompt, system=system, max_tokens=2048)
            try:
                parsed = extract_json(str(response))
                self._last_successful_route = [q["agent"] for q in parsed.get("queries", [])]
            except (json.JSONDecodeError, IndexError):
                pass

        # Final fallback — use cached route, hint, or broadcast
        if parsed is None:
            if self._last_successful_route:
                fallback_agents = self._last_successful_route
                reasoning = f"Routing parse failed — reusing last successful route: {fallback_agents}"
            elif hint:
                fallback_agents = [hint]
                reasoning = f"Routing parse failed — defaulting to hint: {hint}"
            else:
                fallback_agents = list(self.agents.keys())
                reasoning = f"Routing parse failed (no cache, no hint) — broadcasting to ALL: {fallback_agents}"
            _log.warn(reasoning)
            parsed = {
                "reasoning": reasoning,
                "queries": [{"agent": a, "question": question} for a in fallback_agents],
                "can_answer_directly": False,
            }

        # Post-route guard: enforce hint if LLM contradicts without override
        queries = parsed.get("queries", [])
        if hint and queries:
            routed = {q["agent"] for q in queries}
            reasoning = parsed.get("reasoning", "").lower()
            override_signals = ["override", "actually", "both relate", "cross-cutting"]
            if not any(sig in reasoning for sig in override_signals):
                if hint not in routed:
                    parsed["queries"] = [{"agent": hint, "question": question}]
                    _log.detail(f"Post-route guard: override with hint '{hint}'")

        # Audit log
        routed_to = [q["agent"] for q in parsed.get("queries", [])]
        self._log_routing(question, hint, routed_to, parsed.get("reasoning", ""))

        return parsed

    # ── Knowledge query ────────────────────────────────────────────────

    def _query_knowledge(self, queries: list, user_question: str,
                         memory_context: str) -> tuple[str, list[str]]:
        responses: dict[str, str] = {}
        agents_used: list[str] = []

        def _query_one(spec):
            name = spec["agent"]
            q = spec["question"]
            if name in self.agents:
                _log.detail(f"Dispatching to: {name}")
                return name, self.agents[name].query(q)
            return None, None

        if queries:
            futures = [self._executor.submit(_query_one, s) for s in queries]
            for f in futures:
                try:
                    name, resp = f.result()
                    if name:
                        agents_used.append(name)
                        responses[name] = resp
                except Exception as e:
                    _log.error(f"Agent query failed: {e}")
                    continue

        if not responses:
            return "No agents responded.", agents_used
        if len(responses) == 1:
            return next(iter(responses.values())), agents_used
        return self._synthesize(user_question, responses, memory_context), agents_used

    # ── Synthesis ──────────────────────────────────────────────────────

    def _synthesize(self, question: str, responses: dict, memory_context: str = "") -> str:
        # Cap each response so concatenation doesn't exceed the token limit
        max_per_resp = 300_000 // max(len(responses), 1)
        capped = {}
        for agent, resp in responses.items():
            if len(resp) > max_per_resp:
                _log.warn(f"Truncating {agent} response from {len(resp):,} to {max_per_resp:,} chars for synthesis")
                resp = resp[:max_per_resp] + f"\n\n[... truncated from {len(resp):,} chars ...]"
            capped[agent] = resp

        responses_text = "\n\n".join(
            f"=== {agent} ===\n{resp}" for agent, resp in capped.items()
        )
        prompt = f"""Synthesize these responses into a unified answer.

Original question: {question}

Sub-agent responses:
{responses_text}

Provide a coherent, integrated answer. Do not simply concatenate."""

        if memory_context:
            prompt = f"{memory_context}\n\n{prompt}"

        result = self._llm.complete(prompt=prompt, max_tokens=32768)
        return self._continue_if_truncated(prompt, result)
    
    def _continue_if_truncated(self, prompt: str, result: CompletionResult) -> str:
        """Handle truncated responses with continuation."""
        parts = [str(result)]

        for _ in range(MAX_CONTINUATIONS):
            if not getattr(result, "truncated", False):
                break
            _log.warn("Response truncated, continuing...")
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

    # ── Main entry point ───────────────────────────────────────────────

    def query(self, user_question: str,
              memory: Optional["SessionMemory"] = None) -> tuple[str, list[str]]:
        """
        Main query entry point.
        
        Routes based on orchestration_mode:
        - standard: existing routing + synthesis
        - consensus: iterative conflict resolution
        - perspective: multi-perspective solution generation (requires explicit call)
        """
        # Check for LEAD_SUB routing
        if self.lead_sub.current_lead_sub:
            if self.lead_sub.should_route_to_lead_sub(user_question):
                lead_sub_name = self.lead_sub.current_lead_sub
                _log.lead(self.name, f"→ LEAD_SUB: {lead_sub_name}")
                response = self.agents[lead_sub_name].query(user_question)
                return response, [lead_sub_name]
        
        memory_context = memory.get_context() if memory else ""

        # Similarity check
        is_similar, prev = self._check_similar(user_question)
        if is_similar:
            _log.detail("Similar question — returning cached answer")
            return (
                f"I just answered a similar question:\n\n{prev[:2000]}...\n\n"
                "Would you like me to elaborate?",
                [],
            )

        # Cache check
        cached = self._cache.get(user_question, memory_context)
        if cached:
            _log.detail("Cache hit")
            return cached

        # Route based on orchestration mode
        if self.orchestration_mode == "consensus":
            result, metadata = self.orchestration.execute_consensus(
                user_question,
                memory=memory,
                max_iterations=self.orchestration_config.get("max_iterations", 3)
            )
            agents_used = metadata.get("agents_used", [])

            # Cache
            ttl = self._compute_ttl(user_question, result, agents_used)
            self._cache.set(user_question, result, agents_used, memory_context, ttl_hours=ttl)
            self._track_recent(user_question, result)

            return result, agents_used

        if self.orchestration_mode == "perspective":
            # Perspective mode: broadcast to ALL coordinators as independent
            # perspectives, then synthesize via the orchestration engine.
            coordinator_names = [
                name for name, agent in self.agents.items()
                if hasattr(agent, "sub_agents")  # CoordinatorAgent
            ]
            if not coordinator_names:
                coordinator_names = list(self.agents.keys())

            perspective_configs = [
                {
                    "name": f"perspective_{name}",
                    "agent_ref": name,
                    "memory_access": "shared",
                    "role": f"Perspective from {name} coordinator group",
                }
                for name in coordinator_names
            ]

            _log.lead(self.name, f"Perspective mode: {len(coordinator_names)} perspectives")
            result, metadata = self.query_perspective(
                user_question,
                perspective_configs,
                memory=memory,
                bootstrap_qa=True,
            )
            agents_used = coordinator_names

            # Cache
            ttl = self._compute_ttl(user_question, result, agents_used)
            self._cache.set(user_question, result, agents_used, memory_context, ttl_hours=ttl)
            self._track_recent(user_question, result)

            return result, agents_used

        # Standard mode
        routing = self._route_query(user_question, memory_context)

        if routing.get("can_answer_directly"):
            return routing.get("direct_answer", ""), []

        queries = routing.get("queries", [])
        if not queries:
            first_agent = next(iter(self.agents))
            queries = [{"agent": first_agent, "question": user_question}]

        _log.lead(self.name, f"→ {', '.join(q['agent'] for q in queries)}")

        result, agents_used = self._query_knowledge(queries, user_question, memory_context)

        # Cache with dynamic TTL
        ttl = self._compute_ttl(user_question, result, agents_used)
        self._cache.set(user_question, result, agents_used, memory_context, ttl_hours=ttl)

        self._track_recent(user_question, result)

        return result, agents_used

    def query_standard(self, user_question: str,
                       memory: Optional["SessionMemory"] = None) -> tuple[str, list[str]]:
        """
        Force standard routing regardless of orchestration_mode.

        Used by _bootstrap_perspectives to answer Q&A questions without
        recursively re-entering perspective mode.
        """
        memory_context = memory.get_context() if memory else ""

        routing = self._route_query(user_question, memory_context)

        if routing.get("can_answer_directly"):
            return routing.get("direct_answer", ""), []

        queries = routing.get("queries", [])
        if not queries:
            first_agent = next(iter(self.agents))
            queries = [{"agent": first_agent, "question": user_question}]

        _log.detail(f"Standard routing to: {', '.join(q['agent'] for q in queries)}")

        result, agents_used = self._query_knowledge(queries, user_question, memory_context)
        return result, agents_used

    def query_perspective(
        self,
        question: str,
        perspective_configs: list[dict],
        memory: Optional["SessionMemory"] = None,
        bootstrap_qa: bool = True,
    ) -> tuple[str, dict]:
        """
        Execute perspective-based orchestration.
        
        This is explicitly called (not automatic like consensus mode).
        
        Parameters
        ----------
        question : str
            Task/question for perspectives to solve
        perspective_configs : list[dict]
            Perspective agent configurations
        memory : SessionMemory, optional
            Conversation history
        bootstrap_qa : bool
            Whether to run Q&A alignment phase
            
        Returns
        -------
        tuple[str, dict]
            (final_answer, metadata)
        """
        return self.orchestration.execute_perspective(
            question,
            perspective_configs,
            memory=memory,
            bootstrap_qa=bootstrap_qa,
            max_iterations=self.orchestration_config.get("max_iterations", 3)
        )

    # ── TTL ────────────────────────────────────────────────────────────

    def _compute_ttl(self, question: str, response: str, agents_used: list) -> int:
        score = 0
        if len(agents_used) > 1:
            score += 2
        overview_terms = [
            "overview", "explain", "tell me about", "what is", "how does",
            "describe", "summarize", "architecture",
        ]
        if any(t in question.lower() for t in overview_terms):
            score += 2
        if len(response) > 5000:
            score += 2
        elif len(response) > 2000:
            score += 1
        if score >= 4:
            return 168  # 7 days
        if score >= 2:
            return 72   # 3 days
        return 24
