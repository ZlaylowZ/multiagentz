# multiagentz/orchestration.py
"""
Orchestration patterns for multi-agent systems.

Implements advanced coordination strategies:
- Consensus synthesis: iterative conflict resolution between agents
- Perspective mode: independent solution generation with convergence
- Cross-pollination: A/B twin agents exchange outputs for iterative refinement
- LEAD_SUB promotion: designate primary implementation coordinator

Maps to the documented orchestration pattern:
  LEAD (LeadAgent) coordinates multiple SUBS (perspective agents)
  - SUB_MEM: agent with shared memory/context
  - SUB_INC: clean-slate incognito agent
  - Iterative Q&A bootstrapping
  - Parallel solution generation
  - LEAD review and refinement
  - Consensus synthesis
  - LEAD_SUB promotion for implementation

Cross-pollination pattern (A/B twin agents):
  1. Same prompt dispatched to both A and B in parallel
  2. Outputs swapped: A receives B's output, B receives A's
  3. Each refines incorporating counterpart's perspective
  4. Coordinator reconciles the two refined outputs
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Optional

from multiagentz.llm_client import LLMClient
from multiagentz.utils import parse_json_response
from multiagentz import log

if TYPE_CHECKING:
    from multiagentz.lead import LeadAgent
    from multiagentz.memory import SessionMemory


class OrchestrationEngine:
    """
    Advanced multi-agent coordination strategies.

    Extends LeadAgent with consensus building and perspective-based workflows.
    """

    def __init__(self, lead: "LeadAgent"):
        self.lead = lead
        self._llm = lead._llm  # Use lead's configured LLM client

    # ══════════════════════════════════════════════════════════════════
    # CONSENSUS SYNTHESIS MODE
    # ══════════════════════════════════════════════════════════════════

    def execute_consensus(
        self,
        question: str,
        memory: Optional["SessionMemory"] = None,
        max_iterations: int = 3,
    ) -> tuple[str, dict]:
        """
        Execute consensus synthesis mode.

        Flow:
        1. Route to multiple agents (existing lead behavior)
        2. Identify contradictions between responses
        3. Generate refinement questions
        4. Re-query agents with refinements
        5. Iterate until consensus or max iterations
        6. Synthesize final answer noting conflicts/resolutions
        """
        memory_context = memory.get_context() if memory else ""

        # Initial routing and query (use existing lead logic)
        routing = self.lead._route_query(question, memory_context)
        queries = routing.get("queries", [])

        if not queries:
            first_agent = next(iter(self.lead.agents))
            queries = [{"agent": first_agent, "question": question}]

        log.step(f"Consensus: initial routing to {[q['agent'] for q in queries]}")

        # Get initial responses
        responses = self._parallel_query(queries)

        if len(responses) <= 1:
            single_response = next(iter(responses.values()))
            return single_response, {
                "mode": "consensus",
                "iterations": 0,
                "conflicts_found": [],
                "consensus_achieved": True,
                "agents_used": list(responses.keys()),
                "note": "Single agent response, no consensus needed"
            }

        # Iterative consensus building
        iteration = 0
        all_conflicts = []

        while iteration < max_iterations:
            conflict_analysis = self._analyze_conflicts(question, responses)

            has_conflicts = conflict_analysis.get("has_conflicts", False)
            conflicts = conflict_analysis.get("conflicts", [])

            if conflicts:
                all_conflicts.extend(conflicts)

            if not has_conflicts:
                log.ok(f"Consensus achieved after {iteration} iterations")
                break

            log.step(f"Consensus iteration {iteration + 1}: {len(conflicts)} conflicts detected")

            refinements = self._generate_refinements(question, responses, conflicts)
            responses = self._parallel_query(refinements)
            iteration += 1

        final_answer = self._synthesize_with_conflicts(
            question, responses, all_conflicts, iteration
        )

        return final_answer, {
            "mode": "consensus",
            "iterations": iteration,
            "conflicts_found": all_conflicts,
            "consensus_achieved": not conflict_analysis.get("has_conflicts", True),
            "agents_used": list(responses.keys()),
        }

    def _analyze_conflicts(self, question: str, responses: dict[str, str]) -> dict:
        """Identify contradictions between agent responses."""
        responses_text = "\n\n".join(
            f"=== {agent} ===\n{resp}" for agent, resp in responses.items()
        )

        prompt = f"""Analyze these responses for contradictions and conflicts.

Original question: {question}

Agent responses:
{responses_text}

Identify:
1. Direct contradictions (agents disagree on facts)
2. Approach conflicts (different recommended strategies)
3. Missing perspectives (one agent covers something others don't)

Respond with JSON only:
{{
    "has_conflicts": bool,
    "conflicts": [
        {{
            "type": "contradiction|approach|coverage",
            "agents": ["agent1", "agent2"],
            "description": "brief description",
            "specifics": "detailed conflict explanation"
        }}
    ],
    "consensus_points": ["points where all agents agree"]
}}"""

        result = self._llm.complete(
            prompt=prompt,
            system="You detect contradictions and conflicts between agent responses.",
            max_tokens=8192
        )

        return parse_json_response(str(result))

    def _generate_refinements(
        self, question: str, responses: dict[str, str], conflicts: list[dict]
    ) -> list[dict]:
        """Generate refinement questions to resolve conflicts."""
        conflicts_text = "\n".join(
            f"- {c.get('description', '')} ({', '.join(c.get('agents', []))})"
            for c in conflicts
        )

        prompt = f"""Original question: {question}

Conflicts detected:
{conflicts_text}

Generate specific refinement questions for each agent to resolve these conflicts.
Focus on clarifying assumptions, edge cases, and rationale.

Respond with JSON only:
{{
    "queries": [
        {{
            "agent": "agent_name",
            "question": "specific clarification question"
        }}
    ]
}}"""

        result = self._llm.complete(
            prompt=prompt,
            system="You generate targeted refinement questions to resolve conflicts.",
            max_tokens=4096
        )

        parsed = parse_json_response(str(result))
        return parsed.get("queries", [])

    def _synthesize_with_conflicts(
        self, question: str, responses: dict[str, str], conflicts: list, iterations: int
    ) -> str:
        """Final synthesis explicitly noting conflicts and resolutions."""
        responses_text = "\n\n".join(
            f"=== {agent} ===\n{resp}" for agent, resp in responses.items()
        )

        conflicts_text = "None - agents in full agreement" if not conflicts else "\n".join(
            f"- {c.get('description', '')} (resolved in iteration {i+1})"
            for i, c in enumerate(conflicts)
        )

        prompt = f"""Synthesize a unified answer from these agent responses.

Original question: {question}

Agent responses (after {iterations} refinement iterations):
{responses_text}

Conflicts identified and resolved:
{conflicts_text}

Provide a coherent answer that:
1. Integrates all agent perspectives
2. Explicitly notes where agents initially disagreed
3. Explains how conflicts were resolved
4. Highlights consensus points

Be transparent about the consensus-building process."""

        result = self.lead._llm.complete(prompt=prompt, max_tokens=32768)
        return self.lead._continue_if_truncated(prompt, result)

    # ══════════════════════════════════════════════════════════════════
    # PERSPECTIVE MODE (Full Orchestration Pattern)
    # ══════════════════════════════════════════════════════════════════

    def execute_perspective(
        self,
        question: str,
        perspective_configs: list[dict],
        memory: Optional["SessionMemory"] = None,
        bootstrap_qa: bool = True,
        max_iterations: int = 3,
    ) -> tuple[str, dict]:
        """
        Execute perspective-based orchestration pattern.

        All perspectives run IN PARALLEL within each phase.

        Flow:
        1. [Optional] Bootstrap Q&A: each SUB asks questions, LEAD answers
        2. Each SUB generates independent solution proposal
        3. LEAD reviews each solution, provides feedback
        4. SUBS refine based on feedback
        5. Iterate until convergence
        6. Consensus synthesis
        """
        total_phases = 4 if bootstrap_qa else 3
        log.init()

        p_names = [p["name"] for p in perspective_configs]
        log.step(f"Starting perspective orchestration with {len(perspective_configs)} perspectives")
        log.detail(f"Perspectives: {', '.join(p_names)}")

        # Phase 1: Bootstrap Q&A (if enabled)
        aligned_contexts = {}
        if bootstrap_qa:
            log.phase(1, total_phases, "Bootstrap Q&A Alignment")
            aligned_contexts = self._bootstrap_perspectives(
                question, perspective_configs, memory
            )
        else:
            aligned_contexts = {p["name"]: "" for p in perspective_configs}

        # Phase 2: Independent solution generation
        phase_num = 2 if bootstrap_qa else 1
        log.phase(phase_num, total_phases, "Independent Solution Generation (parallel)")
        solutions = self._generate_independent_solutions(
            question, perspective_configs, aligned_contexts
        )

        # Phase 3: Iterative refinement with LEAD feedback
        phase_num += 1
        log.phase(phase_num, total_phases, f"LEAD Review + Refinement (up to {max_iterations} iterations)")
        converged_solutions, convergence_meta = self._iterate_to_convergence(
            question, solutions, perspective_configs, max_iterations
        )

        # Phase 4: Consensus synthesis
        phase_num += 1
        log.phase(phase_num, total_phases, "Final Consensus Synthesis")
        final_answer, synthesis_meta = self._synthesize_perspectives(
            question, converged_solutions, convergence_meta
        )

        log.done(f"{len(perspective_configs)} perspectives synthesized")

        return final_answer, {
            "mode": "perspective",
            "perspectives": p_names,
            "bootstrap_used": bootstrap_qa,
            **convergence_meta,
            **synthesis_meta,
        }

    def _bootstrap_perspectives(
        self,
        question: str,
        perspective_configs: list[dict],
        memory: Optional["SessionMemory"],
    ) -> dict[str, str]:
        """
        Q&A alignment phase — all perspectives run IN PARALLEL.

        Each perspective agent asks questions -> LEAD answers -> iterate until aligned.
        """
        max_qa_turns = 3

        def _align_one(p_config: dict) -> tuple[str, str]:
            p_name = p_config["name"]
            agent_ref = p_config["agent_ref"]
            memory_access = p_config.get("memory_access", "shared")

            if agent_ref not in self.lead.agents:
                log.warn(f"Agent '{agent_ref}' not found for '{p_name}'")
                return p_name, ""

            agent = self.lead.agents[agent_ref]

            if memory_access == "none":
                initial_context = ""
                memory_note = "You have NO access to previous conversation history."
            else:
                initial_context = memory.get_context() if memory else ""
                memory_note = "You have access to conversation history above."

            briefing = f"""## Assignment

You will be assisting with: {question}

{memory_note}

I'll provide context and background information. After reviewing, enter Q&A mode:
- Ask questions to clarify requirements, constraints, architecture
- I (the LEAD agent) will provide authoritative answers
- Continue until you have sufficient understanding
- Signal "ALIGNED" when ready to proceed with solution design

Begin by asking your first round of questions."""

            qa_history = [
                f"## Initial Context\n{initial_context}" if initial_context else "",
                briefing,
            ]

            log.step(f"{p_name}  Starting Q&A alignment...")

            aligned = False
            turn = 0

            while not aligned and turn < max_qa_turns:
                qa_prompt = "\n\n".join(filter(None, qa_history))
                agent_response = agent.query(qa_prompt)

                if "ALIGNED" in agent_response.upper() or "READY TO PROCEED" in agent_response.upper():
                    log.ok(f"{p_name}  Aligned ({turn + 1} turns)")
                    aligned = True
                    break

                questions = self._extract_questions(agent_response)

                if not questions:
                    log.ok(f"{p_name}  Aligned (no questions)")
                    aligned = True
                    break

                log.step(f"{p_name}  Turn {turn + 1}/{max_qa_turns}: {len(questions)} question(s) — LEAD answering...")

                lead_answers = []
                for q in questions:
                    answer, _ = self.lead.query_standard(q, memory=memory)
                    lead_answers.append(f"**Q:** {q}\n**A:** {answer}")

                qa_history.append(f"\n**{p_name} Questions:**\n{agent_response}")
                qa_history.append(f"\n**LEAD Answers:**\n" + "\n\n".join(lead_answers))
                turn += 1

            if not aligned:
                log.warn(f"{p_name}  Max Q&A turns ({max_qa_turns}) reached — moving on")

            return p_name, "\n\n".join(filter(None, qa_history))

        # Run all perspectives in parallel
        log.step(f"Aligning {len(perspective_configs)} perspectives in parallel...")
        contexts: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=len(perspective_configs)) as pool:
            futures = {pool.submit(_align_one, pc): pc for pc in perspective_configs}
            for fut in as_completed(futures):
                p_name, ctx = fut.result()
                contexts[p_name] = ctx

        log.ok(f"All {len(contexts)} perspectives aligned")
        return contexts

    def _extract_questions(self, text: str) -> list[str]:
        """Extract questions from agent response."""
        lines = text.split("\n")
        questions = []

        for line in lines:
            line = line.strip()
            if line.endswith("?"):
                cleaned = re.sub(r"^[\d\-\*\.\)]+\s*", "", line)
                cleaned = re.sub(r"^\*\*.*?\*\*:?\s*", "", cleaned)
                if len(cleaned) > 10:
                    questions.append(cleaned)

        return questions[:10]

    def _generate_independent_solutions(
        self,
        question: str,
        perspective_configs: list[dict],
        aligned_contexts: dict[str, str],
    ) -> dict[str, str]:
        """Each perspective agent generates independent solution — IN PARALLEL."""

        def _solve_one(p_config: dict) -> tuple[str, str]:
            p_name = p_config["name"]
            agent_ref = p_config["agent_ref"]

            if agent_ref not in self.lead.agents:
                return p_name, ""

            agent = self.lead.agents[agent_ref]
            context = aligned_contexts.get(p_name, "")

            prompt = f"""{context}

## Task

Generate your complete proposed solution for:

{question}

Provide:
1. Overall approach and architecture
2. Implementation strategy
3. Key design decisions and rationale
4. Potential challenges and mitigations
5. Integration considerations

Be specific and actionable. This is a solution proposal, not just analysis."""

            log.step(f"{p_name}  Generating solution...")
            solution = agent.query(prompt)
            log.ok(f"{p_name}  Solution ready")
            return p_name, solution

        # Run all perspectives in parallel
        solutions: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=len(perspective_configs)) as pool:
            futures = {pool.submit(_solve_one, pc): pc for pc in perspective_configs}
            for fut in as_completed(futures):
                p_name, sol = fut.result()
                if sol:
                    solutions[p_name] = sol

        log.ok(f"All {len(solutions)} solutions generated")
        return solutions

    def _iterate_to_convergence(
        self,
        question: str,
        solutions: dict[str, str],
        perspective_configs: list[dict],
        max_iterations: int,
    ) -> tuple[dict[str, str], dict]:
        """
        LEAD reviews solutions, provides feedback, perspectives refine.

        Iterate until convergence or max iterations.
        Refinements within each iteration run IN PARALLEL.
        """
        iteration = 0
        feedback_history = []
        all_converged = True  # safe default if max_iterations is 0

        while iteration < max_iterations:
            log.step(f"LEAD reviewing all solutions (iteration {iteration + 1}/{max_iterations})...")

            # LEAD reviews all solutions
            review = self._llm.complete(
                prompt=f"""Review these solution proposals.

Original task: {question}

Proposed solutions:
{self._format_solutions(solutions)}

For each solution, provide:
1. Strengths (what's done well)
2. Weaknesses (gaps, risks, issues)
3. Whether refinement is needed
4. Specific improvements to make

Respond with JSON only:
{{
    "perspective_name": {{
        "strengths": ["strength1", "strength2"],
        "weaknesses": ["weakness1", "weakness2"],
        "refinement_needed": bool,
        "specific_improvements": ["improvement1", "improvement2"]
    }}
}}""",
                system=f"You are the {self.lead.name} LEAD agent providing authoritative technical review.",
                max_tokens=32768
            )

            feedback = parse_json_response(str(review))

            # Guard against empty/malformed JSON — assume refinement needed
            if not feedback:
                log.warn("LEAD review returned invalid JSON — assuming refinement needed for all perspectives")
                feedback = {
                    p["name"]: {"refinement_needed": True, "specific_improvements": ["Re-review needed — previous feedback was malformed"]}
                    for p in perspective_configs
                }

            feedback_history.append(feedback)

            # Check convergence
            all_converged = all(
                not agent_feedback.get("refinement_needed", True)
                for agent_feedback in feedback.values()
            )

            needs_refinement = [
                name for name, fb in feedback.items()
                if fb.get("refinement_needed", True)
            ]

            if all_converged:
                log.ok(f"All perspectives converged ({iteration + 1} iterations)")
                break

            log.step(f"Refinement needed for: {', '.join(needs_refinement)}")

            # Refine solutions IN PARALLEL
            def _refine_one(p_config: dict) -> tuple[str, str]:
                p_name = p_config["name"]
                agent_ref = p_config["agent_ref"]

                if p_name not in solutions or agent_ref not in self.lead.agents:
                    return p_name, solutions.get(p_name, "")

                solution = solutions[p_name]
                agent = self.lead.agents[agent_ref]

                if p_name in feedback:
                    agent_feedback = feedback[p_name]
                    if agent_feedback.get("refinement_needed"):
                        log.step(f"{p_name}  Refining solution...")

                        refinement_prompt = f"""Your previous solution proposal:

{solution}

LEAD feedback:

Strengths:
{self._format_list(agent_feedback.get('strengths', []))}

Weaknesses:
{self._format_list(agent_feedback.get('weaknesses', []))}

Required improvements:
{self._format_list(agent_feedback.get('specific_improvements', []))}

Provide refined solution addressing all feedback points."""

                        refined = agent.query(refinement_prompt)
                        log.ok(f"{p_name}  Refined")
                        return p_name, refined

                return p_name, solution

            refined_solutions: dict[str, str] = {}
            configs_needing_work = [
                pc for pc in perspective_configs
                if pc["name"] in needs_refinement
            ]
            configs_done = [
                pc for pc in perspective_configs
                if pc["name"] not in needs_refinement
            ]

            # Carry forward already-converged solutions
            for pc in configs_done:
                refined_solutions[pc["name"]] = solutions.get(pc["name"], "")

            # Refine the rest in parallel
            if configs_needing_work:
                with ThreadPoolExecutor(max_workers=len(configs_needing_work)) as pool:
                    futures = {pool.submit(_refine_one, pc): pc for pc in configs_needing_work}
                    for fut in as_completed(futures):
                        p_name, refined = fut.result()
                        refined_solutions[p_name] = refined

            solutions = refined_solutions
            iteration += 1

        return solutions, {
            "convergence_iterations": iteration,
            "converged": all_converged,
            "feedback_history": feedback_history,
        }

    def _synthesize_perspectives(
        self,
        question: str,
        solutions: dict[str, str],
        convergence_meta: dict,
    ) -> tuple[str, dict]:
        """Final synthesis of perspective-based solutions."""
        solutions_text = self._format_solutions(solutions)
        iterations = convergence_meta.get("convergence_iterations", 0)
        converged = convergence_meta.get("converged", False)

        log.step("Synthesizing final output...")

        prompt = f"""Synthesize these perspective-based solutions into a unified recommendation.

Original task: {question}

Solutions (after {iterations} refinement iterations):
{solutions_text}

Convergence status: {"Full convergence achieved" if converged else "Partial convergence"}

Provide:
1. Synthesized solution combining best aspects of each perspective
2. Explicit note on where perspectives diverged and why
3. Rationale for final synthesis approach
4. Implementation recommendation

Be transparent about the multi-perspective process and how it informed the final solution."""

        result = self.lead._llm.complete(prompt=prompt, max_tokens=32768)
        final = self.lead._continue_if_truncated(prompt, result)

        log.ok(f"Synthesis complete")

        return final, {
            "synthesis_approach": "perspective-based",
            "perspectives_count": len(solutions),
        }

    # ══════════════════════════════════════════════════════════════════
    # Helper Methods
    # ══════════════════════════════════════════════════════════════════

    def _parallel_query(self, queries: list[dict]) -> dict[str, str]:
        """Execute queries in parallel using lead's executor."""
        responses = {}

        def _query_one(spec):
            agent_name = spec["agent"]
            question = spec["question"]
            if agent_name in self.lead.agents:
                return agent_name, self.lead.agents[agent_name].query(question)
            return None, None

        if queries:
            futures = [self.lead._executor.submit(_query_one, s) for s in queries]
            for f in futures:
                try:
                    name, resp = f.result()
                    if name:
                        responses[name] = resp
                except Exception as e:
                    log.error(f"Parallel query failed: {e}")
                    continue

        return responses

    def _format_solutions(self, solutions: dict[str, str]) -> str:
        """Format solutions for display."""
        return "\n\n".join(
            f"=== {name} ===\n{sol}" for name, sol in solutions.items()
        )

    def _format_list(self, items: list) -> str:
        """Format list as markdown bullets."""
        return "\n".join(f"- {item}" for item in items) if items else "- (none)"


# ══════════════════════════════════════════════════════════════════════
# CROSS-POLLINATION ENGINE
# ══════════════════════════════════════════════════════════════════════

class CrossPollinationEngine:
    """
    Cross-pollination orchestration for A/B twin agent pairs.

    When a coordinator routes to a domain served by twin agents (linked
    via the ``twin`` field in YAML), this engine executes the cross-
    pollination loop instead of a normal single-agent query:

    1. **Parallel dispatch** — same prompt sent to both A and B
    2. **Output swap** — A receives B's output, B receives A's
    3. **Refinement** — each agent refines incorporating counterpart's view
    4. **Reconciliation** — coordinator LLM synthesises the two refined outputs

    Twin pairs may use different models/providers to ensure genuine
    cognitive diversity (e.g., Opus vs Grok).
    """

    def __init__(self, coordinator_llm: LLMClient, max_workers: int = 4):
        self._llm = coordinator_llm
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def execute(
        self,
        question: str,
        agent_a,
        agent_b,
        coordinator_name: str = "",
        max_iterations: int = 1,
    ) -> tuple[str, dict]:
        """Run the cross-pollination loop for a twin pair."""
        a_name = getattr(agent_a, "name", "A")
        b_name = getattr(agent_b, "name", "B")
        cn = coordinator_name or "cross-poll"

        # ── Phase 1: Parallel independent generation ─────────────────
        log.coord(cn, f"Cross-poll: {a_name} ↔ {b_name}")

        fut_a = self._executor.submit(agent_a.query, question)
        fut_b = self._executor.submit(agent_b.query, question)

        out_a = fut_a.result()
        out_b = fut_b.result()

        iteration_log = [
            {"iteration": 0, "a": out_a[:500], "b": out_b[:500]}
        ]

        # ── Phase 2+: Swap-refine cycles ─────────────────────────────
        for i in range(max_iterations):
            log.coord(cn, f"Cross-poll: swap round {i + 1}")

            refine_prompt_for_a = self._build_refine_prompt(
                question, out_a, out_b, b_name, a_name
            )
            refine_prompt_for_b = self._build_refine_prompt(
                question, out_b, out_a, a_name, b_name
            )

            fut_a = self._executor.submit(agent_a.query, refine_prompt_for_a)
            fut_b = self._executor.submit(agent_b.query, refine_prompt_for_b)

            out_a = fut_a.result()
            out_b = fut_b.result()

            iteration_log.append(
                {"iteration": i + 1, "a": out_a[:500], "b": out_b[:500]}
            )

        # ── Phase 3: Coordinator reconciliation ──────────────────────
        log.coord(cn, f"Cross-poll: reconciling {a_name} + {b_name}")
        reconciled = self._reconcile(question, out_a, out_b, a_name, b_name)

        metadata = {
            "mode": "cross_pollination",
            "agent_a": a_name,
            "agent_b": b_name,
            "swap_iterations": max_iterations,
            "iteration_log": iteration_log,
        }

        return reconciled, metadata

    # ── Internal helpers ─────────────────────────────────────────────

    @staticmethod
    def _build_refine_prompt(
        original_question: str,
        own_output: str,
        counterpart_output: str,
        counterpart_name: str,
        own_name: str,
    ) -> str:
        """Build the prompt an agent sees after receiving its counterpart's output."""
        return f"""## Original Question

{original_question}

## Your Previous Response

{own_output}

## Counterpart Response ({counterpart_name})

{counterpart_output}

## Instructions

You have received a response from a counterpart agent ({counterpart_name}) who
independently processed the same question. Review their output and refine your
own response:

1. **Identify strengths** in the counterpart's response that you missed
2. **Identify weaknesses** or errors in the counterpart's response
3. **Incorporate** valid points you overlooked
4. **Defend or revise** points where you disagree
5. **Produce a refined, complete response** that represents your best answer
   after considering both perspectives

Do NOT simply merge the two responses. Think critically and produce your own
refined answer."""

    def _reconcile(
        self,
        question: str,
        output_a: str,
        output_b: str,
        name_a: str,
        name_b: str,
    ) -> str:
        """Coordinator reconciles the two refined outputs into a final answer."""
        prompt = f"""## Reconciliation Task

Two agents independently processed the same question, exchanged outputs, and
each refined their response after reviewing the other's work. Your job is to
produce a single, authoritative answer.

### Original Question

{question}

### Agent {name_a} (Refined Output)

{output_a}

### Agent {name_b} (Refined Output)

{output_b}

### Instructions

Synthesize a unified response that:
1. Integrates the strongest elements from both agents
2. Resolves any remaining disagreements with clear reasoning
3. Notes significant points of divergence and why one view was preferred
4. Produces a complete, actionable answer

Be transparent about which agent contributed which insights."""

        result = self._llm.complete(prompt=prompt, max_tokens=32768)

        # Handle truncation
        parts = [str(result)]
        for _ in range(3):
            if not getattr(result, "truncated", False):
                break
            result = self._llm.complete(
                prompt=(
                    "Continue EXACTLY where this left off. Do not repeat.\n\n"
                    f"--- PARTIAL ---\n{parts[-1][-2000:]}\n--- END ---\n\nContinue:"
                ),
                max_tokens=32768,
            )
            parts.append(str(result))

        return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════
# LEAD_SUB Promotion Extension
# ══════════════════════════════════════════════════════════════════════

class LEADSUBPromotion:
    """
    Manages LEAD_SUB promotion workflow.

    Allows promoting an agent to primary implementation coordinator role.
    LEAD handles escalations, LEAD_SUB handles detailed implementation.
    """

    def __init__(self, lead: "LeadAgent"):
        self.lead = lead
        self._promoted_agent: Optional[str] = None

    def promote(self, agent_name: str) -> str:
        if agent_name not in self.lead.agents:
            return f"Error: Agent '{agent_name}' not found"

        self._promoted_agent = agent_name
        agent = self.lead.agents[agent_name]

        return f"Promoted '{agent_name}' to LEAD_SUB (primary implementation coordinator)"

    def demote(self) -> str:
        if not self._promoted_agent:
            return "No agent currently promoted"

        prev = self._promoted_agent
        self._promoted_agent = None
        return f"Demoted '{prev}' from LEAD_SUB role"

    def should_route_to_lead_sub(self, question: str) -> bool:
        if not self._promoted_agent:
            return False

        escalation_signals = [
            "architecture", "architectural", "design decision",
            "cross-module", "integration", "system-wide",
            "strategic", "breaking change", "overall approach",
            "big picture", "strategy"
        ]

        q_lower = question.lower()
        return not any(sig in q_lower for sig in escalation_signals)

    @property
    def current_lead_sub(self) -> Optional[str]:
        return self._promoted_agent
