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

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional

from multiagentz.llm_client import LLMClient

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
        self._llm = LLMClient()
    
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
        
        Parameters
        ----------
        question : str
            User query
        memory : SessionMemory, optional
            Conversation history
        max_iterations : int
            Maximum refinement cycles
            
        Returns
        -------
        tuple[str, dict]
            (final_answer, metadata)
            metadata contains: iterations, conflicts_found, consensus_achieved
        """
        memory_context = memory.get_context() if memory else ""
        
        # Initial routing and query (use existing lead logic)
        routing = self.lead._route_query(question, memory_context)
        queries = routing.get("queries", [])
        
        if not queries:
            # Fallback to first agent
            first_agent = next(iter(self.lead.agents))
            queries = [{"agent": first_agent, "question": question}]
        
        print(f"[Consensus] Initial routing: {[q['agent'] for q in queries]}")
        
        # Get initial responses
        responses = self._parallel_query(queries)
        
        if len(responses) <= 1:
            # Single agent, no consensus needed
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
            # Detect contradictions
            conflict_analysis = self._analyze_conflicts(question, responses)
            
            has_conflicts = conflict_analysis.get("has_conflicts", False)
            conflicts = conflict_analysis.get("conflicts", [])
            
            if conflicts:
                all_conflicts.extend(conflicts)
            
            if not has_conflicts:
                print(f"[Consensus] Consensus achieved after {iteration} iterations")
                break
            
            print(f"[Consensus] Iteration {iteration + 1}: {len(conflicts)} conflicts detected")
            
            # Generate refinement questions
            refinements = self._generate_refinements(question, responses, conflicts)
            
            # Re-query agents with refinements
            responses = self._parallel_query(refinements)
            iteration += 1
        
        # Final synthesis
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
        
        return self._parse_json_response(str(result))
    
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
        
        parsed = self._parse_json_response(str(result))
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
        
        # Handle continuations if needed
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
        
        Maps to documented pattern:
        - LEAD = self.lead (provides authoritative context)
        - SUBS = perspective agents (generate independent solutions)
        - SUB_MEM = agent with memory_access: shared
        - SUB_INC = agent with memory_access: none
        
        Flow:
        1. [Optional] Bootstrap Q&A: each SUB asks questions, LEAD answers
        2. Each SUB generates independent solution proposal
        3. LEAD reviews each solution, provides feedback
        4. SUBS refine based on feedback
        5. Iterate until convergence
        6. Consensus synthesis
        7. [Optional] Promote winner to LEAD_SUB
        
        Parameters
        ----------
        question : str
            Task/question for agents to solve
        perspective_configs : list[dict]
            List of perspective agent configurations:
            [
                {
                    "name": "sub_mem",
                    "agent_ref": "core",  # which agent to use
                    "memory_access": "shared",
                    "role": "description"
                },
                {
                    "name": "sub_inc",
                    "agent_ref": "core",
                    "memory_access": "none",
                    "role": "description"
                }
            ]
        memory : SessionMemory, optional
            Conversation history
        bootstrap_qa : bool
            Whether to run Q&A alignment phase
        max_iterations : int
            Maximum refinement cycles
            
        Returns
        -------
        tuple[str, dict]
            (final_answer, metadata)
        """
        print(f"\n[Perspective Mode] Initializing {len(perspective_configs)} perspectives")
        
        # Phase 1: Bootstrap Q&A (if enabled)
        aligned_contexts = {}
        if bootstrap_qa:
            print("[Perspective] Phase 1: Bootstrap Q&A alignment")
            aligned_contexts = self._bootstrap_perspectives(
                question, perspective_configs, memory
            )
        else:
            aligned_contexts = {p["name"]: "" for p in perspective_configs}
        
        # Phase 2: Independent solution generation
        print("[Perspective] Phase 2: Independent solution generation")
        solutions = self._generate_independent_solutions(
            question, perspective_configs, aligned_contexts
        )
        
        # Phase 3: Iterative refinement with LEAD feedback
        print("[Perspective] Phase 3: LEAD review and iterative refinement")
        converged_solutions, convergence_meta = self._iterate_to_convergence(
            question, solutions, perspective_configs, max_iterations
        )
        
        # Phase 4: Consensus synthesis
        print("[Perspective] Phase 4: Consensus synthesis")
        final_answer, synthesis_meta = self._synthesize_perspectives(
            question, converged_solutions, convergence_meta
        )
        
        return final_answer, {
            "mode": "perspective",
            "perspectives": [p["name"] for p in perspective_configs],
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
        Q&A alignment phase.
        
        Each perspective agent asks questions → LEAD answers → iterate until aligned.
        """
        contexts = {}
        
        for p_config in perspective_configs:
            p_name = p_config["name"]
            agent_ref = p_config["agent_ref"]
            memory_access = p_config.get("memory_access", "shared")
            
            if agent_ref not in self.lead.agents:
                print(f"[Perspective] Warning: agent '{agent_ref}' not found for '{p_name}'")
                contexts[p_name] = ""
                continue
            
            agent = self.lead.agents[agent_ref]
            
            # Prepare initial context based on memory access
            if memory_access == "none":
                initial_context = ""
                memory_note = "You have NO access to previous conversation history."
            else:
                initial_context = memory.get_context() if memory else ""
                memory_note = "You have access to conversation history above."
            
            # Initial briefing
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
                briefing
            ]
            
            print(f"  [{p_name}] Starting Q&A alignment...")
            
            aligned = False
            max_qa_turns = 10
            turn = 0
            
            while not aligned and turn < max_qa_turns:
                # Agent generates questions
                qa_prompt = "\n\n".join(filter(None, qa_history))
                agent_response = agent.query(qa_prompt)
                
                # Check for alignment signal
                if "ALIGNED" in agent_response.upper() or "READY TO PROCEED" in agent_response.upper():
                    aligned = True
                    print(f"  [{p_name}] Aligned after {turn + 1} Q&A turns")
                    break
                
                # Extract questions
                questions = self._extract_questions(agent_response)
                
                if not questions:
                    # No clear questions, assume aligned
                    print(f"  [{p_name}] No questions detected, assuming aligned")
                    aligned = True
                    break
                
                print(f"  [{p_name}] Turn {turn + 1}: {len(questions)} questions")
                
                # LEAD answers each question
                lead_answers = []
                for q in questions:
                    # Use LEAD's full query method (gets full context + routing)
                    answer, _ = self.lead.query(q, memory=memory)
                    lead_answers.append(f"**Q:** {q}\n**A:** {answer}")
                
                qa_history.append(f"\n**{p_name} Questions:**\n{agent_response}")
                qa_history.append(f"\n**LEAD Answers:**\n" + "\n\n".join(lead_answers))
                
                turn += 1
            
            if not aligned:
                print(f"  [{p_name}] Warning: max Q&A turns reached without explicit alignment")
            
            contexts[p_name] = "\n\n".join(filter(None, qa_history))
        
        return contexts
    
    def _extract_questions(self, text: str) -> list[str]:
        """Extract questions from agent response."""
        # Look for lines ending with ?
        lines = text.split("\n")
        questions = []
        
        for line in lines:
            line = line.strip()
            if line.endswith("?"):
                # Clean up markdown, numbering
                cleaned = re.sub(r"^[\d\-\*\.\)]+\s*", "", line)
                cleaned = re.sub(r"^\*\*.*?\*\*:?\s*", "", cleaned)
                if len(cleaned) > 10:  # Ignore very short questions
                    questions.append(cleaned)
        
        return questions[:10]  # Cap at 10 questions per turn
    
    def _generate_independent_solutions(
        self,
        question: str,
        perspective_configs: list[dict],
        aligned_contexts: dict[str, str],
    ) -> dict[str, str]:
        """Each perspective agent generates independent solution."""
        solutions = {}
        
        for p_config in perspective_configs:
            p_name = p_config["name"]
            agent_ref = p_config["agent_ref"]
            
            if agent_ref not in self.lead.agents:
                continue
            
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
            
            print(f"  [{p_name}] Generating independent solution...")
            solution = agent.query(prompt)
            solutions[p_name] = solution
        
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
        """
        iteration = 0
        feedback_history = []
        
        while iteration < max_iterations:
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
            
            feedback = self._parse_json_response(str(review))
            feedback_history.append(feedback)
            
            print(f"  [LEAD] Iteration {iteration + 1} review complete")
            
            # Check convergence
            all_converged = all(
                not agent_feedback.get("refinement_needed", True)
                for agent_feedback in feedback.values()
            )
            
            if all_converged:
                print(f"  [LEAD] All perspectives converged")
                break
            
            # Refine solutions based on feedback
            refined_solutions = {}
            for p_config in perspective_configs:
                p_name = p_config["name"]
                agent_ref = p_config["agent_ref"]
                
                if p_name not in solutions or agent_ref not in self.lead.agents:
                    continue
                
                solution = solutions[p_name]
                agent = self.lead.agents[agent_ref]
                
                if p_name in feedback:
                    agent_feedback = feedback[p_name]
                    if agent_feedback.get("refinement_needed"):
                        print(f"  [{p_name}] Refining solution...")
                        
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
                        refined_solutions[p_name] = refined
                    else:
                        refined_solutions[p_name] = solution
                else:
                    refined_solutions[p_name] = solution
            
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
                name, resp = f.result()
                if name:
                    responses[name] = resp
        
        return responses
    
    def _parse_json_response(self, text: str) -> dict:
        """Parse JSON from LLM response, handling markdown fences."""
        try:
            # Remove markdown fences
            cleaned = text.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0]
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0]
            
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {}
    
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
        """
        Run the cross-pollination loop for a twin pair.

        Parameters
        ----------
        question : str
            The prompt to process.
        agent_a : SubAgent
            First twin agent.
        agent_b : SubAgent
            Second twin agent (different model/provider recommended).
        coordinator_name : str
            Name of the parent coordinator (for logging).
        max_iterations : int
            Number of swap-refine cycles. Default 1 (one swap round).

        Returns
        -------
        tuple[str, dict]
            (reconciled_answer, metadata)
        """
        a_name = getattr(agent_a, "name", "A")
        b_name = getattr(agent_b, "name", "B")
        prefix = f"[{coordinator_name}]" if coordinator_name else "[CrossPoll]"

        # ── Phase 1: Parallel independent generation ─────────────────
        print(f"{prefix} Cross-pollination: dispatching to {a_name} + {b_name}")

        fut_a = self._executor.submit(agent_a.query, question)
        fut_b = self._executor.submit(agent_b.query, question)

        out_a = fut_a.result()
        out_b = fut_b.result()

        iteration_log = [
            {"iteration": 0, "a": out_a[:500], "b": out_b[:500]}
        ]

        # ── Phase 2+: Swap-refine cycles ─────────────────────────────
        for i in range(max_iterations):
            print(f"{prefix} Cross-pollination iteration {i + 1}: swapping outputs")

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
        print(f"{prefix} Cross-pollination: reconciling outputs")
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
        """
        Promote agent to LEAD_SUB role.
        
        Parameters
        ----------
        agent_name : str
            Name of agent to promote
            
        Returns
        -------
        str
            Confirmation message
        """
        if agent_name not in self.lead.agents:
            return f"Error: Agent '{agent_name}' not found"
        
        self._promoted_agent = agent_name
        
        # Inject LEAD_SUB context into agent
        agent = self.lead.agents[agent_name]
        
        # Note: This modifies the agent's behavior for subsequent queries
        # For SubAgent, we'd inject into system prompt
        # For now, we track state and route differently
        
        return f"✓ Promoted '{agent_name}' to LEAD_SUB (primary implementation coordinator)"
    
    def demote(self) -> str:
        """Remove LEAD_SUB promotion."""
        if not self._promoted_agent:
            return "No agent currently promoted"
        
        prev = self._promoted_agent
        self._promoted_agent = None
        return f"✓ Demoted '{prev}' from LEAD_SUB role"
    
    def should_route_to_lead_sub(self, question: str) -> bool:
        """
        Determine if question should go to LEAD_SUB instead of LEAD.
        
        Implementation details → LEAD_SUB
        Strategic/architectural → LEAD
        """
        if not self._promoted_agent:
            return False
        
        # Escalation signals (bypass LEAD_SUB, go to LEAD)
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
        """Get currently promoted LEAD_SUB agent name."""
        return self._promoted_agent
