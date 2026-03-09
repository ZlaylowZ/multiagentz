# multiagentz/agents/architect.py
"""
ArchitectAgent — decomposes coding tasks into dependency-ordered DAG plans.

The Architect doesn't write code. It produces a task graph that BuilderAgents
execute. Each task is scoped: specific files to read, specific files to produce,
specific validation commands to confirm success.

Used by the builder orchestration mode — not a routable agent in the
standard/consensus/perspective flow.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Optional

from multiagentz.llm_client import LLMClient
from multiagentz.utils import extract_json, parse_json_response
from multiagentz.workspace import Workspace
from multiagentz import log as _log


# ── Planning prompt ─────────────────────────────────────────────────────

PLANNING_PROMPT = """You are a software architect. Decompose the following task
into discrete, dependency-ordered subtasks that a code-writing agent can execute
independently.

## WORKSPACE STRUCTURE
{file_tree}

## RULES
- Each task must be completable by an agent that can only see the files listed in relevant_files.
- Tasks with no dependencies can run in parallel — take advantage of this.
- Include validation commands that will verify each task succeeded.
- Keep tasks small (one file or one logical unit per task).
- Order tasks so dependencies are resolved before dependents.
- Include the specific files the builder should read for context (relevant_files)
  and the files it will create or modify (output_files).

## OUTPUT FORMAT (JSON only)
{{
    "plan_name": "descriptive-slug",
    "tasks": [
        {{
            "id": "t1",
            "description": "Clear, specific instruction for what to build",
            "relevant_files": ["paths the builder needs to see"],
            "output_files": ["paths the builder will create/modify"],
            "depends_on": [],
            "validation": ["command to verify success"]
        }}
    ]
}}

Respond with ONLY the JSON object. No markdown fences, no prose.
"""


REPLAN_PROMPT = """You are a software architect. A subtask in your original plan failed.
Revise the plan to work around or resolve this failure.

## ORIGINAL TASK
{original_task}

## FAILED SUBTASK
ID: {failed_id}
Description: {failed_description}
Error:
{error}

## COMPLETED TASKS (do not redo these)
{completed_tasks}

## WORKSPACE STRUCTURE
{file_tree}

## OUTPUT FORMAT (JSON only)
{{
    "plan_name": "revised-plan",
    "tasks": [
        {{
            "id": "t1",
            "description": "Clear, specific instruction",
            "relevant_files": ["paths to read"],
            "output_files": ["paths to create/modify"],
            "depends_on": [],
            "validation": ["command to verify"]
        }}
    ]
}}

Only include tasks that still need to be done. Do not re-create completed tasks.
Respond with ONLY the JSON object.
"""


class ArchitectAgent:
    """
    Decomposes tasks into dependency-ordered plans for BuilderAgents.

    Not a standard routable agent — used at the orchestration level
    by BuilderOrchestrationEngine and the /build REPL command.
    """

    def __init__(
        self,
        workspace_path: str,
        llm_client: Optional[LLMClient] = None,
        max_tokens: int = 8192,
    ):
        self.workspace = Path(workspace_path).resolve()
        self._llm = llm_client or LLMClient()
        self.max_tokens = max_tokens
        self._workspace_scanner = Workspace(str(self.workspace))

    # ── Public API ──────────────────────────────────────────────────────

    def plan(self, task: str) -> dict:
        """
        Generate a task DAG from a natural language request.

        Returns a dict with 'plan_name' and 'tasks' list.
        Raises ValueError if the plan is structurally invalid.
        """
        t0 = time.time()
        _log.step(f"Architect: planning task: {task[:100]}...")

        file_tree = self._workspace_scanner.file_tree()
        system = PLANNING_PROMPT.format(file_tree=file_tree)

        try:
            response = self._llm.complete(
                prompt=f"Task: {task}",
                system=system,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            _log.error(f"Architect: LLM call failed: {e}")
            raise

        parsed = self._parse_plan(str(response))
        validated = self._validate_plan(parsed)

        elapsed = time.time() - t0
        task_count = len(validated.get("tasks", []))
        _log.ok(
            f"Architect: plan '{validated.get('plan_name', 'unnamed')}' "
            f"with {task_count} tasks ({elapsed:.1f}s)"
        )

        return validated

    def replan(
        self,
        original_task: str,
        failed_task: dict,
        error: str,
        completed_tasks: Optional[list[str]] = None,
    ) -> dict:
        """
        Re-plan after a task fails.

        Returns a revised plan covering only the remaining work.
        """
        _log.step("Architect: replanning after failure...")

        file_tree = self._workspace_scanner.file_tree()
        completed_str = "\n".join(
            f"- {t}" for t in (completed_tasks or [])
        ) or "(none)"

        system = REPLAN_PROMPT.format(
            original_task=original_task,
            failed_id=failed_task.get("id", "unknown"),
            failed_description=failed_task.get("description", "unknown"),
            error=error[-3000:],  # Cap error length
            completed_tasks=completed_str,
            file_tree=file_tree,
        )

        try:
            response = self._llm.complete(
                prompt=f"Revise the plan for: {original_task}",
                system=system,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            _log.error(f"Architect: replan LLM call failed: {e}")
            raise

        parsed = self._parse_plan(str(response))
        validated = self._validate_plan(parsed)

        task_count = len(validated.get("tasks", []))
        _log.ok(f"Architect: revised plan with {task_count} tasks")

        return validated

    # ── Plan parsing ────────────────────────────────────────────────────

    def _parse_plan(self, response_text: str) -> dict:
        """
        Parse plan JSON from LLM response.

        Includes paranoia checks: if the parse succeeds but doesn't
        contain expected top-level keys ('tasks'), log a warning with
        diagnostics so a bad parse surfaces immediately.
        """
        parsed = parse_json_response(response_text)
        if not parsed:
            # Fallback: try extract_json for markdown-fenced JSON
            try:
                parsed = extract_json(response_text)
            except Exception:
                _log.error("Architect: failed to parse plan JSON")
                return {}

        # Paranoia: check for expected structure
        if parsed and "tasks" not in parsed:
            found_keys = list(parsed.keys())[:10]
            _log.warn(
                f"Architect: JSON parsed but missing 'tasks' key. "
                f"Found keys: {found_keys}. "
                f"This may indicate the parser latched onto a nested "
                f"object instead of the plan response. "
                f"Response preview: {response_text[:200]!r}"
            )
        elif parsed:
            tasks = parsed.get("tasks")
            if not isinstance(tasks, list):
                _log.warn(
                    f"Architect: 'tasks' is {type(tasks).__name__}, expected list. "
                    f"Parsed keys: {list(parsed.keys())[:10]}"
                )

        return parsed

    # ── Plan validation ─────────────────────────────────────────────────

    def _validate_plan(self, plan: dict) -> dict:
        """
        Validate plan structure and check for circular dependencies.

        Returns the plan if valid, raises ValueError otherwise.
        """
        if not plan:
            raise ValueError("Empty plan — LLM returned no parseable JSON")

        if "tasks" not in plan:
            raise ValueError("Plan missing 'tasks' key")

        tasks = plan["tasks"]
        if not isinstance(tasks, list) or len(tasks) == 0:
            raise ValueError("Plan 'tasks' must be a non-empty list")

        # Ensure plan_name exists
        if "plan_name" not in plan:
            plan["plan_name"] = "unnamed-plan"

        # Validate each task
        task_ids = set()
        for task in tasks:
            if "id" not in task:
                raise ValueError(f"Task missing 'id': {task}")
            if "description" not in task:
                raise ValueError(f"Task {task['id']} missing 'description'")

            # Defaults
            task.setdefault("depends_on", [])
            task.setdefault("relevant_files", [])
            task.setdefault("output_files", [])
            task.setdefault("validation", [])

            task_ids.add(task["id"])

        # Check dependency references
        for task in tasks:
            for dep in task["depends_on"]:
                if dep not in task_ids:
                    raise ValueError(
                        f"Task {task['id']} depends on unknown task '{dep}'. "
                        f"Known IDs: {task_ids}"
                    )

        # Check for circular dependencies (topological sort)
        self._check_circular_deps(tasks)

        return plan

    def _check_circular_deps(self, tasks: list[dict]):
        """Detect circular dependencies via Kahn's algorithm."""
        # Build adjacency list and in-degree map
        in_degree: dict[str, int] = {}
        graph: dict[str, list[str]] = {}

        for task in tasks:
            tid = task["id"]
            in_degree.setdefault(tid, 0)
            graph.setdefault(tid, [])
            for dep in task["depends_on"]:
                graph.setdefault(dep, [])
                graph[dep].append(tid)
                in_degree[tid] = in_degree.get(tid, 0) + 1

        # Kahn's algorithm
        queue = deque(tid for tid, deg in in_degree.items() if deg == 0)
        sorted_count = 0

        while queue:
            node = queue.popleft()
            sorted_count += 1
            for neighbor in graph.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if sorted_count != len(in_degree):
            raise ValueError(
                "Circular dependency detected in task plan. "
                "Check the depends_on relationships."
            )
