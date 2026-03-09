# multiagentz/task_dag.py
"""
TaskDAG — dependency-ordered execution engine for builder mode.

Walks a plan DAG produced by ArchitectAgent, spawns BuilderAgents for
ready tasks, executes them in parallel, and handles failures with
Architect-mediated replanning.

The DAG is the single source of truth for task ordering. Builders don't
self-organize — the DAG tells them what to do and when.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from multiagentz.agents.builder import BuilderAgent
from multiagentz.llm_client import LLMClient
from multiagentz import log

if TYPE_CHECKING:
    from multiagentz.agents.architect import ArchitectAgent


# ── Task node ───────────────────────────────────────────────────────────

@dataclass
class TaskNode:
    """A single task in the execution DAG."""

    id: str
    spec: dict
    status: str = "pending"   # pending | running | completed | failed
    result: str = ""
    fail_count: int = 0

    def deps_satisfied(self, all_tasks: dict[str, "TaskNode"]) -> bool:
        """Return True if all dependencies are completed."""
        for dep_id in self.spec.get("depends_on", []):
            dep = all_tasks.get(dep_id)
            if dep is None or dep.status != "completed":
                return False
        return True


# ── DAG executor ────────────────────────────────────────────────────────

class TaskDAG:
    """
    Executes a dependency-ordered task plan using BuilderAgents.

    Usage:
        dag = TaskDAG(plan, workspace_path, llm_client, builder_defaults, architect)
        report = dag.execute(original_task="Build feature X")
    """

    def __init__(
        self,
        plan: dict,
        workspace_path: str,
        llm_client: LLMClient,
        builder_defaults: Optional[dict] = None,
        architect: Optional["ArchitectAgent"] = None,
        max_workers: int = 4,
        max_replan_attempts: int = 2,
    ):
        self.plan_name = plan.get("plan_name", "unnamed")
        self.tasks: dict[str, TaskNode] = {
            t["id"]: TaskNode(id=t["id"], spec=t) for t in plan.get("tasks", [])
        }
        self.workspace = Path(workspace_path).resolve()
        self._llm = llm_client
        self._builder_defaults = builder_defaults or {}
        self._architect = architect
        self._max_replan = max_replan_attempts
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    # ── Public API ──────────────────────────────────────────────────────

    def execute(self, original_task: str = "") -> dict:
        """
        Walk the DAG to completion or failure.

        Returns a report dict with completed/failed tasks and summary.
        """
        t0 = time.time()
        total_tasks = len(self.tasks)
        log.init()
        log.phase(1, 1, f"Executing plan: {self.plan_name} ({total_tasks} tasks)")

        iteration = 0
        max_iterations = total_tasks * 3  # Safety bound

        while iteration < max_iterations:
            iteration += 1

            ready = self.get_ready_tasks()
            if not ready:
                # Check for deadlock
                pending = [t for t in self.tasks.values() if t.status == "pending"]
                if pending:
                    pending_ids = [t.id for t in pending]
                    log.error(
                        f"DAG deadlock: {len(pending)} tasks pending but none ready: "
                        f"{pending_ids}"
                    )
                    for t in pending:
                        t.status = "failed"
                        t.result = "Blocked by unsatisfied dependencies (deadlock)"
                break

            log.step(
                f"Wave {iteration}: executing {len(ready)} task(s): "
                f"{[t.id for t in ready]}"
            )

            # Execute ready tasks in parallel
            futures = {
                self._executor.submit(self._execute_task, t): t
                for t in ready
            }

            for future in as_completed(futures):
                task = futures[future]
                try:
                    success, summary = future.result()
                except Exception as e:
                    success = False
                    summary = f"Unexpected error: {e}"

                if success:
                    task.status = "completed"
                    task.result = summary
                    log.ok(f"Task {task.id} completed: {summary[:80]}")
                else:
                    task.fail_count += 1
                    task.result = summary

                    if (
                        task.fail_count <= self._max_replan
                        and self._architect is not None
                    ):
                        log.warn(
                            f"Task {task.id} failed (attempt {task.fail_count}), "
                            f"requesting replan..."
                        )
                        self._handle_replan(task, summary, original_task)
                    else:
                        task.status = "failed"
                        log.error(f"Task {task.id} permanently failed: {summary[:80]}")

        elapsed = time.time() - t0
        report = self._build_report()
        log.done(
            f"Plan '{self.plan_name}' finished: "
            f"{report['completed_count']}/{total_tasks} completed, "
            f"{report['failed_count']} failed ({elapsed:.1f}s)"
        )

        return report

    # ── Task readiness ──────────────────────────────────────────────────

    def get_ready_tasks(self) -> list[TaskNode]:
        """Tasks whose dependencies are all completed and status is pending."""
        return [
            t for t in self.tasks.values()
            if t.status == "pending" and t.deps_satisfied(self.tasks)
        ]

    # ── Task execution ──────────────────────────────────────────────────

    def _execute_task(self, task: TaskNode) -> tuple[bool, str]:
        """Spawn a BuilderAgent for a single task and return (success, summary)."""
        task.status = "running"

        builder = BuilderAgent(
            name=f"builder_{task.id}",
            workspace_path=str(self.workspace),
            relevant_files=task.spec.get("relevant_files", []),
            validation_commands=task.spec.get("validation", []),
            llm_client=self._llm,
            max_retries=self._builder_defaults.get("max_retries", 3),
            max_tokens=self._builder_defaults.get("max_tokens", 16384),
            command_timeout=self._builder_defaults.get("command_timeout", 60),
        )

        result = builder.query(task.spec["description"])

        # Detect failure from BuilderAgent's return format
        is_failure = result.startswith("Task failed after") or result.startswith("Error:")
        return (not is_failure, result)

    # ── Replanning ──────────────────────────────────────────────────────

    def _handle_replan(
        self, failed_task: TaskNode, error: str, original_task: str
    ):
        """Ask the Architect to revise the plan after a task failure."""
        if self._architect is None:
            failed_task.status = "failed"
            return

        completed_ids = [
            t.id for t in self.tasks.values() if t.status == "completed"
        ]

        try:
            new_plan = self._architect.replan(
                original_task=original_task,
                failed_task=failed_task.spec,
                error=error,
                completed_tasks=completed_ids,
            )
        except Exception as e:
            log.error(f"Replanning failed: {e}")
            failed_task.status = "failed"
            return

        new_tasks = new_plan.get("tasks", [])
        if not new_tasks:
            log.warn("Architect returned empty replan — marking task as failed")
            failed_task.status = "failed"
            return

        # Replace the failed task with the new task(s)
        # Remove the old task
        del self.tasks[failed_task.id]

        # Add new tasks, updating dependency references
        for t_spec in new_tasks:
            tid = t_spec["id"]
            # Avoid ID collisions by prefixing with replan round
            if tid in self.tasks:
                tid = f"{tid}_r{failed_task.fail_count}"
                t_spec["id"] = tid
            self.tasks[tid] = TaskNode(id=tid, spec=t_spec)

        log.step(
            f"Replan inserted {len(new_tasks)} replacement task(s): "
            f"{[t['id'] for t in new_tasks]}"
        )

    # ── Reporting ───────────────────────────────────────────────────────

    def _build_report(self) -> dict:
        """Build a summary report of the execution."""
        completed = []
        failed = []

        for task in self.tasks.values():
            entry = {
                "id": task.id,
                "description": task.spec.get("description", ""),
                "result": task.result,
            }
            if task.status == "completed":
                completed.append(entry)
            elif task.status == "failed":
                failed.append(entry)

        # Build human-readable summary
        lines = [f"## Build Report: {self.plan_name}\n"]

        if completed:
            lines.append(f"### Completed ({len(completed)})")
            for c in completed:
                lines.append(f"- **{c['id']}**: {c['description']}")
                if c["result"]:
                    lines.append(f"  Result: {c['result'][:200]}")
            lines.append("")

        if failed:
            lines.append(f"### Failed ({len(failed)})")
            for f_entry in failed:
                lines.append(f"- **{f_entry['id']}**: {f_entry['description']}")
                if f_entry["result"]:
                    lines.append(f"  Error: {f_entry['result'][:300]}")
            lines.append("")

        total = len(self.tasks)
        lines.append(
            f"**Summary**: {len(completed)}/{total} tasks completed, "
            f"{len(failed)} failed."
        )

        return {
            "plan_name": self.plan_name,
            "completed": completed,
            "failed": failed,
            "completed_count": len(completed),
            "failed_count": len(failed),
            "total_tasks": total,
            "summary": "\n".join(lines),
        }
