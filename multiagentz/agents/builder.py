# multiagentz/agents/builder.py
"""
BuilderAgent — executes scoped coding tasks with a write-validate-fix loop.

Unlike SubAgent (read-only analysis), BuilderAgent can:
- Write and patch files in a workspace directory
- Run shell commands (compile, lint, test)
- Iterate on validation failures

It uses structured JSON output from the LLM to determine actions,
then executes them in the workspace. Composes a SubAgent internally
for file loading — no inheritance to keep concerns separate.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Optional

from multiagentz.agents.base import SubAgent
from multiagentz.llm_client import LLMClient
from multiagentz.utils import parse_json_response
from multiagentz.workspace import Workspace
from multiagentz import log as _log


# ── System prompt for structured output ─────────────────────────────────

BUILDER_SYSTEM_PROMPT = """You are a code builder agent. You receive a specific, scoped coding task
and produce file mutations to accomplish it.

You have access to reference files from the project workspace (provided below).
Your job is to write code that integrates correctly with the existing codebase.

## RESPONSE FORMAT
Respond with ONLY a JSON object:
{
    "reasoning": "Brief explanation of your approach",
    "actions": [
        {"type": "write_file", "path": "relative/path.py", "content": "full file content"},
        {"type": "patch_file", "path": "relative/path.py", "search": "old code", "replace": "new code"},
        {"type": "run_command", "command": "python -m py_compile path.py"}
    ],
    "status": "complete",
    "summary": "What was accomplished"
}

## ACTION TYPES
- write_file: Create or overwrite a file. Always provide COMPLETE file content.
- patch_file: Replace a specific string in an existing file. The "search" string must be an exact match.
- run_command: Execute a shell command in the workspace directory.

## RULES
- Write COMPLETE file contents, not snippets — partial files break everything.
- Use only imports and modules visible in the Reference Files.
- Match the project's existing code style (naming conventions, patterns, structure).
- If you need information not in the Reference Files, set status to "needs_more_context"
  and explain what you need in the summary.
- All file paths must be relative to the workspace root.
- Do NOT include markdown fences around the JSON. Return raw JSON only.
"""


class BuilderAgent:
    """
    An agent that writes code via structured LLM output and validates results.

    Composes a SubAgent for file loading. The query() loop:
    1. Load workspace context via internal SubAgent
    2. Ask LLM for structured JSON actions
    3. Execute actions (write files, run commands)
    4. Run validation commands
    5. If validation fails, feed errors back and retry
    """

    def __init__(
        self,
        name: str,
        workspace_path: str,
        relevant_files: Optional[list[str]] = None,
        validation_commands: Optional[list[str]] = None,
        max_retries: int = 3,
        max_tokens: int = 16384,
        command_timeout: int = 60,
        llm_client: Optional[LLMClient] = None,
    ):
        self.name = name
        self.workspace = Path(workspace_path).resolve()
        self.validation_commands = validation_commands or []
        self.max_retries = max_retries
        self._command_timeout = command_timeout

        # Internal reader for workspace file context (composition, not inheritance)
        self._llm = llm_client or LLMClient()
        self._reader = SubAgent(
            name=f"{name}_reader",
            repo_path=str(self.workspace),
            key_files=relevant_files or [],
            llm_client=self._llm,
            max_tokens=max_tokens,
        )
        self.max_tokens = max_tokens
        self._workspace_scanner = Workspace(str(self.workspace))

    # ── Public interface ────────────────────────────────────────────────

    @property
    def description(self) -> str:
        return f"Builder agent: {self.name}"

    def query(self, task_description: str, include_files: Optional[list[str]] = None) -> str:
        """
        Execute a coding task with a write-validate-fix loop.

        Returns a summary of what was accomplished or a failure message.
        """
        t0 = time.time()
        self._log(f"Starting task: {task_description[:100]}...")

        # Load workspace file context once
        file_context = self._reader._load_file_contents()
        validation_errors = ""

        for attempt in range(self.max_retries):
            # Build prompt with optional error feedback
            prompt = task_description
            if validation_errors:
                prompt += (
                    f"\n\n## PREVIOUS ATTEMPT FAILED (attempt {attempt}/{self.max_retries})\n"
                    f"{validation_errors}\n\n"
                    "Fix the errors above. Return a complete corrected action plan."
                )

            # Build system prompt with workspace context
            system = self._build_full_system(file_context)

            # Get structured actions from LLM
            self._log(f"Requesting actions from LLM (attempt {attempt + 1}/{self.max_retries})")
            try:
                response = self._llm.complete(
                    prompt=prompt,
                    system=system,
                    max_tokens=self.max_tokens,
                )
            except Exception as e:
                _log.error(f"{self.name}: LLM call failed: {e}")
                return f"Error: LLM call failed: {e}"

            # Parse JSON response
            parsed = self._parse_actions(str(response))
            if not parsed or "actions" not in parsed:
                _log.warn(f"{self.name}: Failed to parse structured response")
                if attempt < self.max_retries - 1:
                    validation_errors = (
                        "Your previous response was not valid JSON. "
                        "You MUST respond with ONLY a JSON object. No markdown, no prose."
                    )
                    continue
                return f"Failed to parse builder response after {self.max_retries} attempts."

            # Log reasoning
            reasoning = parsed.get("reasoning", "")
            if reasoning:
                self._log(f"Reasoning: {reasoning[:150]}")

            # Execute actions
            action_count = len(parsed.get("actions", []))
            self._log(f"Executing {action_count} actions...")
            exec_results = self._execute_actions(parsed)

            # Collect written/patched files for validation templating
            written_files = [
                r["path"] for r in exec_results
                if r.get("ok") and r.get("type") in ("write_file", "patch_file")
            ]

            # Check for execution failures
            exec_failures = [r for r in exec_results if not r.get("ok", False)]
            if exec_failures:
                failure_msgs = "; ".join(
                    f"{r.get('type')}: {r.get('error', 'unknown')}"
                    for r in exec_failures
                )
                _log.warn(f"{self.name}: {len(exec_failures)} action(s) failed: {failure_msgs}")

            # Run validation against the files that were actually written
            validation_errors = self._validate(written_files)
            if not validation_errors:
                # Success! Invalidate reader cache since workspace changed
                self._reader.invalidate_cache()
                summary = parsed.get("summary", "Task completed successfully.")
                elapsed = time.time() - t0
                self._log(f"Completed ({elapsed:.1f}s)")
                _log.ok(f"{self.name}: {summary[:100]}")
                return summary

            _log.warn(
                f"{self.name}: Validation failed (attempt {attempt + 1}/{self.max_retries})"
            )
            # Loop continues — validation_errors will be injected into next prompt

        elapsed = time.time() - t0
        return (
            f"Task failed after {self.max_retries} attempts ({elapsed:.1f}s). "
            f"Last errors:\n{validation_errors}"
        )

    # ── System prompt assembly ──────────────────────────────────────────

    def _build_full_system(self, file_context: str) -> str:
        """Build the full system prompt with workspace context."""
        tree = self._workspace_scanner.file_tree()

        parts = [BUILDER_SYSTEM_PROMPT]

        if tree:
            parts.append(f"\n## WORKSPACE STRUCTURE\n{tree}")

        if file_context:
            parts.append(f"\n## REFERENCE FILES\n{file_context}")
        else:
            parts.append(
                "\n## REFERENCE FILES\n"
                "No reference files loaded. You may need to create files from scratch."
            )

        parts.append(f"\n\nWorkspace root: {self.workspace}")

        return "\n".join(parts)

    # ── Action parsing ──────────────────────────────────────────────────

    def _parse_actions(self, response_text: str) -> dict:
        """
        Parse structured JSON from LLM response.

        Uses brace-matching parser (extract_json_robust) which handles
        nested JSON in write_file content fields correctly — unlike
        fence-based splitting which breaks when file content contains
        markdown fences or nested JSON.

        Includes paranoia checks: if the parse succeeds but doesn't
        contain expected top-level keys, log a warning with diagnostics
        so bad parses surface immediately instead of silently doing nothing.
        """
        parsed = parse_json_response(response_text)

        if parsed and "actions" not in parsed:
            found_keys = list(parsed.keys())[:10]
            _log.warn(
                f"{self.name}: JSON parsed but missing 'actions' key. "
                f"Found keys: {found_keys}. "
                f"This may indicate the brace-matcher latched onto a nested "
                f"object instead of the outer response. "
                f"Response preview: {response_text[:200]!r}"
            )
        elif parsed:
            actions = parsed.get("actions")
            if not isinstance(actions, list):
                _log.warn(
                    f"{self.name}: 'actions' is {type(actions).__name__}, "
                    f"expected list. Parsed keys: {list(parsed.keys())[:10]}"
                )

        return parsed

    # ── Action execution ────────────────────────────────────────────────

    def _execute_actions(self, parsed: dict) -> list[dict]:
        """Execute the action list from the LLM response."""
        results = []
        for action in parsed.get("actions", []):
            action_type = action.get("type")

            if action_type == "write_file":
                result = self._action_write_file(action)
            elif action_type == "patch_file":
                result = self._action_patch_file(action)
            elif action_type == "run_command":
                result = self._run_command(action.get("command", ""))
            else:
                result = {
                    "type": action_type, "ok": False,
                    "error": f"Unknown action type: {action_type}",
                }

            results.append(result)

        return results

    def _action_write_file(self, action: dict) -> dict:
        """Write a file to the workspace."""
        rel_path = action.get("path", "")
        content = action.get("content", "")

        path = (self.workspace / rel_path).resolve()

        # Security: ensure path is within workspace
        if not path.is_relative_to(self.workspace):
            _log.error(f"{self.name}: Path traversal rejected: {rel_path}")
            return {
                "type": "write_file", "path": rel_path, "ok": False,
                "error": "Path traversal detected — path is outside workspace",
            }

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            self._log(f"Wrote {rel_path} ({len(content):,} chars)")
            return {"type": "write_file", "path": rel_path, "ok": True}
        except Exception as e:
            return {"type": "write_file", "path": rel_path, "ok": False, "error": str(e)}

    def _action_patch_file(self, action: dict) -> dict:
        """Patch an existing file by replacing a search string."""
        rel_path = action.get("path", "")
        search = action.get("search", "")
        replace = action.get("replace", "")

        path = (self.workspace / rel_path).resolve()

        # Security: ensure path is within workspace
        if not path.is_relative_to(self.workspace):
            _log.error(f"{self.name}: Path traversal rejected: {rel_path}")
            return {
                "type": "patch_file", "path": rel_path, "ok": False,
                "error": "Path traversal detected — path is outside workspace",
            }

        if not path.exists():
            return {
                "type": "patch_file", "path": rel_path, "ok": False,
                "error": "File not found",
            }

        try:
            content = path.read_text(encoding="utf-8")
            if search not in content:
                return {
                    "type": "patch_file", "path": rel_path, "ok": False,
                    "error": "Search string not found in file",
                }
            content = content.replace(search, replace, 1)
            path.write_text(content, encoding="utf-8")
            self._log(f"Patched {rel_path}")
            return {"type": "patch_file", "path": rel_path, "ok": True}
        except Exception as e:
            return {"type": "patch_file", "path": rel_path, "ok": False, "error": str(e)}

    # ── Validation ──────────────────────────────────────────────────────

    def _validate(self, written_files: Optional[list[str]] = None) -> str:
        """
        Run validation commands.

        If a command contains {file}, it is expanded against each file
        that was written/patched in this round. This ensures validation
        runs against the *actual* output files, not the Architect's
        predicted output_files (which may diverge).

        Returns empty string if all pass, error details otherwise.
        """
        if not self.validation_commands:
            return ""

        written_files = written_files or []
        errors = []

        for cmd_template in self.validation_commands:
            if "{file}" in cmd_template:
                # Expand template against each written file
                if not written_files:
                    # No files written — skip templated commands
                    continue
                for filepath in written_files:
                    cmd = cmd_template.replace("{file}", filepath)
                    result = self._run_command(cmd)
                    if result.get("returncode", 1) != 0:
                        stderr = result.get("stderr", "")
                        stdout = result.get("stdout", "")
                        output = stderr or stdout or "unknown error"
                        errors.append(f"$ {cmd}\n{output}")
            else:
                # No template — run command as-is
                result = self._run_command(cmd_template)
                if result.get("returncode", 1) != 0:
                    stderr = result.get("stderr", "")
                    stdout = result.get("stdout", "")
                    output = stderr or stdout or "unknown error"
                    errors.append(f"$ {cmd_template}\n{output}")

        return "\n\n".join(errors)

    # ── Command execution ───────────────────────────────────────────────

    def _run_command(self, command: str) -> dict:
        """Execute a shell command in the workspace directory."""
        if not command.strip():
            return {"type": "run_command", "command": command, "ok": False,
                    "error": "Empty command"}

        self._log(f"Running: {command[:100]}")
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=self._command_timeout,
            )
            return {
                "type": "run_command",
                "command": command,
                "returncode": proc.returncode,
                "stdout": proc.stdout[-5000:],   # Cap output
                "stderr": proc.stderr[-5000:],
                "ok": proc.returncode == 0,
            }
        except subprocess.TimeoutExpired:
            return {
                "type": "run_command", "command": command,
                "ok": False, "returncode": -1,
                "error": f"Timed out after {self._command_timeout}s",
            }
        except Exception as e:
            return {
                "type": "run_command", "command": command,
                "ok": False, "returncode": -1,
                "error": str(e),
            }

    # ── Logging ─────────────────────────────────────────────────────────

    def _log(self, msg: str):
        _log.agent(self.name, msg)
