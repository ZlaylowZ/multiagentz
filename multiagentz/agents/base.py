# multiagentz/agents/base.py
"""
SubAgent — the universal building block.

A SubAgent is a domain expert backed by a local code/doc repo. It loads
key files into the LLM context window and answers questions grounded
exclusively in that content. Everything domain-specific (name, description,
system prompt, file list) is injected via constructor or YAML config —
no subclassing required for simple agents.
"""

from __future__ import annotations

import os
import fnmatch
from pathlib import Path
from typing import Optional

from multiagentz.llm_client import LLMClient, CompletionResult


# ── Defaults ────────────────────────────────────────────────────────────

READABLE_EXTENSIONS: set[str] = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".yaml", ".yml",
    ".md", ".txt", ".rst", ".toml", ".cfg", ".ini",
    ".html", ".css", ".scss", ".sql", ".sh",
    ".svelte", ".vue", ".go", ".rs", ".rb", ".php", ".java", ".kt",
}

EXCLUDE_PATTERNS: list[str] = [
    "__pycache__", ".venv", "venv", ".env", "*.pyc", ".pytest_cache",
    "*.egg-info", ".mypy_cache", ".ruff_cache",
    "node_modules", "package-lock.json", "pnpm-lock.yaml",
    ".DS_Store", "Thumbs.db", ".git",
    "dist", "build", "*.so", "*.dylib",
    "*.min.js", "*.min.css", "*.map",
]

MAX_CONTEXT_CHARS: int = 600_000   # ~150K tokens
MAX_FILE_BYTES: int = 500_000      # 500 KB per file
MAX_WALK_DEPTH: int = 8
MAX_CONTINUATIONS: int = 6


class SubAgent:
    """
    A domain-expert agent backed by a local file tree.

    Can be instantiated directly from config (no subclassing needed):

        agent = SubAgent(
            name="backend",
            repo_path="/path/to/repo",
            description="Expert on the backend API.",
            system_prompt="You are an expert on ...",
            key_files=["src/", "README.md"],
        )

    Or subclassed for advanced behavior (tiered loading, custom query logic).
    """

    def __init__(
        self,
        name: str,
        repo_path: str,
        description: str = "",
        system_prompt: str = "",
        key_files: Optional[list[str]] = None,
        max_tokens: int = 32768,
        readable_extensions: Optional[set[str]] = None,
        exclude_patterns: Optional[list[str]] = None,
        max_context_chars: int = MAX_CONTEXT_CHARS,
        llm_client: Optional[LLMClient] = None
    ):
        self.name = name
        self.repo_path = Path(repo_path).resolve()
        self._description = description or f"Expert on {name}"
        self._system_prompt_text = system_prompt
        self.key_files = key_files or []
        self.max_tokens = max_tokens
        self.readable_extensions = readable_extensions or READABLE_EXTENSIONS
        self.exclude_patterns = exclude_patterns or EXCLUDE_PATTERNS
        self.max_context_chars = max_context_chars
        self._llm = llm_client or LLMClient()

    # ── Public interface ────────────────────────────────────────────────

    @property
    def description(self) -> str:
        return self._description

    def query(self, question: str, include_files: Optional[list[str]] = None) -> str:
        """Answer a question using loaded file context + LLM."""
        file_context = self._load_file_contents()

        if include_files:
            for fp in include_files:
                try:
                    content = Path(fp).read_text(encoding="utf-8")
                    if len(content) > 200_000:
                        content = content[:200_000] + "\n... [truncated]"
                    file_context += f"\n\n=== {fp} ===\n{content}"
                except Exception as e:
                    print(f"[{self.name}] Warning: couldn't read {fp}: {e}")

        full_system = self._build_full_system(file_context)

        try:
            response = self._llm.complete(
                prompt=question, system=full_system, max_tokens=self.max_tokens,
            )
            return self._continue_if_truncated(question, full_system, response)
        except Exception as e:
            return f"Error querying {self.name}: {e}"

    # ── System prompt assembly ──────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        """Override in subclasses for dynamic prompts; default uses the stored text."""
        return self._system_prompt_text

    def _build_full_system(self, file_context: str) -> str:
        base = self._build_system_prompt()

        grounding = """

RESPONSE GUIDELINES:
- Ground every claim in the Reference Files provided below. These files are
  your ONLY source of truth. Do not invent function names, class names,
  file names, parameter names, or behaviors that do not appear in them.
- If the Reference Files do not contain enough information to fully answer
  the question, say so explicitly.
- Be concise and direct. Focus on answering the specific question.
"""

        if file_context:
            files_section = f"""
## Reference Files (YOUR ONLY SOURCE OF TRUTH)
{file_context}"""
        else:
            files_section = """
## Reference Files
No reference files were loaded. State this clearly and decline to make
specific claims about code structure or behavior.
"""

        return f"{base}{grounding}{files_section}\n\nRepository location: {self.repo_path}"

    # ── Continuation loop ───────────────────────────────────────────────

    def _continue_if_truncated(
        self, question: str, full_system: str, response: CompletionResult,
    ) -> str:
        parts = [str(response)]

        for _ in range(MAX_CONTINUATIONS):
            if not getattr(response, "truncated", False):
                break

            print(f"[{self.name}] Response truncated — requesting continuation…")
            continuation = (
                f"{question}\n\n"
                "--- PARTIAL RESPONSE SO FAR (do NOT repeat this) ---\n"
                f"{parts[-1][-2000:]}\n"
                "--- END PARTIAL ---\n\n"
                "Continue EXACTLY where the partial response left off. "
                "Do not repeat any content already provided."
            )
            response = self._llm.complete(
                prompt=continuation, system=full_system, max_tokens=self.max_tokens,
            )
            parts.append(str(response))

        full = "\n".join(parts)

        if getattr(response, "truncated", False):
            full += (
                "\n\n---\n"
                "⚠️ **Response truncated** — output exceeded max generation length. "
                "Ask a more focused question or request output in sections."
            )

        return full

    # ── File loading ────────────────────────────────────────────────────

    def _resolve_key_files(self) -> list[str]:
        resolved = []
        for rel in self.key_files:
            full = self.repo_path / rel
            if not full.exists():
                print(f"[{self.name}] Warning: key file not found: {rel}")
                continue
            if full.is_file():
                resolved.append(str(full))
            elif full.is_dir():
                dir_files = self._walk_directory(full)
                resolved.extend(dir_files)
                if dir_files:
                    print(f"[{self.name}] Loaded {len(dir_files)} files from {rel}")
        return resolved

    def _walk_directory(self, dirpath: Path, max_depth: int = MAX_WALK_DEPTH) -> list[str]:
        files = []
        for root, dirs, filenames in os.walk(dirpath):
            depth = len(Path(root).relative_to(dirpath).parts)
            if depth > max_depth:
                dirs.clear()
                continue
            dirs[:] = [d for d in dirs if not self._should_exclude(d)]
            for fn in sorted(filenames):
                if self._should_exclude(fn):
                    continue
                fp = Path(root) / fn
                if fp.suffix.lower() not in self.readable_extensions:
                    continue
                try:
                    if fp.stat().st_size > MAX_FILE_BYTES:
                        continue
                except OSError:
                    continue
                files.append(str(fp))
        return files

    def _should_exclude(self, name: str) -> bool:
        base = Path(name).name
        return any(fnmatch.fnmatch(base, pat) for pat in self.exclude_patterns)

    def _load_file_contents(self) -> str:
        sections = []
        total = 0
        loaded = 0
        skipped = 0

        for fp in self._resolve_key_files():
            if self._should_exclude(fp):
                continue
            if total > self.max_context_chars:
                skipped += 1
                continue
            try:
                content = Path(fp).read_text(encoding="utf-8")
                if len(content) > 200_000:
                    content = content[:200_000] + "\n... [truncated]"
                try:
                    display = str(Path(fp).relative_to(self.repo_path))
                except ValueError:
                    display = fp
                sections.append(f"=== {display} ===\n{content}")
                total += len(content)
                loaded += 1
            except IsADirectoryError:
                pass
            except Exception as e:
                print(f"[{self.name}] Warning: couldn't read {fp}: {e}")

        if skipped:
            print(f"[{self.name}] Warning: skipped {skipped} files (context limit)")
        if loaded:
            print(f"[{self.name}] Loaded {loaded} files ({total:,} chars)")

        return "\n\n".join(sections)
