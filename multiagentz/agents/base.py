# multiagentz/agents/base.py
"""
SubAgent — the universal building block.

A SubAgent is a domain expert backed by a local code/doc repo. It loads
key files into the LLM context window and answers questions grounded
exclusively in that content. Everything domain-specific (name, description,
system prompt, file list) is injected via constructor or YAML config —
no subclassing required for simple agents.

File contents are cached after the first load and reused for subsequent
queries, eliminating redundant disk reads and log noise.
"""

from __future__ import annotations

import os
import fnmatch
import time
from pathlib import Path
from typing import Optional

from multiagentz.llm_client import LLMClient, CompletionResult
from multiagentz import log as _log


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
        self._file_cache: Optional[str] = None  # Cached file context
        self._file_cache_count: int = 0          # Number of cached files
        self._file_cache_chars: int = 0          # Total cached chars

    # ── Public interface ────────────────────────────────────────────────

    @property
    def description(self) -> str:
        return self._description

    def query(self, question: str, include_files: Optional[list[str]] = None) -> str:
        """Answer a question using loaded file context + LLM."""
        t0 = time.time()
        file_context = self._load_file_contents()

        if include_files:
            for fp in include_files:
                try:
                    content = Path(fp).read_text(encoding="utf-8")
                    if len(content) > 200_000:
                        content = content[:200_000] + "\n... [truncated]"
                    file_context += f"\n\n=== {fp} ===\n{content}"
                except Exception as e:
                    _log.warn(f"{self.name}: couldn't read extra file {fp}: {e}")

        full_system = self._build_full_system(file_context)

        try:
            q_preview = question[:120].replace("\n", " ")
            self._log(f"Sending to LLM ({self._llm.provider}/{self._llm.model})  \"{q_preview}...\"")
            response = self._llm.complete(
                prompt=question, system=full_system, max_tokens=self.max_tokens,
            )
            result = self._continue_if_truncated(question, full_system, response)
            elapsed = time.time() - t0
            self._log(f"Done ({elapsed:.1f}s, {len(result):,} chars)")
            return result
        except Exception as e:
            _log.error(f"{self.name}: {e}")
            return f"Error querying {self.name}: {e}"

    def _log(self, msg: str):
        """Route to centralized logger (verbose only for sub-agents)."""
        _log.agent(self.name, msg)

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

            _log.warn(f"{self.name}: response truncated, continuing...")
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
                self._log(f"Warning: key file not found: {rel}")
                continue
            if full.is_file():
                resolved.append(str(full))
            elif full.is_dir():
                dir_files = self._walk_directory(full)
                resolved.extend(dir_files)
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
        """Load file contents, using cache after the first call."""
        if self._file_cache is not None:
            return self._file_cache

        # First load — read from disk and cache
        t0 = time.time()
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
                self._log(f"Warning: couldn't read {fp}: {e}")

        elapsed = time.time() - t0
        if skipped:
            self._log(f"Warning: {skipped} files skipped (context limit reached)")
        self._log(f"Loaded {loaded} files ({total:,} chars) from disk in {elapsed:.2f}s")

        self._file_cache = "\n\n".join(sections)
        self._file_cache_count = loaded
        self._file_cache_chars = total
        return self._file_cache

    def invalidate_cache(self):
        """Force re-read of files on next query (e.g. after file changes)."""
        self._file_cache = None
        self._file_cache_count = 0
        self._file_cache_chars = 0
