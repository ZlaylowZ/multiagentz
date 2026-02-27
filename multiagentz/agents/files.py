# multiagentz/agents/files.py
"""
FileHandlerAgent — watches arbitrary files/directories and answers
questions about their contents. Not tied to any specific project.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Optional

from multiagentz.llm_client import LLMClient
from multiagentz.agents.base import READABLE_EXTENSIONS, MAX_FILE_BYTES


class FileHandlerAgent:
    """General-purpose file/directory reader with persistent watch list."""

    SKIP_DIRS = {
        "__pycache__", "node_modules", ".git", ".venv", "venv",
        "dist", "build", ".next", ".cache", "coverage", ".turbo",
        ".pytest_cache", ".mypy_cache", ".ruff_cache", "egg-info",
    }

    MAX_CONTEXT_CHARS = 600_000

    def __init__(self, watched_paths: Optional[list[str]] = None, llm_client: Optional[LLMClient] = None):
        self.name = "files"
        self.watched_paths: list[Path] = []
        self._llm = llm_client or LLMClient()
        # Per-working-directory watch file (not global)
        self._watch_file = Path(".maz_watched.json").resolve()
        self._load_watched()
        if watched_paths:
            for p in watched_paths:
                self.add_path(p)

    @property
    def description(self) -> str:
        return "Reads and answers questions about user-specified files and directories."

    # ── Watch list management ───────────────────────────────────────────

    def _load_watched(self):
        if self._watch_file.exists():
            try:
                self.watched_paths = [
                    Path(p) for p in json.loads(self._watch_file.read_text())
                    if Path(p).exists()
                ]
            except Exception:
                self.watched_paths = []

    def _save_watched(self):
        try:
            self._watch_file.write_text(json.dumps([str(p) for p in self.watched_paths]))
        except Exception:
            pass

    def add_path(self, path: str) -> str:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"Path not found: {path}"
        if p in self.watched_paths:
            return f"Already watching: {p}"
        self.watched_paths.append(p)
        self._save_watched()
        return f"Added: {p}"

    def remove_path(self, path: str) -> str:
        p = Path(path).expanduser().resolve()
        if p in self.watched_paths:
            self.watched_paths.remove(p)
            self._save_watched()
            return f"Removed: {p}"
        return f"Not in watch list: {path}"

    def list_watched(self) -> list[str]:
        return [str(p) for p in self.watched_paths]

    def clear_watched(self) -> str:
        count = len(self.watched_paths)
        self.watched_paths = []
        self._save_watched()
        return f"Cleared {count} watched paths"

    def get_context_stats(self) -> dict:
        ctx = self._build_context()
        return {
            "watched_paths": len(self.watched_paths),
            "context_chars": len(ctx),
            "context_tokens_approx": len(ctx) // 4,
        }

    # ── Context building ────────────────────────────────────────────────

    def _build_context(self) -> str:
        if not self.watched_paths:
            return "No files or folders are currently being watched."

        sections = []
        total = 0
        files_read = 0

        for path in self.watched_paths:
            if not path.exists():
                sections.append(f"=== {path} ===\n[Path no longer exists]")
                continue

            if path.is_file():
                content = self._read_safe(path)
                sections.append(f"=== FILE: {path} ===\n{content}")
                total += len(content)
                files_read += 1

            elif path.is_dir():
                tree = self._build_tree(path)
                sections.append(f"=== DIRECTORY TREE: {path} ===\n{tree}")
                total += len(tree)

                for root, dirs, files in os.walk(path):
                    dirs[:] = [d for d in dirs if d not in self.SKIP_DIRS and not d.startswith(".")]
                    for fn in sorted(files):
                        if total > self.MAX_CONTEXT_CHARS:
                            break
                        fp = Path(root) / fn
                        if fp.suffix.lower() not in READABLE_EXTENSIONS:
                            continue
                        if fp.name.startswith("."):
                            continue
                        try:
                            if fp.stat().st_size > MAX_FILE_BYTES:
                                continue
                        except OSError:
                            continue
                        content = self._read_safe(fp)
                        try:
                            rel = fp.relative_to(path)
                        except ValueError:
                            rel = fp.name
                        sections.append(f"\n--- {rel} ---\n{content}")
                        total += len(content)
                        files_read += 1
                    if total > self.MAX_CONTEXT_CHARS:
                        break

        sections.insert(0, f"[Context: {files_read} files, {total:,} chars]")
        return "\n\n".join(sections)

    def _read_safe(self, fp: Path, max_chars: int = 200_000) -> str:
        try:
            content = fp.read_text(encoding="utf-8", errors="replace")[:max_chars]
            if len(content) == max_chars:
                content += "\n... [truncated]"
            return content
        except Exception as e:
            return f"[Error reading: {e}]"

    def _build_tree(self, dirpath: Path, max_depth: int = 3) -> str:
        lines: list[str] = []

        def walk(p: Path, depth: int, prefix: str = ""):
            if depth > max_depth:
                return
            try:
                items = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
            except PermissionError:
                return
            for item in items:
                if item.name in self.SKIP_DIRS or item.name.startswith("."):
                    continue
                if item.is_dir():
                    lines.append(f"{prefix}{item.name}/")
                    walk(item, depth + 1, prefix + "  ")
                else:
                    lines.append(f"{prefix}{item.name}")

        walk(dirpath, 0)
        return "\n".join(lines[:500])

    # ── Query ───────────────────────────────────────────────────────────

    def query(self, question: str) -> str:
        if not self.watched_paths:
            return "No files or folders are being watched. Use /watch <path> to add files."

        context = self._build_context()
        prompt = f"""I have access to the following files and their contents:

{context}

---

Question: {question}"""

        return self._llm.complete(
            prompt=prompt,
            system=(
                "You are a file analysis agent. You read actual file contents and "
                "answer questions about code structure, patterns, and relationships. "
                "Reference specific files and quote relevant code. Be precise."
            ),
            max_tokens=32768,
        )
