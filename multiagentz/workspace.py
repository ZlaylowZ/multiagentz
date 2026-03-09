# multiagentz/workspace.py
"""
Workspace — file tree scanning utility for builder agents.

Provides project structure snapshots used by ArchitectAgent (for planning)
and BuilderAgent (for context). Pure utility — no LLM dependency.

Constants are duplicated from agents/base.py to avoid circular imports
(workspace ← agents/__init__ ← builder ← workspace).
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Optional


# ── Shared constants (mirrored from agents/base.py) ─────────────────────

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

MAX_FILE_BYTES: int = 500_000      # 500 KB per file
MAX_WALK_DEPTH: int = 8


class Workspace:
    """
    A scannable project workspace.

    Provides file tree views, file listing, and file reading with
    the same filtering rules used by SubAgent.
    """

    def __init__(
        self,
        root_path: str,
        readable_extensions: Optional[set[str]] = None,
        exclude_patterns: Optional[list[str]] = None,
        max_depth: int = MAX_WALK_DEPTH,
    ):
        self.root = Path(root_path).resolve()
        self.readable_extensions = readable_extensions or READABLE_EXTENSIONS
        self.exclude_patterns = exclude_patterns or EXCLUDE_PATTERNS
        self.max_depth = max_depth

    # ── Public API ──────────────────────────────────────────────────────

    def file_tree(self) -> str:
        """
        Return an indented tree representation of the workspace.

        Used by ArchitectAgent to give the LLM a view of project structure.
        """
        lines: list[str] = [f"{self.root.name}/"]
        self._build_tree(self.root, "", lines, depth=0)
        return "\n".join(lines)

    def list_files(self) -> list[str]:
        """Return a flat list of all readable file paths (relative to root)."""
        files: list[str] = []
        for root_dir, dirs, filenames in os.walk(self.root):
            depth = len(Path(root_dir).relative_to(self.root).parts)
            if depth > self.max_depth:
                dirs.clear()
                continue
            dirs[:] = sorted(d for d in dirs if not self._should_exclude(d))
            for fn in sorted(filenames):
                if self._should_exclude(fn):
                    continue
                fp = Path(root_dir) / fn
                if fp.suffix.lower() not in self.readable_extensions:
                    continue
                try:
                    if fp.stat().st_size > MAX_FILE_BYTES:
                        continue
                except OSError:
                    continue
                try:
                    files.append(str(fp.relative_to(self.root)))
                except ValueError:
                    files.append(str(fp))
        return files

    def read_file(self, rel_path: str) -> str:
        """Read a single file from the workspace. Returns empty string on error."""
        fp = (self.root / rel_path).resolve()
        # Security: ensure path is within workspace
        if not fp.is_relative_to(self.root):
            return ""
        try:
            if fp.stat().st_size > MAX_FILE_BYTES:
                return ""
            return fp.read_text(encoding="utf-8")
        except Exception:
            return ""

    # ── Internal ────────────────────────────────────────────────────────

    def _build_tree(
        self, dirpath: Path, prefix: str, lines: list[str], depth: int
    ):
        """Recursively build indented tree lines."""
        if depth > self.max_depth:
            return

        entries = []
        try:
            for entry in sorted(dirpath.iterdir(), key=lambda e: (e.is_file(), e.name)):
                if self._should_exclude(entry.name):
                    continue
                entries.append(entry)
        except PermissionError:
            return

        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
            extension = "    " if is_last else "\u2502   "

            if entry.is_dir():
                lines.append(f"{prefix}{connector}{entry.name}/")
                self._build_tree(entry, prefix + extension, lines, depth + 1)
            elif entry.is_file():
                if entry.suffix.lower() in self.readable_extensions:
                    lines.append(f"{prefix}{connector}{entry.name}")

    def _should_exclude(self, name: str) -> bool:
        """Check if a file or directory name matches exclusion patterns."""
        return any(fnmatch.fnmatch(name, pat) for pat in self.exclude_patterns)
