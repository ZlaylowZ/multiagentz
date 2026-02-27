# multiagentz/log.py
"""
Pretty, human-readable logging for multi-agent orchestration.

Produces clean, color-coded terminal output with visual hierarchy:

    ◆  LEAD agent decisions
    →  Orchestration steps (actions in progress)
    ✓  Success / completion
    ⚠  Warnings
    ✗  Errors

Normal mode shows only high-level flow. Set ``log.verbose = True``
for full debug detail (coordinator routing, sub-agent file loading,
LLM calls).
"""

import sys
import time

# ── ANSI color codes (auto-detected) ────────────────────────

_OK = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"{code}{text}\033[0m" if _OK else text


_DIM    = "\033[2m"
_BLUE   = "\033[34m"
_CYAN   = "\033[36m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_MAG    = "\033[35m"
_B_BLUE = "\033[1;34m"
_B_CYAN = "\033[1;36m"
_B_GRN  = "\033[1;32m"
_B_YEL  = "\033[1;33m"
_B_RED  = "\033[1;31m"
_B_MAG  = "\033[1;35m"
_B_WHT  = "\033[1;37m"

# ── Timer state ──────────────────────────────────────────────

_t0: float = 0.0
verbose: bool = False


def _ts() -> str:
    """Elapsed timestamp with trailing space, or empty string."""
    if not _t0:
        return ""
    e = time.time() - _t0
    if e < 60:
        s = f"{e:4.0f}s"
    else:
        m, sec = divmod(int(e), 60)
        s = f"{m}m{sec:02d}s"
    return _c(_DIM, f"[{s:>5}] ")


def _pr(text: str):
    print(text, flush=True)


# ── Public API ───────────────────────────────────────────────

def init():
    """Start / reset the orchestration timer."""
    global _t0
    _t0 = time.time()


def phase(num: int, total: int, title: str):
    """Major phase banner — highly visible."""
    bar = _c(_B_BLUE, "━" * 52)
    label = _c(_B_WHT, f"  Phase {num}/{total}")
    desc = _c(_B_BLUE, f"  {title}")
    _pr(f"\n{bar}\n{label}{desc}\n{bar}")


def done(msg: str = ""):
    """Final completion banner."""
    if not _t0:
        return
    e = time.time() - _t0
    m, s = divmod(int(e), 60)
    bar = _c(_B_GRN, "━" * 52)
    label = _c(_B_GRN, "  ✓ Done")
    timing = _c(_DIM, f"  ({m}m {s:02d}s)")
    extra = f"  {msg}" if msg else ""
    _pr(f"\n{bar}\n{label}{timing}{extra}\n{bar}\n")


def step(msg: str):
    """Orchestration step — action in progress."""
    _pr(f" {_ts()}{_c(_B_BLUE, '→')} {msg}")


def ok(msg: str):
    """Success / sub-task completion."""
    _pr(f" {_ts()}{_c(_B_GRN, '✓')} {_c(_GREEN, msg)}")


def warn(msg: str):
    """Warning — always visible."""
    _pr(f" {_ts()}{_c(_B_YEL, '⚠')} {_c(_YELLOW, msg)}")


def error(msg: str):
    """Error — always visible."""
    _pr(f" {_ts()}{_c(_B_RED, '✗')} {_c(_RED, msg)}")


def lead(name: str, msg: str):
    """LEAD agent message — always visible."""
    _pr(f" {_ts()}{_c(_B_MAG, '◆')} {_c(_MAG, name)}  {msg}")


def coord(name: str, msg: str):
    """Coordinator-level detail — verbose only."""
    if not verbose:
        return
    _pr(f" {_ts()}  {_c(_B_CYAN, '▸')} {_c(_CYAN, name)}  {msg}")


def agent(name: str, msg: str):
    """Sub-agent detail — verbose only."""
    if not verbose:
        return
    _pr(f" {_ts()}    {_c(_DIM, '·')} {_c(_DIM, name)}  {_c(_DIM, msg)}")


def detail(msg: str):
    """Extra detail — verbose only."""
    if not verbose:
        return
    _pr(f"            {_c(_DIM, msg)}")
