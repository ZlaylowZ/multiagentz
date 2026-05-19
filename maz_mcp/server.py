"""
MAZ MCP Server — FastMCP wrapper for multiagentz orchestration.

Thick tools (6, always registered):
    maz_configure, maz_query, maz_consensus,
    maz_perspective, maz_cross_pollinate, maz_status

Run with:
    python -m maz_mcp
    maz-mcp  (if installed)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional

# MCP SDK
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    try:
        from fastmcp import FastMCP
    except ImportError:
        print("ERROR: MCP SDK not installed. Run: pip install mcp", file=sys.stderr)
        sys.exit(1)
from mcp.server.fastmcp.server import TransportSecuritySettings


# =============================================================================
# Configuration
# =============================================================================

def get_config() -> Dict[str, Any]:
    """Load configuration from environment."""
    return {
        "default_stack": os.getenv("MAZ_DEFAULT_STACK", ""),
        "default_model": os.getenv("MAZ_LLM_MODEL", os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")),
        "anthropic_key": bool(os.getenv("ANTHROPIC_API_KEY")),
        "openai_key": bool(os.getenv("OPENAI_API_KEY")),
        "xai_key": bool(os.getenv("XAI_API_KEY")),
        "google_key": bool(os.getenv("GOOGLE_API_KEY")),
        "max_cached_stacks": int(os.getenv("MAZ_MAX_CACHED_STACKS", "10")),
    }


# =============================================================================
# Stack Cache (module-level, thread-safe)
# =============================================================================

_stack_cache: Dict[str, Any] = {}  # hash -> LeadAgent
_stack_lock = threading.Lock()
_default_lead = None
_default_lead_lock = threading.Lock()

BUNDLED_STACKS_DIR = Path(__file__).parent.parent / "stacks" / "cloud"


def _yaml_hash(yaml_str: str) -> str:
    return hashlib.sha256(yaml_str.encode()).hexdigest()[:16]


def _load_stack_from_yaml(yaml_str: str):
    """Load a stack from YAML string, with caching."""
    cache_key = _yaml_hash(yaml_str)

    with _stack_lock:
        if cache_key in _stack_cache:
            return _stack_cache[cache_key]

    # Write to temp file and load
    from multiagentz.stack import load_stack

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_str)
        tmp_path = f.name

    try:
        lead = load_stack(tmp_path)
    finally:
        os.unlink(tmp_path)

    with _stack_lock:
        # LRU eviction
        config = get_config()
        max_cached = config["max_cached_stacks"]
        if len(_stack_cache) >= max_cached:
            oldest_key = next(iter(_stack_cache))
            del _stack_cache[oldest_key]
        _stack_cache[cache_key] = lead

    return lead


def _load_stack_from_path(config_path: str):
    """Load a stack from a file path."""
    from multiagentz.stack import load_stack
    return load_stack(config_path)


def _get_default_stack():
    """Get or create the default stack."""
    global _default_lead

    with _default_lead_lock:
        if _default_lead is not None:
            return _default_lead

    # Check for configured default stack
    config = get_config()
    default_path = config["default_stack"]

    if default_path and Path(default_path).exists():
        lead = _load_stack_from_path(default_path)
    else:
        # Load bundled cloud stack
        bundled = BUNDLED_STACKS_DIR / "default.yaml"
        if bundled.exists():
            lead = _load_stack_from_path(str(bundled))
        else:
            # Fallback: create minimal stack programmatically
            lead = _create_minimal_stack()

    with _default_lead_lock:
        _default_lead = lead

    return lead


def _create_minimal_stack():
    """Create a minimal 3-agent stack for general-purpose use."""
    from multiagentz.agents.base import SubAgent
    from multiagentz.lead import LeadAgent
    from multiagentz.llm_client import LLMClient

    agents = {
        "technical": SubAgent(
            name="technical",
            repo_path=".",
            description="Senior software engineer and technical architect",
            system_prompt=(
                "You are a senior software engineer and technical architect with 20+ years of experience. "
                "You specialize in system design, distributed systems, API architecture, database design, "
                "performance optimization, and code quality. Provide detailed, actionable technical guidance."
            ),
        ),
        "strategy": SubAgent(
            name="strategy",
            repo_path=".",
            description="Strategic business consultant and operations expert",
            system_prompt=(
                "You are a seasoned business strategy consultant with expertise in market analysis, "
                "competitive strategy, growth planning, operations optimization, and risk assessment. "
                "Provide clear strategic recommendations backed by frameworks and industry knowledge."
            ),
        ),
        "research": SubAgent(
            name="research",
            repo_path=".",
            description="Research analyst and data scientist",
            system_prompt=(
                "You are an expert research analyst and data scientist. You excel at literature review, "
                "data analysis, statistical methodology, trend identification, and synthesizing complex "
                "information into clear insights. Be rigorous, cite your reasoning, and flag uncertainty."
            ),
        ),
    }

    keywords = {
        "technical": {"code", "software", "architecture", "api", "database", "system", "performance", "debug", "deploy", "engineering"},
        "strategy": {"business", "strategy", "market", "growth", "risk", "competition", "pricing", "revenue", "operations"},
        "research": {"research", "data", "analysis", "trends", "study", "statistics", "methodology", "survey"},
    }

    return LeadAgent(
        name="maz-cloud",
        agents=agents,
        keywords=keywords,
    )


def _resolve_stack(stack_yaml: str = ""):
    """Resolve which stack to use: custom YAML or default."""
    if stack_yaml and stack_yaml.strip():
        return _load_stack_from_yaml(stack_yaml)
    return _get_default_stack()


def _stack_info(lead) -> Dict[str, Any]:
    """Extract stack info for status/configure responses."""
    agents_info = {}
    for name, agent in lead.agents.items():
        agents_info[name] = {
            "type": type(agent).__name__,
            "description": getattr(agent, "description", ""),
        }

    return {
        "name": lead.name,
        "mode": getattr(lead, "_orchestration_mode", "standard"),
        "agents": agents_info,
        "agent_count": len(lead.agents),
    }


def _format_error(e: Exception) -> Dict[str, Any]:
    """Format exception into agent-friendly error response."""
    return {
        "error": True,
        "error_type": type(e).__name__,
        "message": str(e),
    }


# =============================================================================
# MCP Server
# =============================================================================

mcp = FastMCP(
    name="maz",
    host="0.0.0.0",
    port=8080,
    stateless_http=True,
    instructions=(
        "MAZ (multiagentz) — multi-agent orchestration framework. "
        "6 tools: maz_configure, maz_query, maz_consensus, maz_perspective, maz_cross_pollinate, maz_status. "
        "Workflow: optionally maz_configure (load custom stack) → maz_query / maz_consensus / maz_perspective / maz_cross_pollinate. "
        "If no stack is configured, a default general-purpose 3-agent stack (technical, strategy, research) is used. "
        "maz_query routes questions to the best expert agent(s) and synthesizes responses. "
        "maz_consensus runs iterative conflict resolution across agents. "
        "maz_perspective runs 4-phase deep analysis: bootstrap Q&A → independent solutions → refinement → synthesis. "
        "maz_cross_pollinate runs A/B twin analysis: parallel query → output swap → refinement → reconciliation. "
        "All query tools accept optional stack_yaml to use a custom agent stack inline. "
        "Use maz_status to check server state and available agents."
    ),
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
        allowed_hosts=["*"],
        allowed_origins=[
            "https://claude.ai",
            "https://chatgpt.com",
            "https://chat.openai.com",
            "http://localhost:3000",
        ],
    ),
)


# =============================================================================
# Register Tools
# =============================================================================

from .tools import register_thick_tools

register_thick_tools(mcp)


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Run the MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="MAZ MCP Server")
    parser.add_argument("--transport", choices=["stdio", "http", "sse"], default="stdio",
                        help="Transport type (default: stdio)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port for HTTP/SSE transport (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host for HTTP/SSE transport (default: 127.0.0.1)")
    parser.add_argument("--version", action="store_true",
                        help="Show version and exit")

    args = parser.parse_args()

    if args.version:
        from . import __version__
        print(f"maz-mcp version {__version__}")
        sys.exit(0)

    config = get_config()
    print("Starting MAZ MCP Server...", file=sys.stderr)
    print(f"  Default model: {config['default_model']}", file=sys.stderr)
    print(f"  Anthropic key: {'configured' if config['anthropic_key'] else 'NOT SET'}", file=sys.stderr)
    print(f"  OpenAI key: {'configured' if config['openai_key'] else 'NOT SET'}", file=sys.stderr)
    print(f"  Default stack: {config['default_stack'] or 'built-in'}", file=sys.stderr)
    print(f"  Transport: {args.transport}", file=sys.stderr)

    if args.transport == "http":
        os.environ.setdefault("HOST", args.host)
        os.environ.setdefault("PORT", str(args.port))
        print(f"  Listening: {args.host}:{args.port}", file=sys.stderr)
        mcp.run(transport="streamable-http")
    elif args.transport == "sse":
        os.environ.setdefault("HOST", args.host)
        os.environ.setdefault("PORT", str(args.port))
        print(f"  Listening: {args.host}:{args.port}", file=sys.stderr)
        mcp.run(transport="sse")
    else:
        print("  Transport: stdio", file=sys.stderr)
        mcp.run()


if __name__ == "__main__":
    main()
