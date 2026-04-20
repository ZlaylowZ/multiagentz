"""
MAZ MCP Server — Multi-agent orchestration as a Cloud Run MCP service.

Install: pip install multiagentz[mcp]
Run: python -m maz_mcp (or maz-mcp CLI)

Exposes 6 thick MCP tools:
    maz_configure, maz_query, maz_consensus,
    maz_perspective, maz_cross_pollinate, maz_status
"""

from .server import mcp, main, get_config

__version__ = "0.1.0"
__all__ = ["mcp", "main", "get_config"]
