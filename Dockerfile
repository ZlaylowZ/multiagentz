FROM python:3.12-slim

WORKDIR /app

# ── Core multiagentz package ──────────────────────────────────────
COPY multiagentz/     ./multiagentz/
COPY pyproject.toml   ./
RUN pip install --no-cache-dir ".[all]"

# ── MCP dependencies ──────────────────────────────────────────────
RUN pip install --no-cache-dir "mcp>=1.0.0,<2.0.0" uvicorn starlette

# ── MCP server + bundled stacks ───────────────────────────────────
COPY maz_mcp/         ./maz_mcp/
COPY stacks/          ./stacks/
COPY entrypoint.py    ./

CMD ["python", "entrypoint.py"]
