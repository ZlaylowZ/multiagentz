FROM python:3.12.15-slim@sha256:cb33f7f771ecec2ef1225cf39b159591a040391bd73e4c01f00fdfc4c47611d2

# ── Security (eMudhra/emSign CA incident, 2026-10-22) ───────────
# 1. Upgrade the OS CA bundle first, as root, before anything talks TLS.
RUN apt-get update \
    && apt-get install -y --no-install-recommends --only-upgrade ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Install patched pip 26.2.2 BEFORE any other pip install. The official
#    3.12 image ships pip 25.0.1 and will not be bumped upstream; pip
#    verifies the sha256 in the URL fragment.
RUN python -m pip install --no-cache-dir "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"

# ── Core multiagentz package ──────────────────────────────────────
COPY multiagentz/     ./multiagentz/
COPY pyproject.toml   ./
# certifi>=2026.10.19 is declared in pyproject.toml; it is repeated here so
# the floor holds in the image regardless of transitive resolution order.
RUN pip install --no-cache-dir ".[all]" "certifi>=2026.10.19"

# ── MCP dependencies ──────────────────────────────────────────────
RUN pip install --no-cache-dir "mcp>=1.0.0,<2.0.0" uvicorn starlette

# ── MCP server + bundled stacks ───────────────────────────────────
COPY maz_mcp/         ./maz_mcp/
COPY stacks/          ./stacks/
COPY entrypoint.py    ./

CMD ["python", "entrypoint.py"]
