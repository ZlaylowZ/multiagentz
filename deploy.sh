#!/usr/bin/env bash
# deploy.sh — Pre-flight check + Cloud Run deploy for maz-mcp
# Usage: ./deploy.sh
set -euo pipefail

REGION="us-central1"
SERVICE="maz-mcp"

# ── Colors ────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo -e "${YELLOW}━━━ MAZ MCP Pre-Deploy Checks ━━━${NC}"

# 1. Verify core package exists
if [[ ! -f "multiagentz/__init__.py" ]]; then
    echo -e "${RED}✗ FATAL: multiagentz/ package not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ multiagentz package found${NC}"

# 2. Verify MCP server exists
if [[ ! -f "maz_mcp/server.py" ]]; then
    echo -e "${RED}✗ FATAL: maz_mcp/server.py not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ maz_mcp server found${NC}"

# 3. Verify entrypoint
if [[ ! -f "entrypoint.py" ]]; then
    echo -e "${RED}✗ FATAL: entrypoint.py not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ entrypoint.py found${NC}"

# 4. Verify Dockerfile
if [[ ! -f "Dockerfile" ]]; then
    echo -e "${RED}✗ FATAL: Dockerfile not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Dockerfile found${NC}"

# 5. Check tool count
TOOL_COUNT=$(grep -c '@mcp.tool()' maz_mcp/tools.py || true)
echo -e "${GREEN}✓ ${TOOL_COUNT} thick tools registered${NC}"

echo ""
echo -e "${YELLOW}━━━ Deploying to Cloud Run ━━━${NC}"

gcloud run deploy "$SERVICE" \
    --source . \
    --region "$REGION" \
    --allow-unauthenticated \
    --min-instances=2 \
    --max-instances=4 \
    --cpu=2 \
    --memory=1Gi \
    --execution-environment=gen2 \
    --set-secrets="ANTHROPIC_API_KEY=anthropic-api-key:latest" \
    --set-env-vars="MAZ_LLM_MODEL=claude-sonnet-4-20250514"

echo ""
echo -e "${GREEN}━━━ Deploy complete ━━━${NC}"
echo "  Test with: maz_status()"
echo "  Then: maz_query(question='What is the best approach to microservices?')"
