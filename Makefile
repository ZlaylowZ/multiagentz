SHELL    := /bin/bash
SERVICE  := maz-mcp
REGION   := us-central1

.PHONY: help deploy verify verify-tools run run-http docker-build docker-run clean

help: ## Show this help
	@grep -E '^[a-z_-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ── Deploy ───────────────────────────────────────────────────

deploy: verify ## Verify + deploy to Cloud Run
	./deploy.sh

# ── Verification ─────────────────────────────────────────────

verify: verify-tools ## Run all checks

verify-tools: ## Confirm thick tools register correctly
	@python3 -c "\
	from maz_mcp.server import mcp; \
	tools = list(mcp._tool_manager._tools.keys()); \
	assert len(tools) == 6, f'Expected 6 thick tools, got {len(tools)}: {tools}'; \
	print(f'OK: {len(tools)} thick tools registered: {tools}')"

# ── Local Dev ────────────────────────────────────────────────

run: ## Run MCP server locally (stdio)
	python3 -m maz_mcp

run-http: ## Run MCP server locally (streamable-http :8000)
	python3 -m maz_mcp --transport http --port 8000

# ── Docker ───────────────────────────────────────────────────

docker-build: ## Build Docker image locally
	docker build -t $(SERVICE) .

docker-run: docker-build ## Build + run in Docker
	docker run --rm -p 8080:8080 \
		-e ANTHROPIC_API_KEY="$${ANTHROPIC_API_KEY}" \
		$(SERVICE)

# ── Cleanup ──────────────────────────────────────────────────

clean: ## Remove build artifacts
	rm -rf dist build *.egg-info __pycache__ maz_mcp/__pycache__
