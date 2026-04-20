import os
import sys
import uvicorn
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from maz_mcp.server import mcp

ALLOWED_ORIGINS = [
    "https://claude.ai",
    "https://chatgpt.com",
    "https://chat.openai.com",
    "http://localhost:3000",
]


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Optional bearer token auth. Set MAZ_AUTH_TOKEN to enable."""

    def __init__(self, app, token: str):
        super().__init__(app)
        self.token = token

    async def dispatch(self, request: Request, call_next):
        # Allow CORS preflight through
        if request.method == "OPTIONS":
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth[7:] != self.token:
            return JSONResponse(
                {"error": "Unauthorized", "message": "Invalid or missing Bearer token"},
                status_code=401,
            )
        return await call_next(request)


if __name__ == "__main__":
    app = mcp.streamable_http_app()

    # CORS must be added first
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Accept", "Authorization", "Mcp-Session-Id", "mcp-protocol-version"],
        expose_headers=["Mcp-Session-Id"],
        allow_credentials=False,
    )

    # Optional bearer auth — set MAZ_AUTH_TOKEN to enable
    auth_token = os.environ.get("MAZ_AUTH_TOKEN", "").strip()
    if auth_token:
        app.add_middleware(BearerAuthMiddleware, token=auth_token)
        print("  Auth: Bearer token ENABLED", file=sys.stderr)
    else:
        print("  Auth: DISABLED (set MAZ_AUTH_TOKEN to enable)", file=sys.stderr)

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
