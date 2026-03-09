# multiagentz/agents/service.py
"""
ServiceAgent — an HTTP proxy agent for external services.

Instead of using a local LLM + file context, this agent delegates
queries to an external HTTP service. Designed for integrating hosted
agent services (like AAAP/NukezAgent) into a multiagentz stack.

The agent auto-provisions an instance on first use and reuses it
for subsequent queries within the session.
"""

from __future__ import annotations

import time
from typing import Optional

import httpx

from multiagentz import log as _log


class ServiceAgent:
    """
    An agent that proxies queries to an external HTTP service.

    Implements the same query(question) -> str interface as SubAgent,
    but sends the question to a remote service instead of a local LLM.

    YAML config example:
        nukez_agent:
          type: service
          base_url: http://localhost:8080
          endpoint: /v1/chat
          description: "Nukez storage specialist agent."
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        description: str = "",
        endpoint: str = "/v1/chat",
        instance_id: Optional[str] = None,
        timeout: int = 120,
    ):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self._description = description or f"External service agent: {name}"
        self.endpoint = endpoint
        self.timeout = timeout

        # Instance state (lazy-provisioned on first query)
        self._instance_id = instance_id
        self._api_key: Optional[str] = None

    # ── Public interface ────────────────────────────────────────────────

    @property
    def description(self) -> str:
        return self._description

    def query(self, question: str, include_files: Optional[list[str]] = None) -> str:
        """Send a question to the external service and return its response."""
        t0 = time.time()

        # Provision instance on first call
        if not self._instance_id:
            self._provision_instance()

        # Build and send the chat request
        url = f"{self.base_url}{self.endpoint}"
        payload = {
            "instance_id": self._instance_id,
            "message": question,
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        self._log(f"POST {url} (instance={self._instance_id})")

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except httpx.ConnectError:
            msg = f"Cannot connect to {self.base_url} — is the service running?"
            _log.error(f"{self.name}: {msg}")
            return f"Error: {msg}"
        except httpx.TimeoutException:
            msg = f"Request to {self.base_url} timed out after {self.timeout}s"
            _log.error(f"{self.name}: {msg}")
            return f"Error: {msg}"
        except httpx.HTTPStatusError as e:
            msg = f"HTTP {e.response.status_code} from {url}: {e.response.text[:500]}"
            _log.error(f"{self.name}: {msg}")
            return f"Error: {msg}"
        except Exception as e:
            _log.error(f"{self.name}: {e}")
            return f"Error querying {self.name}: {e}"

        # Parse the ChatResponse
        result = self._parse_response(data)

        elapsed = time.time() - t0
        self._log(f"Done ({elapsed:.1f}s, {len(result):,} chars)")
        return result

    # ── Instance provisioning ───────────────────────────────────────────

    def _provision_instance(self):
        """Provision a new instance via POST /v1/instances."""
        url = f"{self.base_url}/v1/instances"
        self._log(f"Provisioning instance at {url}")

        try:
            with httpx.Client(timeout=30) as client:
                resp = client.post(
                    url,
                    json={"developer_name": f"maz-{self.name}"},
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                data = resp.json()
            self._instance_id = data["instance_id"]
            self._api_key = data.get("api_key")
            self._log(f"Instance provisioned: {self._instance_id}")
        except Exception as e:
            _log.error(f"{self.name}: Failed to provision instance: {e}")
            # Use a fallback instance_id so we can still attempt chat
            self._instance_id = "dev_test"
            _log.warn(f"{self.name}: Using fallback instance_id '{self._instance_id}'")

    # ── Response parsing ────────────────────────────────────────────────

    def _parse_response(self, data: dict) -> str:
        """Parse a ChatResponse dict into a string result."""
        resp_type = data.get("type", "error")
        message = data.get("message", "")

        if resp_type == "result":
            # Include structured data summary if present
            structured = data.get("structured")
            if structured:
                return f"{message}\n\n[Structured data: {structured}]"
            return message

        elif resp_type == "sign_request":
            # The service needs a developer signature to proceed
            envelope_id = data.get("envelope_id", "unknown")
            return (
                f"[Signing Required] {message}\n\n"
                f"The storage operation requires a developer signature. "
                f"Envelope ID: {envelope_id}\n"
                f"This is expected for operations that modify on-chain state."
            )

        elif resp_type == "error":
            error_code = data.get("error_code", "unknown")
            retryable = data.get("retryable", False)
            return (
                f"[Service Error] {message}\n"
                f"Error code: {error_code}, Retryable: {retryable}"
            )

        else:
            return f"[Unknown response type: {resp_type}] {message}"

    # ── Logging ─────────────────────────────────────────────────────────

    def _log(self, msg: str):
        _log.agent(self.name, msg)
