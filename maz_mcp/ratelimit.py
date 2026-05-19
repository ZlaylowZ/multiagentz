"""
Token-bucket rate limiter for MAZ MCP tools.

Per-tool budgets reflect actual Anthropic API cost:
  maz_query           ~2-3 LLM calls   → cost_weight 3
  maz_consensus       ~6-10 LLM calls  → cost_weight 8
  maz_perspective     ~8-12 LLM calls  → cost_weight 10
  maz_cross_pollinate ~5 LLM calls     → cost_weight 5
  maz_configure       0 LLM calls      → cost_weight 0
  maz_status          0 LLM calls      → cost_weight 0

Default budget: 60 tokens/min, refill 1/sec.
Override via MAZ_RATE_LIMIT_TOKENS_PER_MIN env var.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Dict, Optional

# Tool → approximate LLM call cost weight
TOOL_COSTS: Dict[str, int] = {
    "maz_query": 3,
    "maz_consensus": 8,
    "maz_perspective": 10,
    "maz_cross_pollinate": 5,
    "maz_configure": 0,
    "maz_status": 0,
}

_DEFAULT_BUCKET_SIZE = 60  # max burst
_DEFAULT_REFILL_RATE = 1.0  # tokens per second


class TokenBucket:
    """Thread-safe token bucket."""

    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def consume(self, cost: int) -> tuple[bool, float]:
        """Try to consume `cost` tokens. Returns (allowed, wait_seconds)."""
        if cost == 0:
            return True, 0.0

        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now

            if self.tokens >= cost:
                self.tokens -= cost
                return True, 0.0

            wait = (cost - self.tokens) / self.refill_rate
            return False, round(wait, 1)

    @property
    def available(self) -> float:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            return min(self.capacity, self.tokens + elapsed * self.refill_rate)


# Global bucket (shared across all requests on this instance)
_bucket: Optional[TokenBucket] = None
_bucket_lock = threading.Lock()

# Metrics
_metrics_lock = threading.Lock()
_total_requests = 0
_total_cost = 0
_rejected_requests = 0


def _get_bucket() -> TokenBucket:
    global _bucket
    if _bucket is None:
        with _bucket_lock:
            if _bucket is None:
                cap = int(os.getenv("MAZ_RATE_LIMIT_TOKENS_PER_MIN", str(_DEFAULT_BUCKET_SIZE)))
                rate = cap / 60.0  # convert per-min to per-sec
                _bucket = TokenBucket(capacity=cap, refill_rate=rate)
    return _bucket


def check_rate_limit(tool_name: str) -> Optional[Dict]:
    """Check if a tool call is allowed under the rate limit.

    Returns None if allowed, or an error dict if rejected.
    """
    # Bypass if rate limiting is disabled
    if os.getenv("MAZ_RATE_LIMIT_DISABLED", "").lower() in ("1", "true", "yes"):
        return None

    cost = TOOL_COSTS.get(tool_name, 1)
    if cost == 0:
        return None

    bucket = _get_bucket()
    allowed, wait_secs = bucket.consume(cost)

    global _total_requests, _total_cost, _rejected_requests
    with _metrics_lock:
        _total_requests += 1
        if allowed:
            _total_cost += cost
        else:
            _rejected_requests += 1

    if allowed:
        return None

    return {
        "error": True,
        "error_type": "RateLimitExceeded",
        "message": f"Rate limit exceeded for {tool_name} (cost={cost}). Retry in {wait_secs}s.",
        "retry_after_seconds": wait_secs,
        "retryable": True,
    }


def get_metrics() -> Dict:
    """Return rate limiter metrics for maz_status."""
    bucket = _get_bucket()
    with _metrics_lock:
        return {
            "total_requests": _total_requests,
            "total_cost_tokens": _total_cost,
            "rejected_requests": _rejected_requests,
            "bucket_available": round(bucket.available, 1),
            "bucket_capacity": bucket.capacity,
            "rate_limit_disabled": os.getenv("MAZ_RATE_LIMIT_DISABLED", "").lower() in ("1", "true", "yes"),
        }
