# multiagentz/utils.py
"""
Shared utilities for routing, JSON parsing, and other common operations.
"""

from __future__ import annotations

import json
import re


def extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown fences."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    return json.loads(text.strip())


def extract_json_robust(text: str) -> dict:
    """
    Extract JSON from LLM response using brace-matching.

    More robust than extract_json for builder responses where the JSON
    content field may itself contain markdown fences, nested JSON,
    or other content that breaks naive split-based extraction.

    Strategy:
    1. Try direct json.loads (LLM returned raw JSON)
    2. Find the outermost { ... } by brace-depth counting (skipping strings)
    3. Fall back to fence-based extraction
    """
    text = text.strip()

    # Strategy 1: Direct parse (best case — raw JSON)
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try with trailing comma cleanup
            try:
                return json.loads(_strip_trailing_commas(text))
            except json.JSONDecodeError:
                pass

    # Strategy 2: Brace-matching — find outermost { } respecting strings
    result = _extract_outermost_json(text)
    if result is not None:
        return result

    # Strategy 3: Fence-based fallback (original method)
    return extract_json(text)


def _strip_trailing_commas(text: str) -> str:
    """
    Remove trailing commas before } and ] in JSON-like text.

    LLMs sometimes produce [1, 2, 3,] or {"a": 1,} which is invalid JSON.
    This is applied as a preprocessing step before json.loads.
    """
    return re.sub(r",\s*([}\]])", r"\1", text)


def _extract_outermost_json(text: str) -> dict | None:
    """
    Find and parse the outermost JSON object in text using brace-depth
    tracking that correctly skips string literals (handles escaped quotes).

    Returns parsed dict or None if no valid JSON object found.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    i = start

    while i < len(text):
        ch = text[i]

        if escape_next:
            escape_next = False
            i += 1
            continue

        if ch == "\\":
            escape_next = True
            i += 1
            continue

        if ch == '"' and not escape_next:
            in_string = not in_string
            i += 1
            continue

        if in_string:
            i += 1
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # Try again with trailing comma cleanup (LLMs produce these)
                    try:
                        return json.loads(_strip_trailing_commas(candidate))
                    except json.JSONDecodeError:
                        pass
                    # This brace pair wasn't valid JSON; try next '{'
                    start = text.find("{", i + 1)
                    if start == -1:
                        return None
                    i = start
                    depth = 0
                    continue

        i += 1

    return None


def extract_routing_task(question: str) -> str:
    """
    Extract the core task from a question for routing purposes.

    Refinement prompts contain the full previous solution + LEAD feedback
    which overwhelms the routing classifier. Strip that down to just the
    task description.
    """
    refinement_markers = [
        "Your previous solution proposal:",
        "Your previous solution:",
        "LEAD feedback:",
        "Required improvements:",
        "Provide refined solution",
    ]
    is_refinement = any(marker in question for marker in refinement_markers)

    if is_refinement:
        parts = []
        for marker in ["Weaknesses:", "Required improvements:", "Provide refined solution"]:
            idx = question.find(marker)
            if idx != -1:
                parts.append(question[idx:idx + 500])
        if parts:
            return (
                "[Refinement request] The following areas need improvement:\n\n"
                + "\n\n".join(parts)
            )

    if len(question) > 2000:
        return (
            question[:1500]
            + "\n\n[... question continues for "
            + f"{len(question):,} total chars ...]\n\n"
            + question[-500:]
        )

    return question


def parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response, handling markdown fences. Returns {} on failure."""
    try:
        return extract_json_robust(text)
    except (json.JSONDecodeError, IndexError, TypeError):
        # Final fallback: original method
        try:
            return extract_json(text)
        except (json.JSONDecodeError, IndexError):
            return {}
