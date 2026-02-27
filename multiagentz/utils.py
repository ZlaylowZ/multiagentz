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
        return extract_json(text)
    except (json.JSONDecodeError, IndexError):
        return {}
