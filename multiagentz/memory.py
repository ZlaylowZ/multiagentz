# multiagentz/memory.py
"""
Rolling-window session memory for conversation history.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    routed_to: Optional[list[str]] = None


class SessionMemory:
    def __init__(self, max_turns: int = 50):
        self.turns: list[Turn] = []
        self.max_turns = max_turns

    def add_user(self, content: str) -> None:
        self.turns.append(Turn(role="user", content=content))
        self._trim()

    def add_assistant(self, content: str, routed_to: list[str] = None) -> None:
        self.turns.append(Turn(role="assistant", content=content, routed_to=routed_to))
        self._trim()

    def _trim(self) -> None:
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def get_context(self, last_n: int = 20) -> str:
        recent = self.turns[-last_n:]
        if not recent:
            return ""
        lines = ["## Recent Conversation History"]
        for t in recent:
            prefix = "User" if t.role == "user" else "Assistant"
            content = t.content[:4000] + "..." if len(t.content) > 4000 else t.content
            lines.append(f"\n**{prefix}**: {content}")
        return "\n".join(lines)

    def clear(self) -> None:
        self.turns = []
