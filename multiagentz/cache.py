# multiagentz/cache.py
"""
Disk-backed persistent cache with TTL support.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple


class PersistentCache:
    def __init__(self, cache_dir: str = ".maz_cache", default_ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.default_ttl = timedelta(hours=default_ttl_hours)

    def _key_to_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _hash_query(self, question: str, context: str = "") -> str:
        content = f"{question}:{context}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get(self, question: str, context: str = "") -> Optional[Tuple[str, list]]:
        key = self._hash_query(question, context)
        path = self._key_to_path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            cached_at = datetime.fromisoformat(data["cached_at"])
            ttl_hours = data.get("ttl_hours", self.default_ttl.total_seconds() / 3600)
            if datetime.now() - cached_at > timedelta(hours=ttl_hours):
                path.unlink()
                return None
            return data["response"], data["agents_used"]
        except (json.JSONDecodeError, KeyError):
            path.unlink()
            return None

    def set(
        self, question: str, response: str, agents_used: list,
        context: str = "", ttl_hours: Optional[int] = None,
    ) -> None:
        key = self._hash_query(question, context)
        path = self._key_to_path(key)
        data = {
            "question": question,
            "context_hash": hashlib.sha256(context.encode()).hexdigest()[:8] if context else None,
            "response": response,
            "agents_used": agents_used,
            "cached_at": datetime.now().isoformat(),
            "ttl_hours": ttl_hours or self.default_ttl.total_seconds() / 3600,
        }
        path.write_text(json.dumps(data, indent=2))

    def clear(self) -> int:
        count = 0
        for p in self.cache_dir.glob("*.json"):
            p.unlink()
            count += 1
        return count

    def stats(self) -> dict:
        entries = list(self.cache_dir.glob("*.json"))
        total_size = sum(p.stat().st_size for p in entries)
        return {
            "entries": len(entries),
            "size_kb": round(total_size / 1024, 2),
            "cache_dir": str(self.cache_dir),
        }
