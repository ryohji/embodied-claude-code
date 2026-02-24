"""Echo buffer core logic — persistent, decaying resonance store."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path

# 半減期: 12時間。12時間後に強度0.5、24時間後0.25、3日後約0.016
HALF_LIFE_HOURS = 12.0

# この強度を下回ったエコーは「消えた」とみなす
MIN_STRENGTH = 0.05

DEFAULT_STORE_PATH = Path.home() / ".claude" / "echo_buffer.json"


@dataclass
class Echo:
    id: str
    content: str
    base_strength: float
    added_at: float  # unix timestamp

    def current_strength(self, now: float | None = None) -> float:
        if now is None:
            now = time.time()
        age_hours = (now - self.added_at) / 3600.0
        return self.base_strength * (0.5 ** (age_hours / HALF_LIFE_HOURS))


class EchoBuffer:
    def __init__(self, store_path: Path = DEFAULT_STORE_PATH) -> None:
        self.store_path = store_path
        self.frozen = False
        self._echoes: list[Echo] = self._load()

    # ---- persistence ----

    def _load(self) -> list[Echo]:
        if not self.store_path.exists():
            return []
        try:
            with open(self.store_path, encoding="utf-8") as f:
                data = json.load(f)
            return [Echo(**e) for e in data]
        except Exception:
            return []

    def _save(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump([asdict(e) for e in self._echoes], f, indent=2, ensure_ascii=False)

    # ---- public API ----

    def add(self, content: str, base_strength: float = 1.0) -> str:
        """エコーを追加する。凍結中は "frozen" を返す。"""
        if self.frozen:
            return "frozen"
        echo = Echo(
            id=str(uuid.uuid4()),
            content=content,
            base_strength=min(max(base_strength, 0.0), 1.0),
            added_at=time.time(),
        )
        self._echoes.append(echo)
        self._prune()
        self._save()
        return echo.id

    def read(self, top_k: int = 5) -> list[dict]:
        """減衰適用後の強度順でエコーを返す。"""
        now = time.time()
        active = [
            (e, e.current_strength(now))
            for e in self._echoes
            if e.current_strength(now) >= MIN_STRENGTH
        ]
        active.sort(key=lambda x: x[1], reverse=True)
        return [
            {
                "content": e.content,
                "strength": round(s, 3),
                "age_hours": round((now - e.added_at) / 3600.0, 1),
            }
            for e, s in active[:top_k]
        ]

    def clear(self) -> None:
        """バッファを全消去する。"""
        self._echoes = []
        self._save()

    def freeze(self, enabled: bool) -> None:
        """新規追加を停止/再開する。"""
        self.frozen = enabled

    def status(self) -> dict:
        """現在状態を返す。"""
        now = time.time()
        active = [
            (e, e.current_strength(now))
            for e in self._echoes
            if e.current_strength(now) >= MIN_STRENGTH
        ]
        active.sort(key=lambda x: x[1], reverse=True)
        return {
            "total_stored": len(self._echoes),
            "active": len(active),
            "frozen": self.frozen,
            "half_life_hours": HALF_LIFE_HOURS,
            "echoes": [
                {
                    "content": e.content[:80] + ("..." if len(e.content) > 80 else ""),
                    "strength": round(s, 3),
                    "age_hours": round((now - e.added_at) / 3600.0, 1),
                }
                for e, s in active
            ],
        }

    # ---- internal ----

    def _prune(self) -> None:
        """MIN_STRENGTH を下回ったエコーを除去する。"""
        now = time.time()
        self._echoes = [
            e for e in self._echoes
            if e.current_strength(now) >= MIN_STRENGTH
        ]
