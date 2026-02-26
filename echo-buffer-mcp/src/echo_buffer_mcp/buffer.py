"""Echo buffer core logic — persistent, decaying resonance store."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path

# 半減期: 6交換回。6回のecho_add後に強度が半減。
# 実時間ではなく会話的距離で減衰させる——セッション間の空白時間で不当に消えないように。
HALF_LIFE_STEPS = 6
DECAY_PER_STEP = 0.5 ** (1.0 / HALF_LIFE_STEPS)  # ≈ 0.891 per exchange

# この強度を下回ったエコーは「消えた」とみなす
MIN_STRENGTH = 0.05

DEFAULT_STORE_PATH = Path.home() / ".claude" / "echo_buffer.json"


@dataclass
class Echo:
    id: str
    content: str
    base_strength: float
    added_at: float  # unix timestamp（表示用のみ。減衰には使わない）

    def strength_at_age(self, ordinal_age: int) -> float:
        """ordinal_age: このエコーより後に追加されたエコーの数（0=最新）"""
        return self.base_strength * (DECAY_PER_STEP ** ordinal_age)


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
        """交換回数ベースの減衰を適用した強度順でエコーを返す。"""
        sorted_echoes = sorted(self._echoes, key=lambda e: e.added_at)
        total = len(sorted_echoes)
        active = []
        for i, e in enumerate(sorted_echoes):
            ordinal_age = total - 1 - i  # 0=最新、古いほど大きい
            s = e.strength_at_age(ordinal_age)
            if s >= MIN_STRENGTH:
                active.append((e, s, ordinal_age))
        active.sort(key=lambda x: x[1], reverse=True)
        return [
            {
                "content": e.content,
                "strength": round(s, 3),
                "age_steps": ordinal_age,
            }
            for e, s, ordinal_age in active[:top_k]
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
        sorted_echoes = sorted(self._echoes, key=lambda e: e.added_at)
        total = len(sorted_echoes)
        active = []
        for i, e in enumerate(sorted_echoes):
            ordinal_age = total - 1 - i
            s = e.strength_at_age(ordinal_age)
            if s >= MIN_STRENGTH:
                active.append((e, s, ordinal_age))
        active.sort(key=lambda x: x[1], reverse=True)
        return {
            "total_stored": len(self._echoes),
            "active": len(active),
            "frozen": self.frozen,
            "half_life_steps": HALF_LIFE_STEPS,
            "echoes": [
                {
                    "content": e.content[:80] + ("..." if len(e.content) > 80 else ""),
                    "strength": round(s, 3),
                    "age_steps": ordinal_age,
                }
                for e, s, ordinal_age in active
            ],
        }

    # ---- internal ----

    def _prune(self) -> None:
        """MIN_STRENGTH を下回ったエコーを除去する。"""
        sorted_echoes = sorted(self._echoes, key=lambda e: e.added_at)
        total = len(sorted_echoes)
        self._echoes = [
            e for i, e in enumerate(sorted_echoes)
            if e.strength_at_age(total - 1 - i) >= MIN_STRENGTH
        ]
