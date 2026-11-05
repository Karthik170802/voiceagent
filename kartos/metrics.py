from __future__ import annotations

import statistics
import time

STAGES = [
    ("endpointing", "speech_end", "endpoint"),
    ("stt", "endpoint", "stt"),
    ("llm_ttft", "stt", "llm_first_token"),
    ("tts_ttfb", "llm_first_token", "tts_first_chunk"),
    ("playback_start", "tts_first_chunk", "audio_start"),
    ("TOTAL", "speech_end", "audio_start"),
]


class TurnMetrics:
    def __init__(self, turn: int):
        self.turn = turn
        self.marks: dict[str, float] = {}

    def mark(self, name: str, t: float | None = None) -> None:
        self.marks[name] = time.monotonic() if t is None else t

    def mark_once(self, name: str) -> None:
        self.marks.setdefault(name, time.monotonic())

    def summary(self) -> dict[str, int]:
        out = {}
        for label, a, b in STAGES:
            if a in self.marks and b in self.marks:
                out[label] = max(0, round((self.marks[b] - self.marks[a]) * 1000))
        return out


class MetricsLog:
    def __init__(self):
        self.turns: list[TurnMetrics] = []
        self.current: TurnMetrics = TurnMetrics(0)

    def new_turn(self) -> TurnMetrics:
        self.current = TurnMetrics(len(self.turns) + 1)
        self.turns.append(self.current)
        return self.current

    def report(self) -> str:
        totals = [s["TOTAL"] for t in self.turns if "TOTAL" in (s := t.summary())]
        if not totals:
            return "no completed turns"
        p50 = statistics.median(totals)
        p95 = sorted(totals)[min(len(totals) - 1, int(len(totals) * 0.95))]
        return f"turns={len(totals)}  response latency p50={p50:.0f}ms  p95={p95:.0f}ms"
