"""Interruptible audio output. Plays in 20 ms frames so pause/flush take effect immediately,
and tracks how much of each sentence was actually heard (to truncate history on barge-in)."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import numpy as np


class Sink(Protocol):
    async def write(self, pcm: np.ndarray) -> None: ...


@dataclass
class _Rec:
    text: str
    n: int
    played: int = 0


class Player:
    def __init__(self, sink: Sink, rate: int, on_first_audio: Optional[Callable[[], None]] = None):
        self.sink = sink
        self.rate = rate
        self.frame = int(rate * 0.02)
        self.on_first_audio = on_first_audio
        self._q: asyncio.Queue = asyncio.Queue()
        self._resume = asyncio.Event()
        self._resume.set()
        self._gen = 0
        self._records: list[_Rec] = []
        self._first_sent = False

    # ---- control ----
    def new_turn(self) -> None:
        self._records.clear()
        self._first_sent = False

    def enqueue(self, text: str, pcm: np.ndarray) -> None:
        rec = _Rec(text, len(pcm))
        self._records.append(rec)
        self._q.put_nowait((rec, pcm))

    def pause(self) -> None:
        self._resume.clear()

    def resume(self) -> None:
        self._resume.set()

    def flush(self) -> None:
        self._gen += 1
        while not self._q.empty():
            self._q.get_nowait()
            self._q.task_done()
        self._resume.set()

    async def drain(self) -> None:
        await self._q.join()

    def spoken_text(self) -> str:
        parts: list[str] = []
        for r in self._records:
            if r.played >= r.n:
                parts.append(r.text)
                continue
            if r.played > 0:
                words = r.text.split()
                k = int(len(words) * r.played / r.n)
                if k:
                    parts.append(" ".join(words[:k]) + "…")
            break
        return " ".join(parts)

    # ---- worker ----
    async def run(self) -> None:
        while True:
            rec, pcm = await self._q.get()
            gen = self._gen
            try:
                for i in range(0, len(pcm), self.frame):
                    if gen != self._gen:
                        break
                    await self._resume.wait()
                    if gen != self._gen:
                        break
                    if not self._first_sent:
                        self._first_sent = True
                        if self.on_first_audio:
                            self.on_first_audio()
                    await self.sink.write(pcm[i : i + self.frame])
                    rec.played = min(rec.n, i + self.frame)
            finally:
                self._q.task_done()
