"""Hardware I/O (sounddevice) plus a simulated sink for tests and demos."""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

import numpy as np

from .config import Config


class Microphone:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    async def frames(self) -> AsyncIterator[np.ndarray]:
        import sounddevice as sd

        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def cb(indata, n, t, status):
            loop.call_soon_threadsafe(q.put_nowait, indata[:, 0].copy())

        with sd.InputStream(
            samplerate=self.cfg.sample_rate, channels=1, dtype="int16",
            blocksize=self.cfg.frame_samples, callback=cb,
        ):
            while True:
                yield await q.get()


class SpeakerSink:
    def __init__(self, rate: int):
        import sounddevice as sd

        self.stream = sd.OutputStream(samplerate=rate, channels=1, dtype="int16", latency="low")
        self.stream.start()

    async def write(self, pcm: np.ndarray) -> None:
        await asyncio.to_thread(self.stream.write, pcm.reshape(-1, 1))

    def close(self) -> None:
        self.stream.stop()
        self.stream.close()


class NullSink:
    """Pretends to play audio in (scaled) real time."""

    def __init__(self, rate: int, speed: float = 1.0):
        self.rate, self.speed = rate, speed

    async def write(self, pcm: np.ndarray) -> None:
        await asyncio.sleep(len(pcm) / self.rate / self.speed)
