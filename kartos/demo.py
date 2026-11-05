"""Hardware-free simulation: synthetic 'speech' (tones) drives the full pipeline with mock providers.

Scenario: user asks a question -> agent answers -> user says "mm-hmm" mid-answer (agent should
pause and resume) -> user interrupts with a new request (agent should stop and answer it).
"""
from __future__ import annotations

import asyncio

import numpy as np

from .agent import VoiceAgent
from .audio import NullSink
from .config import Config
from .metrics import MetricsLog
from .player import Player
from .providers import MockLLM, MockSTT, MockTTS
from .vad import EnergyVAD

# (seconds, tone Hz or None for silence)
SCENARIO = [
    (0.6, None), (1.4, 300),   # "What time is it?"
    (3.0, None), (0.4, 500),   # agent talking... user: "mm-hmm"
    (2.0, None), (1.6, 700),   # agent talking... user barges in: "Actually, tell me a joke instead."
    (6.0, None),
]


async def _mic(cfg: Config, speed: float):
    rng = np.random.default_rng(0)
    for secs, freq in SCENARIO:
        for i in range(int(secs * 1000 / cfg.frame_ms)):
            t = (np.arange(cfg.frame_samples) + i * cfg.frame_samples) / cfg.sample_rate
            noise = rng.normal(0, 20, cfg.frame_samples)
            tone = 6000 * np.sin(2 * np.pi * freq * t) if freq else 0
            yield (noise + tone).astype(np.int16)
            await asyncio.sleep(cfg.frame_ms / 1000 / speed)


async def run_demo(speed: float = 1.0, quiet: bool = False) -> tuple[list[str], MetricsLog]:
    cfg = Config()
    events: list[str] = []

    def on_event(name: str, **kw):
        events.append(name)
        if not quiet:
            detail = " ".join(f"{k}={v!r}" for k, v in kw.items())
            print(f"{name:<13} {detail}")

    player = Player(NullSink(cfg.out_rate, speed), cfg.out_rate)
    agent = VoiceAgent(cfg, EnergyVAD(cfg), MockSTT(), MockLLM(speed), MockTTS(), player, on_event)
    await agent.run(_mic(cfg, speed))
    return events, agent.metrics


if __name__ == "__main__":
    import sys

    sp = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    _, metrics = asyncio.run(run_demo(sp))
    print("\n" + metrics.report())
