"""Generates synthetic audio test scenarios for evaluation and regression testing."""
from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class ScenarioSpec:
    name: str
    filename: str
    description: str
    timeline: list[tuple[float, float | None]]  # (duration_seconds, tone_hz or None for silence)
    expected_turns: int
    expected_backchannels: int
    expected_interruptions: int


DEFAULT_SPECS = [
    ScenarioSpec(
        name="single_turn_qa",
        filename="single_turn_qa.wav",
        description="User asks 'What time is it?'; agent answers fully without interruption.",
        timeline=[
            (0.5, None),
            (1.2, 300.0),  # "What time is it?"
            (4.0, None),
        ],
        expected_turns=1,
        expected_backchannels=0,
        expected_interruptions=0,
    ),
    ScenarioSpec(
        name="backchannel_duck",
        filename="backchannel_duck.wav",
        description="Agent speaks; user murmurs 'Mm-hmm.'; agent ducks and resumes without interruption.",
        timeline=[
            (0.5, None),
            (1.2, 300.0),  # "What time is it?"
            (2.2, None),  # agent responds...
            (0.35, 500.0),  # user: "Mm-hmm."
            (3.5, None),  # agent finishes
        ],
        expected_turns=1,
        expected_backchannels=1,
        expected_interruptions=0,
    ),
    ScenarioSpec(
        name="barge_in_interruption",
        filename="barge_in_interruption.wav",
        description="Agent speaks; user barges in with 'Actually, tell me a joke instead.'; agent stops and replies.",
        timeline=[
            (0.5, None),
            (1.2, 300.0),  # "What time is it?"
            (1.8, None),  # agent speaks
            (1.5, 700.0),  # user barges in: "Actually, tell me a joke instead."
            (4.5, None),  # agent answers new query
        ],
        expected_turns=2,
        expected_backchannels=0,
        expected_interruptions=1,
    ),
]


def render_scenario_pcm(
    timeline: list[tuple[float, float | None]],
    sample_rate: int = 16000,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chunks = []
    for secs, freq in timeline:
        n_samples = int(secs * sample_rate)
        noise = rng.normal(0, 20, n_samples)
        if freq is not None:
            t = np.arange(n_samples) / sample_rate
            tone = 6000.0 * np.sin(2.0 * np.pi * freq * t)
            chunk = (noise + tone).astype(np.int16)
        else:
            chunk = noise.astype(np.int16)
        chunks.append(chunk)
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int16)


def write_wav(path: Path | str, pcm: np.ndarray, sample_rate: int = 16000) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.astype("<i2").tobytes())
    return p


def create_sample_scenarios(dir_path: Path | str) -> list[tuple[ScenarioSpec, Path]]:
    out_dir = Path(dir_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in DEFAULT_SPECS:
        wav_path = out_dir / spec.filename
        if not wav_path.exists():
            pcm = render_scenario_pcm(spec.timeline)
            write_wav(wav_path, pcm)
        results.append((spec, wav_path))
    return results
