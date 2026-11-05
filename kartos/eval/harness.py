"""Evaluation harness for replaying audio WAV files through VoiceAgent and calculating benchmarks."""
from __future__ import annotations

import asyncio
import statistics
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Callable, List, Optional

import numpy as np

from kartos.agent import VoiceAgent
from kartos.audio import NullSink
from kartos.config import Config
from kartos.player import Player
from kartos.providers import MockLLM, MockSTT, MockTTS, resample
from kartos.vad import EnergyVAD


@dataclass
class Scenario:
    name: str
    wav_path: Path
    description: str = ""
    expected_turns: int = 1
    expected_backchannels: int = 0
    expected_interruptions: int = 0


@dataclass
class ScenarioResult:
    scenario: Scenario
    turns_detected: int
    backchannels_detected: int
    interruptions_detected: int
    false_interruptions: int
    turn_accuracy: float
    turn_latencies_ms: list[int]
    p50_latency_ms: float
    p95_latency_ms: float
    events: list[str]
    passed: bool
    details: str = ""

    def summary(self) -> dict:
        return {
            "scenario": self.scenario.name,
            "passed": self.passed,
            "turns": f"{self.turns_detected}/{self.scenario.expected_turns}",
            "backchannels": f"{self.backchannels_detected}/{self.scenario.expected_backchannels}",
            "interruptions": f"{self.interruptions_detected}/{self.scenario.expected_interruptions}",
            "false_interruptions": self.false_interruptions,
            "turn_accuracy": f"{self.turn_accuracy * 100:.1f}%",
            "p50_ms": round(self.p50_latency_ms, 1),
            "p95_ms": round(self.p95_latency_ms, 1),
        }


@dataclass
class EvalReport:
    results: list[ScenarioResult] = field(default_factory=list)
    total_scenarios: int = 0
    passed_scenarios: int = 0
    end_of_turn_accuracy: float = 0.0
    total_false_interruptions: int = 0
    overall_p50_ms: float = 0.0
    overall_p95_ms: float = 0.0

    def print_summary(self) -> None:
        print("\n" + "=" * 80)
        print("                  kartOS VOICE AGENT EVALUATION REPORT")
        print("=" * 80)
        print(f"{'Scenario':<25} {'Turns':<10} {'Backch':<10} {'Interr':<10} {'FalseInt':<10} {'p50(ms)':<10} {'Status'}")
        print("-" * 80)
        for r in self.results:
            s = r.summary()
            status_str = "PASS" if r.passed else "FAIL"
            print(f"{s['scenario']:<25} {s['turns']:<10} {s['backchannels']:<10} {s['interruptions']:<10} {s['false_interruptions']:<10} {s['p50_ms']:<10} {status_str}")
        print("-" * 80)
        print(f"Scenarios: {self.passed_scenarios}/{self.total_scenarios} passed ({(self.passed_scenarios / max(1, self.total_scenarios)) * 100:.1f}%)")
        print(f"End-of-Turn Accuracy:     {self.end_of_turn_accuracy * 100:.1f}%")
        print(f"Total False Interruptions: {self.total_false_interruptions}")
        print(f"Aggregate Latency:        p50 = {self.overall_p50_ms:.0f} ms | p95 = {self.overall_p95_ms:.0f} ms")
        print("=" * 80 + "\n")

    def to_dict(self) -> dict:
        return {
            "total_scenarios": self.total_scenarios,
            "passed_scenarios": self.passed_scenarios,
            "end_of_turn_accuracy": self.end_of_turn_accuracy,
            "total_false_interruptions": self.total_false_interruptions,
            "overall_p50_ms": self.overall_p50_ms,
            "overall_p95_ms": self.overall_p95_ms,
            "scenarios": [r.summary() for r in self.results],
        }


async def wav_to_frames(
    path: Path | str,
    frame_samples: int = 512,
    sample_rate: int = 16000,
    speed: float = 1.0,
) -> AsyncIterator[np.ndarray]:
    p = Path(path)
    with wave.open(str(p), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())

    pcm = np.frombuffer(raw, dtype="<i2")
    if n_channels > 1:
        pcm = pcm[::n_channels]
    if sr != sample_rate:
        pcm = resample(pcm, sr, sample_rate)

    frame_ms = frame_samples * 1000 / sample_rate
    sleep_s = (frame_ms / 1000) / max(0.01, speed)

    for i in range(0, len(pcm), frame_samples):
        chunk = pcm[i : i + frame_samples]
        if len(chunk) < frame_samples:
            chunk = np.pad(chunk, (0, frame_samples - len(chunk)))
        yield chunk
        if speed > 0:
            await asyncio.sleep(sleep_s)


class EvalHarness:
    def __init__(
        self,
        cfg: Optional[Config] = None,
        stt_provider: Optional[any] = None,
        llm_provider: Optional[any] = None,
        tts_provider: Optional[any] = None,
        vad_provider: Optional[any] = None,
    ):
        self.cfg = cfg or Config()
        self.stt = stt_provider
        self.llm = llm_provider
        self.tts = tts_provider
        self.vad = vad_provider

    async def run_scenario(self, scenario: Scenario, speed: float = 1.0, verbose: bool = False) -> ScenarioResult:
        cfg = self.cfg
        events: list[str] = []

        def on_event(name: str, **kw):
            events.append(name)
            if verbose:
                detail = " ".join(f"{k}={v!r}" for k, v in kw.items())
                print(f"[{scenario.name}] {name:<12} {detail}")

        stt = self.stt or MockSTT()
        llm = self.llm or MockLLM(speed=speed)
        tts = self.tts or MockTTS()
        vad = self.vad or EnergyVAD(cfg)
        player = Player(NullSink(cfg.out_rate, speed=speed), cfg.out_rate)

        agent = VoiceAgent(cfg, vad, stt, llm, tts, player, on_event)
        frames_iter = wav_to_frames(scenario.wav_path, cfg.frame_samples, cfg.sample_rate, speed)
        await agent.run(frames_iter)

        actual_turns = len(agent.metrics.turns)
        actual_backchannels = events.count("backchannel")
        actual_interruptions = events.count("interrupted")
        false_interruptions = max(0, actual_interruptions - scenario.expected_interruptions)

        turn_diff = abs(actual_turns - scenario.expected_turns)
        turn_acc = max(0.0, 1.0 - (turn_diff / max(1, scenario.expected_turns)))

        latencies = [
            s["TOTAL"]
            for t in agent.metrics.turns
            if "TOTAL" in (s := t.summary())
        ]
        p50 = float(statistics.median(latencies)) if latencies else 0.0
        p95 = float(sorted(latencies)[min(len(latencies) - 1, int(len(latencies) * 0.95))]) if latencies else 0.0

        passed = (
            actual_turns == scenario.expected_turns
            and actual_backchannels == scenario.expected_backchannels
            and actual_interruptions == scenario.expected_interruptions
            and false_interruptions == 0
        )

        return ScenarioResult(
            scenario=scenario,
            turns_detected=actual_turns,
            backchannels_detected=actual_backchannels,
            interruptions_detected=actual_interruptions,
            false_interruptions=false_interruptions,
            turn_accuracy=turn_acc,
            turn_latencies_ms=latencies,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            events=events,
            passed=passed,
        )

    async def run_all(
        self,
        scenarios: list[Scenario],
        speed: float = 1.0,
        verbose: bool = False,
    ) -> EvalReport:
        results = []
        all_latencies = []
        for sc in scenarios:
            res = await self.run_scenario(sc, speed=speed, verbose=verbose)
            results.append(res)
            all_latencies.extend(res.turn_latencies_ms)

        passed_count = sum(1 for r in results if r.passed)
        avg_turn_acc = sum(r.turn_accuracy for r in results) / max(1, len(results))
        total_false_interr = sum(r.false_interruptions for r in results)

        p50_all = float(statistics.median(all_latencies)) if all_latencies else 0.0
        p95_all = (
            float(sorted(all_latencies)[min(len(all_latencies) - 1, int(len(all_latencies) * 0.95))])
            if all_latencies
            else 0.0
        )

        return EvalReport(
            results=results,
            total_scenarios=len(results),
            passed_scenarios=passed_count,
            end_of_turn_accuracy=avg_turn_acc,
            total_false_interruptions=total_false_interr,
            overall_p50_ms=p50_all,
            overall_p95_ms=p95_all,
        )
