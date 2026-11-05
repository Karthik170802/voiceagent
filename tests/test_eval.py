import asyncio
from pathlib import Path
import pytest

from kartos.eval.harness import EvalHarness, Scenario
from kartos.eval.scenarios import create_sample_scenarios


def test_eval_harness_end_to_end(tmp_path: Path):
    created = create_sample_scenarios(tmp_path)
    assert len(created) == 3

    scenarios = [
        Scenario(
            name=spec.name,
            wav_path=wav_path,
            description=spec.description,
            expected_turns=spec.expected_turns,
            expected_backchannels=spec.expected_backchannels,
            expected_interruptions=spec.expected_interruptions,
        )
        for spec, wav_path in created
    ]

    harness = EvalHarness()
    report = asyncio.run(harness.run_all(scenarios, speed=8.0))

    assert report.total_scenarios == 3
    assert report.passed_scenarios == 3
    assert report.end_of_turn_accuracy == 1.0
    assert report.total_false_interruptions == 0
    assert report.overall_p50_ms > 0
    assert report.overall_p95_ms >= report.overall_p50_ms

    d = report.to_dict()
    assert d["passed_scenarios"] == 3
    assert len(d["scenarios"]) == 3
