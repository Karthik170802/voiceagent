"""CLI interface for running voice agent evaluations.

Usage:
    python -m kartos.eval
    python -m kartos.eval --speed 4
    python -m kartos.eval --json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .harness import EvalHarness, Scenario
from .scenarios import DEFAULT_SPECS, create_sample_scenarios


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="kartOS Voice Agent Evaluation Harness")
    p.add_argument(
        "--speed",
        type=float,
        default=4.0,
        help="Replay speed multiplier (default: 4.0 for fast test, 1.0 for real-time)",
    )
    p.add_argument(
        "--dir",
        type=str,
        default="./eval_scenarios",
        help="Directory to store or load WAV test scenarios (default: ./eval_scenarios)",
    )
    p.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Run only a specific scenario by name",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON evaluation metrics",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose event stream during replay",
    )
    return p.parse_args()


async def amain(args: argparse.Namespace) -> int:
    scenario_dir = Path(args.dir)
    created = create_sample_scenarios(scenario_dir)

    scenarios = []
    for spec, wav_path in created:
        if args.scenario and spec.name != args.scenario:
            continue
        scenarios.append(
            Scenario(
                name=spec.name,
                wav_path=wav_path,
                description=spec.description,
                expected_turns=spec.expected_turns,
                expected_backchannels=spec.expected_backchannels,
                expected_interruptions=spec.expected_interruptions,
            )
        )

    if not scenarios:
        print(f"Error: no scenarios found matching filter '{args.scenario}'", file=sys.stderr)
        return 1

    harness = EvalHarness()
    report = await harness.run_all(scenarios, speed=args.speed, verbose=args.verbose)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        report.print_summary()

    return 0 if report.passed_scenarios == report.total_scenarios else 1


def main() -> None:
    args = parse_args()
    code = asyncio.run(amain(args))
    sys.exit(code)


if __name__ == "__main__":
    main()
