"""Run a live voice agent from your microphone.

    export ANTHROPIC_API_KEY=...
    python main.py --piper-model ./en_US-lessac-medium.onnx
    python main.py --demo            # no mic, no keys: simulated conversation
"""
from __future__ import annotations

import argparse
import asyncio

from kartos.agent import VoiceAgent
from kartos.audio import Microphone, SpeakerSink
from kartos.config import Config
from kartos.player import Player
from kartos.providers import AnthropicLLM, FasterWhisperSTT, PiperTTS
from kartos.tools import default_tools
from kartos.vad import make_vad


async def amain(args: argparse.Namespace) -> None:
    cfg = Config(headphones=not args.speakers)
    sink = SpeakerSink(cfg.out_rate)
    agent = VoiceAgent(
        cfg,
        make_vad(cfg),
        FasterWhisperSTT(args.whisper_model),
        AnthropicLLM(cfg.model, cfg.max_tokens, default_tools(), cfg.tool_filler),
        PiperTTS(args.piper_model),
        Player(sink, cfg.out_rate),
    )
    print("Listening. Speak naturally; interrupt any time. Ctrl+C to quit.")
    try:
        await agent.run(Microphone(cfg).frames())
    finally:
        sink.close()
        print("\n" + agent.metrics.report())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="simulated conversation with mock providers")
    p.add_argument("--speed", type=float, default=1.0, help="demo speed multiplier")
    p.add_argument("--whisper-model", default="base.en")
    p.add_argument("--piper-model", default="en_US-lessac-medium.onnx")
    p.add_argument("--speakers", action="store_true", help="using speakers (not headphones): stricter VAD while agent talks")
    args = p.parse_args()
    if args.demo:
        from kartos.demo import run_demo
        _, metrics = asyncio.run(run_demo(args.speed))
        print("\n" + metrics.report())
        return
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
