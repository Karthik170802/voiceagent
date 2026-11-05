import asyncio
import time
import numpy as np

from kartos.agent import VoiceAgent
from kartos.audio import NullSink
from kartos.chunker import SentenceChunker, clean_for_tts
from kartos.config import Config
from kartos.demo import run_demo
from kartos.player import Player
from kartos.providers import MockLLM, MockTTS
from kartos.tools import default_tools
from kartos.turn import EndOfTurnDetector, classify
from kartos.vad import EnergyVAD


def chunks(tokens, **kw):
    c = SentenceChunker(**kw)
    out = []
    for t in tokens:
        out += c.feed(t)
    return out + c.flush()


def test_first_sentence_is_emitted_immediately():
    out = chunks(["Sure", ". ", "Here is", " the longer", " second sentence", " for you", "."])
    assert out[0] == "Sure."


def test_decimals_and_abbreviations_not_split():
    out = chunks(["Dr. Smith paid 3.5 dollars", " for it. ", "Done", "."])
    assert out[0] == "Dr. Smith paid 3.5 dollars for it."


def test_clause_split_for_first_chunk():
    out = chunks(["Well, ", "let me think about that for a second ", "and then answer."], first_min_chars=5)
    assert out[0] == "Well,"


def test_clean_for_tts():
    assert clean_for_tts("**Hi** [link](http://x.com) `code` \U0001F600") == "Hi link code"


def test_end_of_turn_adaptive():
    d = EndOfTurnDetector(Config())
    assert d.required_silence_ms("What time is it?") < d.required_silence_ms("so what I wanted to say is")
    assert d.required_silence_ms("my number is 5 5 5") == Config().incomplete_silence_ms
    assert d.required_silence_ms("I need to book a flight,") == Config().incomplete_silence_ms


def test_classify():
    assert classify("Mm-hmm.") == "backchannel"
    assert classify("Uh-huh") == "backchannel"
    assert classify("Yeah, but what about pricing?") == "interrupt"
    assert classify("wait") == "interrupt"
    assert classify("   ") == "empty"


def test_tools():
    t = default_tools()
    assert asyncio.run(t.call("calculate", {"expression": "(12.5*4)/2"})) == "25.0"
    assert asyncio.run(t.call("calculate", {"expression": "__import__('os')"})).startswith("error")


def test_full_pipeline_backchannel_and_barge_in():
    events, metrics = asyncio.run(run_demo(speed=8.0, quiet=True))
    assert "backchannel" in events            # paused then resumed, not interrupted
    assert events.count("interrupted") == 1   # real interruption handled once
    assert events.count("metrics") == 2       # two replies reached the speaker
    assert metrics.report().startswith("turns=2")


def test_speculative_stt_non_blocking_and_cancellation():
    class SlowMockSTT:
        def __init__(self, delay: float = 0.08):
            self.delay = delay
            self.cancelled_count = 0
            self.call_count = 0

        async def transcribe(self, pcm: np.ndarray, sample_rate: int) -> str:
            self.call_count += 1
            try:
                await asyncio.sleep(self.delay)
                return "hello world."
            except asyncio.CancelledError:
                self.cancelled_count += 1
                raise

    async def run_test():
        cfg = Config()
        events = []
        slow_stt = SlowMockSTT(delay=0.08)
        player = Player(NullSink(cfg.out_rate, speed=10.0), cfg.out_rate)
        agent = VoiceAgent(
            cfg, EnergyVAD(cfg), slow_stt, MockLLM(speed=10.0), MockTTS(), player,
            lambda name, **kw: events.append(name)
        )

        sr = cfg.sample_rate
        frames_during_stt = 0

        async def frame_generator():
            nonlocal frames_during_stt
            # 1. Speech for 10 frames (320 ms)
            for _ in range(10):
                t = np.arange(cfg.frame_samples) / sr
                tone = (6000 * np.sin(2 * np.pi * 300 * t)).astype(np.int16)
                yield tone
                await asyncio.sleep(0.005)

            # 2. Silence: speculative STT starts when sil_ms >= 250 ms (frame 8)
            for _ in range(10):
                sil = np.zeros(cfg.frame_samples, dtype=np.int16)
                if agent._stt_task and not agent._stt_task.done():
                    frames_during_stt += 1
                yield sil
                await asyncio.sleep(0.005)

            assert frames_during_stt > 0, "Audio frames must continue flowing while STT is running"
            assert agent._stt_task and not agent._stt_task.done(), "STT task should be in flight"

            # 3. Speech resumes -> cancels background STT!
            for _ in range(6):
                t = np.arange(cfg.frame_samples) / sr
                tone = (6000 * np.sin(2 * np.pi * 300 * t)).astype(np.int16)
                yield tone
                await asyncio.sleep(0.005)

            assert slow_stt.cancelled_count >= 1, "STT task should have been cancelled on new speech"

            # 4. Final silence (25 frames) -> turn finalizes cleanly
            for _ in range(25):
                sil = np.zeros(cfg.frame_samples, dtype=np.int16)
                yield sil
                await asyncio.sleep(0.01)

        await agent.run(frame_generator())
        assert "user" in events
        assert slow_stt.cancelled_count >= 1

    asyncio.run(run_test())
