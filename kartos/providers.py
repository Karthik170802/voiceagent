"""Pluggable STT / LLM / TTS. Each is a tiny interface; add a provider by writing one class."""
from __future__ import annotations

import asyncio
from typing import AsyncIterator, Protocol

import numpy as np

from .tools import ToolRegistry


class STT(Protocol):
    async def transcribe(self, pcm: np.ndarray, sample_rate: int) -> str: ...


class LLM(Protocol):
    def stream(self, history: list[dict], system: str) -> AsyncIterator[str]: ...


class TTS(Protocol):
    async def synth(self, text: str) -> tuple[np.ndarray, int]: ...


def resample(pcm: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or len(pcm) == 0:
        return pcm
    n = int(len(pcm) * dst / src)
    x = np.linspace(0, len(pcm) - 1, n)
    return np.interp(x, np.arange(len(pcm)), pcm.astype(np.float32)).astype(np.int16)


# ------------------------------------------------------------------ real providers
class FasterWhisperSTT:
    def __init__(self, model_size: str = "base.en", language: str | None = "en"):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model_size, device="auto", compute_type="int8")
        self.language = language

    def _run(self, pcm: np.ndarray) -> str:
        audio = pcm.astype(np.float32) / 32768.0
        segs, _ = self.model.transcribe(
            audio, language=self.language, beam_size=1, vad_filter=False,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segs).strip()

    async def transcribe(self, pcm: np.ndarray, sample_rate: int) -> str:
        return await asyncio.to_thread(self._run, pcm)


class AnthropicLLM:
    """Streams text; transparently runs tool-use rounds, with a spoken filler while tools run."""

    def __init__(self, model: str, max_tokens: int = 300, tools: ToolRegistry | None = None,
                 filler: str = "One moment. ", max_tool_rounds: int = 3):
        import anthropic

        self.client = anthropic.AsyncAnthropic()
        self.model, self.max_tokens = model, max_tokens
        self.tools, self.filler, self.rounds = tools, filler, max_tool_rounds

    async def stream(self, history: list[dict], system: str) -> AsyncIterator[str]:
        msgs = [dict(m) for m in history]
        for rnd in range(self.rounds + 1):
            kw = dict(model=self.model, max_tokens=self.max_tokens, system=system, messages=msgs)
            if self.tools:
                kw["tools"] = self.tools.schemas()
            async with self.client.messages.stream(**kw) as s:
                async for text in s.text_stream:
                    yield text
                final = await s.get_final_message()
            if final.stop_reason != "tool_use" or not self.tools:
                return
            if rnd == 0 and self.filler:
                yield self.filler
            msgs.append({"role": "assistant", "content": final.content})
            results = []
            for b in final.content:
                if b.type == "tool_use":
                    out = await self.tools.call(b.name, b.input)
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
            msgs.append({"role": "user", "content": results})


class PiperTTS:
    """Local neural TTS through the `piper` CLI (pip install piper-tts)."""

    def __init__(self, model_path: str, sample_rate: int = 22050):
        self.model_path, self.sr = model_path, sample_rate

    async def synth(self, text: str) -> tuple[np.ndarray, int]:
        proc = await asyncio.create_subprocess_exec(
            "piper", "--model", self.model_path, "--output-raw",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate(text.encode() + b"\n")
        return np.frombuffer(out, dtype=np.int16), self.sr


# ------------------------------------------------------------------ mocks (no keys, no models)
_TONE_TEXT = {300: "What time is it?", 500: "Mm-hmm.", 700: "Actually, tell me a joke instead."}


class MockSTT:
    """Identifies the 'utterance' from the dominant tone frequency in the buffer (see demo.py)."""

    async def transcribe(self, pcm: np.ndarray, sample_rate: int) -> str:
        await asyncio.sleep(0.02)
        spec = np.abs(np.fft.rfft(pcm.astype(np.float32)))
        if spec.max() < 1e5:
            return ""
        freq = np.fft.rfftfreq(len(pcm), 1 / sample_rate)[int(spec.argmax())]
        return _TONE_TEXT[min(_TONE_TEXT, key=lambda f: abs(f - freq))]


class MockLLM:
    def __init__(self, speed: float = 1.0):
        self.speed = speed

    async def stream(self, history: list[dict], system: str) -> AsyncIterator[str]:
        last = history[-1]["content"].lower()
        if "joke" in last:
            reply = "Sure! Why did the developer go broke? Because he used up all his cache."
        else:
            reply = ("Right now it is a quarter past three in the afternoon. "
                     "I can also set a reminder, look something up, or just chat for a while. "
                     "Let me know whatever would be most useful to you today.")
        await asyncio.sleep(0.15 / self.speed)
        for w in reply.split(" "):
            await asyncio.sleep(0.02 / self.speed)
            yield w + " "


class MockTTS:
    """Synthesizes a quiet tone, 0.3 s per word, so timing behaves like real speech."""

    async def synth(self, text: str) -> tuple[np.ndarray, int]:
        await asyncio.sleep(0.03)
        n = int(24000 * 0.3 * max(1, len(text.split())))
        t = np.arange(n) / 24000
        return (np.sin(2 * np.pi * 220 * t) * 2000).astype(np.int16), 24000
