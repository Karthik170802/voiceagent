"""The orchestrator: mic frames in, interruptible speech out.

Flow per utterance
  frames -> VAD -> utterance buffer -> (speculative) STT -> adaptive end-of-turn
         -> LLM stream -> sentence chunker -> TTS -> interruptible player

Turn-taking behaviours
  * Adaptive endpointing: wait longer after "and..." than after "What time is it?"
  * Soft barge-in: agent audio pauses ("ducks") when the user starts talking; if the
    utterance turns out to be a backchannel ("mm-hmm") it resumes, otherwise it stops
    for good and history is truncated to what was actually heard.
  * Utterance merging: if the user keeps talking while the agent is still thinking,
    the pending reply is cancelled and both fragments are answered together.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import AsyncIterator, Callable, Optional

import numpy as np

from .chunker import SentenceChunker, clean_for_tts
from .config import Config
from .metrics import MetricsLog
from .player import Player
from .providers import LLM, STT, TTS, resample
from .turn import EndOfTurnDetector, classify
from .vad import VAD


def _print_event(name: str, **kw) -> None:
    detail = " ".join(f"{k}={v!r}" for k, v in kw.items())
    print(f"[{time.strftime('%H:%M:%S')}] {name:<13} {detail}")


class VoiceAgent:
    def __init__(self, cfg: Config, vad: VAD, stt: STT, llm: LLM, tts: TTS, player: Player,
                 on_event: Callable[..., None] = _print_event):
        self.cfg, self.vad, self.stt, self.llm, self.tts, self.player = cfg, vad, stt, llm, tts, player
        self.on_event = on_event
        self.eot = EndOfTurnDetector(cfg)
        self.metrics = MetricsLog()
        self.history: list[dict] = []

        self._resp_task: Optional[asyncio.Task] = None
        self._player_task: Optional[asyncio.Task] = None
        self._stt_task: Optional[asyncio.Task] = None
        self._audio_started = False
        self._t_stt_done: Optional[float] = None

        self._preroll: deque = deque(maxlen=cfg.frames(cfg.preroll_ms))
        self._start_frames = cfg.frames(cfg.speech_start_ms)
        self._reset_utterance()
        player.on_first_audio = self._on_first_audio

    # ------------------------------------------------------------ state
    @property
    def responding(self) -> bool:
        return self._resp_task is not None and not self._resp_task.done()

    @property
    def speaking(self) -> bool:
        return self.responding and self._audio_started

    def _cancel_stt(self) -> None:
        if self._stt_task and not self._stt_task.done():
            self._stt_task.cancel()
        self._stt_task = None

    def _reset_utterance(self) -> None:
        self._cancel_stt()
        self._collecting = False
        self._ducked = False
        self._buf: list[np.ndarray] = []
        self._speech_run = 0
        self._speech_frames = 0
        self._silence_frames = 0
        self._cache: Optional[str] = None
        self._t_stt_done = None
        self._preroll.clear()

    # ------------------------------------------------------------ main loop
    async def run(self, frames: AsyncIterator[np.ndarray]) -> None:
        self._player_task = asyncio.create_task(self.player.run())
        try:
            async for frame in frames:
                await self._on_frame(frame)
            if self._collecting:
                await self._finalize()
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        self._cancel_stt()
        for t in (self._resp_task, self._player_task):
            if t and not t.done():
                t.cancel()
        await asyncio.gather(*(t for t in (self._resp_task, self._player_task) if t), return_exceptions=True)

    async def _on_frame(self, frame: np.ndarray) -> None:
        cfg = self.cfg
        strict = (not cfg.headphones) and self.speaking and not self._ducked
        is_speech = self.vad.is_speech(frame, strict=strict)

        if not self._collecting:
            self._preroll.append(frame)
            self._speech_run = self._speech_run + 1 if is_speech else 0
            if self._speech_run >= self._start_frames:
                self._collecting = True
                self._buf = list(self._preroll)
                self._speech_frames = self._speech_run
                self._silence_frames = 0
                self._cache = None
                self._t_stt_done = None
                self.on_event("speech_start")
            return

        self._buf.append(frame)
        if is_speech:
            self._cancel_stt()
            self._speech_frames += 1
            self._silence_frames = 0
            self._cache = None
            self._t_stt_done = None
            self._t_last_speech = time.monotonic()
            if self.speaking and not self._ducked and self._speech_frames * cfg.frame_ms >= cfg.duck_after_ms:
                self.player.pause()
                self._ducked = True
                self.on_event("duck")
            if len(self._buf) * cfg.frame_ms > cfg.max_utterance_s * 1000:
                await self._finalize()
        else:
            self._silence_frames += 1
            sil_ms = self._silence_frames * cfg.frame_ms
            if sil_ms >= cfg.min_silence_ms:
                await self._maybe_endpoint(sil_ms)

    # ------------------------------------------------------------ end of turn
    async def _transcribe(self) -> str:
        pcm = np.concatenate(self._buf)
        return (await self.stt.transcribe(pcm, self.cfg.sample_rate)).strip()

    async def _run_stt_task(self, pcm: np.ndarray) -> None:
        try:
            text = (await self.stt.transcribe(pcm, self.cfg.sample_rate)).strip()
            self._cache = text
            self._t_stt_done = time.monotonic()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.on_event("error", error=f"stt_error: {e}")

    async def _maybe_endpoint(self, sil_ms: float) -> None:
        if self._cache is None and (self._stt_task is None or self._stt_task.done()):
            pcm = np.concatenate(self._buf)
            self._stt_task = asyncio.create_task(self._run_stt_task(pcm))

        if self._cache is not None:
            need = self.eot.required_silence_ms(self._cache)
            if self.speaking and classify(self._cache) == "backchannel":
                need = self.cfg.min_silence_ms  # resume the agent quickly after "mm-hmm"
            if sil_ms >= min(need, self.cfg.max_silence_ms):
                await self._finalize()
        elif sil_ms >= self.cfg.max_silence_ms:
            if self._stt_task and not self._stt_task.done():
                try:
                    await self._stt_task
                except asyncio.CancelledError:
                    return
            await self._finalize()

    async def _finalize(self) -> None:
        t_decision = time.monotonic()
        if self._cache is None:
            if self._stt_task and not self._stt_task.done():
                try:
                    await self._stt_task
                except asyncio.CancelledError:
                    pass
            if self._cache is None:
                self._cache = await self._transcribe()
                self._t_stt_done = time.monotonic()
        text = self._cache
        t_stt = getattr(self, "_t_stt_done", t_decision) or t_decision
        speech_ms = self._speech_frames * self.cfg.frame_ms
        t_speech_end = getattr(self, "_t_last_speech", t_decision)
        was_ducked = self._ducked
        self._reset_utterance()

        if speech_ms < self.cfg.min_speech_ms or not text:
            if was_ducked:
                self.player.resume()
            return
        self.on_event("user", text=text)

        if self.speaking:
            if classify(text) == "backchannel":
                self.player.resume()
                self.on_event("backchannel", text=text)
                return
            await self._interrupt()
        elif self.responding:  # still thinking: cancel and merge with the new fragment
            await self._cancel_response()
            self.on_event("merged")

        m = self.metrics.new_turn()
        m.mark("speech_end", t_speech_end)
        m.mark("endpoint", t_decision)
        m.mark("stt", t_stt)
        self._start_response(text)

    # ------------------------------------------------------------ interruption
    async def _cancel_response(self) -> None:
        task = self._resp_task
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.player.flush()

    async def _interrupt(self) -> None:
        self.player.pause()  # instant silence; history truncation happens in _respond's cancel path
        await self._cancel_response()
        self.on_event("interrupted")

    # ------------------------------------------------------------ response
    def _add_user(self, text: str) -> None:
        if self.history and self.history[-1]["role"] == "user":
            self.history[-1]["content"] += " " + text
        else:
            self.history.append({"role": "user", "content": text})

    def _trimmed_history(self) -> list[dict]:
        h = self.history[-self.cfg.max_history_messages:]
        while h and h[0]["role"] != "user":
            h = h[1:]
        return h

    def _start_response(self, text: str) -> None:
        self._add_user(text)
        self._audio_started = False
        self.player.new_turn()
        self._resp_task = asyncio.create_task(self._respond())

    def _on_first_audio(self) -> None:
        self._audio_started = True
        m = self.metrics.current
        m.mark("audio_start")
        self.on_event("metrics", **m.summary())

    async def _respond(self) -> None:
        m = self.metrics.current
        chunker = SentenceChunker()
        q: asyncio.Queue = asyncio.Queue()
        full: list[str] = []

        async def produce() -> None:
            try:
                async for tok in self.llm.stream(self._trimmed_history(), self.cfg.system_prompt):
                    m.mark_once("llm_first_token")
                    full.append(tok)
                    for s in chunker.feed(tok):
                        await q.put(s)
                for s in chunker.flush():
                    await q.put(s)
            finally:
                q.put_nowait(None)

        async def consume() -> None:
            while (s := await q.get()) is not None:
                s = clean_for_tts(s)
                if not s:
                    continue
                pcm, sr = await self.tts.synth(s)
                m.mark_once("tts_first_chunk")
                self.player.enqueue(s, resample(pcm, sr, self.cfg.out_rate))
                self.on_event("agent", text=s)

        prod, cons = asyncio.create_task(produce()), asyncio.create_task(consume())
        try:
            await asyncio.gather(prod, cons)
            await self.player.drain()
            self.history.append({"role": "assistant", "content": "".join(full).strip()})
        except asyncio.CancelledError:
            prod.cancel()
            cons.cancel()
            await asyncio.gather(prod, cons, return_exceptions=True)
            heard = self.player.spoken_text()  # only what the user actually heard
            if heard:
                self.history.append({"role": "assistant", "content": heard + " [user interrupted]"})
            raise
        except Exception as e:
            prod.cancel()
            cons.cancel()
            self.on_event("error", error=repr(e))
