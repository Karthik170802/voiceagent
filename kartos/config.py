from __future__ import annotations

import os
from dataclasses import dataclass

SYSTEM_PROMPT = """You are a friendly voice assistant on a live call. Your words are spoken aloud.
Rules:
- Reply in one to three short sentences unless asked for more.
- Plain spoken language only: no markdown, lists, emojis, or code blocks.
- Say numbers, dates and units the way a person would say them.
- If the user's request is unclear, ask one short clarifying question.
- If you were interrupted, don't repeat yourself; respond to what the user just said."""


@dataclass
class Config:
    # ---- audio ----
    sample_rate: int = 16000          # mic rate
    frame_samples: int = 512          # 32 ms @ 16 kHz (Silero VAD's native frame size)
    out_rate: int = 24000             # speaker rate
    headphones: bool = True          # False => stricter VAD while agent speaks (echo guard)

    # ---- VAD ----
    vad_threshold: float = 0.5
    energy_min_rms: float = 400.0
    speech_start_ms: int = 96         # sustained speech needed to open an utterance
    preroll_ms: int = 320             # audio kept from before speech start

    # ---- end-of-turn ----
    turn_detector: str = "heuristic"  # "heuristic" | "model"
    turn_detector_model_path: str | None = None
    turn_detector_threshold: float = 0.5
    min_silence_ms: int = 250         # when speculative transcription kicks in
    base_silence_ms: int = 600
    complete_silence_ms: int = 350    # sentence looks finished
    incomplete_silence_ms: int = 1400 # "and...", "my number is 5 5..."
    max_silence_ms: int = 2500
    min_speech_ms: int = 160          # shorter blips are ignored
    max_utterance_s: float = 30.0

    # ---- barge-in ----
    duck_after_ms: int = 240          # pause agent audio after this much user speech

    # ---- LLM / TTS chunking ----
    model: str = os.getenv("KARTOS_MODEL", "claude-haiku-5-5")
    max_tokens: int = 300
    system_prompt: str = SYSTEM_PROMPT
    max_history_messages: int = 24
    tool_filler: str = "One moment. "

    @property
    def frame_ms(self) -> float:
        return self.frame_samples * 1000 / self.sample_rate

    def frames(self, ms: float) -> int:
        return max(1, round(ms / self.frame_ms))
