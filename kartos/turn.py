"""Turn-taking brain: adaptive end-of-turn timing and backchannel / interruption classification."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional, Protocol

from .config import Config

_DANGLING = {
    "and", "but", "or", "so", "because", "cause", "if", "then", "that", "which", "who",
    "the", "a", "an", "to", "of", "with", "for", "in", "on", "at", "from", "by", "about",
    "my", "your", "is", "are", "was", "like", "also", "um", "uh", "er", "hmm",
}

_SINGLE = {
    "mm", "mmm", "mhm", "hmm", "hm", "uh", "huh", "um", "yeah", "yep", "yes", "yup", "ok",
    "okay", "right", "sure", "ah", "oh", "cool", "nice", "wow", "really", "interesting",
    "gotcha", "alright",
}
_PHRASES = {"i see", "got it", "i get it", "go on", "makes sense"}
_STOP = ("stop", "wait", "hold on", "pause", "cancel", "never mind", "nevermind", "shut up", "enough")


def _words(text: str) -> list[str]:
    return re.sub(r"[^\w\s']", " ", text.lower()).split()


def classify(text: str) -> str:
    """'empty' | 'backchannel' | 'interrupt'. Only used while the agent is speaking."""
    words = _words(text)
    if not words:
        return "empty"
    joined = " ".join(words)
    if any(s in joined for s in _STOP):
        return "interrupt"
    if joined in _PHRASES or (len(words) <= 2 and all(w in _SINGLE for w in words)):
        return "backchannel"
    return "interrupt"


class EndOfTurnDetector:
    """Decides how long to wait after the user stops talking, based on what they said.

    A finished sentence ("What's the weather?") gets a short wait; a trailing
    conjunction or half-dictated number gets a long one. This is the cheap heuristic
    version; swap in a model (LiveKit turn-detector, Pipecat Smart Turn) behind the
    same `required_silence_ms` interface.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def required_silence_ms(self, text: str) -> int:
        c = self.cfg
        t = text.strip()
        words = _words(t)
        if not words:
            return c.base_silence_ms
        if t[-1] in ",…-–:;" or t.endswith(".."):
            return min(c.incomplete_silence_ms, c.max_silence_ms)
        if words[-1] in _DANGLING or words[-1].isdigit():
            return min(c.incomplete_silence_ms, c.max_silence_ms)
        if t[-1] in ".?!" and len(words) >= 2:
            return c.complete_silence_ms
        return c.base_silence_ms


class TurnDetector(Protocol):
    def required_silence_ms(self, text: str) -> int: ...


class LearnedEndOfTurnDetector:
    """Model-based end-of-turn detector conforming to `required_silence_ms`.

    Uses an ONNX classification model or probability predictor to determine turn completion.
    Gracefully falls back to heuristic EndOfTurnDetector if model files are absent or fail to load.
    """

    def __init__(
        self,
        cfg: Config,
        model_path: str | Path | None = None,
        predictor: Optional[Callable[[str], float]] = None,
    ):
        self.cfg = cfg
        self.model_path = model_path or cfg.turn_detector_model_path
        self.threshold = cfg.turn_detector_threshold
        self.heuristic = EndOfTurnDetector(cfg)
        self.predictor = predictor
        self.session = None

        if self.predictor is None and self.model_path:
            self._load_onnx()

    def _load_onnx(self) -> None:
        try:
            p = Path(self.model_path)
            if not p.is_file():
                return
            import onnxruntime as ort
            self.session = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        except Exception:
            self.session = None

    def predict_completion_prob(self, text: str) -> Optional[float]:
        if self.predictor is not None:
            try:
                return float(self.predictor(text))
            except Exception:
                return None

        if self.session is not None:
            try:
                import numpy as np
                input_name = self.session.get_inputs()[0].name
                out = self.session.run(None, {input_name: np.array([text], dtype=object)})
                val = float(out[0].flat[0])
                if val < 0.0 or val > 1.0:
                    val = 1.0 / (1.0 + np.exp(-val))
                return val
            except Exception:
                return None

        return None

    def required_silence_ms(self, text: str) -> int:
        prob = self.predict_completion_prob(text)
        if prob is None:
            # Fall back gracefully to heuristic
            return self.heuristic.required_silence_ms(text)

        c = self.cfg
        if prob >= self.threshold:
            return c.complete_silence_ms
        elif prob < (self.threshold * 0.5):
            return min(c.incomplete_silence_ms, c.max_silence_ms)
        return c.base_silence_ms


def make_turn_detector(cfg: Config) -> TurnDetector:
    """Factory selecting the turn detector based on Config (heuristic vs model)."""
    if getattr(cfg, "turn_detector", "heuristic") == "model":
        return LearnedEndOfTurnDetector(cfg)
    return EndOfTurnDetector(cfg)
