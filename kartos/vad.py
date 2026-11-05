"""Voice activity detection. Silero if installed, otherwise an adaptive energy gate."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .config import Config


class VAD(Protocol):
    def is_speech(self, frame: np.ndarray, strict: bool = False) -> bool: ...


class EnergyVAD:
    """Zero-dependency fallback: RMS gate over an adaptive noise floor."""

    def __init__(self, cfg: Config):
        self.min_rms = cfg.energy_min_rms
        self.floor = 100.0

    def is_speech(self, frame: np.ndarray, strict: bool = False) -> bool:
        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
        thr = max(self.min_rms, self.floor * 3.0) * (2.0 if strict else 1.0)
        speech = rms > thr
        if not speech:
            self.floor = 0.95 * self.floor + 0.05 * rms
        return speech


class SileroVAD:
    def __init__(self, cfg: Config):
        import torch
        from silero_vad import load_silero_vad

        self.torch = torch
        self.model = load_silero_vad()
        self.sr = cfg.sample_rate
        self.thr = cfg.vad_threshold

    def is_speech(self, frame: np.ndarray, strict: bool = False) -> bool:
        x = self.torch.from_numpy(frame.astype(np.float32) / 32768.0)
        with self.torch.no_grad():
            p = float(self.model(x, self.sr).item())
        return p > (0.8 if strict else self.thr)


def make_vad(cfg: Config) -> VAD:
    try:
        return SileroVAD(cfg)
    except Exception:
        print("[vad] silero-vad not available, using energy VAD")
        return EnergyVAD(cfg)
