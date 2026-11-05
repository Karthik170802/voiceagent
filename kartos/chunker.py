"""Turns a token stream into speakable chunks so TTS can start before the LLM finishes."""
from __future__ import annotations

import re

_ABBREV = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "st", "e.g", "i.e", "approx", "inc"}
_SENT = re.compile(r'[.!?]+["\')\]]*(?=\s)|\n')
_CLAUSE = re.compile(r"[,;:\u2014\u2013](?=\s)")


class SentenceChunker:
    """First chunk is emitted ASAP (any full sentence, or a clause once long enough) to cut
    time-to-first-audio. Later chunks are merged up to `min_chars` for better prosody."""

    def __init__(self, first_min_chars: int = 20, min_chars: int = 40, max_chars: int = 240):
        self.first_min = first_min_chars
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.buf = ""
        self.emitted = 0

    def feed(self, text: str) -> list[str]:
        self.buf += text
        out: list[str] = []
        while (cut := self._find_cut()) is not None:
            chunk, self.buf = self.buf[:cut].strip(), self.buf[cut:]
            if chunk:
                out.append(chunk)
                self.emitted += 1
        return out

    def flush(self) -> list[str]:
        rest, self.buf = self.buf.strip(), ""
        return [rest] if rest else []

    def _is_abbrev(self, idx: int) -> bool:
        if self.buf[idx] != "." or (idx + 1 < len(self.buf) and self.buf[idx + 1] == "."):
            return False
        m = re.search(r"([A-Za-z]+(?:\.[A-Za-z]+)*)$", self.buf[:idx])
        return bool(m) and (m.group(1).lower() in _ABBREV or len(m.group(1)) == 1)

    def _find_cut(self) -> int | None:
        first = self.emitted == 0
        min_len = 1 if first else self.min_chars
        for m in _SENT.finditer(self.buf):
            if self._is_abbrev(m.start()):
                continue
            if len(self.buf[: m.end()].strip()) >= min_len:
                return m.end()
        if first:
            for m in _CLAUSE.finditer(self.buf):
                if m.end() >= self.first_min:
                    return m.end()
        if len(self.buf) > self.max_chars:
            sp = self.buf.rfind(" ", 0, self.max_chars)
            if sp > 0:
                return sp + 1
        return None


def clean_for_tts(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[*_`#>~|]+", "", text)
    text = re.sub("[\U0001F300-\U0001FAFF\u2600-\u27BF]", "", text)
    return re.sub(r"\s+", " ", text).strip()
