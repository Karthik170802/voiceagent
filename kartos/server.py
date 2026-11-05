"""Browser front end server: serves web/index.html and a WebSocket voice endpoint.

Protocol (one WebSocket per call, path /ws)
  browser -> server : binary, int16 little-endian, 16 kHz mono (any chunk size)
  server  -> browser: binary, int16 little-endian, 24 kHz mono (paced ~real time)
                      text, JSON events: {"type":"event","name":"user","text":"..."} etc.

Run:  python -m kartos.server            (real STT/LLM/TTS)
      python -m kartos.server --llm mock --tts mock   (plumbing test, no key / no voice model)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
from aiohttp import WSMsgType, web

from .agent import VoiceAgent
from .config import Config
from .player import Player
from .providers import AnthropicLLM, FasterWhisperSTT, MockLLM, MockSTT, MockTTS, PiperTTS
from .tools import default_tools
from .vad import EnergyVAD, make_vad

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

STT_KEY = web.AppKey("stt", any)
TTS_KEY = web.AppKey("tts", any)
VAD_FACTORY_KEY = web.AppKey("make_vad", any)
LLM_FACTORY_KEY = web.AppKey("make_llm", any)


class WSSink:
    """Sends audio to the browser, paced to real time so pause/flush/barge-in stay accurate
    (the Player must not run far ahead of what the user has actually heard)."""

    def __init__(self, ws: web.WebSocketResponse, rate: int, lead: float = 0.12):
        self.ws, self.rate, self.lead = ws, rate, lead
        self._t = 0.0

    async def write(self, pcm: np.ndarray) -> None:
        if self.ws.closed:
            return
        await self.ws.send_bytes(pcm.astype("<i2").tobytes())
        now = time.monotonic()
        self._t = max(self._t, now) + len(pcm) / self.rate
        delay = self._t - now - self.lead
        if delay > 0:
            await asyncio.sleep(delay)


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)
    app = request.app
    cfg = Config(headphones=request.query.get("speakers") != "1")  # browser AEC => treat as headphones
    pending: set[asyncio.Task] = set()

    def on_event(name: str, **kw) -> None:
        if ws.closed:
            return
        t = asyncio.get_running_loop().create_task(
            ws.send_str(json.dumps({"type": "event", "name": name, **kw}, default=str)))
        pending.add(t)
        t.add_done_callback(pending.discard)

    make_vad_fn = app.get(VAD_FACTORY_KEY) or app.get("make_vad")
    stt_provider = app.get(STT_KEY) or app.get("stt")
    make_llm_fn = app.get(LLM_FACTORY_KEY) or app.get("make_llm")
    tts_provider = app.get(TTS_KEY) or app.get("tts")

    agent = VoiceAgent(cfg, make_vad_fn(cfg), stt_provider, make_llm_fn(cfg), tts_provider,
                       Player(WSSink(ws, cfg.out_rate), cfg.out_rate), on_event)

    async def frames():
        buf = np.empty(0, dtype=np.int16)
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                buf = np.concatenate([buf, np.frombuffer(msg.data, dtype="<i2")])
                while len(buf) >= cfg.frame_samples:
                    yield buf[: cfg.frame_samples].copy()
                    buf = buf[cfg.frame_samples:]
            elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                break

    try:
        await agent.run(frames())
    finally:
        print("call ended:", agent.metrics.report())
    return ws


async def index(request: web.Request) -> web.FileResponse:
    if (STATIC_DIR / "index.html").exists():
        return web.FileResponse(STATIC_DIR / "index.html")
    return web.FileResponse(WEB_DIR / "index.html")


def create_app(stt: str = "whisper", llm: str = "anthropic", tts: str = "piper",
               whisper_model: str = "base.en", piper_model: str = "en_US-lessac-medium.onnx") -> web.Application:
    app = web.Application()
    stt_inst = MockSTT() if stt == "mock" else FasterWhisperSTT(whisper_model)  # loaded once, shared
    tts_inst = MockTTS() if tts == "mock" else PiperTTS(piper_model)
    vad_fn = EnergyVAD if stt == "mock" else make_vad
    if llm == "mock":
        llm_fn = lambda cfg: MockLLM()
    else:
        llm_fn = lambda cfg: AnthropicLLM(cfg.model, cfg.max_tokens, default_tools(), cfg.tool_filler)

    app[STT_KEY] = stt_inst
    app[TTS_KEY] = tts_inst
    app[VAD_FACTORY_KEY] = vad_fn
    app[LLM_FACTORY_KEY] = llm_fn

    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    if STATIC_DIR.exists():
        app.router.add_static("/static/", STATIC_DIR)
    return app


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--stt", choices=["whisper", "mock"], default="whisper")
    p.add_argument("--llm", choices=["anthropic", "mock"], default="anthropic")
    p.add_argument("--tts", choices=["piper", "mock"], default="piper")
    p.add_argument("--whisper-model", default="base.en")
    p.add_argument("--piper-model", default="en_US-lessac-medium.onnx")
    a = p.parse_args()
    app = create_app(a.stt, a.llm, a.tts, a.whisper_model, a.piper_model)
    print(f"kartOS web: http://{a.host}:{a.port}")
    web.run_app(app, host=a.host, port=a.port, print=None)


if __name__ == "__main__":
    main()
