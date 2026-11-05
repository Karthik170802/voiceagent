import asyncio
import json

import numpy as np
import pytest

pytest.importorskip("aiohttp")
from aiohttp import WSMsgType
from aiohttp.test_utils import TestClient, TestServer

from kartos.server import create_app


def tone(freq, secs, sr=16000):
    t = np.arange(int(sr * secs)) / sr
    return (6000 * np.sin(2 * np.pi * freq * t)).astype(np.int16)


def test_websocket_roundtrip():
    async def go():
        client = TestClient(TestServer(create_app(stt="mock", llm="mock", tts="mock")))
        await client.start_server()
        events, audio_bytes = [], 0
        ws = await client.ws_connect("/ws")

        async def reader():
            nonlocal audio_bytes
            async for m in ws:
                if m.type == WSMsgType.BINARY:
                    audio_bytes += len(m.data)
                elif m.type == WSMsgType.TEXT:
                    events.append(json.loads(m.data)["name"])

        task = asyncio.create_task(reader())
        rng = np.random.default_rng(0)
        pcm = np.concatenate([rng.normal(0, 20, 8000).astype(np.int16), tone(300, 1.4),
                              rng.normal(0, 20, 16000 * 4).astype(np.int16)])
        for i in range(0, len(pcm), 320):                  # 20 ms chunks, real time
            await ws.send_bytes(pcm[i:i + 320].astype("<i2").tobytes())
            await asyncio.sleep(0.02)
        await ws.close()
        await asyncio.wait_for(task, 5)
        await client.close()
        return events, audio_bytes

    events, audio_bytes = asyncio.run(go())
    assert "user" in events and "agent" in events and "metrics" in events
    assert audio_bytes > 24000 * 2 * 0.5          # got at least ~0.5 s of reply audio
