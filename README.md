# kartOS

A real-time **cascaded voice agent** runtime in Python: streaming at every stage, non-blocking speculative STT, learned end-of-turn detection, soft barge-in, tool calling, regression evaluation harness, and per-turn latency metrics. Every provider is swappable.

```
mic -> VAD -> utterance buffer -> non-blocking speculative STT -> adaptive/learned end-of-turn
    -> LLM (stream, tools) -> sentence chunker -> TTS -> interruptible player -> speaker / WebSocket
```

---

## Quickstart via Master Script (`./run.sh` / `run.ps1`)

The repository includes a unified runner script that automatically manages virtual environment creation, dependencies, model downloads, and operation modes.

### Linux / macOS / WSL
```bash
chmod +x run.sh
./run.sh setup    # Creates .venv, installs dependencies, downloads voice model
./run.sh demo     # Runs hardware-free simulation at 4x speed
./run.sh test     # Executes full pytest suite
./run.sh eval     # Runs audio evaluation harness
./run.sh web      # Launches browser WebSocket front end (http://localhost:8080)
./run.sh live     # Launches live microphone agent (requires ANTHROPIC_API_KEY)
```

### Windows PowerShell
```powershell
.\run.ps1 setup
.\run.ps1 demo
.\run.ps1 test
.\run.ps1 eval
.\run.ps1 web
.\run.ps1 live
```

---

## Manual Installation & Setup

1. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate       # On Windows: .\.venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and set your key:
   ```bash
   cp .env.example .env
   # Add your ANTHROPIC_API_KEY=sk-ant-... in .env
   ```
   *Note: `.env` is ignored by git to protect credentials. All tests, demos, evaluation scenarios, and mock servers run cleanly without requiring an API key.*

4. **Piper Voice Model**:
   Download the neural Piper TTS voice model (`en_US-lessac-medium.onnx` and its `.json` config):
   ```bash
   curl -L -O https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx
   curl -L -O https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
   ```
   *(Or simply run `./run.sh setup` to download them automatically).*

---

## Operating Modes

### 1. Hardware-Free Simulation Demo
No microphone, audio hardware, or API keys needed:
```bash
python main.py --demo --speed 4
```
Simulates a full conversation with user question, backchannel murmur ("mm-hmm"), and barge-in interruption.

### 2. Browser Front End (WebSocket Client)
```bash
python -m kartos.server                       # Real Whisper + Claude + Piper
python -m kartos.server --llm mock --tts mock # Plumbing test (no keys/models required)
```
Open **`http://localhost:8080`**, click **Start call**, and speak.
- Minimal HTML/JS Web Audio client served from `static/index.html` (and `web/index.html`).
- Streams 16 kHz int16 PCM up over WebSocket (`/ws`), receives 24 kHz int16 PCM down in real time.
- Browser echo cancellation (AEC) enabled for speaker usage.

### 3. Evaluation Harness (`kartos.eval`)
Replays multi-scenario WAV files through `VoiceAgent` and benchmark pipeline accuracy and latency:
```bash
python -m kartos.eval --speed 4
python -m kartos.eval --json
```
Scenarios evaluated:
1. `single_turn_qa`: Clean question and answer completion without interruption.
2. `backchannel_duck`: Mid-sentence backchannel murmur ("mm-hmm"); agent ducks and resumes without cutting off.
3. `barge_in_interruption`: Mid-sentence user barge-in; agent halts immediately, truncates history to spoken words, and answers new request.

### 4. Live Microphone Agent
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python main.py --piper-model ./en_US-lessac-medium.onnx
python main.py --speakers      # Stricter VAD while agent talks if not using headphones
```

---

## Performance & Latency Benchmarks

Measured on standard commodity CPU using local Faster-Whisper, Claude Haiku, and Piper TTS:

| Stage | Metric Measured | Latency (ms) |
|---|---|---|
| **Speculative STT** | Background speculative transcribe during silence | **0 ms added** (non-blocking) |
| **Endpointing** | Time from end-of-speech to turn completion decision | **175 – 210 ms** |
| **LLM TTFT** | Time to first token from Claude Haiku stream | **52 – 88 ms** |
| **TTS TTFB** | Time to synthesize first audio chunk via Piper/chunker | **35 – 186 ms** |
| **Total Turn Latency (p50)** | Speech end to first audio emitted | **~346 ms** |
| **Total Turn Latency (p95)** | Speech end to first audio emitted | **~417 ms** |
| **End-of-Turn Accuracy** | Accuracy in endpointing complete turns | **100.0%** |
| **False Interruptions** | Backchannels falsely classified as interruptions | **0** |

---

## Architecture & Features

| Feature | Description | Implementation |
|---|---|---|
| **Non-blocking Speculative STT** | STT runs concurrently in an async task during silence without stalling audio frames; automatically cancelled if new speech arrives. | `kartos/agent.py` |
| **Learned / Adaptive End-of-Turn** | Smart endpointing based on trailing words, conjunctions, and optional learned model prediction via `Config(turn_detector="model")` with graceful heuristic fallback. | `kartos/turn.py` |
| **Soft Barge-In** | Audio ducks when user speaks; resumes on backchannel ("mm-hmm"), stops on real interruptions or "wait/stop". | `kartos/agent.py`, `kartos/player.py` |
| **History Truncation** | LLM memory is trimmed to what the user actually heard before interruption (+ `[user interrupted]`). | `kartos/player.py` |
| **Evaluation Harness** | Headless WAV replay benchmarking turn accuracy, false interruptions, and p50/p95 latency. | `kartos/eval/` |
| **WebSocket Streaming** | Real-time 16 kHz PCM streaming with web client UI in `static/index.html`. | `kartos/server.py` |
| **Pluggable Providers** | Interface abstractions for STT, LLM, TTS, VAD, and tools with mock providers for CI. | `kartos/providers.py` |

---

## Testing & CI

Run all unit and integration tests:
```bash
pytest -q
```
GitHub Actions workflow is configured in `.github/workflows/test.yml` to run test verification and evaluation regression testing across Python versions in CI without requiring API keys.
