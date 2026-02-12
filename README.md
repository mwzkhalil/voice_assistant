# Voice Assistant (Urdu) -- Realtime Streaming System

Urdu voice assistant with dual transport support (Asterisk ARI telephony and public WebSocket), real time streaming ASR/LLM/TTS pipeline, Redis backed session management, and full observability.

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Configuration Reference](#configuration-reference)
- [Running Locally](#running-locally)
- [Public Deployment](#public-deployment)
- [WebSocket Protocol Specification](#websocket-protocol-specification)
- [Performance Characteristics](#performance-characteristics)
- [Observability](#observability)
- [Failure Handling](#failure-handling)
- [Security Model](#security-model)
- [Load Testing](#load-testing)
- [Development Guidelines](#development-guidelines)

---

## Project Overview

This system is a real-time, streaming Urdu voice assistant designed for telecom and public facing environments. It accepts audio input from either SIP/GSM phone calls (via Asterisk ARI) or WebSocket clients (browsers, mobile apps, WebRTC frontends), processes speech through a streaming ASR-LLM-TTS pipeline, and returns synthesized audio responses with sub-1.5-second target latency.

### Architectural Goals

- **Zero blocking calls.** Every component is fully async. Model inference runs in dedicated thread pool executors to avoid event loop starvation.
- **Transport-agnostic pipeline.** The streaming pipeline operates against an abstract `AudioTransport` interface. ARI/RTP and WebSocket transports are interchangeable at deployment time via a single environment variable.
- **Per-call isolation.** Each call/session gets its own pipeline instance with independent asyncio task group, bounded queues, VAD tracker, and cancellation scope. No shared mutable state between sessions.
- **Strict state machine.** Every session enforces a validated state machine (INIT, ANSWERED, LISTENING, THINKING, SPEAKING, INTERRUPTED, TERMINATED) with Redis-backed persistence and distributed locking.
- **Barge-in support.** When the VAD detects user speech during the SPEAKING state, the system immediately cancels LLM and TTS generation, flushes all queues, stops audio output, and transitions back to LISTENING.
- **Circuit breaker.** The LLM service implements a three-state circuit breaker (CLOSED, OPEN, HALF_OPEN) with configurable failure threshold and recovery timeout to prevent cascading failures.
- **No closed-source API dependencies.** All models (ASR, LLM, TTS) run locally on GPU. The LLM is served via Ollama.

### Enterprise Constraints

- Python 3.11+ required
- AsyncIO only, no synchronous blocking calls anywhere
- No global mutable state
- No monolithic scripts
- Full type hints on all function signatures
- Structured JSON logging only, no print statements
- Pydantic models for all schemas
- Dependency injection for all services

---

## System Architecture

### High-Level Architecture

```
                         +------------------+
                         |   Browser /      |
                         |   Mobile App     |
                         +--------+---------+
                                  |
                           WSS /ws/realtime
                                  |
                         +--------v---------+
                         |     Nginx        |
                         |  (TLS, rate      |
                         |   limiting)      |
                         +--------+---------+
                                  |
                         +--------v---------+
                         |    FastAPI        |
                         |    (main.py)      |
                         +--------+---------+
                                  |
           +----------------------+----------------------+
           |                                             |
  TRANSPORT_MODE=WS                            TRANSPORT_MODE=ARI
           |                                             |
  +--------v---------+                       +-----------v----------+
  | WebSocketTransport|                       |    ARITransport      |
  | (base64 PCM over |                       |  (wraps RTPBridge)   |
  |  WebSocket)       |                       +----------+-----------+
  +--------+---------+                                   |
           |                                    +--------v--------+
           |                                    |   RTP Bridge    |
           |                                    | (UDP, ulaw/PCM) |
           |                                    +--------+--------+
           |                                             |
           +---------------------+-----------------------+
                                 |
                    +------------v-----------+
                    |   AudioTransport       |
                    |   (abstract interface) |
                    +------------+-----------+
                                 |
                    +------------v-----------+
                    |   StreamingPipeline    |
                    |   (per-call instance)  |
                    +--+---+---+---+---+----+
                       |   |   |   |   |
            +----------+   |   |   |   +----------+
            |              |   |   |              |
    +-------v--+   +------v-+ | +-v------+  +----v------+
    | Audio In |   | ASR    | | | TTS    |  | Audio Out |
    | (VAD)    |   | Service| | | Service|  | (paced)   |
    +----------+   +--------+ | +--------+  +-----------+
                        +------v------+
                        | LLM Service |
                        | (Ollama)    |
                        +-------------+
                               |
                    +----------v----------+
                    | Redis               |
                    | (sessions, locks)   |
                    +---------------------+
```

### Component Breakdown

| Layer | Module | Responsibility |
|-------|--------|---------------|
| Core | `core/config.py` | Pydantic BaseSettings for all environment variables |
| Core | `core/logger.py` | structlog JSON logging with correlation IDs |
| Core | `core/exceptions.py` | Structured exception hierarchy (12 exception classes) |
| Core | `core/metrics.py` | Prometheus metrics registration and MetricsCollector facade |
| Core | `core/lifecycle.py` | Ordered startup/shutdown hooks, FastAPI lifespan integration |
| Core | `core/dependency_injection.py` | ServiceContainer with lazy initialization, transport-mode-aware |
| Transport | `transport/base.py` | Abstract `AudioTransport` interface |
| Transport | `transport/ari_transport.py` | ARI/RTP adapter wrapping `RTPBridge` |
| Transport | `transport/websocket_transport.py` | WebSocket transport with base64 PCM and metadata channel |
| Services | `services/vad_service.py` | WebRTC VAD with per-call state tracker |
| Services | `services/asr_service.py` | Auto-detects Wav2Vec2 (CTC) or Whisper (seq2seq), thread-pool inference |
| Services | `services/llm_service.py` | OpenAI AsyncClient for Ollama, circuit breaker, retry, streaming |
| Services | `services/tts_service.py` | VITS/MMS model, sentence-level synthesis, ulaw/PCM codec |
| Services | `services/session_manager.py` | Redis-backed session persistence with state machine validation |
| Services | `services/streaming_pipeline.py` | Per-call pipeline orchestrator: Audio-in -> VAD -> ASR -> LLM -> TTS -> Audio-out |
| ARI | `ari/ari_client.py` | ARI WebSocket client with auto-reconnect and REST-over-WebSocket |
| ARI | `ari/event_router.py` | Event dispatch with handler registry and error isolation |
| ARI | `ari/call_controller.py` | Call lifecycle: answer, ExternalMedia, bridge, pipeline, teardown |
| ARI | `ari/rtp_bridge.py` | Per-call UDP socket, RTP parse/serialize, ulaw codec, port pool |
| API | `api/health.py` | Liveness and readiness probes (transport-mode aware) |
| API | `api/metrics_endpoint.py` | Prometheus metrics endpoint with GPU memory reporting |
| API | `api/realtime.py` | Public WebSocket endpoint with session lifecycle management |
| API | `api/auth.py` | API key validation, per-IP rate limiting, session limits |
| API | `api/websocket_debug.py` | Debug WebSocket for real-time event monitoring |

### Per-Call Data Flow

Each call/session creates an isolated pipeline with five concurrent asyncio tasks connected by bounded queues:

```
Audio-In -> [VAD] -> ASR Audio Queue (100) -> [ASR] -> ASR Text Queue (10)
     -> [LLM] -> LLM Text Queue (50) -> [TTS] -> TTS Audio Queue (20) -> Audio-Out
```

1. **Audio Ingest Task**: Reads PCM frames from the transport, runs frame-level VAD. When speech is detected, frames are pushed to the ASR audio queue. When silence exceeds the configured threshold after speech, the queue is signaled with end-of-stream and the session transitions to THINKING.

2. **ASR Task**: Accumulates audio frames into chunks of configured duration (default 2000ms) with overlap (default 200ms). Runs inference in a thread pool executor. Yields partial transcripts and combines them into a final transcript.

3. **LLM Task**: Takes the final transcript, retrieves conversation history from the Redis-backed session, constructs the message array with the Urdu system prompt, and streams tokens from Ollama. Tokens are pushed to the TTS text queue as they arrive.

4. **TTS Task**: Accumulates tokens and splits at Urdu sentence boundaries. Synthesizes each sentence fragment in a thread pool executor. The first audio chunk triggers transition to SPEAKING and records end-to-end latency.

5. **Audio Output Task**: Reads synthesized PCM from the TTS audio queue and writes it through the transport. When the queue receives end-of-stream, transitions back to LISTENING.

### State Machine

```
States:  INIT -> ANSWERED -> LISTENING -> THINKING -> SPEAKING -> LISTENING -> ...
                                                   -> INTERRUPTED -> LISTENING
         Any non-terminal state -> TERMINATED
```

| From | Valid Targets |
|------|--------------|
| INIT | ANSWERED, TERMINATED |
| ANSWERED | LISTENING, TERMINATED |
| LISTENING | THINKING, TERMINATED |
| THINKING | SPEAKING, TERMINATED |
| SPEAKING | INTERRUPTED, LISTENING, TERMINATED |
| INTERRUPTED | LISTENING, TERMINATED |
| TERMINATED | (terminal -- no transitions) |

Invalid transitions raise `InvalidStateTransition` with the current state, target state, and call ID in the exception context. All transitions are serialized through Redis distributed locks.

### Circuit Breaker

The LLM service implements a circuit breaker to prevent cascading failures when the Ollama backend is degraded:

- **CLOSED** (normal): Requests pass through. Failures are counted.
- **OPEN** (tripped): After `circuit_breaker_threshold` consecutive failures (default 5), all requests are immediately rejected with `CircuitBreakerOpen` for `circuit_breaker_recovery_timeout` seconds (default 30).
- **HALF_OPEN** (probe): After recovery timeout, one request is allowed through. Success closes the circuit; failure reopens it.

The circuit breaker state is exported as a Prometheus gauge (`voice_assistant_llm_circuit_breaker_state`).

### Barge-In Handling

When the VAD detects speech during the SPEAKING state:

1. `_llm_cancel` and `_tts_cancel` events are set
2. ASR text, LLM text, and TTS audio queues are flushed
3. Transport `stop_output()` is called immediately
4. Session transitions SPEAKING -> INTERRUPTED -> LISTENING
5. Cancel events are cleared for the next turn
6. VAD tracker is reset

This ensures the user can interrupt the assistant mid-response with sub-frame latency.

---

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.11+ |
| Framework | FastAPI + Uvicorn | >=0.115.0 / >=0.32.0 |
| Async Runtime | asyncio + uvloop (Linux) | -- |
| WebSocket | websockets | >=14.0 |
| Validation | Pydantic + pydantic-settings | >=2.10.0 / >=2.7.0 |
| LLM Client | openai (AsyncOpenAI) | >=1.60.0 |
| ML Runtime | PyTorch + torchaudio | >=2.5.0 |
| Models | HuggingFace transformers | >=4.47.0 |
| VAD | webrtcvad | >=2.0.10 |
| Session Store | Redis (hiredis) | >=5.2.0 |
| Metrics | prometheus-client | >=0.21.0 |
| Logging | structlog | >=24.4.0 |
| Reverse Proxy | Nginx | 1.27+ |
| Container | Docker + NVIDIA CUDA Runtime | 12.2.2 |

### Model Dependencies

| Role | Model | Architecture |
|------|-------|-------------|
| ASR | `sajadkawa/ns_finetune_urdu_asr_org` | Wav2Vec2 CTC (auto-detected; Whisper seq2seq also supported) |
| LLM | `mahwiz/zoma` | Served via Ollama (OpenAI-compatible `/v1/` endpoint) |
| TTS | `suhaibrashid17/MMS_TTS_Urdu_3` | VITS (MMS) |

### GPU Requirements

- NVIDIA GPU with CUDA 12.x support
- Minimum 6 GB VRAM (Tesla T4 16 GB recommended)
- ASR and TTS models are loaded to GPU at startup and held in memory
- CPU fallback is supported (set `ASR_DEVICE=cpu` and `TTS_DEVICE=cpu`)

---

## Configuration Reference

All configuration is sourced from environment variables. A `.env` file is supported. No hardcoded credentials exist in the codebase.

### Transport Mode

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SERVER_TRANSPORT_MODE` | `WS` or `ARI` | `WS` | `WS`: public WebSocket mode (no Asterisk required). `ARI`: Asterisk telephony mode. |

### Server

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SERVER_HOST` | str | `0.0.0.0` | Server bind host |
| `SERVER_PORT` | int | `8080` | Server bind port |
| `SERVER_WORKERS` | int | `1` | Uvicorn workers. Must be 1 when using GPU (models are not fork-safe). |
| `SERVER_LOG_LEVEL` | str | `INFO` | Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `SERVER_DEBUG` | bool | `false` | Enables /docs and /redoc endpoints |
| `SERVER_CORS_ORIGINS` | str | `*` | Comma-separated CORS origins |
| `SERVER_MAX_CONCURRENT_CALLS` | int | `200` | Maximum concurrent calls/sessions across all transports |

### Authentication (WebSocket mode)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AUTH_ENABLED` | bool | `true` | Enable API key authentication |
| `AUTH_API_KEYS` | str | `""` | Comma-separated valid API keys. Empty string disables key validation. |
| `AUTH_HEADER_NAME` | str | `X-API-Key` | HTTP header name for API key |
| `AUTH_RATE_LIMIT_PER_MINUTE` | int | `60` | Maximum requests per IP per sliding 60-second window |
| `AUTH_MAX_SESSIONS_PER_IP` | int | `5` | Maximum concurrent WebSocket sessions per IP |
| `AUTH_MAX_AUDIO_DURATION_SECONDS` | float | `600.0` | Maximum session duration in seconds (10 minutes) |
| `AUTH_MAX_CHUNK_SIZE_BYTES` | int | `32768` | Maximum base64-decoded audio chunk size per message |
| `AUTH_IDLE_SESSION_TIMEOUT` | float | `300.0` | Session terminated after this many seconds without client activity |

### Redis

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `REDIS_URL` | str | `redis://localhost:6379/0` | Redis connection URL |
| `REDIS_POOL_MAX_SIZE` | int | `20` | Maximum connections in pool |
| `REDIS_SOCKET_TIMEOUT` | float | `5.0` | Socket timeout in seconds |
| `REDIS_SESSION_TTL` | int | `3600` | Session auto-expiry TTL in seconds |
| `REDIS_LOCK_TIMEOUT` | float | `10.0` | Distributed lock acquisition timeout |

### LLM (Ollama)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `OLLAMA_BASE_URL` | str | `http://localhost:11434/v1` | Ollama OpenAI-compatible endpoint |
| `OLLAMA_MODEL_NAME` | str | `mahwiz/zoma` | Model name to request from Ollama |
| `OLLAMA_MAX_TOKENS` | int | `512` | Maximum tokens per response |
| `OLLAMA_TEMPERATURE` | float | `0.7` | Sampling temperature |
| `OLLAMA_REQUEST_TIMEOUT` | float | `30.0` | Per-request timeout in seconds |
| `OLLAMA_CIRCUIT_BREAKER_THRESHOLD` | int | `5` | Consecutive failures before circuit opens |
| `OLLAMA_CIRCUIT_BREAKER_RECOVERY_TIMEOUT` | float | `30.0` | Seconds before circuit transitions to half-open |
| `OLLAMA_RETRY_MAX_ATTEMPTS` | int | `3` | Retry attempts with exponential backoff |
| `OLLAMA_RETRY_BASE_DELAY` | float | `0.5` | Base delay in seconds (doubles each retry) |

### ASR

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ASR_MODEL_NAME` | str | `sajadkawa/ns_finetune_urdu_asr_org` | HuggingFace model identifier |
| `ASR_DEVICE` | str | `auto` | `auto` (GPU if available, else CPU), `cuda`, `cpu` |
| `ASR_SAMPLE_RATE` | int | `16000` | Model input sample rate in Hz |
| `ASR_CHUNK_DURATION_MS` | int | `2000` | Audio chunk duration for inference |
| `ASR_CHUNK_OVERLAP_MS` | int | `200` | Overlap between consecutive chunks for context |
| `ASR_MAX_AUDIO_BUFFER_SECONDS` | float | `30.0` | Maximum audio buffer before truncation |
| `ASR_INFERENCE_TIMEOUT` | float | `10.0` | Per-inference timeout |

### TTS

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TTS_MODEL_NAME` | str | `suhaibrashid17/MMS_TTS_Urdu_3` | HuggingFace model identifier |
| `TTS_DEVICE` | str | `auto` | `auto`, `cuda`, `cpu` |
| `TTS_OUTPUT_SAMPLE_RATE` | int | `8000` | Output sample rate. Use `16000` for WebSocket mode, `8000` for ARI/ulaw. |
| `TTS_MAX_TEXT_LENGTH` | int | `500` | Maximum characters per synthesis call |
| `TTS_INFERENCE_TIMEOUT` | float | `15.0` | Per-synthesis timeout |
| `TTS_SENTENCE_DELIMITERS` | str | `۔؟!،.?!,;` | Characters used to split LLM output for incremental synthesis |

### VAD

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `VAD_AGGRESSIVENESS` | int | `2` | WebRTC VAD aggressiveness (0=least, 3=most aggressive) |
| `VAD_FRAME_DURATION_MS` | int | `30` | Frame size: 10, 20, or 30 ms |
| `VAD_SILENCE_THRESHOLD_MS` | int | `800` | Silence duration after speech to trigger end-of-utterance |
| `VAD_SPEECH_THRESHOLD_MS` | int | `250` | Minimum speech duration before triggering ASR |

### RTP (ARI mode only)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `RTP_LISTEN_HOST` | str | `0.0.0.0` | UDP bind host |
| `RTP_PORT_RANGE_START` | int | `10000` | Lowest port in the allocation pool |
| `RTP_PORT_RANGE_END` | int | `20000` | Highest port in the allocation pool |
| `RTP_CODEC` | str | `ulaw` | `ulaw` (G.711 mu-law) or `slin16` |
| `RTP_PACKET_DURATION_MS` | int | `20` | RTP packet duration |

### Asterisk ARI (ARI mode only)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ARI_HOST` | str | `localhost` | Asterisk host |
| `ARI_PORT` | int | `8088` | ARI HTTP port |
| `ARI_USERNAME` | str | `asterisk` | ARI username |
| `ARI_PASSWORD` | str | `asterisk` | ARI password |
| `ARI_APP_NAME` | str | `voice-assistant` | Stasis application name |

---

## Running Locally

### Prerequisites

- Python 3.11+
- Redis server running on `localhost:6379`
- Ollama running on `localhost:11434` with `mahwiz/zoma` model pulled
- NVIDIA GPU with CUDA 12.x (optional; CPU fallback works)

### Install

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env: set AUTH_API_KEYS, OLLAMA_BASE_URL, device settings
```

### Run

```bash
python main.py
```

### Docker

```bash
docker compose build
docker compose up -d
```

### Verify

```bash
# Liveness
curl http://localhost:8080/health

# Readiness (checks Redis, models)
curl http://localhost:8080/ready

# Prometheus metrics
curl http://localhost:8080/metrics

# WebSocket test (CLI)
python -m tests.ws_client \
  --url ws://localhost:8080/ws/realtime \
  --api-key YOUR_KEY \
  --speech 3.0 \
  --silence 1.5

# Browser test client
# Open http://localhost:8080/static/index.html
```

---

## Public Deployment

### VPS Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 4 cores | 8 cores |
| RAM | 8 GB | 16 GB |
| GPU | NVIDIA T4 (16 GB) | NVIDIA A10/A30 (24 GB) |
| Disk | 40 GB (models cached) | 80 GB SSD |
| OS | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |
| CUDA | 12.2+ | 12.2+ |
| Docker | 24.0+ with nvidia-container-toolkit | 24.0+ |

### HTTPS Setup

1. Place TLS certificates in `nginx/ssl/`:

```bash
cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem nginx/ssl/fullchain.pem
cp /etc/letsencrypt/live/yourdomain.com/privkey.pem nginx/ssl/privkey.pem
```

2. The provided `nginx/nginx.conf` handles:
   - HTTP -> HTTPS redirect on port 80
   - TLS termination on port 443 (TLS 1.2 and 1.3)
   - WebSocket upgrade for `/ws/realtime` and `/ws/debug`
   - Rate limiting zones: 60 req/min for API, 10 connections/min for WebSocket, 5 concurrent WebSocket connections per IP
   - Security headers (HSTS, X-Frame-Options, X-Content-Type-Options)

3. Configure `.env`:

```bash
TRANSPORT_MODE=WS
AUTH_ENABLED=true
AUTH_API_KEYS=production-key-1,production-key-2
SERVER_MAX_CONCURRENT_CALLS=200
TTS_OUTPUT_SAMPLE_RATE=16000
```

4. Launch:

```bash
docker compose up -d
```

5. The system is accessible at:

```
https://yourdomain.com/ws/realtime?api_key=production-key-1
https://yourdomain.com/static/index.html
https://yourdomain.com/health
https://yourdomain.com/ready
https://yourdomain.com/metrics
```

### Firewall Configuration

```bash
# Allow HTTP/HTTPS
ufw allow 80/tcp
ufw allow 443/tcp

# ARI mode only: allow RTP range
ufw allow 10000:10100/udp

# Block direct access to application port
ufw deny 8080/tcp
```

### Resource Limits

The `docker-compose.yml` enforces:

| Service | Memory Limit | CPU Limit |
|---------|-------------|-----------|
| voice-assistant | 8 GB | 4.0 cores |
| redis | 512 MB | 1.0 core |
| nginx | 128 MB | 0.5 cores |

Redis is configured with `maxmemory 256mb` and `allkeys-lru` eviction.

---

## WebSocket Protocol Specification

### Endpoint

```
WS  /ws/realtime?api_key=<key>
WSS /ws/realtime?api_key=<key>    (behind Nginx TLS)
```

Authentication can also be provided via the `X-API-Key` header during the WebSocket handshake.

### Client-to-Server Messages

#### Audio Chunk

```json
{
  "type": "audio_chunk",
  "data": "<base64-encoded PCM 16-bit signed, 16 kHz, mono>"
}
```

The `data` field must decode to at most `AUTH_MAX_CHUNK_SIZE_BYTES` bytes (default 32768). Chunks exceeding this limit receive an error response and are discarded.

#### Ping

```json
{
  "type": "ping"
}
```

#### Stop Session

```json
{
  "type": "stop"
}
```

### Server-to-Client Messages

#### Audio Chunk

```json
{
  "type": "audio_chunk",
  "data": "<base64-encoded PCM 16-bit signed, 16 kHz, mono>",
  "timestamp": 1739367600.123
}
```

#### Partial Transcript

```json
{
  "type": "partial_transcript",
  "text": "آپ کیسے",
  "is_final": false,
  "timestamp": 1739367600.456
}
```

#### Final Transcript

```json
{
  "type": "final_transcript",
  "text": "آپ کیسے ہیں",
  "is_final": true,
  "timestamp": 1739367600.789
}
```

#### State Change

```json
{
  "type": "state_change",
  "state": "LISTENING",
  "timestamp": 1739367601.000
}
```

Possible `state` values: `LISTENING`, `THINKING`, `SPEAKING`, `INTERRUPTED`.

#### Error

```json
{
  "type": "error",
  "message": "Session idle timeout",
  "timestamp": 1739367601.500
}
```

#### Pong

```json
{
  "type": "pong",
  "timestamp": 1739367601.100
}
```

### Connection Lifecycle

1. Client opens WebSocket with API key.
2. Server validates API key, rate limit, session limit.
3. On success: server accepts, creates Redis session, transitions to ANSWERED then LISTENING, sends `state_change` message.
4. Client streams audio chunks. Server streams audio + transcript + state messages.
5. Session ends on: client `stop` message, client disconnect, idle timeout, max duration exceeded, or server error.
6. Server closes WebSocket, stops pipeline, transitions session to TERMINATED, cleans up resources.

### WebSocket Close Codes

| Code | Meaning |
|------|---------|
| 1000 | Normal closure |
| 1008 | Invalid API key |
| 1013 | Service not ready (try again later) |
| 4029 | Rate limit or session limit exceeded |

---

## Performance Characteristics

### Target Latency

| Metric | Target | Measurement |
|--------|--------|-------------|
| End-to-end (speech end to first TTS audio) | < 1.5 seconds | `voice_assistant_pipeline_e2e_latency_seconds` |
| ASR inference per chunk | < 500 ms | `voice_assistant_asr_latency_seconds` |
| LLM time-to-first-token | < 1.0 second | `voice_assistant_llm_latency_seconds` |
| TTS per sentence | < 500 ms | `voice_assistant_tts_latency_seconds` |

### Concurrency

- Default max concurrent sessions: 200
- Each session consumes approximately 5 asyncio tasks, 5 bounded queues, and one VAD tracker
- GPU VRAM is the primary bottleneck: ASR and TTS models are shared across sessions (inference serialized via thread pool with 2 workers each)
- Redis connections pooled (default max 20)

### GPU Memory Usage

| Component | Approximate VRAM |
|-----------|-----------------|
| ASR model (Wav2Vec2 300M) | ~1.2 GB |
| TTS model (VITS/MMS) | ~0.8 GB |
| PyTorch overhead | ~0.5 GB |
| **Total baseline** | **~2.5 GB** |

Inference activations add temporary VRAM per concurrent inference. With 2 ASR workers and 2 TTS workers, peak usage is approximately 4-5 GB on a Tesla T4.

### CPU Fallback

When `ASR_DEVICE=cpu` and `TTS_DEVICE=cpu`:
- Inference runs in thread pool executors (2 workers each)
- Latency increases 3-5x compared to GPU
- Suitable for development and testing, not recommended for production with >10 concurrent sessions

### Backpressure

All inter-stage queues are bounded:

| Queue | Max Size | Overflow Behavior |
|-------|----------|-------------------|
| ASR Audio Queue | 100 frames (~3s at 30ms) | Drops frame, logs warning |
| ASR Text Queue | 10 | Blocks producer (bounded put) |
| LLM Text Queue | 50 | Blocks producer |
| TTS Audio Queue | 20 | Blocks producer |
| WS Inbound Queue | 200 | Drops frame, increments `rtp_packets_dropped_total{reason="ws_inbound_full"}` |
| WS Outbound Queue | 100 | Drops frame, increments `rtp_packets_dropped_total{reason="ws_outbound_full"}` |

### Scaling Recommendations

- **Vertical scaling**: Upgrade to a larger GPU (A10, A30) for higher throughput.
- **Horizontal scaling**: Deploy multiple instances behind a load balancer with sticky sessions (WebSocket affinity). Redis provides shared session state.
- **Workers**: Must remain at 1 per instance when using GPU. Scale by adding instances, not workers.

---

## Observability

### Logging

All logs are JSON-formatted via structlog. Every log entry includes:

| Field | Source |
|-------|--------|
| `timestamp` | ISO 8601 |
| `level` | Log level |
| `event` | Event name |
| `service` | `"voice-assistant"` (static) |
| `correlation_id` | Per-request, from `X-Request-ID` header or auto-generated UUID |
| `call_id` | Per-call/session UUID (when in call context) |
| `module` | Python module name |
| `func_name` | Function name |
| `lineno` | Line number |

No print statements exist in the codebase. Third-party loggers (websockets, urllib3, httpx, asyncio) are set to WARNING level.

### Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Liveness probe. Returns 200 if the process is running. |
| `/ready` | GET | Readiness probe. Returns 200 if Redis, models, and transport are operational. Returns 503 otherwise. |
| `/metrics` | GET | Prometheus text format. All metrics listed below. |
| `/ws/debug` | WS | Real-time event stream for monitoring (auth-gated). |

### Prometheus Metrics

#### Call Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `voice_assistant_calls_total` | Counter | `status` (answered, rejected, error) | Total calls/sessions |
| `voice_assistant_calls_active` | Gauge | -- | Currently active calls/sessions |
| `voice_assistant_call_duration_seconds` | Histogram | -- | Call duration. Buckets: 1, 5, 10, 30, 60, 120, 300, 600 |

#### ASR Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `voice_assistant_asr_latency_seconds` | Histogram | -- | Per-chunk inference latency. Buckets: 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0 |
| `voice_assistant_asr_errors_total` | Counter | `error_type` | ASR errors by type |
| `voice_assistant_asr_chunks_processed_total` | Counter | -- | Total audio chunks processed |

#### LLM Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `voice_assistant_llm_latency_seconds` | Histogram | -- | Time-to-first-token. Buckets: 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0 |
| `voice_assistant_llm_total_latency_seconds` | Histogram | -- | Total generation latency. Buckets: 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0 |
| `voice_assistant_llm_tokens_generated_total` | Counter | -- | Total tokens generated |
| `voice_assistant_llm_errors_total` | Counter | `error_type` | LLM errors by type |
| `voice_assistant_llm_circuit_breaker_state` | Gauge | -- | 0=closed, 1=open, 2=half-open |

#### TTS Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `voice_assistant_tts_latency_seconds` | Histogram | -- | Per-sentence synthesis latency. Buckets: 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0 |
| `voice_assistant_tts_errors_total` | Counter | `error_type` | TTS errors by type |
| `voice_assistant_tts_chunks_generated_total` | Counter | -- | Total audio chunks generated |

#### Transport Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `voice_assistant_rtp_packets_received_total` | Counter | -- | Inbound packets (RTP or WS chunks) |
| `voice_assistant_rtp_packets_sent_total` | Counter | -- | Outbound packets |
| `voice_assistant_rtp_packets_dropped_total` | Counter | `reason` | Dropped: buffer_full, decode_error, ws_inbound_full, ws_outbound_full |

#### Pipeline Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `voice_assistant_pipeline_e2e_latency_seconds` | Histogram | -- | End-to-end latency. Buckets: 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0 |
| `voice_assistant_pipeline_interrupts_total` | Counter | -- | Total barge-in interrupts |

#### System Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `voice_assistant_gpu_memory_used_bytes` | Gauge | -- | GPU VRAM used (updated on /metrics scrape) |
| `voice_assistant_gpu_memory_total_bytes` | Gauge | -- | GPU VRAM total |
| `voice_assistant_redis_connections_active` | Gauge | -- | Active Redis pool connections |
| `voice_assistant_timeouts_total` | Counter | `component` (asr, llm, tts, ari, pipeline) | Timeouts by component |
| `voice_assistant_errors_total` | Counter | `category` (asr, llm, tts, ari, session, pipeline, system) | Errors by category |

---

## Failure Handling

Every failure mode is handled explicitly. No unhandled exceptions are allowed in production.

| Failure | Detection | Response | Impact on Other Sessions |
|---------|-----------|----------|------------------------|
| **GPU OOM** | PyTorch CUDA OOM exception during inference | Caught in thread pool executor. Logs structured error, increments error counter. Affected session receives error; pipeline continues for next utterance if possible. `torch.cuda.empty_cache()` called on model unload. | None. Models are shared but inference is serialized per thread pool worker. |
| **Redis disconnection** | `ConnectionError` on any Redis operation | Session operations fail with `SessionError`. Affected call cannot persist state transitions. Rate limiting falls back to in-memory. Health check returns `redis: error`. | Other sessions already in flight continue using cached session data. New sessions fail at creation. |
| **ARI WebSocket drop** | `ConnectionClosed` in receive loop | Auto-reconnect with exponential backoff (1s base, 30s max). All pending ARI commands fail with `ARIConnectionError`. Active calls continue with their existing RTP bridges. | None if reconnect succeeds. If reconnect fails after max attempts, new calls cannot be set up. |
| **LLM timeout** | `asyncio.TimeoutError` after `OLLAMA_REQUEST_TIMEOUT` | Increments timeout counter. If within retry budget, retries with exponential backoff. If circuit breaker threshold exceeded, circuit opens and rejects all requests for recovery period. | None. Circuit breaker is per-service, not per-session. |
| **TTS timeout** | `asyncio.TimeoutError` after `TTS_INFERENCE_TIMEOUT` | Logs error, increments timeout counter. Affected sentence is skipped. Pipeline continues with next sentence fragment. | None. |
| **WebSocket disconnect** | `WebSocketDisconnect` exception or receive timeout | Pipeline stopped, session transitioned to TERMINATED, per-IP session count decremented, resources cleaned up. | None. |
| **Malformed audio** | Invalid base64, oversized chunks, wrong PCM format | Error message sent to client. Chunk discarded. Session continues. | None. |
| **Model load failure** | Exception during `load_model()` startup | Application fails to start. Container health check fails. Orchestrator restarts container. | All sessions affected (startup failure). |

---

## Security Model

### API Key Validation

- API keys are configured via `AUTH_API_KEYS` (comma-separated).
- Validated against: `X-API-Key` header or `api_key` query parameter.
- Invalid keys result in WebSocket close code 1008 or HTTP 403.
- When `AUTH_API_KEYS` is empty, authentication is effectively disabled (development only).

### Rate Limiting

- **Algorithm**: Sliding window (60-second window).
- **Storage**: Redis sorted set when available, in-memory dict fallback.
- **Scope**: Per source IP (respects `X-Forwarded-For` from reverse proxy).
- **Default**: 60 requests per minute per IP.
- **Enforcement**: Applied before WebSocket acceptance and on HTTP endpoints.
- **Nginx layer**: Additional rate limiting at 60 req/min for API, 10 WebSocket connections/min per IP, max 5 concurrent WebSocket connections per IP.

### Session Limits

| Limit | Default | Enforcement |
|-------|---------|-------------|
| Concurrent sessions per IP | 5 | Checked before WebSocket acceptance |
| Max session duration | 600 seconds | Checked every 1 second in session loop |
| Max idle time | 300 seconds | WebSocket receive timeout |
| Max audio chunk size | 32768 bytes | Checked per received message |
| Max concurrent global sessions | 200 | Checked before WebSocket acceptance |

### Transport Security

- **TLS is required in production.** The Nginx configuration enforces HTTPS with TLS 1.2/1.3.
- WebSocket connections are upgraded from HTTPS (WSS).
- Direct access to the application port (8080) should be blocked by firewall.
- The application itself does not terminate TLS. All TLS is handled by Nginx.

### Production Hardening Recommendations

1. Set `AUTH_API_KEYS` to strong, randomly generated keys (minimum 32 characters).
2. Set `SERVER_DEBUG=false` to disable /docs and /redoc.
3. Restrict `SERVER_CORS_ORIGINS` to your domain instead of `*`.
4. Run the container as non-root (`appuser`, UID 1001, enforced in Dockerfile).
5. Enable Redis authentication (`requirepass`) in production.
6. Rotate API keys periodically.
7. Monitor `voice_assistant_errors_total` and `voice_assistant_timeouts_total` for anomaly detection.
8. Set `SERVER_MAX_CONCURRENT_CALLS` based on measured GPU capacity under load.

---

## Load Testing

### CLI WebSocket Tester

Single session test:

```bash
python -m tests.ws_client \
  --url wss://yourdomain.com/ws/realtime \
  --api-key YOUR_KEY \
  --speech 5.0 \
  --silence 2.0 \
  --listen 15.0
```

### WebSocket Load Test

Concurrent session stress test:

```bash
python -m tests.ws_client \
  --url wss://yourdomain.com/ws/realtime \
  --api-key YOUR_KEY \
  --load-test \
  --concurrency 50 \
  --duration 60
```

Output reports: total sessions, success/failure count, chunks sent, chunks per second.

### RTP Load Test (ARI mode)

```bash
python -m tests.load_test \
  --target yourdomain.com \
  --port 10000 \
  --calls 100 \
  --duration 10
```

### Health Endpoint Load Test

```bash
python -m tests.simulate_call --host yourdomain.com --port 443 --load-test --concurrency 50
```

### Metrics to Monitor During Load

| Metric | Healthy Range |
|--------|--------------|
| `voice_assistant_calls_active` | < `SERVER_MAX_CONCURRENT_CALLS` |
| `voice_assistant_pipeline_e2e_latency_seconds` (p95) | < 1.5 seconds |
| `voice_assistant_asr_latency_seconds` (p95) | < 1.0 seconds |
| `voice_assistant_llm_latency_seconds` (p95) | < 2.0 seconds |
| `voice_assistant_gpu_memory_used_bytes` | < 90% of total |
| `voice_assistant_rtp_packets_dropped_total` | Near zero |
| `voice_assistant_llm_circuit_breaker_state` | 0 (closed) |
| `voice_assistant_errors_total` rate | Near zero |

---

## Development Guidelines

### Async-Only Rule

Every function that performs I/O must be `async`. Synchronous blocking calls are forbidden. Model inference runs in `ThreadPoolExecutor` via `loop.run_in_executor()`.

### No Blocking Calls

The following are prohibited:
- `time.sleep()` (use `asyncio.sleep()`)
- Synchronous HTTP clients (use `openai.AsyncOpenAI`, `httpx.AsyncClient`)
- Synchronous Redis operations (use `redis.asyncio`)
- `threading.Lock` in async code (use `asyncio.Lock`)

### Type Hints

All function signatures must have complete type annotations. `from __future__ import annotations` is used in every module for forward reference support. Pydantic models are used for all structured data.

### Structured Logging

- Use `get_logger(__name__)` from `core/logger.py`.
- Never use `print()`.
- Log events must be descriptive: `logger.info("session_created", call_id=call_id, channel_id=channel_id)`.
- Include `call_id` in all per-call log messages (set via `set_call_id()`).

### Exception Handling

- Every exception must be caught and logged with structured context.
- Use the exception hierarchy from `core/exceptions.py`.
- Exception context dicts are logged and returned in API error responses.
- Never allow an unhandled exception to crash the event loop.

### Testing

- Unit tests use `pytest` with `pytest-asyncio` (auto mode).
- All services are mockable via dependency injection.
- Fixtures in `tests/conftest.py` provide mock Redis, ARI, ASR, TTS, LLM, and VAD instances.
- State machine tests cover every valid transition and every invalid transition.
- Run tests: `pytest tests/ -v`

### Code Quality

- No TODO comments in production code.
- No commented-out dead code.
- No unused imports.
- No mock placeholders or fake implementations.
- Every file must be fully runnable as part of the complete system.
