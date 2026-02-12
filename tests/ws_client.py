"""
CLI WebSocket tester for the realtime voice assistant.

Connects to /ws/realtime, sends synthetic audio, and displays
server responses in real-time.

Usage:
    python -m tests.ws_client --url ws://localhost:8080/ws/realtime --api-key YOUR_KEY
    python -m tests.ws_client --url wss://yourdomain.com/ws/realtime --api-key YOUR_KEY --duration 10
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import struct
import sys
import time

import numpy as np

try:
    import websockets
    import websockets.client
except ImportError:
    websockets = None  # type: ignore[assignment]


def generate_urdu_speech_audio(
    duration_ms: int = 30,
    sample_rate: int = 16000,
    frequency: float = 300.0,
    amplitude: float = 0.3,
) -> bytes:
    """Generate synthetic speech-like audio as PCM 16-bit bytes."""
    num_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)
    # Multi-frequency to simulate speech
    signal = (
        np.sin(2 * np.pi * frequency * t) * 0.5
        + np.sin(2 * np.pi * (frequency * 1.5) * t) * 0.3
        + np.sin(2 * np.pi * (frequency * 2.1) * t) * 0.2
    )
    pcm = (signal * amplitude * 32767).astype(np.int16)
    return pcm.tobytes()


def generate_silence(duration_ms: int = 30, sample_rate: int = 16000) -> bytes:
    """Generate silence as PCM 16-bit bytes."""
    num_samples = int(sample_rate * duration_ms / 1000)
    return b"\x00" * (num_samples * 2)


async def run_client(
    url: str,
    api_key: str = "",
    speech_duration: float = 3.0,
    silence_duration: float = 1.5,
    listen_duration: float = 10.0,
) -> None:
    """
    Connect to the realtime WebSocket and run a test session.

    Args:
        url: WebSocket URL (ws:// or wss://)
        api_key: API key for authentication
        speech_duration: Duration of speech audio to send (seconds)
        silence_duration: Duration of silence after speech (seconds)
        listen_duration: Duration to listen for responses (seconds)
    """
    if websockets is None:
        sys.exit("websockets is required: pip install websockets")

    # Add API key to URL
    if api_key:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}api_key={api_key}"

    sys.stdout.write(f"Connecting to {url}...\n")

    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key

    try:
        async with websockets.connect(url, additional_headers=headers) as ws:  # type: ignore[attr-defined]
            sys.stdout.write("Connected!\n\n")

            # Start receiver task
            receiver_task = asyncio.create_task(receive_messages(ws))

            # Send speech audio
            frame_duration_ms = 30
            num_speech_frames = int(speech_duration * 1000 / frame_duration_ms)
            sys.stdout.write(
                f"Sending {num_speech_frames} speech frames ({speech_duration}s)...\n"
            )

            for i in range(num_speech_frames):
                freq = 250 + (i % 40) * 8
                audio = generate_urdu_speech_audio(
                    duration_ms=frame_duration_ms,
                    frequency=freq,
                )
                b64 = base64.b64encode(audio).decode("ascii")
                await ws.send(json.dumps({
                    "type": "audio_chunk",
                    "data": b64,
                }))
                await asyncio.sleep(frame_duration_ms / 1000)

            # Send silence
            num_silence_frames = int(silence_duration * 1000 / frame_duration_ms)
            sys.stdout.write(
                f"Sending {num_silence_frames} silence frames ({silence_duration}s)...\n"
            )

            for _ in range(num_silence_frames):
                silence = generate_silence(frame_duration_ms)
                b64 = base64.b64encode(silence).decode("ascii")
                await ws.send(json.dumps({
                    "type": "audio_chunk",
                    "data": b64,
                }))
                await asyncio.sleep(frame_duration_ms / 1000)

            # Listen for responses
            sys.stdout.write(f"\nListening for responses ({listen_duration}s)...\n")
            await asyncio.sleep(listen_duration)

            # Disconnect
            await ws.send(json.dumps({"type": "stop"}))
            receiver_task.cancel()
            try:
                await receiver_task
            except asyncio.CancelledError:
                pass

            sys.stdout.write("\nTest complete.\n")

    except Exception as exc:
        sys.stdout.write(f"Connection failed: {exc}\n")


async def receive_messages(ws: Any) -> None:
    """Receive and display WebSocket messages."""
    audio_chunks = 0
    try:
        async for msg in ws:
            data = json.loads(msg)
            msg_type = data.get("type", "unknown")

            if msg_type == "audio_chunk":
                audio_chunks += 1
                b64 = data.get("data", "")
                audio_bytes = len(base64.b64decode(b64)) if b64 else 0
                if audio_chunks % 10 == 1:
                    sys.stdout.write(
                        f"  [AUDIO] Chunk #{audio_chunks} ({audio_bytes} bytes)\n"
                    )

            elif msg_type in ("partial_transcript", "final_transcript"):
                text = data.get("text", "")
                is_final = data.get("is_final", False)
                marker = "FINAL" if is_final else "PARTIAL"
                sys.stdout.write(f"  [{marker}] {text}\n")

            elif msg_type == "state_change":
                state = data.get("state", "")
                sys.stdout.write(f"  [STATE] {state}\n")

            elif msg_type == "error":
                message = data.get("message", "")
                sys.stdout.write(f"  [ERROR] {message}\n")

            elif msg_type == "pong":
                pass

            else:
                sys.stdout.write(f"  [UNKNOWN] {json.dumps(data)}\n")

    except asyncio.CancelledError:
        pass
    except Exception as exc:
        sys.stdout.write(f"  [RECV ERROR] {exc}\n")

    sys.stdout.write(f"\nTotal audio chunks received: {audio_chunks}\n")


async def ws_load_test(
    url: str,
    api_key: str = "",
    concurrency: int = 10,
    duration: float = 30.0,
) -> None:
    """
    Load test: open multiple concurrent WebSocket sessions.

    Args:
        url: WebSocket URL
        api_key: API key
        concurrency: Number of concurrent sessions
        duration: Duration per session in seconds
    """
    if websockets is None:
        sys.exit("websockets is required: pip install websockets")

    results: dict[str, int] = {"success": 0, "error": 0, "chunks_sent": 0}

    async def session(session_id: int) -> None:
        target_url = url
        if api_key:
            sep = "&" if "?" in target_url else "?"
            target_url = f"{target_url}{sep}api_key={api_key}"

        headers = {}
        if api_key:
            headers["X-API-Key"] = api_key

        try:
            async with websockets.connect(target_url, additional_headers=headers) as ws:  # type: ignore[attr-defined]
                end_time = time.monotonic() + duration
                frame_ms = 30
                i = 0

                while time.monotonic() < end_time:
                    freq = 250 + (i % 40) * 8
                    audio = generate_urdu_speech_audio(frame_ms, frequency=freq)
                    b64 = base64.b64encode(audio).decode("ascii")
                    await ws.send(json.dumps({
                        "type": "audio_chunk",
                        "data": b64,
                    }))
                    results["chunks_sent"] += 1
                    i += 1
                    await asyncio.sleep(frame_ms / 1000)

                await ws.send(json.dumps({"type": "stop"}))
                results["success"] += 1

        except Exception as exc:
            results["error"] += 1

    sys.stdout.write(
        f"Load test: {concurrency} sessions, {duration}s each, to {url}\n\n"
    )

    start = time.monotonic()
    tasks = [asyncio.create_task(session(i)) for i in range(concurrency)]
    await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.monotonic() - start

    sys.stdout.write(f"\n{'='*50}\n")
    sys.stdout.write(f"RESULTS\n")
    sys.stdout.write(f"{'='*50}\n")
    sys.stdout.write(f"Duration:       {elapsed:.1f}s\n")
    sys.stdout.write(f"Sessions:       {concurrency}\n")
    sys.stdout.write(f"Successful:     {results['success']}\n")
    sys.stdout.write(f"Failed:         {results['error']}\n")
    sys.stdout.write(f"Chunks sent:    {results['chunks_sent']}\n")
    sys.stdout.write(f"Chunks/sec:     {results['chunks_sent'] / elapsed:.0f}\n")
    sys.stdout.write(f"{'='*50}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="WebSocket Realtime Tester")
    parser.add_argument(
        "--url",
        default="ws://localhost:8080/ws/realtime",
        help="WebSocket URL",
    )
    parser.add_argument("--api-key", default="", help="API key")
    parser.add_argument("--speech", type=float, default=3.0, help="Speech duration (s)")
    parser.add_argument("--silence", type=float, default=1.5, help="Silence duration (s)")
    parser.add_argument("--listen", type=float, default=10.0, help="Listen duration (s)")
    parser.add_argument("--load-test", action="store_true", help="Run load test")
    parser.add_argument("--concurrency", type=int, default=10, help="Load test concurrency")
    parser.add_argument("--duration", type=float, default=30.0, help="Load test duration (s)")

    args = parser.parse_args()

    if args.load_test:
        asyncio.run(
            ws_load_test(args.url, args.api_key, args.concurrency, args.duration)
        )
    else:
        asyncio.run(
            run_client(args.url, args.api_key, args.speech, args.silence, args.listen)
        )


if __name__ == "__main__":
    main()
