"""
CLI call simulator for testing the voice assistant.

Connects to the voice assistant's debug WebSocket and simulates
call events. Can also generate synthetic audio for end-to-end testing.

Usage:
    python -m tests.simulate_call --host localhost --port 8080
    python -m tests.simulate_call --health-check
    python -m tests.simulate_call --load-test --concurrency 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import sys
import time
from typing import Any

import numpy as np

try:
    import websockets
    import websockets.client
except ImportError:
    websockets = None  # type: ignore[assignment]

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]


def generate_sine_wave(
    frequency: float = 440.0,
    duration_ms: int = 20,
    sample_rate: int = 8000,
    amplitude: float = 0.3,
) -> bytes:
    """Generate a sine wave as ulaw-encoded bytes (simulating voice)."""
    num_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)
    pcm = (np.sin(2 * np.pi * frequency * t) * amplitude * 32767).astype(np.int16)
    return pcm.tobytes()


def generate_silence(duration_ms: int = 20, sample_rate: int = 8000) -> bytes:
    """Generate silence as PCM bytes."""
    num_samples = int(sample_rate * duration_ms / 1000)
    return b"\x00" * (num_samples * 2)


async def health_check(host: str, port: int) -> None:
    """Check health and readiness endpoints."""
    if httpx is None:
        sys.exit("httpx is required: pip install httpx")

    base_url = f"http://{host}:{port}"

    async with httpx.AsyncClient() as client:
        # Health check
        try:
            resp = await client.get(f"{base_url}/health", timeout=5.0)
            status = "OK" if resp.status_code == 200 else "FAIL"
            sys.stdout.write(f"[HEALTH] {status} ({resp.status_code}): {resp.json()}\n")
        except Exception as exc:
            sys.stdout.write(f"[HEALTH] FAIL: {exc}\n")

        # Readiness check
        try:
            resp = await client.get(f"{base_url}/ready", timeout=5.0)
            status = "OK" if resp.status_code == 200 else "NOT READY"
            sys.stdout.write(f"[READY]  {status} ({resp.status_code}): {json.dumps(resp.json(), indent=2)}\n")
        except Exception as exc:
            sys.stdout.write(f"[READY]  FAIL: {exc}\n")

        # Metrics check
        try:
            resp = await client.get(f"{base_url}/metrics", timeout=5.0)
            lines = resp.text.strip().split("\n")
            sys.stdout.write(f"[METRICS] OK - {len(lines)} metric lines\n")
            # Show key metrics
            for line in lines:
                if any(k in line for k in [
                    "calls_active", "calls_total", "gpu_memory",
                    "asr_latency", "llm_latency", "tts_latency",
                ]):
                    if not line.startswith("#"):
                        sys.stdout.write(f"  {line}\n")
        except Exception as exc:
            sys.stdout.write(f"[METRICS] FAIL: {exc}\n")


async def debug_websocket_monitor(host: str, port: int, duration: int = 30) -> None:
    """Connect to the debug WebSocket and monitor events."""
    if websockets is None:
        sys.exit("websockets is required: pip install websockets")

    uri = f"ws://{host}:{port}/ws/debug?token=debug"
    sys.stdout.write(f"Connecting to {uri}...\n")

    try:
        async with websockets.connect(uri) as ws:  # type: ignore[attr-defined]
            sys.stdout.write("Connected. Monitoring events...\n\n")
            deadline = time.monotonic() + duration

            while time.monotonic() < deadline:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    data = json.loads(msg)
                    sys.stdout.write(f"[EVENT] {json.dumps(data, indent=2)}\n")
                except asyncio.TimeoutError:
                    sys.stdout.write(".")
                    sys.stdout.flush()
                except Exception as exc:
                    sys.stdout.write(f"\n[ERROR] {exc}\n")
                    break

    except Exception as exc:
        sys.stdout.write(f"Connection failed: {exc}\n")


async def simulate_rtp_call(
    host: str,
    rtp_port: int,
    speech_duration_seconds: float = 3.0,
    silence_after_seconds: float = 1.0,
) -> None:
    """
    Simulate an RTP audio stream to the voice assistant.

    Sends synthetic speech audio followed by silence to trigger
    the ASR -> LLM -> TTS pipeline.
    """
    sys.stdout.write(f"Simulating RTP to {host}:{rtp_port}\n")

    loop = asyncio.get_event_loop()

    # Create UDP socket
    transport, _ = await loop.create_datagram_endpoint(
        asyncio.DatagramProtocol,
        remote_addr=(host, rtp_port),
    )

    seq = 0
    ts = 0
    ssrc = 0x12345678
    packet_duration_ms = 20
    samples_per_packet = 160  # 8kHz * 20ms

    # Send speech audio
    speech_packets = int(speech_duration_seconds * 1000 / packet_duration_ms)
    sys.stdout.write(f"Sending {speech_packets} speech packets ({speech_duration_seconds}s)...\n")

    for i in range(speech_packets):
        # Generate audio (varying frequency to simulate speech)
        freq = 300 + (i % 50) * 10
        audio = generate_sine_wave(freq, packet_duration_ms)

        # Build RTP packet
        byte0 = (2 << 6)  # V=2
        byte1 = 0  # PT=0 (PCMU)
        header = struct.pack("!BBHII", byte0, byte1, seq & 0xFFFF, ts & 0xFFFFFFFF, ssrc)

        # For simplicity, send PCM as-is (real impl would encode to ulaw)
        transport.sendto(header + audio[:samples_per_packet])

        seq += 1
        ts += samples_per_packet
        await asyncio.sleep(packet_duration_ms / 1000)

    # Send silence
    silence_packets = int(silence_after_seconds * 1000 / packet_duration_ms)
    sys.stdout.write(f"Sending {silence_packets} silence packets ({silence_after_seconds}s)...\n")

    for _ in range(silence_packets):
        silence = generate_silence(packet_duration_ms)
        byte0 = (2 << 6)
        byte1 = 0
        header = struct.pack("!BBHII", byte0, byte1, seq & 0xFFFF, ts & 0xFFFFFFFF, ssrc)
        transport.sendto(header + silence[:samples_per_packet])

        seq += 1
        ts += samples_per_packet
        await asyncio.sleep(packet_duration_ms / 1000)

    transport.close()
    sys.stdout.write("RTP simulation complete.\n")


async def load_test(
    host: str,
    port: int,
    concurrency: int = 10,
    duration_seconds: float = 30.0,
) -> None:
    """
    Simulate multiple concurrent health check requests.

    This is a simple load test that hammers the health endpoint
    to verify the server handles concurrent connections.
    """
    if httpx is None:
        sys.exit("httpx is required: pip install httpx")

    base_url = f"http://{host}:{port}"
    results: dict[str, int] = {"success": 0, "error": 0}
    latencies: list[float] = []

    async def worker(worker_id: int) -> None:
        async with httpx.AsyncClient() as client:
            deadline = time.monotonic() + duration_seconds
            while time.monotonic() < deadline:
                start = time.monotonic()
                try:
                    resp = await client.get(f"{base_url}/health", timeout=5.0)
                    if resp.status_code == 200:
                        results["success"] += 1
                    else:
                        results["error"] += 1
                except Exception:
                    results["error"] += 1
                latencies.append(time.monotonic() - start)
                await asyncio.sleep(0.01)  # Small delay

    sys.stdout.write(
        f"Load test: {concurrency} workers for {duration_seconds}s against {base_url}\n"
    )

    tasks = [asyncio.create_task(worker(i)) for i in range(concurrency)]
    await asyncio.gather(*tasks)

    total = results["success"] + results["error"]
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    p95_lat = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0

    sys.stdout.write(f"\nResults:\n")
    sys.stdout.write(f"  Total requests:  {total}\n")
    sys.stdout.write(f"  Success:         {results['success']}\n")
    sys.stdout.write(f"  Errors:          {results['error']}\n")
    sys.stdout.write(f"  RPS:             {total / duration_seconds:.1f}\n")
    sys.stdout.write(f"  Avg latency:     {avg_lat * 1000:.1f}ms\n")
    sys.stdout.write(f"  P95 latency:     {p95_lat * 1000:.1f}ms\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice Assistant Call Simulator")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server HTTP port")
    parser.add_argument("--rtp-port", type=int, default=10000, help="RTP port for audio sim")
    parser.add_argument("--health-check", action="store_true", help="Run health checks")
    parser.add_argument("--monitor", action="store_true", help="Monitor debug WebSocket")
    parser.add_argument("--monitor-duration", type=int, default=30, help="Monitor duration (s)")
    parser.add_argument("--simulate-rtp", action="store_true", help="Simulate RTP audio")
    parser.add_argument("--speech-duration", type=float, default=3.0, help="Speech duration (s)")
    parser.add_argument("--load-test", action="store_true", help="Run load test")
    parser.add_argument("--concurrency", type=int, default=10, help="Load test concurrency")
    parser.add_argument("--duration", type=float, default=30.0, help="Load test duration (s)")

    args = parser.parse_args()

    if args.health_check:
        asyncio.run(health_check(args.host, args.port))
    elif args.monitor:
        asyncio.run(debug_websocket_monitor(args.host, args.port, args.monitor_duration))
    elif args.simulate_rtp:
        asyncio.run(simulate_rtp_call(args.host, args.rtp_port, args.speech_duration))
    elif args.load_test:
        asyncio.run(load_test(args.host, args.port, args.concurrency, args.duration))
    else:
        # Default: run health check
        asyncio.run(health_check(args.host, args.port))


if __name__ == "__main__":
    main()
