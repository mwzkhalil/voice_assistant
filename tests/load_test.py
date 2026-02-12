"""
Concurrent call load simulation.

Simulates multiple concurrent call sessions hitting the voice assistant
to test scalability and resource handling.

Usage:
    python -m tests.load_test --target localhost --port 8080 --calls 50 --duration 60
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import sys
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CallStats:
    """Statistics for a single simulated call."""
    call_id: int
    start_time: float = 0.0
    end_time: float = 0.0
    packets_sent: int = 0
    errors: int = 0
    success: bool = False


@dataclass
class LoadTestResults:
    """Aggregated load test results."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_packets_sent: int = 0
    total_errors: int = 0
    call_stats: list[CallStats] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    def summary(self) -> str:
        duration = self.end_time - self.start_time
        durations = [
            s.end_time - s.start_time
            for s in self.call_stats
            if s.end_time > 0
        ]
        avg_duration = sum(durations) / len(durations) if durations else 0

        lines = [
            "=" * 60,
            "LOAD TEST RESULTS",
            "=" * 60,
            f"Duration:             {duration:.1f}s",
            f"Total calls:          {self.total_calls}",
            f"Successful:           {self.successful_calls}",
            f"Failed:               {self.failed_calls}",
            f"Total packets sent:   {self.total_packets_sent}",
            f"Total errors:         {self.total_errors}",
            f"Avg call duration:    {avg_duration:.1f}s",
            f"Calls per second:     {self.total_calls / duration:.2f}" if duration > 0 else "",
            "=" * 60,
        ]
        return "\n".join(lines)


async def simulate_single_call(
    call_id: int,
    target_host: str,
    rtp_port_base: int,
    speech_duration: float = 3.0,
    silence_duration: float = 1.0,
) -> CallStats:
    """
    Simulate a single call by sending RTP audio.

    Args:
        call_id: Unique call identifier for this simulation.
        target_host: Target host for RTP.
        rtp_port_base: Base RTP port (call_id is used as offset).
        speech_duration: Duration of speech audio to send.
        silence_duration: Duration of silence after speech.

    Returns:
        CallStats with results.
    """
    stats = CallStats(call_id=call_id)
    stats.start_time = time.monotonic()
    rtp_port = rtp_port_base + (call_id % 100)

    loop = asyncio.get_event_loop()

    try:
        transport, _ = await loop.create_datagram_endpoint(
            asyncio.DatagramProtocol,
            remote_addr=(target_host, rtp_port),
        )
    except Exception as exc:
        stats.errors += 1
        stats.end_time = time.monotonic()
        return stats

    seq = 0
    ts = 0
    ssrc = 0x10000000 + call_id
    packet_duration_ms = 20
    samples_per_packet = 160

    try:
        # Speech phase
        num_speech_packets = int(speech_duration * 1000 / packet_duration_ms)
        for i in range(num_speech_packets):
            freq = 300 + (i % 50) * 10
            t_arr = np.linspace(0, packet_duration_ms / 1000, samples_per_packet, endpoint=False)
            audio = (np.sin(2 * np.pi * freq * t_arr) * 0.3 * 32767).astype(np.int16)

            byte0 = (2 << 6)
            byte1 = 0
            header = struct.pack(
                "!BBHII", byte0, byte1,
                seq & 0xFFFF, ts & 0xFFFFFFFF, ssrc,
            )
            transport.sendto(header + audio.tobytes()[:samples_per_packet])
            stats.packets_sent += 1
            seq += 1
            ts += samples_per_packet
            await asyncio.sleep(packet_duration_ms / 1000)

        # Silence phase
        num_silence_packets = int(silence_duration * 1000 / packet_duration_ms)
        silence = b"\x00" * (samples_per_packet * 2)
        for _ in range(num_silence_packets):
            byte0 = (2 << 6)
            byte1 = 0
            header = struct.pack(
                "!BBHII", byte0, byte1,
                seq & 0xFFFF, ts & 0xFFFFFFFF, ssrc,
            )
            transport.sendto(header + silence[:samples_per_packet])
            stats.packets_sent += 1
            seq += 1
            ts += samples_per_packet
            await asyncio.sleep(packet_duration_ms / 1000)

        stats.success = True

    except Exception as exc:
        stats.errors += 1
    finally:
        transport.close()
        stats.end_time = time.monotonic()

    return stats


async def run_load_test(
    target_host: str,
    port: int,
    num_calls: int,
    call_duration: float,
    ramp_up_seconds: float = 5.0,
) -> LoadTestResults:
    """
    Execute a load test with specified concurrency.

    Args:
        target_host: Target host.
        port: Base RTP port.
        num_calls: Number of concurrent calls to simulate.
        call_duration: Duration of each call in seconds.
        ramp_up_seconds: Time to ramp up to full concurrency.

    Returns:
        LoadTestResults with aggregated statistics.
    """
    results = LoadTestResults()
    results.start_time = time.monotonic()
    results.total_calls = num_calls

    ramp_delay = ramp_up_seconds / num_calls if num_calls > 0 else 0

    tasks: list[asyncio.Task[CallStats]] = []
    for i in range(num_calls):
        task = asyncio.create_task(
            simulate_single_call(
                call_id=i,
                target_host=target_host,
                rtp_port_base=port,
                speech_duration=call_duration * 0.7,
                silence_duration=call_duration * 0.3,
            )
        )
        tasks.append(task)
        if ramp_delay > 0:
            await asyncio.sleep(ramp_delay)

    # Wait for all calls to complete
    call_results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in call_results:
        if isinstance(result, Exception):
            results.failed_calls += 1
            results.total_errors += 1
        elif isinstance(result, CallStats):
            results.call_stats.append(result)
            results.total_packets_sent += result.packets_sent
            results.total_errors += result.errors
            if result.success:
                results.successful_calls += 1
            else:
                results.failed_calls += 1

    results.end_time = time.monotonic()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice Assistant Load Tester")
    parser.add_argument("--target", default="localhost", help="Target host")
    parser.add_argument("--port", type=int, default=10000, help="Base RTP port")
    parser.add_argument("--calls", type=int, default=10, help="Number of concurrent calls")
    parser.add_argument("--duration", type=float, default=5.0, help="Call duration seconds")
    parser.add_argument("--ramp-up", type=float, default=2.0, help="Ramp up period seconds")

    args = parser.parse_args()

    sys.stdout.write(
        f"Starting load test: {args.calls} calls to {args.target}:{args.port}\n"
    )
    sys.stdout.write(f"Call duration: {args.duration}s, Ramp up: {args.ramp_up}s\n\n")

    results = asyncio.run(
        run_load_test(
            target_host=args.target,
            port=args.port,
            num_calls=args.calls,
            call_duration=args.duration,
            ramp_up_seconds=args.ramp_up,
        )
    )

    sys.stdout.write(results.summary() + "\n")


if __name__ == "__main__":
    main()
