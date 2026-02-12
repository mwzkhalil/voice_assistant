"""
RTP bridge for per-call audio I/O.

Manages UDP sockets with asyncio DatagramProtocol for receiving
and sending RTP packets. Handles codec conversion between
telephony (G.711 ulaw 8kHz) and model input (PCM 16kHz).
Includes port allocation pool and jitter buffer.
"""

from __future__ import annotations

import asyncio
import struct
import threading
import time
from typing import Any, Optional

import numpy as np

from core.config import RTPSettings
from core.exceptions import RTPError, ResourceExhausted
from core.logger import get_logger
from core.metrics import (
    RTP_PACKETS_DROPPED,
    RTP_PACKETS_RECEIVED,
    RTP_PACKETS_SENT,
    MetricsCollector,
)

logger = get_logger(__name__)

# RTP Header: V=2, P=0, X=0, CC=0, M, PT, SeqNum, Timestamp, SSRC = 12 bytes
RTP_HEADER_SIZE = 12
RTP_VERSION = 2
PT_PCMU = 0  # G.711 mu-law payload type
PT_L16 = 11  # Linear 16-bit payload type

# G.711 mu-law encode/decode tables (ITU-T G.711)
_ULAW_BIAS = 0x84
_ULAW_CLIP = 32635

_ULAW_DECODE_TABLE = [
    -32124, -31100, -30076, -29052, -28028, -27004, -25980, -24956,
    -23932, -22908, -21884, -20860, -19836, -18812, -17788, -16764,
    -15996, -15484, -14972, -14460, -13948, -13436, -12924, -12412,
    -11900, -11388, -10876, -10364, -9852, -9340, -8828, -8316,
    -7932, -7676, -7420, -7164, -6908, -6652, -6396, -6140,
    -5884, -5628, -5372, -5116, -4860, -4604, -4348, -4092,
    -3900, -3772, -3644, -3516, -3388, -3260, -3132, -3004,
    -2876, -2748, -2620, -2492, -2364, -2236, -2108, -1980,
    -1884, -1820, -1756, -1692, -1628, -1564, -1500, -1436,
    -1372, -1308, -1244, -1180, -1116, -1052, -988, -924,
    -876, -844, -812, -780, -748, -716, -684, -652,
    -620, -588, -556, -524, -492, -460, -428, -396,
    -372, -356, -340, -324, -308, -292, -276, -260,
    -244, -228, -212, -196, -180, -164, -148, -132,
    -120, -112, -104, -96, -88, -80, -72, -64,
    -56, -48, -40, -32, -24, -16, -8, 0,
    32124, 31100, 30076, 29052, 28028, 27004, 25980, 24956,
    23932, 22908, 21884, 20860, 19836, 18812, 17788, 16764,
    15996, 15484, 14972, 14460, 13948, 13436, 12924, 12412,
    11900, 11388, 10876, 10364, 9852, 9340, 8828, 8316,
    7932, 7676, 7420, 7164, 6908, 6652, 6396, 6140,
    5884, 5628, 5372, 5116, 4860, 4604, 4348, 4092,
    3900, 3772, 3644, 3516, 3388, 3260, 3132, 3004,
    2876, 2748, 2620, 2492, 2364, 2236, 2108, 1980,
    1884, 1820, 1756, 1692, 1628, 1564, 1500, 1436,
    1372, 1308, 1244, 1180, 1116, 1052, 988, 924,
    876, 844, 812, 780, 748, 716, 684, 652,
    620, 588, 556, 524, 492, 460, 428, 396,
    372, 356, 340, 324, 308, 292, 276, 260,
    244, 228, 212, 196, 180, 164, 148, 132,
    120, 112, 104, 96, 88, 80, 72, 64,
    56, 48, 40, 32, 24, 16, 8, 0,
]

# Pre-computed numpy decode array for vectorized conversion
_ULAW_DECODE_ARRAY = np.array(_ULAW_DECODE_TABLE, dtype=np.int16)


def ulaw_decode_batch(ulaw_data: bytes) -> np.ndarray:
    """Vectorized mu-law to PCM 16-bit conversion."""
    indices = np.frombuffer(ulaw_data, dtype=np.uint8)
    return _ULAW_DECODE_ARRAY[indices]


def pcm_to_ulaw_batch(pcm: np.ndarray) -> bytes:
    """Vectorized PCM 16-bit to mu-law conversion."""
    # Ensure int16
    if pcm.dtype != np.int16:
        pcm = np.clip(pcm, -32768, 32767).astype(np.int16)

    sign = (pcm >> 8) & 0x80
    magnitude = np.where(sign != 0, -pcm, pcm).astype(np.int32)
    magnitude = np.clip(magnitude, 0, _ULAW_CLIP)
    magnitude = magnitude + _ULAW_BIAS

    # Find exponent (number of leading zeros approach via lookup)
    exponent = np.zeros(len(magnitude), dtype=np.uint8)
    shifted = (magnitude >> 7).astype(np.uint8)
    for exp_val in range(7, -1, -1):
        mask = shifted >= (1 << exp_val)
        exponent = np.where(mask & (exponent == 0), exp_val, exponent)
        # Fix: we need the highest bit set
    # Simpler approach using bit operations
    exponent = np.zeros(len(magnitude), dtype=np.int32)
    temp = magnitude >> 7
    for i in range(7, 0, -1):
        exponent = np.where((temp >= (1 << i)) & (exponent == 0), i, exponent)
    exponent = np.where((exponent == 0) & (temp >= 1), 0, exponent).astype(np.uint8)

    mantissa = ((magnitude >> (exponent.astype(np.int32) + 3)) & 0x0F).astype(np.uint8)
    ulaw_byte = ~(sign.astype(np.uint8) | (exponent << 4) | mantissa) & 0xFF

    return ulaw_byte.astype(np.uint8).tobytes()


def resample_8k_to_16k(pcm_8k: np.ndarray) -> np.ndarray:
    """Simple 2x upsampling from 8kHz to 16kHz using linear interpolation."""
    length = len(pcm_8k)
    indices = np.linspace(0, length - 1, length * 2)
    return np.interp(indices, np.arange(length), pcm_8k.astype(np.float64)).astype(np.int16)


def resample_16k_to_8k(pcm_16k: np.ndarray) -> np.ndarray:
    """Simple 2x downsampling from 16kHz to 8kHz using decimation with averaging."""
    length = len(pcm_16k)
    if length % 2 != 0:
        pcm_16k = np.append(pcm_16k, [0])
    return ((pcm_16k[::2].astype(np.int32) + pcm_16k[1::2].astype(np.int32)) // 2).astype(np.int16)


class RTPPacket:
    """Parsed RTP packet."""

    __slots__ = (
        "version", "padding", "extension", "csrc_count", "marker",
        "payload_type", "sequence_number", "timestamp", "ssrc", "payload",
    )

    def __init__(
        self,
        version: int = RTP_VERSION,
        padding: bool = False,
        extension: bool = False,
        csrc_count: int = 0,
        marker: bool = False,
        payload_type: int = PT_PCMU,
        sequence_number: int = 0,
        timestamp: int = 0,
        ssrc: int = 0,
        payload: bytes = b"",
    ) -> None:
        self.version = version
        self.padding = padding
        self.extension = extension
        self.csrc_count = csrc_count
        self.marker = marker
        self.payload_type = payload_type
        self.sequence_number = sequence_number
        self.timestamp = timestamp
        self.ssrc = ssrc
        self.payload = payload

    @classmethod
    def parse(cls, data: bytes) -> RTPPacket:
        """Parse raw UDP data into an RTPPacket."""
        if len(data) < RTP_HEADER_SIZE:
            raise RTPError("RTP packet too short", length=len(data))

        byte0, byte1, seq, ts, ssrc = struct.unpack("!BBHII", data[:RTP_HEADER_SIZE])

        version = (byte0 >> 6) & 0x03
        if version != RTP_VERSION:
            raise RTPError("Invalid RTP version", version=version)

        padding = bool((byte0 >> 5) & 0x01)
        extension = bool((byte0 >> 4) & 0x01)
        csrc_count = byte0 & 0x0F
        marker = bool((byte1 >> 7) & 0x01)
        payload_type = byte1 & 0x7F

        header_size = RTP_HEADER_SIZE + (csrc_count * 4)
        if len(data) < header_size:
            raise RTPError("RTP packet truncated", expected=header_size, got=len(data))

        payload = data[header_size:]

        # Handle extension header
        if extension and len(payload) >= 4:
            ext_length = struct.unpack("!HH", payload[:4])[1]
            ext_total = 4 + ext_length * 4
            if len(payload) >= ext_total:
                payload = payload[ext_total:]

        # Handle padding
        if padding and len(payload) > 0:
            pad_length = payload[-1]
            payload = payload[:-pad_length]

        return cls(
            version=version,
            padding=padding,
            extension=extension,
            csrc_count=csrc_count,
            marker=marker,
            payload_type=payload_type,
            sequence_number=seq,
            timestamp=ts,
            ssrc=ssrc,
            payload=payload,
        )

    def serialize(self) -> bytes:
        """Serialize RTPPacket to bytes for UDP transmission."""
        byte0 = (
            (self.version << 6)
            | (int(self.padding) << 5)
            | (int(self.extension) << 4)
            | (self.csrc_count & 0x0F)
        )
        byte1 = (int(self.marker) << 7) | (self.payload_type & 0x7F)

        header = struct.pack(
            "!BBHII",
            byte0,
            byte1,
            self.sequence_number & 0xFFFF,
            self.timestamp & 0xFFFFFFFF,
            self.ssrc & 0xFFFFFFFF,
        )
        return header + self.payload


class PortAllocator:
    """
    Thread-safe port allocator for RTP sockets.

    Manages a pool of available UDP ports from a configured range.
    """

    def __init__(self, start: int, end: int) -> None:
        self._available: set[int] = set(range(start, end + 1))
        self._in_use: set[int] = set()
        self._lock = threading.Lock()

        logger.info(
            "port_allocator_initialized",
            range_start=start,
            range_end=end,
            total_ports=len(self._available),
        )

    def allocate(self) -> int:
        """Allocate a free port. Raises ResourceExhausted if none available."""
        with self._lock:
            if not self._available:
                raise ResourceExhausted("RTP ports", limit=len(self._in_use))
            port = self._available.pop()
            self._in_use.add(port)
            return port

    def release(self, port: int) -> None:
        """Return a port to the available pool."""
        with self._lock:
            self._in_use.discard(port)
            self._available.add(port)

    @property
    def available_count(self) -> int:
        with self._lock:
            return len(self._available)


class RTPProtocol(asyncio.DatagramProtocol):
    """Asyncio DatagramProtocol for RTP packet I/O."""

    def __init__(self, inbound_queue: asyncio.Queue[bytes]) -> None:
        self._inbound_queue = inbound_queue
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._remote_addr: Optional[tuple[str, int]] = None

    @property
    def transport(self) -> asyncio.DatagramTransport | None:
        return self._transport

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self._transport = transport  # type: ignore[assignment]

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        """Handle incoming RTP packet."""
        RTP_PACKETS_RECEIVED.inc()

        # Learn remote address from first packet
        if self._remote_addr is None:
            self._remote_addr = addr
            logger.info("rtp_remote_learned", host=addr[0], port=addr[1])

        try:
            self._inbound_queue.put_nowait(data)
        except asyncio.QueueFull:
            RTP_PACKETS_DROPPED.labels(reason="buffer_full").inc()

    def error_received(self, exc: Exception) -> None:
        logger.warning("rtp_protocol_error", error=str(exc))

    def connection_lost(self, exc: Optional[Exception]) -> None:
        if exc:
            logger.warning("rtp_connection_lost", error=str(exc))

    def send_to_remote(self, data: bytes) -> None:
        """Send RTP data to the learned remote address."""
        if self._transport and self._remote_addr:
            self._transport.sendto(data, self._remote_addr)
            RTP_PACKETS_SENT.inc()

    def set_remote_addr(self, host: str, port: int) -> None:
        """Explicitly set the remote RTP address."""
        self._remote_addr = (host, port)


class RTPBridge:
    """
    Per-call RTP bridge managing audio I/O over UDP.

    Handles:
    - UDP socket binding via asyncio DatagramProtocol
    - RTP packet parse/construct
    - Codec conversion (ulaw <-> PCM 16kHz)
    - Jitter buffering for inbound audio
    - Paced output for outbound audio
    """

    def __init__(
        self,
        call_id: str,
        local_port: int,
        settings: RTPSettings,
    ) -> None:
        self._call_id = call_id
        self._local_port = local_port
        self._settings = settings
        self._protocol: Optional[RTPProtocol] = None
        self._transport: Optional[asyncio.DatagramTransport] = None

        # Inbound: raw RTP packets
        self._raw_inbound: asyncio.Queue[bytes] = asyncio.Queue(maxsize=500)
        # Inbound: decoded PCM frames (16kHz, 16-bit)
        self._pcm_inbound: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)

        # Outbound state
        self._out_seq: int = 0
        self._out_ts: int = 0
        self._out_ssrc: int = hash(call_id) & 0xFFFFFFFF
        self._output_stopped = False

        # Decode task
        self._decode_task: Optional[asyncio.Task] = None
        self._cancel_event = asyncio.Event()

        logger.info(
            "rtp_bridge_created",
            call_id=call_id,
            local_port=local_port,
        )

    @property
    def local_port(self) -> int:
        return self._local_port

    async def start(self) -> None:
        """Bind UDP socket and start decode loop."""
        loop = asyncio.get_event_loop()

        transport, protocol = await loop.create_datagram_endpoint(
            lambda: RTPProtocol(self._raw_inbound),
            local_addr=(self._settings.listen_host, self._local_port),
        )

        self._transport = transport  # type: ignore[assignment]
        self._protocol = protocol  # type: ignore[assignment]

        # Start background decode loop
        self._decode_task = asyncio.create_task(
            self._decode_loop(), name=f"rtp_decode_{self._call_id}"
        )

        logger.info(
            "rtp_bridge_started",
            call_id=self._call_id,
            bind=f"{self._settings.listen_host}:{self._local_port}",
        )

    async def stop(self) -> None:
        """Stop the RTP bridge and release resources."""
        self._cancel_event.set()

        if self._decode_task:
            self._decode_task.cancel()
            try:
                await self._decode_task
            except asyncio.CancelledError:
                pass

        if self._transport:
            self._transport.close()

        logger.info("rtp_bridge_stopped", call_id=self._call_id)

    def set_remote(self, host: str, port: int) -> None:
        """Set the remote RTP endpoint address."""
        if self._protocol:
            self._protocol.set_remote_addr(host, port)
            logger.info(
                "rtp_remote_set",
                call_id=self._call_id,
                host=host,
                port=port,
            )

    def stop_output(self) -> None:
        """Immediately stop audio output (for barge-in)."""
        self._output_stopped = True
        # Flush output-related inbound PCM
        while not self._pcm_inbound.empty():
            try:
                self._pcm_inbound.get_nowait()
            except asyncio.QueueEmpty:
                break

    def resume_output(self) -> None:
        """Resume audio output after barge-in reset."""
        self._output_stopped = False

    async def _decode_loop(self) -> None:
        """
        Background task: decode raw RTP packets to PCM frames.

        Parses RTP headers, extracts ulaw payload, converts to
        PCM 16kHz 16-bit, and queues for consumption.
        """
        while not self._cancel_event.is_set():
            try:
                raw = await asyncio.wait_for(
                    self._raw_inbound.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            try:
                packet = RTPPacket.parse(raw)

                # Decode ulaw payload to PCM 16-bit at 8kHz
                pcm_8k = ulaw_decode_batch(packet.payload)

                # Resample 8kHz -> 16kHz for models
                pcm_16k = resample_8k_to_16k(pcm_8k)

                # Convert to bytes and queue
                pcm_bytes = pcm_16k.astype(np.int16).tobytes()

                try:
                    self._pcm_inbound.put_nowait(pcm_bytes)
                except asyncio.QueueFull:
                    RTP_PACKETS_DROPPED.labels(reason="decode_buffer_full").inc()

            except RTPError:
                RTP_PACKETS_DROPPED.labels(reason="decode_error").inc()
            except Exception:
                RTP_PACKETS_DROPPED.labels(reason="decode_error").inc()
                logger.exception("rtp_decode_error", call_id=self._call_id)

    async def read_audio(self, frame_size: int) -> bytes | None:
        """
        Read decoded PCM audio from the inbound queue.

        Args:
            frame_size: Expected frame size in bytes.

        Returns:
            PCM 16-bit audio bytes, or None if bridge is stopping.
        """
        if self._cancel_event.is_set():
            return None

        try:
            pcm = await asyncio.wait_for(
                self._pcm_inbound.get(), timeout=2.0
            )
            # Pad or trim to expected frame size
            if len(pcm) < frame_size:
                pcm = pcm + b'\x00' * (frame_size - len(pcm))
            elif len(pcm) > frame_size:
                # Put remaining back
                remainder = pcm[frame_size:]
                pcm = pcm[:frame_size]
                try:
                    self._pcm_inbound.put_nowait(remainder)
                except asyncio.QueueFull:
                    pass
            return pcm
        except asyncio.TimeoutError:
            return None

    async def write_audio(self, pcm_data: bytes) -> None:
        """
        Send PCM audio out via RTP.

        Converts PCM 16kHz 16-bit to ulaw 8kHz and packetizes into
        RTP packets for transmission.

        Args:
            pcm_data: Raw PCM 16-bit audio bytes at 16kHz.
        """
        if self._output_stopped or self._protocol is None:
            return

        # Convert to numpy
        pcm_16k = np.frombuffer(pcm_data, dtype=np.int16)

        # Downsample 16kHz -> 8kHz
        pcm_8k = resample_16k_to_8k(pcm_16k)

        # Encode to ulaw
        ulaw_data = pcm_to_ulaw_batch(pcm_8k)

        # Packetize into RTP (160 samples per packet at 8kHz = 20ms)
        samples_per_packet = self._settings.packet_duration_ms * 8  # 8 samples/ms at 8kHz
        offset = 0

        while offset < len(ulaw_data):
            chunk = ulaw_data[offset:offset + samples_per_packet]
            if len(chunk) == 0:
                break

            # Pad short final packet
            if len(chunk) < samples_per_packet:
                chunk = chunk + b'\x7f' * (samples_per_packet - len(chunk))  # ulaw silence

            packet = RTPPacket(
                payload_type=PT_PCMU,
                sequence_number=self._out_seq,
                timestamp=self._out_ts,
                ssrc=self._out_ssrc,
                payload=chunk,
            )

            self._protocol.send_to_remote(packet.serialize())

            self._out_seq = (self._out_seq + 1) & 0xFFFF
            self._out_ts = (self._out_ts + samples_per_packet) & 0xFFFFFFFF
            offset += samples_per_packet

            # Pace output to match real-time (20ms per packet)
            await asyncio.sleep(self._settings.packet_duration_ms / 1000.0)


# Module-level port allocator singleton
_port_allocator: Optional[PortAllocator] = None


def get_port_allocator(settings: RTPSettings) -> PortAllocator:
    """Get or create the singleton port allocator."""
    global _port_allocator
    if _port_allocator is None:
        _port_allocator = PortAllocator(settings.port_range_start, settings.port_range_end)
    return _port_allocator
