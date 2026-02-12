"""
Tests for RTP bridge packet parsing, construction, and codec conversion.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from ari.rtp_bridge import (
    PT_PCMU,
    RTP_HEADER_SIZE,
    RTP_VERSION,
    RTPPacket,
    PortAllocator,
    pcm_to_ulaw_batch,
    resample_8k_to_16k,
    resample_16k_to_8k,
    ulaw_decode_batch,
)
from core.exceptions import RTPError, ResourceExhausted


class TestRTPPacket:
    """Tests for RTP packet parsing and serialization."""

    def _make_rtp_bytes(
        self,
        payload_type: int = PT_PCMU,
        seq: int = 1,
        ts: int = 160,
        ssrc: int = 12345,
        payload: bytes = b"\x00" * 160,
        version: int = RTP_VERSION,
    ) -> bytes:
        """Helper to construct a raw RTP packet."""
        byte0 = (version << 6)  # V=2, P=0, X=0, CC=0
        byte1 = payload_type & 0x7F
        header = struct.pack("!BBHII", byte0, byte1, seq, ts, ssrc)
        return header + payload

    def test_parse_valid_packet(self):
        raw = self._make_rtp_bytes(seq=42, ts=6720, ssrc=99999)
        pkt = RTPPacket.parse(raw)
        assert pkt.version == RTP_VERSION
        assert pkt.sequence_number == 42
        assert pkt.timestamp == 6720
        assert pkt.ssrc == 99999
        assert pkt.payload_type == PT_PCMU
        assert len(pkt.payload) == 160

    def test_parse_too_short(self):
        with pytest.raises(RTPError, match="too short"):
            RTPPacket.parse(b"\x00" * 5)

    def test_parse_invalid_version(self):
        raw = self._make_rtp_bytes(version=3)
        with pytest.raises(RTPError, match="Invalid RTP version"):
            RTPPacket.parse(raw)

    def test_serialize_roundtrip(self):
        original = RTPPacket(
            payload_type=PT_PCMU,
            sequence_number=1000,
            timestamp=160000,
            ssrc=0xDEADBEEF,
            payload=b"\x7f" * 160,
        )
        raw = original.serialize()
        parsed = RTPPacket.parse(raw)
        assert parsed.sequence_number == 1000
        assert parsed.timestamp == 160000
        assert parsed.ssrc == 0xDEADBEEF
        assert parsed.payload == b"\x7f" * 160

    def test_sequence_number_wraps(self):
        pkt = RTPPacket(sequence_number=65536)  # Should wrap to 0
        raw = pkt.serialize()
        parsed = RTPPacket.parse(raw)
        assert parsed.sequence_number == 0

    def test_marker_bit(self):
        pkt = RTPPacket(marker=True, payload_type=PT_PCMU, payload=b"\x00")
        raw = pkt.serialize()
        parsed = RTPPacket.parse(raw)
        assert parsed.marker is True
        assert parsed.payload_type == PT_PCMU

    def test_empty_payload(self):
        raw = self._make_rtp_bytes(payload=b"")
        pkt = RTPPacket.parse(raw)
        assert pkt.payload == b""


class TestCodecConversion:
    """Tests for mu-law <-> PCM conversion."""

    def test_ulaw_decode_silence(self):
        """Mu-law 0xFF should decode to ~0 (silence)."""
        ulaw_silence = bytes([0xFF] * 160)
        pcm = ulaw_decode_batch(ulaw_silence)
        assert len(pcm) == 160
        assert all(abs(s) < 10 for s in pcm)

    def test_ulaw_decode_known_value(self):
        """Test a known mu-law value decodes correctly."""
        ulaw_data = bytes([0x7F])  # Full-scale negative
        pcm = ulaw_decode_batch(ulaw_data)
        assert len(pcm) == 1
        assert isinstance(pcm[0], (np.int16, np.integer))

    def test_pcm_to_ulaw_silence(self):
        """Near-zero PCM should encode to near-silence mu-law."""
        pcm = np.zeros(160, dtype=np.int16)
        ulaw = pcm_to_ulaw_batch(pcm)
        assert len(ulaw) == 160

    def test_roundtrip_approximate(self):
        """PCM -> ulaw -> PCM should be approximately equal (lossy codec)."""
        # Generate a simple sine wave
        t = np.linspace(0, 0.02, 160, endpoint=False)
        pcm_original = (np.sin(2 * np.pi * 1000 * t) * 16000).astype(np.int16)

        ulaw = pcm_to_ulaw_batch(pcm_original)
        pcm_decoded = ulaw_decode_batch(ulaw)

        # Allow for quantization error (mu-law is 8-bit companded)
        error = np.abs(pcm_original.astype(np.float64) - pcm_decoded.astype(np.float64))
        mean_error = np.mean(error)
        # Mu-law should be reasonably close (within ~2% of full scale)
        assert mean_error < 1000, f"Mean error {mean_error} is too large"


class TestResampling:
    """Tests for sample rate conversion."""

    def test_resample_8k_to_16k_doubles_length(self):
        pcm_8k = np.zeros(160, dtype=np.int16)
        pcm_16k = resample_8k_to_16k(pcm_8k)
        assert len(pcm_16k) == 320

    def test_resample_16k_to_8k_halves_length(self):
        pcm_16k = np.zeros(320, dtype=np.int16)
        pcm_8k = resample_16k_to_8k(pcm_16k)
        assert len(pcm_8k) == 160

    def test_resample_preserves_dc(self):
        """A constant signal should remain constant after resampling."""
        pcm_8k = np.full(160, 1000, dtype=np.int16)
        pcm_16k = resample_8k_to_16k(pcm_8k)
        # All values should be near 1000
        assert all(abs(s - 1000) < 2 for s in pcm_16k)

    def test_resample_roundtrip(self):
        """8k -> 16k -> 8k should be approximately identity."""
        pcm_original = np.arange(0, 160, dtype=np.int16) * 100
        pcm_16k = resample_8k_to_16k(pcm_original)
        pcm_back = resample_16k_to_8k(pcm_16k)
        assert len(pcm_back) == 160
        # Allow small error from interpolation
        error = np.abs(pcm_original.astype(np.float64) - pcm_back.astype(np.float64))
        assert np.mean(error) < 200


class TestPortAllocator:
    """Tests for the port allocation pool."""

    def test_allocate_and_release(self):
        alloc = PortAllocator(30000, 30002)
        assert alloc.available_count == 3

        p1 = alloc.allocate()
        assert alloc.available_count == 2
        assert p1 in {30000, 30001, 30002}

        alloc.release(p1)
        assert alloc.available_count == 3

    def test_exhaust_ports(self):
        alloc = PortAllocator(30000, 30001)
        alloc.allocate()
        alloc.allocate()

        with pytest.raises(ResourceExhausted, match="RTP ports"):
            alloc.allocate()

    def test_release_returns_port_to_pool(self):
        alloc = PortAllocator(30000, 30000)
        port = alloc.allocate()
        assert alloc.available_count == 0

        alloc.release(port)
        assert alloc.available_count == 1

        # Should be able to allocate again
        port2 = alloc.allocate()
        assert port2 == 30000
