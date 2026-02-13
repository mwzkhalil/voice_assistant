"""
 configuration management via Pydantic BaseSettings.

All configuration is sourced from environment variables with validated defaults.
No hardcoded credentials. Supports .env file loading.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class TransportMode(str, Enum):
    """Transport mode selection."""
    WS = "WS"       # WebSocket (public realtime, default)
    ARI = "ARI"     # Asterisk ARI (telephony)


class ARISettings(BaseSettings):
    """Asterisk ARI connection settings."""

    model_config = SettingsConfigDict(env_prefix="ARI_")

    host: str = Field(default="localhost", description="Asterisk ARI host")
    port: int = Field(default=8088, ge=1, le=65535, description="Asterisk ARI HTTP port")
    ws_port: int = Field(default=8088, ge=1, le=65535, description="Asterisk ARI WebSocket port")
    username: str = Field(default="asterisk", description="ARI username")
    password: str = Field(default="asterisk", description="ARI password")
    app_name: str = Field(default="voice-assistant", description="Stasis application name")
    secure: bool = Field(default=False, description="Use WSS/HTTPS")
    reconnect_interval: float = Field(default=1.0, ge=0.1, description="Base reconnect interval seconds")
    reconnect_max_interval: float = Field(default=30.0, ge=1.0, description="Max reconnect interval seconds")
    reconnect_max_attempts: int = Field(default=0, ge=0, description="Max reconnect attempts, 0=unlimited")
    command_timeout: float = Field(default=5.0, ge=0.5, description="ARI command timeout seconds")

    @property
    def ws_url(self) -> str:
        scheme = "wss" if self.secure else "ws"
        return f"{scheme}://{self.host}:{self.ws_port}/ari/events?app={self.app_name}&subscribeAll=true"

    @property
    def base_url(self) -> str:
        scheme = "https" if self.secure else "http"
        return f"{scheme}://{self.host}:{self.port}/ari"


class RedisSettings(BaseSettings):
    """Redis connection settings."""

    model_config = SettingsConfigDict(env_prefix="REDIS_")

    url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    pool_min_size: int = Field(default=5, ge=1, description="Minimum connection pool size")
    pool_max_size: int = Field(default=20, ge=1, description="Maximum connection pool size")
    socket_timeout: float = Field(default=5.0, ge=0.1, description="Socket timeout seconds")
    socket_connect_timeout: float = Field(default=5.0, ge=0.1, description="Connection timeout seconds")
    retry_on_timeout: bool = Field(default=True, description="Retry on timeout")
    health_check_interval: int = Field(default=30, ge=0, description="Health check interval seconds")
    session_ttl: int = Field(default=3600, ge=60, description="Session TTL in seconds")
    lock_timeout: float = Field(default=10.0, ge=1.0, description="Distributed lock timeout seconds")


class OllamaSettings(BaseSettings):
    """Ollama / LLM settings (OpenAI-compatible endpoint)."""

    model_config = SettingsConfigDict(env_prefix="OLLAMA_")

    base_url: str = Field(default="http://localhost:11434/v1", description="Ollama OpenAI-compatible base URL")
    model_name: str = Field(default="mahwiz/zoma", description="LLM model name")
    api_key: str = Field(default="ollama", description="API key (required by client, ignored by Ollama)")
    max_tokens: int = Field(default=512, ge=1, le=4096, description="Maximum tokens per response")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    request_timeout: float = Field(default=30.0, ge=1.0, description="Per-request timeout seconds")
    system_prompt: str = Field(
        default=(
            "آپ ایک مددگار اردو زبان کا اسسٹنٹ ہیں۔ "
            "آپ صارفین کے سوالات کا جواب اردو میں دیتے ہیں۔ "
            "جواب مختصر اور واضح رکھیں۔"
        ),
        description="System prompt always injected into LLM context",
    )
    circuit_breaker_threshold: int = Field(default=5, ge=1, description="Failures before circuit opens")
    circuit_breaker_recovery_timeout: float = Field(default=30.0, ge=5.0, description="Circuit breaker recovery seconds")
    retry_max_attempts: int = Field(default=3, ge=1, description="Max retry attempts")
    retry_base_delay: float = Field(default=0.5, ge=0.1, description="Base retry delay seconds")


class ASRSettings(BaseSettings):
    """Automatic Speech Recognition settings."""

    model_config = SettingsConfigDict(env_prefix="ASR_")

    model_name: str = Field(
        default="sajadkawa/ns_finetune_urdu_asr_org",
        description="HuggingFace ASR model identifier",
    )
    device: str = Field(default="auto", description="Device: auto, cuda, cpu")
    sample_rate: int = Field(default=16000, ge=8000, le=48000, description="Model input sample rate")
    chunk_duration_ms: int = Field(default=2000, ge=500, le=10000, description="Processing chunk duration ms")
    chunk_overlap_ms: int = Field(default=200, ge=0, le=2000, description="Chunk overlap ms for context")
    max_audio_buffer_seconds: float = Field(default=30.0, ge=5.0, description="Max audio buffer length seconds")
    inference_timeout: float = Field(default=10.0, ge=1.0, description="Inference timeout seconds")
    batch_size: int = Field(default=1, ge=1, description="Inference batch size")

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        allowed = {"auto", "cuda", "cpu", "cuda:0", "cuda:1"}
        if v not in allowed:
            raise ValueError(f"device must be one of {allowed}")
        return v


class TTSSettings(BaseSettings):
    """Text-to-Speech settings."""

    model_config = SettingsConfigDict(env_prefix="TTS_")

    model_name: str = Field(
        default="suhaibrashid17/MMS_TTS_Urdu_3",
        description="HuggingFace TTS model identifier",
    )
    device: str = Field(default="auto", description="Device: auto, cuda, cpu")
    output_sample_rate: int = Field(default=8000, ge=8000, le=48000, description="Output sample rate for RTP")
    max_text_length: int = Field(default=500, ge=50, description="Maximum text length per synthesis call")
    inference_timeout: float = Field(default=15.0, ge=1.0, description="Inference timeout seconds")
    sentence_delimiters: str = Field(
        default="۔؟!،.?!,;",
        description="Characters that delimit sentence boundaries for chunked synthesis",
    )

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        allowed = {"auto", "cuda", "cpu", "cuda:0", "cuda:1"}
        if v not in allowed:
            raise ValueError(f"device must be one of {allowed}")
        return v


class VADSettings(BaseSettings):
    """Voice Activity Detection settings."""

    model_config = SettingsConfigDict(env_prefix="VAD_")

    aggressiveness: int = Field(default=2, ge=0, le=3, description="WebRTC VAD aggressiveness 0-3")
    frame_duration_ms: int = Field(default=30, description="VAD frame duration ms (10, 20, or 30)")
    silence_threshold_ms: int = Field(default=800, ge=200, le=3000, description="Silence duration to end utterance")
    speech_threshold_ms: int = Field(default=250, ge=50, le=1000, description="Min speech duration to trigger")
    sample_rate: int = Field(default=16000, description="VAD input sample rate")

    @field_validator("frame_duration_ms")
    @classmethod
    def validate_frame_duration(cls, v: int) -> int:
        if v not in (10, 20, 30):
            raise ValueError("frame_duration_ms must be 10, 20, or 30")
        return v


class RTPSettings(BaseSettings):
    """RTP bridge settings."""

    model_config = SettingsConfigDict(env_prefix="RTP_")

    listen_host: str = Field(default="0.0.0.0", description="RTP listen host")
    port_range_start: int = Field(default=10000, ge=1024, le=65000, description="RTP port range start")
    port_range_end: int = Field(default=20000, ge=1025, le=65535, description="RTP port range end")
    jitter_buffer_ms: int = Field(default=60, ge=0, le=500, description="Jitter buffer size ms")
    packet_duration_ms: int = Field(default=20, ge=10, le=40, description="RTP packet duration ms")
    codec: str = Field(default="ulaw", description="RTP codec: ulaw or slin16")

    @field_validator("codec")
    @classmethod
    def validate_codec(cls, v: str) -> str:
        if v not in ("ulaw", "slin16"):
            raise ValueError("codec must be 'ulaw' or 'slin16'")
        return v


class AuthSettings(BaseSettings):
    """API key authentication and rate limiting settings."""

    model_config = SettingsConfigDict(env_prefix="AUTH_")

    enabled: bool = Field(default=True, description="Enable API key authentication")
    api_keys: str = Field(
        default="",
        description="Comma-separated valid API keys (empty = auth disabled)",
    )
    header_name: str = Field(default="X-API-Key", description="API key header name")
    rate_limit_per_minute: int = Field(default=60, ge=1, description="Requests per IP per minute")
    max_sessions_per_ip: int = Field(default=5, ge=1, description="Max concurrent sessions per IP")
    max_audio_duration_seconds: float = Field(default=600.0, ge=30.0, description="Max audio duration per session")
    max_chunk_size_bytes: int = Field(default=32768, ge=1024, description="Max audio chunk size in bytes")
    idle_session_timeout: float = Field(default=300.0, ge=30.0, description="Idle session timeout seconds")

    def get_api_keys(self) -> set[str]:
        """Parse comma-separated API keys into a set."""
        if not self.api_keys:
            return set()
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


class ServerSettings(BaseSettings):
    """FastAPI / Uvicorn server settings."""

    model_config = SettingsConfigDict(env_prefix="SERVER_")

    host: str = Field(default="0.0.0.0", description="Server bind host")
    port: int = Field(default=8080, ge=1, le=65535, description="Server bind port")
    workers: int = Field(default=1, ge=1, description="Uvicorn workers (1 for GPU)")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Application log level")
    debug: bool = Field(default=False, description="Enable debug mode")
    cors_origins: str = Field(default="*", description="Comma-separated CORS origins")
    max_concurrent_calls: int = Field(default=200, ge=1, description="Maximum concurrent calls/sessions")
    transport_mode: TransportMode = Field(default=TransportMode.WS, description="Transport mode: WS or ARI")


class AppSettings(BaseSettings):
    """Root application settings aggregating all sub-configurations."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    ari: ARISettings = Field(default_factory=ARISettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    asr: ASRSettings = Field(default_factory=ASRSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    vad: VADSettings = Field(default_factory=VADSettings)
    rtp: RTPSettings = Field(default_factory=RTPSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)


_settings: Optional[AppSettings] = None


def get_settings() -> AppSettings:
    """Return the singleton application settings instance."""
    global _settings
    if _settings is None:
        _settings = AppSettings()
    return _settings


def reset_settings() -> None:
    """Reset settings singleton (for testing)."""
    global _settings
    _settings = None
