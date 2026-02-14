"""Configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class ListenConfig:
    """Configuration for the audio-listen-mcp server."""

    whisper_engine: str
    whisper_model: str
    language: str
    audio_device: str | None
    sample_rate: int
    default_duration: int
    max_duration: int
    vad_silence_duration: float
    vad_silence_threshold: int

    @classmethod
    def from_env(cls) -> ListenConfig:
        """Create configuration from environment variables."""
        load_dotenv()
        return cls(
            whisper_engine=os.environ.get("WHISPER_ENGINE", "mlx"),
            whisper_model=os.environ.get("WHISPER_MODEL", "small"),
            language=os.environ.get("WHISPER_LANGUAGE", "ja"),
            audio_device=os.environ.get("AUDIO_DEVICE"),
            sample_rate=int(os.environ.get("AUDIO_SAMPLE_RATE", "16000")),
            default_duration=int(os.environ.get("LISTEN_DEFAULT_DURATION", "5")),
            max_duration=int(os.environ.get("LISTEN_MAX_DURATION", "30")),
            vad_silence_duration=float(os.environ.get("VAD_SILENCE_DURATION", "2.0")),
            vad_silence_threshold=int(os.environ.get("VAD_SILENCE_THRESHOLD", "500")),
        )
