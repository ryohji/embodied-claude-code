"""Configuration for audio-speak-mcp server."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class SpeakConfig:
    """TTS configuration loaded from environment variables."""

    tts_engine: str
    tts_voice: str
    tts_rate: int | None
    elevenlabs_api_key: str | None
    elevenlabs_voice_id: str | None
    elevenlabs_model_id: str
    kokoro_voice: str
    kokoro_model_id: str
    kokoro_speed: float
    kokoro_lang_code: str

    @classmethod
    def from_env(cls) -> SpeakConfig:
        load_dotenv()

        rate_str = os.getenv("TTS_RATE")
        tts_rate = int(rate_str) if rate_str else None

        kokoro_speed_str = os.getenv("KOKORO_SPEED", "1.0")
        kokoro_speed = float(kokoro_speed_str)

        return cls(
            tts_engine=os.getenv("TTS_ENGINE", "macos"),
            tts_voice=os.getenv("TTS_VOICE", "Kyoko"),
            tts_rate=tts_rate,
            elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY"),
            elevenlabs_voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
            elevenlabs_model_id=os.getenv("ELEVENLABS_MODEL_ID", "eleven_v3"),
            kokoro_voice=os.getenv("KOKORO_VOICE", "jf_alpha"),
            kokoro_model_id=os.getenv(
                "KOKORO_MODEL_ID", "mlx-community/Kokoro-82M-bf16"
            ),
            kokoro_speed=kokoro_speed,
            kokoro_lang_code=os.getenv("KOKORO_LANG_CODE", "j"),
        )
