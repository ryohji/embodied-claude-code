"""TTS engine abstraction for audio-speak-mcp."""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from .config import SpeakConfig

logger = logging.getLogger(__name__)


class TTSEngine(ABC):
    """Abstract base class for TTS engines."""

    @abstractmethod
    async def say(self, text: str, voice: str | None = None, rate: int | None = None) -> str:
        """Speak the given text. Returns a status message."""
        ...

    @abstractmethod
    async def list_voices(self) -> list[dict[str, str]]:
        """Return a list of available voices."""
        ...


class MacOSTTSEngine(TTSEngine):
    """TTS engine using macOS built-in say command."""

    def __init__(self, default_voice: str, default_rate: int | None) -> None:
        self._default_voice = default_voice
        self._default_rate = default_rate

    async def say(self, text: str, voice: str | None = None, rate: int | None = None) -> str:
        voice = voice or self._default_voice
        rate = rate or self._default_rate

        # Write text to a temp file to avoid shell injection via special characters
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(text)
            text_file = f.name

        try:
            cmd = ["say", "-v", voice]
            if rate is not None:
                cmd.extend(["-r", str(rate)])
            cmd.extend(["-f", text_file])

            logger.info("Running: %s", " ".join(cmd))
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()

            if proc.returncode != 0:
                err = stderr.decode().strip()
                return f"say コマンドがエラーを返しました (code {proc.returncode}): {err}"

            return f"発話完了（voice={voice}, {len(text)}文字）"
        finally:
            Path(text_file).unlink(missing_ok=True)

    async def list_voices(self) -> list[dict[str, str]]:
        proc = await asyncio.create_subprocess_exec(
            "say", "-v", "?",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()

        voices: list[dict[str, str]] = []
        for line in stdout.decode().splitlines():
            # Format: "Name    lang  # Sample text"
            if not line.strip():
                continue
            parts = line.split("#", 1)
            meta = parts[0].strip()
            sample = parts[1].strip() if len(parts) > 1 else ""

            # Split meta into name and language code
            # e.g. "Kyoko             ja_JP"
            tokens = meta.split()
            if len(tokens) >= 2:
                name = tokens[0]
                lang = tokens[1]
            else:
                name = meta
                lang = ""

            voices.append({"name": name, "language": lang, "sample": sample})

        return voices


class ElevenLabsTTSEngine(TTSEngine):
    """TTS engine using ElevenLabs API with mpv for playback."""

    def __init__(
        self,
        api_key: str,
        default_voice_id: str | None,
        model_id: str,
    ) -> None:
        self._api_key = api_key
        self._default_voice_id = default_voice_id
        self._model_id = model_id

    async def say(self, text: str, voice: str | None = None, rate: int | None = None) -> str:
        try:
            from elevenlabs import ElevenLabs
        except ImportError:
            return (
                "elevenlabs パッケージがインストールされていません。"
                "pip install elevenlabs でインストールしてください。"
            )

        if not shutil.which("mpv"):
            return "mpv が見つかりません。brew install mpv でインストールしてください。"

        voice_id = voice or self._default_voice_id
        if not voice_id:
            return "ElevenLabs の voice_id が指定されていません。ELEVENLABS_VOICE_ID を設定してください。"

        # Generate audio using ElevenLabs API
        try:
            client = ElevenLabs(api_key=self._api_key)
            audio_generator = await asyncio.to_thread(
                client.text_to_speech.convert,
                text=text,
                voice_id=voice_id,
                model_id=self._model_id,
            )

            # Write audio to temp file
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                for chunk in audio_generator:
                    f.write(chunk)
                audio_path = f.name

            # Play with mpv
            proc = await asyncio.create_subprocess_exec(
                "mpv", "--no-video", "--really-quiet", audio_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            Path(audio_path).unlink(missing_ok=True)

            if proc.returncode != 0:
                return f"mpv の再生でエラーが発生しました (code {proc.returncode})"

            return f"発話完了（ElevenLabs, voice_id={voice_id}, {len(text)}文字）"
        except Exception as e:
            logger.exception("ElevenLabs TTS error")
            return f"ElevenLabs でエラーが発生しました: {e!s}"

    async def list_voices(self) -> list[dict[str, str]]:
        try:
            from elevenlabs import ElevenLabs
        except ImportError:
            return []

        try:
            client = ElevenLabs(api_key=self._api_key)
            response = await asyncio.to_thread(client.voices.get_all)
            return [
                {"name": v.name or "", "voice_id": v.voice_id, "language": ""}
                for v in response.voices
            ]
        except Exception as e:
            logger.exception("Failed to list ElevenLabs voices")
            return []


def create_engine(config: SpeakConfig) -> TTSEngine:
    """Create a TTS engine based on configuration."""
    if config.tts_engine == "elevenlabs":
        if config.elevenlabs_api_key:
            logger.info("Using ElevenLabs TTS engine")
            return ElevenLabsTTSEngine(
                api_key=config.elevenlabs_api_key,
                default_voice_id=config.elevenlabs_voice_id,
                model_id=config.elevenlabs_model_id,
            )
        else:
            logger.warning(
                "ElevenLabs API key not set, falling back to macOS TTS"
            )

    logger.info("Using macOS TTS engine (voice=%s)", config.tts_voice)
    return MacOSTTSEngine(
        default_voice=config.tts_voice,
        default_rate=config.tts_rate,
    )
