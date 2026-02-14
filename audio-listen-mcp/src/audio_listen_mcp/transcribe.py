"""Whisper engine abstraction for speech-to-text transcription."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class WhisperEngine(ABC):
    """Abstract base class for Whisper transcription engines."""

    @abstractmethod
    async def transcribe(self, audio_path: str, language: str) -> str:
        """Transcribe the audio file at the given path.

        Returns the transcribed text.
        """
        ...


class MLXWhisperEngine(WhisperEngine):
    """Whisper engine using mlx-whisper (Apple Silicon optimized)."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        # mlx-whisper uses HuggingFace model IDs
        self._model_path = f"mlx-community/whisper-{model_name}-mlx"
        logger.info("MLXWhisperEngine initialized with model: %s", self._model_path)

    async def transcribe(self, audio_path: str, language: str) -> str:
        import mlx_whisper

        logger.info("Transcribing %s with mlx-whisper", audio_path)
        result = await asyncio.to_thread(
            mlx_whisper.transcribe,
            audio_path,
            path_or_hf_repo=self._model_path,
            language=language,
        )
        text = result.get("text", "").strip()
        logger.info("Transcription result: %s", text[:100])
        return text


class PyTorchWhisperEngine(WhisperEngine):
    """Whisper engine using OpenAI's PyTorch implementation (fallback)."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None

    async def _ensure_model(self) -> None:
        if self._model is None:
            import whisper

            logger.info("Loading PyTorch Whisper model: %s", self._model_name)
            self._model = await asyncio.to_thread(
                whisper.load_model, self._model_name
            )
            logger.info("PyTorch Whisper model loaded")

    async def transcribe(self, audio_path: str, language: str) -> str:
        await self._ensure_model()

        logger.info("Transcribing %s with PyTorch Whisper", audio_path)
        result = await asyncio.to_thread(
            self._model.transcribe, audio_path, language=language
        )
        text = result.get("text", "").strip()
        logger.info("Transcription result: %s", text[:100])
        return text


def create_engine(engine_name: str, model_name: str) -> WhisperEngine:
    """Create a Whisper engine instance with graceful fallback.

    Tries the requested engine first, then falls back to alternatives.
    """
    errors: list[str] = []

    # Determine engine order based on preference
    if engine_name == "mlx":
        engine_order = [("mlx", MLXWhisperEngine), ("pytorch", PyTorchWhisperEngine)]
    elif engine_name == "pytorch":
        engine_order = [("pytorch", PyTorchWhisperEngine), ("mlx", MLXWhisperEngine)]
    else:
        engine_order = [("mlx", MLXWhisperEngine), ("pytorch", PyTorchWhisperEngine)]
        errors.append(f"Unknown engine '{engine_name}', trying available engines")

    for name, engine_cls in engine_order:
        try:
            engine = engine_cls(model_name)
            # Verify the engine's dependencies are importable
            if name == "mlx":
                import mlx_whisper  # noqa: F401
            elif name == "pytorch":
                import whisper  # noqa: F401
            if name != engine_name:
                logger.warning(
                    "Requested engine '%s' unavailable, using '%s' instead",
                    engine_name, name,
                )
            else:
                logger.info("Using Whisper engine: %s", name)
            return engine
        except ImportError as e:
            errors.append(f"{name}: {e}")
            logger.debug("Engine '%s' not available: %s", name, e)

    raise ImportError(
        "No Whisper engine available. Install one of:\n"
        "  pip install mlx-whisper    (recommended for Apple Silicon)\n"
        "  pip install openai-whisper (PyTorch, slower)\n"
        f"\nDetailed errors:\n" + "\n".join(f"  - {e}" for e in errors)
    )
