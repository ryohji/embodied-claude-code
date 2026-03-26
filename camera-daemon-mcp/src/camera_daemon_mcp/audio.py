"""Audio streaming from Tapo camera RTSP stream."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from aiohttp import web

logger = logging.getLogger(__name__)

SAMPLE_RATE = 8000
CHUNK_DURATION = 0.1  # 100ms chunks
CHUNK_BYTES = int(SAMPLE_RATE * 2 * CHUNK_DURATION)  # 1600 bytes (16-bit mono)


class SileroVAD:
    """Silero VAD wrapper using onnxruntime."""

    MODEL_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
    MODEL_PATH = Path.home() / ".local" / "share" / "camera-daemon-mcp" / "silero_vad.onnx"
    # 256 samples @ 8kHz = 32ms per frame
    FRAME_SAMPLES = 256
    FRAME_BYTES = FRAME_SAMPLES * 2  # 512 bytes
    _SR = 8000

    def __init__(self) -> None:
        import onnxruntime as ort
        import numpy as np
        self._np = np
        model_path = self._ensure_model()
        self._session = ort.InferenceSession(str(model_path))
        # Combined LSTM state (2, 1, 128)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._sr = np.array(self._SR, dtype=np.int64)

    def _ensure_model(self) -> Path:
        """Download the model if not already cached."""
        path = self.MODEL_PATH
        if not path.exists():
            import urllib.request
            logger.info("Downloading Silero VAD model to %s", path)
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(self.MODEL_URL, path)
            logger.info("Silero VAD model downloaded")
        return path

    def is_speech(self, frame: bytes) -> float:
        """Run inference on a 256-sample (512-byte) 8kHz PCM frame.

        Returns speech probability (0.0–1.0).
        """
        np = self._np
        n_samples = len(frame) // 2
        samples = np.frombuffer(frame[:n_samples * 2], dtype=np.int16).astype(np.float32) / 32768.0
        audio = samples[np.newaxis, :]  # shape: (1, 256)

        ort_inputs = {
            "input": audio,
            "sr":    self._sr,
            "state": self._state,
        }
        ort_outputs = self._session.run(None, ort_inputs)
        prob = float(ort_outputs[0].squeeze())  # output = speech probability
        self._state = ort_outputs[1]
        return prob

    def reset(self) -> None:
        """Reset LSTM state (call between utterances if needed)."""
        self._state = self._np.zeros((2, 1, 128), dtype=self._np.float32)


def _build_ffmpeg_cmd(rtsp_url: str, max_duration: int) -> list[str]:
    return [
        "ffmpeg", "-y",
        "-analyzeduration", "0",
        "-fflags", "nobuffer",
        "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-vn",
        "-af", "dynaudnorm=f=150:g=15",
        "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        "-t", str(max_duration),
        "-f", "s16le",
        "pipe:1",
    ]


async def stream_audio_fixed(
    response: web.StreamResponse,
    rtsp_url: str,
    duration: int,
) -> None:
    """Stream fixed-duration PCM audio from RTSP to HTTP response."""
    cmd = _build_ffmpeg_cmd(rtsp_url, duration)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        while True:
            chunk = await proc.stdout.read(CHUNK_BYTES)
            if not chunk:
                break
            await response.write(chunk)
    finally:
        if proc.returncode is None:
            proc.terminate()
        await proc.wait()


async def stream_audio_vad(
    response: web.StreamResponse,
    rtsp_url: str,
    max_duration: int,
    silence_duration: float = 1.5,
    vad_threshold: float = 0.5,
) -> None:
    """Stream PCM audio from RTSP with Silero VAD end-of-speech detection."""
    vad = await asyncio.to_thread(SileroVAD)  # モデルロード（初回はダウンロード）
    cmd = _build_ffmpeg_cmd(rtsp_url, max_duration)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        speech_detected = False
        silence_start: float | None = None
        buf = bytearray()

        while True:
            chunk = await proc.stdout.read(SileroVAD.FRAME_BYTES)
            if not chunk:
                break

            await response.write(chunk)
            buf.extend(chunk)

            # Process complete 256-sample (8kHz) frames from the buffer
            while len(buf) >= SileroVAD.FRAME_BYTES:
                frame = bytes(buf[:SileroVAD.FRAME_BYTES])
                buf = buf[SileroVAD.FRAME_BYTES:]
                prob = await asyncio.to_thread(vad.is_speech, frame)

                if prob >= vad_threshold:
                    if not speech_detected:
                        logger.info("Speech detected (prob=%.2f)", prob)
                    speech_detected = True
                    silence_start = None
                elif speech_detected:
                    if silence_start is None:
                        silence_start = asyncio.get_event_loop().time()
                    elapsed = asyncio.get_event_loop().time() - silence_start
                    if elapsed >= silence_duration:
                        logger.info("Silence for %.1fs after speech, stopping VAD", elapsed)
                        return
    finally:
        if proc.returncode is None:
            proc.terminate()
        await proc.wait()
