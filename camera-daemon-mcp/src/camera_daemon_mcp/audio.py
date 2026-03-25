"""Audio streaming from Tapo camera RTSP stream."""

from __future__ import annotations

import asyncio
import logging
import math
import struct

from aiohttp import web

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.1  # 100ms chunks
CHUNK_BYTES = int(SAMPLE_RATE * 2 * CHUNK_DURATION)  # 3200 bytes (16-bit mono)


def _build_ffmpeg_cmd(rtsp_url: str, max_duration: int) -> list[str]:
    return [
        "ffmpeg", "-y",
        "-analyzeduration", "0",
        "-fflags", "nobuffer",
        "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-vn",
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
    silence_threshold: int = 500,
) -> None:
    """Stream PCM audio from RTSP with RMS-based VAD end-of-speech detection."""
    cmd = _build_ffmpeg_cmd(rtsp_url, max_duration)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        speech_detected = False
        silence_start: float | None = None

        while True:
            chunk = await proc.stdout.read(CHUNK_BYTES)
            if not chunk:
                break

            await response.write(chunk)

            n_samples = len(chunk) // 2
            if n_samples > 0:
                samples = struct.unpack(f"<{n_samples}h", chunk[:n_samples * 2])
                rms = math.sqrt(sum(s * s for s in samples) / n_samples)
            else:
                rms = 0.0

            if rms >= silence_threshold:
                if not speech_detected:
                    logger.info("Speech detected (RMS=%.0f)", rms)
                speech_detected = True
                silence_start = None
            elif speech_detected:
                if silence_start is None:
                    silence_start = asyncio.get_event_loop().time()
                elapsed = asyncio.get_event_loop().time() - silence_start
                if elapsed >= silence_duration:
                    logger.info("Silence for %.1fs after speech, stopping VAD capture", elapsed)
                    break
    finally:
        if proc.returncode is None:
            proc.terminate()
        await proc.wait()
