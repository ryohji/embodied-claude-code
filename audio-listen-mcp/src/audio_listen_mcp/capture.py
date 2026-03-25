"""Audio capture using ffmpeg's avfoundation on macOS."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import struct
import tempfile
import wave

import httpx

from .config import ListenConfig

logger = logging.getLogger(__name__)


async def _run_vad_capture(
    proc: asyncio.subprocess.Process,
    sample_rate: int,
    silence_duration: float,
    silence_threshold: int,
) -> str:
    """Read PCM data from proc.stdout, detect end-of-speech via VAD, save as WAV.

    The caller is responsible for launching the process and its cleanup on error.
    """
    chunk_duration = 0.1  # 100ms chunks
    chunk_bytes = int(sample_rate * 2 * chunk_duration)

    audio_chunks: list[bytes] = []
    speech_detected = False
    silence_start: float | None = None

    try:
        while True:
            chunk = await proc.stdout.read(chunk_bytes)
            if not chunk:
                break
            audio_chunks.append(chunk)
            n_samples = len(chunk) // 2
            if n_samples == 0:
                continue
            samples = struct.unpack(f"<{n_samples}h", chunk[:n_samples * 2])
            rms = math.sqrt(sum(s * s for s in samples) / n_samples)
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
                    logger.info("Silence for %.1fs after speech, stopping", elapsed)
                    break
    finally:
        if proc.returncode is None:
            proc.terminate()
            await proc.wait()

    if not audio_chunks:
        raise RuntimeError("No audio data captured")

    pcm_data = b"".join(audio_chunks)
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)

    actual_duration = len(pcm_data) / (sample_rate * 2)
    logger.info("Recorded %.1fs to %s", actual_duration, wav_path)
    return wav_path


class AudioCapture:
    """Captures audio from the local microphone via ffmpeg."""

    def __init__(self, config: ListenConfig) -> None:
        self._config = config

    async def record(self, duration: int) -> str:
        """Record audio from the microphone for the given duration.

        Returns the path to the recorded WAV file. The caller is responsible
        for deleting the file after use.
        """
        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        device = self._config.audio_device or ":0"
        if not device.startswith(":"):
            device = f":{device}"

        cmd = [
            "ffmpeg",
            "-y",
            "-f", "avfoundation",
            "-i", device,
            "-acodec", "pcm_s16le",
            "-ar", str(self._config.sample_rate),
            "-ac", "1",
            "-t", str(duration),
            wav_path,
        ]

        logger.info("Recording %d seconds from device %s", duration, device)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            # Clean up the temp file on failure
            try:
                os.unlink(wav_path)
            except OSError:
                pass
            error_msg = stderr.decode(errors="replace").strip()
            raise RuntimeError(f"ffmpeg recording failed (exit {proc.returncode}): {error_msg}")

        logger.info("Recorded to %s", wav_path)
        return wav_path

    async def record_with_vad(
        self,
        max_duration: int,
        silence_duration: float,
        silence_threshold: int,
    ) -> str:
        """Record audio with Voice Activity Detection.

        Starts recording and waits for speech. Once speech is detected,
        recording continues until silence persists for `silence_duration`
        seconds, then stops automatically.

        Returns the path to the recorded WAV file.
        """
        device = self._config.audio_device or ":0"
        if not device.startswith(":"):
            device = f":{device}"

        sample_rate = self._config.sample_rate
        chunk_duration = 0.1  # 100ms chunks
        chunk_bytes = int(sample_rate * 2 * chunk_duration)  # 16bit mono

        cmd = [
            "ffmpeg",
            "-y",
            "-f", "avfoundation",
            "-i", device,
            "-acodec", "pcm_s16le",
            "-ar", str(sample_rate),
            "-ac", "1",
            "-t", str(max_duration),
            "-f", "s16le",
            "pipe:1",
        ]

        logger.info(
            "Recording with VAD (max=%ds, silence=%.1fs, threshold=%d) from %s",
            max_duration, silence_duration, silence_threshold, device,
        )

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        return await _run_vad_capture(proc, sample_rate, silence_duration, silence_threshold)

    async def list_devices(self) -> list[dict[str, str]]:
        """List available audio input devices using ffmpeg avfoundation.

        Returns a list of dicts with 'index' and 'name' keys.
        """
        cmd = [
            "ffmpeg",
            "-f", "avfoundation",
            "-list_devices", "true",
            "-i", "",
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        output = stderr.decode(errors="replace")

        devices: list[dict[str, str]] = []
        in_audio_section = False

        for line in output.splitlines():
            if "AVFoundation audio devices:" in line:
                in_audio_section = True
                continue
            if in_audio_section:
                # Lines look like: [AVFoundation ...] [0] MacBook Air Microphone
                match = re.search(r"\[(\d+)\]\s+(.+)$", line)
                if match:
                    devices.append({
                        "index": match.group(1),
                        "name": match.group(2).strip(),
                    })
                elif devices:
                    # End of audio device section
                    break

        return devices


class TapoAudioCapture:
    """Captures audio from Tapo camera microphone via RTSP stream."""

    def __init__(self, config: "ListenConfig") -> None:
        self._config = config

    def _rtsp_url(self) -> str:
        host = self._config.tapo_host or "192.168.0.1"
        user = self._config.tapo_username or "admin"
        pw = self._config.tapo_password or ""
        return f"rtsp://{user}:{pw}@{host}:554/stream1"

    def _masked_url(self) -> str:
        host = self._config.tapo_host or "192.168.0.1"
        return f"rtsp://****@{host}:554/stream1"

    def _base_cmd(self) -> list[str]:
        return [
            "ffmpeg", "-y",
            "-analyzeduration", "0",
            "-fflags", "nobuffer",
            "-rtsp_transport", "tcp",
            "-i", self._rtsp_url(),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(self._config.sample_rate),
            "-ac", "1",
        ]

    async def record(self, duration: int) -> str:
        """Record audio from RTSP stream for a fixed duration."""
        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        cmd = self._base_cmd() + ["-t", str(duration), wav_path]
        logger.info("Recording %ds from Tapo camera %s", duration, self._masked_url())
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=duration + 15)
        except asyncio.TimeoutError:
            proc.terminate()
            await proc.wait()
            raise RuntimeError("ffmpeg RTSP recording timed out")
        if proc.returncode != 0:
            try:
                os.unlink(wav_path)
            except OSError:
                pass
            raise RuntimeError(
                f"ffmpeg RTSP recording failed: {stderr.decode(errors='replace').strip()}"
            )
        return wav_path

    async def record_with_vad(
        self,
        max_duration: int,
        silence_duration: float,
        silence_threshold: int,
    ) -> str:
        """Record audio from RTSP stream with Voice Activity Detection."""
        cmd = self._base_cmd() + [
            "-t", str(max_duration),
            "-f", "s16le", "pipe:1",
        ]
        logger.info(
            "Recording with VAD from Tapo camera %s (max=%ds)",
            self._masked_url(), max_duration,
        )
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        return await _run_vad_capture(
            proc, self._config.sample_rate, silence_duration, silence_threshold
        )

    async def list_devices(self) -> list[dict[str, str]]:
        """Return camera info as a single-item device list."""
        return [{"index": "tapo", "name": f"Tapo Camera ({self._config.tapo_host})"}]


def _pcm_to_wav(pcm_data: bytes, sample_rate: int) -> str:
    """Save raw PCM (s16le, mono) bytes to a temporary WAV file and return the path."""
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return wav_path


class CameraDaemonAudioCapture:
    """Captures audio from camera-daemon's GET /audio HTTP endpoint."""

    SAMPLE_RATE = 8000

    def __init__(self, config: ListenConfig) -> None:
        self._config = config
        self._base_url = (config.camera_daemon_url or "http://localhost:8080").rstrip("/")

    async def record(self, duration: int) -> str:
        """Fetch fixed-duration PCM from camera-daemon and return path to WAV file."""
        url = f"{self._base_url}/audio"
        params = {"duration": duration}
        timeout = httpx.Timeout(duration + 15)
        logger.info("Fetching %ds audio from camera-daemon: %s", duration, url)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("GET", url, params=params) as resp:
                resp.raise_for_status()
                chunks: list[bytes] = []
                async for chunk in resp.aiter_bytes():
                    chunks.append(chunk)
        pcm_data = b"".join(chunks)
        if not pcm_data:
            raise RuntimeError("camera-daemon returned empty audio data")
        wav_path = _pcm_to_wav(pcm_data, self.SAMPLE_RATE)
        actual_duration = len(pcm_data) / (self.SAMPLE_RATE * 2)
        logger.info("Recorded %.1fs to %s", actual_duration, wav_path)
        return wav_path

    async def record_with_vad(
        self,
        max_duration: int,
        silence_duration: float,
        silence_threshold: int,
    ) -> str:
        """Fetch VAD-terminated PCM from camera-daemon and return path to WAV file."""
        url = f"{self._base_url}/audio"
        params = {
            "max_duration": max_duration,
            "silence_duration": silence_duration,
            "vad_threshold": "0.5",  # Silero probability threshold (固定値)
        }
        timeout = httpx.Timeout(max_duration + 15)
        logger.info(
            "Fetching VAD audio from camera-daemon: %s (max=%ds, silence=%.1fs)",
            url, max_duration, silence_duration,
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("GET", url, params=params) as resp:
                resp.raise_for_status()
                chunks: list[bytes] = []
                async for chunk in resp.aiter_bytes():
                    chunks.append(chunk)
        pcm_data = b"".join(chunks)
        if not pcm_data:
            raise RuntimeError("camera-daemon returned empty audio data")
        wav_path = _pcm_to_wav(pcm_data, self.SAMPLE_RATE)
        actual_duration = len(pcm_data) / (self.SAMPLE_RATE * 2)
        logger.info("Recorded %.1fs to %s", actual_duration, wav_path)
        return wav_path

    async def list_devices(self) -> list[dict[str, str]]:
        """Return camera-daemon endpoint info as a single-item device list."""
        return [{"index": "daemon", "name": f"Camera Daemon ({self._base_url})"}]
