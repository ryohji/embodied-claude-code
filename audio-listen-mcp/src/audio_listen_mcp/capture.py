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

from .config import ListenConfig

logger = logging.getLogger(__name__)


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

        audio_chunks: list[bytes] = []
        speech_detected = False
        silence_start: float | None = None

        try:
            while True:
                chunk = await proc.stdout.read(chunk_bytes)
                if not chunk:
                    break

                audio_chunks.append(chunk)

                # Calculate RMS amplitude
                n_samples = len(chunk) // 2
                if n_samples == 0:
                    continue
                samples = struct.unpack(f"<{n_samples}h", chunk[:n_samples * 2])
                rms = math.sqrt(sum(s * s for s in samples) / n_samples)

                if rms >= silence_threshold:
                    # Speech detected
                    if not speech_detected:
                        logger.info("Speech detected (RMS=%.0f)", rms)
                    speech_detected = True
                    silence_start = None
                elif speech_detected:
                    # Silence after speech
                    if silence_start is None:
                        silence_start = asyncio.get_event_loop().time()
                    elapsed = asyncio.get_event_loop().time() - silence_start
                    if elapsed >= silence_duration:
                        logger.info(
                            "Silence for %.1fs after speech, stopping", elapsed
                        )
                        break
        finally:
            if proc.returncode is None:
                proc.terminate()
                await proc.wait()

        if not audio_chunks:
            raise RuntimeError("No audio data captured")

        # Write collected PCM data as WAV
        pcm_data = b"".join(audio_chunks)
        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)

        actual_duration = len(pcm_data) / (sample_rate * 2)
        logger.info("Recorded %.1fs to %s", actual_duration, wav_path)
        return wav_path

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
