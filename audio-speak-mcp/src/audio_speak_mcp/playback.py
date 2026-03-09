from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

if TYPE_CHECKING:
    from .go2rtc import Go2RTCProcess

logger = logging.getLogger(__name__)


def _post_and_poll(audio_path: str, api_url: str, stream_name: str) -> None:
    """POST audio to go2rtc and poll until playback completes (blocking)."""
    abs_path = str(Path(audio_path).resolve())
    src = f"ffmpeg:{abs_path}#audio=pcma#input=file"
    url = (
        f"{api_url}/api/streams"
        f"?dst={quote(stream_name, safe='')}"
        f"&src={quote(src, safe='')}"
    )

    # Add the ffmpeg producer to the stream (body is empty)
    req = urllib.request.Request(url, method="POST", data=b"")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
    except Exception as exc:
        raise RuntimeError(f"go2rtc POST failed: {exc}") from exc

    # Check if backchannel consumer with senders exists
    has_sender = False
    for consumer in body.get("consumers", []):
        if consumer.get("senders"):
            has_sender = True
            break

    if not has_sender:
        logger.warning(
            "go2rtc: no audio sender established on stream '%s' — camera may not support backchannel",
            stream_name,
        )
        return

    # Find ffmpeg producer ID
    ffmpeg_producer_id = None
    for p in body.get("producers", []):
        if p.get("format_name") == "wav" or "ffmpeg" in p.get("source", ""):
            ffmpeg_producer_id = p.get("id")
            break

    if ffmpeg_producer_id:
        logger.info("go2rtc: audio producer started (id=%s), polling...", ffmpeg_producer_id)
        # Poll until producer disappears (playback done)
        status_url = f"{api_url}/api/streams"
        for _ in range(60):
            time.sleep(0.5)
            try:
                with urllib.request.urlopen(status_url, timeout=5) as r:
                    streams = json.loads(r.read())
                stream = streams.get(stream_name, {})
                still_playing = any(
                    p.get("id") == ffmpeg_producer_id
                    for p in stream.get("producers", [])
                )
                if not still_playing:
                    break
            except Exception:
                break

    logger.info("go2rtc: audio playback complete")


async def play_with_go2rtc(audio_path: str, go2rtc: "Go2RTCProcess") -> None:
    """Send an audio file to go2rtc and play it through the camera speaker."""
    await asyncio.to_thread(
        _post_and_poll, audio_path, go2rtc.api_url, go2rtc.stream_name
    )
