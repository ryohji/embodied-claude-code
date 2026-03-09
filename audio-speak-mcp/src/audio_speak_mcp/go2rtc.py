"""go2rtc binary management for Tapo camera streaming."""

from __future__ import annotations

import asyncio
import logging
import os
import stat
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import IO, Optional

import yaml

logger = logging.getLogger(__name__)

GO2RTC_VERSION = "1.9.14"
GO2RTC_URL = f"https://github.com/AlexxIT/go2rtc/releases/download/v{GO2RTC_VERSION}/go2rtc_mac_arm64.zip"
GO2RTC_BIN = Path.home() / ".local" / "bin" / "go2rtc"


def _download_binary() -> Path:
    """Download go2rtc zip, extract binary, make it executable (blocking)."""
    GO2RTC_BIN.parent.mkdir(parents=True, exist_ok=True)
    zip_path = GO2RTC_BIN.with_suffix(".zip")
    urllib.request.urlretrieve(GO2RTC_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        # zip contains a single file named "go2rtc"
        zf.extract("go2rtc", GO2RTC_BIN.parent)
    zip_path.unlink(missing_ok=True)
    current = stat.S_IMODE(GO2RTC_BIN.stat().st_mode)
    GO2RTC_BIN.chmod(current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return GO2RTC_BIN


async def ensure_binary() -> Path:
    """Return path to go2rtc binary, downloading if necessary."""
    if GO2RTC_BIN.exists():
        return GO2RTC_BIN
    return await asyncio.to_thread(_download_binary)


def generate_config(
    camera_host: str,
    username: str,
    password: str,
    cloud_password: str,
    stream_name: str = "camera",
) -> str:
    """Generate go2rtc YAML config string for a Tapo camera.

    Two stream entries are required:
    - rtsp://: provides the video/audio stream
    - tapo://: enables the backchannel audio (cloud_password only, no username)
    """
    import shutil
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    config = {
        "streams": {
            stream_name: [
                f"rtsp://{username}:{password}@{camera_host}:554/stream1",
                f"tapo://{cloud_password}@{camera_host}",
            ],
        },
        "ffmpeg": {
            "bin": ffmpeg_bin,
        },
        "api": {
            "listen": ":1984",
        },
        "log": {
            "level": "info",
        },
    }
    return yaml.dump(config, default_flow_style=False, allow_unicode=True)


class Go2RTCProcess:
    """Manages a go2rtc child process."""

    def __init__(
        self,
        camera_host: str,
        username: str,
        password: str,
        cloud_password: str,
        api_url: str = "http://localhost:1984",
        stream_name: str = "camera",
    ) -> None:
        self._camera_host = camera_host
        self._username = username
        self._password = password
        self._cloud_password = cloud_password
        self._api_url = api_url
        self._stream_name = stream_name
        self._process: Optional[asyncio.subprocess.Process] = None
        self._config_path: Optional[str] = None
        self._log_file: Optional[IO] = None

    @property
    def api_url(self) -> str:
        return self._api_url

    @property
    def stream_name(self) -> str:
        return self._stream_name

    async def start(self) -> None:
        """Start the go2rtc process. Does nothing if already running."""
        if await self.is_running():
            logger.info("go2rtc already running at %s", self._api_url)
            return

        if self._process is not None and self._process.returncode is None:
            return

        binary = await ensure_binary()

        config_yaml = generate_config(
            self._camera_host,
            self._username,
            self._password,
            self._cloud_password,
            self._stream_name,
        )

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            prefix="go2rtc_",
            delete=False,
        ) as f:
            f.write(config_yaml)
            self._config_path = f.name

        log_path = "/tmp/go2rtc.log"
        logger.info("go2rtc log: %s", log_path)
        self._log_file = open(log_path, "w")  # noqa: SIM115

        self._process = await asyncio.create_subprocess_exec(
            str(binary),
            "-config",
            self._config_path,
            stdout=self._log_file,
            stderr=self._log_file,
        )

        await asyncio.sleep(1.5)

        if self._process.returncode is not None:
            try:
                log_content = Path(log_path).read_text()
            except OSError:
                log_content = ""
            raise RuntimeError(f"go2rtc exited immediately: {log_content}")

        if not await self.is_running():
            self._process.terminate()
            raise RuntimeError("go2rtc started but not responding")

    def stop(self) -> None:
        """Terminate the go2rtc process and clean up the temp config file."""
        if self._process is not None:
            try:
                self._process.terminate()
            except ProcessLookupError:
                pass
            self._process = None

        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

        if self._config_path is not None:
            try:
                os.unlink(self._config_path)
            except FileNotFoundError:
                pass
            self._config_path = None

    async def is_running(self) -> bool:
        """Return True if the go2rtc API endpoint is reachable."""
        url = f"{self._api_url}/api"

        def _check() -> bool:
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    return resp.status == 200
            except Exception:
                return False

        return await asyncio.to_thread(_check)
