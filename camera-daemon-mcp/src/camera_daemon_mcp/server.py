"""MCP Server that manages go2rtc lifecycle and provides HTTP API for camera access."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from aiohttp import web
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool

from .camera import TapoCamera
from .config import CameraConfig, ServerConfig
from .go2rtc import Go2RTCProcess

logger = logging.getLogger(__name__)

HTTP_PORT = int(os.environ.get("HTTP_PORT", "8080"))


class CameraDaemonMCPServer:
    def __init__(self) -> None:
        self._server = Server("camera-daemon")
        self._go2rtc = Go2RTCProcess(
            camera_host=os.environ.get("TAPO_CAMERA_HOST", ""),
            username=os.environ.get("TAPO_USERNAME", ""),
            password=os.environ.get("TAPO_PASSWORD", ""),
            cloud_password=os.environ.get("TAPO_CLOUD_PASSWORD", ""),
        )
        self._server_config = ServerConfig.from_env()
        self._camera: TapoCamera | None = None
        self._setup_mcp_handlers()

    def _setup_mcp_handlers(self) -> None:
        @self._server.list_tools()
        async def list_tools() -> list[Tool]:
            return []  # ツールは公開しない（HTTP API 経由でのみアクセス）

    # --- HTTP ハンドラ ---

    async def _handle_see(self, request: web.Request) -> web.Response:
        if self._camera is None:
            return web.Response(status=503, text="カメラ未接続")
        result = await self._camera.capture_image()
        return web.json_response({
            "image": result.image_base64,
            "mime_type": "image/jpeg",
            "width": result.width,
            "height": result.height,
            "timestamp": result.timestamp,
        })

    async def _handle_ptz(self, request: web.Request) -> web.Response:
        if self._camera is None:
            return web.Response(status=503, text="カメラ未接続")
        body = await request.json()
        direction = body.get("direction", "")
        degrees = body.get("degrees", 30)
        match direction:
            case "left":
                await self._camera.pan_left(degrees)
            case "right":
                await self._camera.pan_right(degrees)
            case "up":
                await self._camera.tilt_up(degrees)
            case "down":
                await self._camera.tilt_down(degrees)
            case _:
                return web.Response(status=400, text=f"不明な direction: {direction}")
        return web.json_response({"status": "moved", "direction": direction, "degrees": degrees})

    async def _handle_look_around(self, request: web.Request) -> web.Response:
        if self._camera is None:
            return web.Response(status=503, text="カメラ未接続")
        captures = await self._camera.look_around()
        directions = ["Center", "Left", "Right", "Up"]
        result = []
        for i, capture in enumerate(captures):
            result.append({
                "image": capture.image_base64,
                "mime_type": "image/jpeg",
                "direction": directions[i] if i < len(directions) else f"Angle {i}",
            })
        return web.json_response({"captures": result})

    async def _handle_info(self, request: web.Request) -> web.Response:
        if self._camera is None:
            return web.Response(status=503, text="カメラ未接続")
        info = await self._camera.get_device_info()
        return web.json_response(info)

    async def _handle_presets(self, request: web.Request) -> web.Response:
        if self._camera is None:
            return web.Response(status=503, text="カメラ未接続")
        presets = await self._camera.get_presets()
        return web.json_response(presets)

    async def _handle_go_to_preset(self, request: web.Request) -> web.Response:
        if self._camera is None:
            return web.Response(status=503, text="カメラ未接続")
        preset_id = request.match_info["preset_id"]
        await self._camera.go_to_preset(preset_id)
        return web.json_response({"status": "moved", "preset_id": preset_id})

    async def _handle_say_to_camera(self, request: web.Request) -> web.Response:
        body = await request.json()
        text = body.get("text", "")
        voice = body.get("voice", "Kyoko")
        rate = body.get("rate")
        if not shutil.which("ffmpeg"):
            return web.Response(status=500, text="ffmpeg が見つかりません")
        audio_file = tempfile.mktemp(suffix=".aiff")
        text_file = tempfile.mktemp(suffix=".txt")
        try:
            Path(text_file).write_text(text, encoding="utf-8")
            cmd = ["say", "-v", voice, "-o", audio_file]
            if rate is not None:
                cmd.extend(["-r", str(rate)])
            cmd.extend(["-f", text_file])
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                return web.Response(status=500, text=f"say エラー: {stderr.decode().strip()}")
            from .playback import play_with_go2rtc
            await play_with_go2rtc(audio_file, self._go2rtc)
            return web.json_response({"status": "spoken", "text": text, "voice": voice})
        finally:
            Path(text_file).unlink(missing_ok=True)
            Path(audio_file).unlink(missing_ok=True)

    async def _handle_stream_url(self, request: web.Request) -> web.Response:
        return web.json_response({
            "url": self._go2rtc.api_url,
            "stream": self._go2rtc.stream_name,
        })

    def _make_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/see", self._handle_see)
        app.router.add_post("/ptz", self._handle_ptz)
        app.router.add_get("/look_around", self._handle_look_around)
        app.router.add_get("/info", self._handle_info)
        app.router.add_get("/presets", self._handle_presets)
        app.router.add_post("/preset/{preset_id}", self._handle_go_to_preset)
        app.router.add_post("/say", self._handle_say_to_camera)
        app.router.add_get("/stream_url", self._handle_stream_url)
        return app

    async def run(self) -> None:
        await self._go2rtc.start()
        try:
            config = CameraConfig.from_env()
            self._camera = TapoCamera(config, self._server_config.capture_dir)
            await self._camera.connect()
        except Exception as e:
            logger.warning("Camera connection failed: %s", e)
            self._camera = None

        # HTTP サーバーと MCP stdio を並行起動
        app = self._make_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "localhost", HTTP_PORT)
        await site.start()
        logger.info("HTTP server listening on http://localhost:%d", HTTP_PORT)

        try:
            async with stdio_server() as (read_stream, write_stream):
                await self._server.run(
                    read_stream,
                    write_stream,
                    self._server.create_initialization_options(),
                )
        finally:
            await runner.cleanup()
            self._go2rtc.stop()
            if self._camera is not None:
                await self._camera.disconnect()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(CameraDaemonMCPServer().run())
