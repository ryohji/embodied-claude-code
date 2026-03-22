"""MCP Server that manages go2rtc lifecycle and provides HTTP API for camera access."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path

from aiohttp import web
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool

from .camera import TapoCamera
from .config import CameraConfig, ServerConfig
from .go2rtc import Go2RTCProcess
from .playback import play_with_go2rtc

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

    async def _handle_image(self, request: web.Request) -> web.Response:
        if self._camera is None:
            return web.Response(status=503, text="カメラ未接続")
        pan_str = request.rel_url.query.get("pan")
        tilt_str = request.rel_url.query.get("tilt")
        if pan_str is not None or tilt_str is not None:
            # 片側省略の場合は現在位置を取得して補完
            if pan_str is None or tilt_str is None:
                pos = await self._camera.get_hw_position()
                if pos is None:
                    return web.Response(status=503, text="カメラ位置を取得できません")
                pan = float(pan_str) if pan_str is not None else pos.pan
                tilt = float(tilt_str) if tilt_str is not None else pos.tilt
            else:
                pan = float(pan_str)
                tilt = float(tilt_str)
            await self._camera.absolute_move(pan, tilt)
        result = await self._camera.capture_image()
        return web.json_response({
            "image": result.image_base64,
            "mime_type": "image/jpeg",
            "width": result.width,
            "height": result.height,
            "timestamp": result.timestamp,
        })

    async def _handle_direction_get(self, request: web.Request) -> web.Response:
        if self._camera is None:
            return web.Response(status=503, text="カメラ未接続")
        pos = await self._camera.get_hw_position()
        if pos is None:
            return web.Response(status=503, text="カメラ位置を取得できません")
        return web.json_response({"pan": pos.pan, "tilt": pos.tilt})

    async def _handle_direction_post(self, request: web.Request) -> web.Response:
        if self._camera is None:
            return web.Response(status=503, text="カメラ未接続")
        try:
            body = await request.json()
            pan = float(body["pan"])
            tilt = float(body["tilt"])
        except Exception:
            return web.Response(status=400, text='body must be {"pan": number, "tilt": number}')
        result = await self._camera.absolute_move(pan, tilt)
        if result.success:
            return web.json_response({"status": "ok"})
        return web.Response(status=500, text=result.message)

    async def _handle_audio_post(self, request: web.Request) -> web.Response:
        content_type = request.content_type or ""
        if "mpeg" in content_type or "mp3" in content_type:
            ext = ".mp3"
        elif "aiff" in content_type or "aif" in content_type:
            ext = ".aiff"
        else:
            ext = ".wav"

        data = await request.read()
        if not data:
            return web.Response(status=400, text="リクエストボディが空です")

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(data)
            tmp_path = f.name

        try:
            await play_with_go2rtc(tmp_path, self._go2rtc)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        return web.json_response({"status": "ok"})

    async def _handle_info(self, request: web.Request) -> web.Response:
        if self._camera is None:
            return web.Response(status=503, text="カメラ未接続")
        info = await self._camera.get_device_info()
        return web.json_response(info)

    async def _handle_stream_url(self, request: web.Request) -> web.Response:
        return web.json_response({
            "url": self._go2rtc.api_url,
            "stream": self._go2rtc.stream_name,
        })

    def _make_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/image", self._handle_image)
        app.router.add_post("/audio", self._handle_audio_post)
        app.router.add_get("/direction", self._handle_direction_get)
        app.router.add_post("/direction", self._handle_direction_post)
        app.router.add_get("/info", self._handle_info)
        app.router.add_get("/stream_url", self._handle_stream_url)
        return app

    async def run(self) -> None:
        # HTTP サーバーを先に起動（MCP ハンドシェイクをブロックしないため）
        app = self._make_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "localhost", HTTP_PORT)
        await site.start()
        logger.info("HTTP server listening on http://localhost:%d", HTTP_PORT)

        async def _init_camera() -> None:
            await self._go2rtc.start()
            try:
                config = CameraConfig.from_env()
                self._camera = TapoCamera(config, self._server_config.capture_dir)
                await self._camera.connect()
            except Exception as e:
                logger.warning("Camera connection failed: %s", e)
                self._camera = None

        try:
            async with stdio_server() as (read_stream, write_stream):
                # カメラ初期化はバックグラウンドで実行
                asyncio.create_task(_init_camera())
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
