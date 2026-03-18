"""MCP Server that manages go2rtc lifecycle and provides HTTP API for camera access."""

from __future__ import annotations

import asyncio
import logging
import os

from aiohttp import web
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool

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

    async def _handle_image(self, request: web.Request) -> web.Response:
        if self._camera is None:
            return web.Response(status=503, text="カメラ未接続")
        pan_str = request.rel_url.query.get("pan")
        tilt_str = request.rel_url.query.get("tilt")
        if pan_str is not None:
            pan = int(pan_str)
            if pan > 0:
                await self._camera.pan_right(abs(pan))
            elif pan < 0:
                await self._camera.pan_left(abs(pan))
        if tilt_str is not None:
            tilt = int(tilt_str)
            if tilt > 0:
                await self._camera.tilt_up(abs(tilt))
            elif tilt < 0:
                await self._camera.tilt_down(abs(tilt))
        result = await self._camera.capture_image()
        return web.json_response({
            "image": result.image_base64,
            "mime_type": "image/jpeg",
            "width": result.width,
            "height": result.height,
            "timestamp": result.timestamp,
        })

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
        app.router.add_get("/info", self._handle_info)
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
