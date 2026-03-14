"""MCP Server that manages go2rtc lifecycle for Tapo camera streaming."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool

from .go2rtc import Go2RTCProcess

logger = logging.getLogger(__name__)


class CameraDaemonMCPServer:
    def __init__(self) -> None:
        self._server = Server("camera-daemon")
        self._go2rtc = Go2RTCProcess(
            camera_host=os.environ.get("TAPO_CAMERA_HOST", ""),
            username=os.environ.get("TAPO_USERNAME", ""),
            password=os.environ.get("TAPO_PASSWORD", ""),
            cloud_password=os.environ.get("TAPO_CLOUD_PASSWORD", ""),
        )
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        @self._server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="get_stream_url",
                    description=(
                        "go2rtc の API URL とストリーム名を返す。"
                        "他の MCP が go2rtc エンドポイントを知るための窓口。"
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                ),
                Tool(
                    name="say_to_camera",
                    description=(
                        "テキストをカメラのスピーカーから発話する。"
                        "macOS say コマンドで音声生成し、go2rtc バックチャンネル経由でカメラスピーカーに送出する。"
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "発話するテキスト",
                            },
                            "voice": {
                                "type": "string",
                                "description": "音声名（省略時は Kyoko）",
                            },
                            "rate": {
                                "type": "integer",
                                "description": "発話速度 WPM（省略時はデフォルト）",
                            },
                        },
                        "required": ["text"],
                    },
                ),
            ]

        @self._server.call_tool()
        async def call_tool(
            name: str, arguments: dict[str, Any]
        ) -> list[TextContent]:
            try:
                match name:
                    case "get_stream_url":
                        return CallToolResult(
                            content=[],
                            structuredContent={
                                "url": self._go2rtc.api_url,
                                "stream": self._go2rtc.stream_name,
                            },
                        )
                    case "say_to_camera":
                        text = arguments.get("text", "")
                        voice = arguments.get("voice", "Kyoko")
                        rate = arguments.get("rate")
                        if not shutil.which("ffmpeg"):
                            return [TextContent(type="text", text="ffmpeg が見つかりません")]
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
                                return [TextContent(type="text", text=f"say エラー: {stderr.decode().strip()}")]
                            from .playback import play_with_go2rtc
                            await play_with_go2rtc(audio_file, self._go2rtc)
                            return CallToolResult(
                                content=[],
                                structuredContent={"status": "spoken", "text": text, "voice": voice},
                            )
                        finally:
                            Path(text_file).unlink(missing_ok=True)
                            Path(audio_file).unlink(missing_ok=True)

                    case _:
                        return [TextContent(
                            type="text",
                            text=f"不明なツール: {name}",
                        )]
            except Exception as e:
                logger.exception("Error in tool %s", name)
                return [TextContent(type="text", text=f"エラー: {e!s}")]

    async def run(self) -> None:
        await self._go2rtc.start()
        try:
            async with stdio_server() as (read_stream, write_stream):
                await self._server.run(
                    read_stream,
                    write_stream,
                    self._server.create_initialization_options(),
                )
        finally:
            self._go2rtc.stop()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(CameraDaemonMCPServer().run())
