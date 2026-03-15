"""MCP Server that manages go2rtc lifecycle for Tapo camera streaming."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, ImageContent, TextContent, Tool

from .camera import TapoCamera
from .config import CameraConfig, ServerConfig
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
        self._server_config = ServerConfig.from_env()
        self._camera: TapoCamera | None = None
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
                Tool(
                    name="see",
                    description="カメラで現在の映像を撮影して返す。",
                    inputSchema={"type": "object", "properties": {}, "required": []},
                ),
                Tool(
                    name="look_left",
                    description="カメラを左に向ける（パン）。",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "degrees": {
                                "type": "integer",
                                "description": "移動量（度）、デフォルト30",
                                "default": 30,
                                "minimum": 1,
                                "maximum": 90,
                            }
                        },
                        "required": [],
                    },
                ),
                Tool(
                    name="look_right",
                    description="カメラを右に向ける（パン）。",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "degrees": {
                                "type": "integer",
                                "description": "移動量（度）、デフォルト30",
                                "default": 30,
                                "minimum": 1,
                                "maximum": 90,
                            }
                        },
                        "required": [],
                    },
                ),
                Tool(
                    name="look_up",
                    description="カメラを上に向ける（チルト）。",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "degrees": {
                                "type": "integer",
                                "description": "移動量（度）、デフォルト20",
                                "default": 20,
                                "minimum": 1,
                                "maximum": 90,
                            }
                        },
                        "required": [],
                    },
                ),
                Tool(
                    name="look_down",
                    description="カメラを下に向ける（チルト）。",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "degrees": {
                                "type": "integer",
                                "description": "移動量（度）、デフォルト20",
                                "default": 20,
                                "minimum": 1,
                                "maximum": 90,
                            }
                        },
                        "required": [],
                    },
                ),
                Tool(
                    name="look_around",
                    description="複数の角度から部屋を撮影して返す。",
                    inputSchema={"type": "object", "properties": {}, "required": []},
                ),
                Tool(
                    name="camera_info",
                    description="カメラデバイス情報を取得する。",
                    inputSchema={"type": "object", "properties": {}, "required": []},
                ),
                Tool(
                    name="camera_presets",
                    description="保存済みカメラプリセット一覧を取得する。",
                    inputSchema={"type": "object", "properties": {}, "required": []},
                ),
                Tool(
                    name="camera_go_to_preset",
                    description="指定したプリセットにカメラを移動する。",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "preset_id": {
                                "type": "string",
                                "description": "プリセットID",
                            }
                        },
                        "required": ["preset_id"],
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

                    case "see":
                        if self._camera is None:
                            return [TextContent(type="text", text="カメラ未接続")]
                        result = await self._camera.capture_image()
                        return [
                            ImageContent(type="image", data=result.image_base64, mimeType="image/jpeg"),
                            TextContent(type="text", text=f"撮影: {result.timestamp} ({result.width}x{result.height})"),
                        ]

                    case "look_left":
                        if self._camera is None:
                            return [TextContent(type="text", text="カメラ未接続")]
                        degrees = arguments.get("degrees", 30)
                        await self._camera.pan_left(degrees)
                        return CallToolResult(content=[], structuredContent={"status": "moved", "direction": "left", "degrees": degrees})

                    case "look_right":
                        if self._camera is None:
                            return [TextContent(type="text", text="カメラ未接続")]
                        degrees = arguments.get("degrees", 30)
                        await self._camera.pan_right(degrees)
                        return CallToolResult(content=[], structuredContent={"status": "moved", "direction": "right", "degrees": degrees})

                    case "look_up":
                        if self._camera is None:
                            return [TextContent(type="text", text="カメラ未接続")]
                        degrees = arguments.get("degrees", 20)
                        await self._camera.tilt_up(degrees)
                        return CallToolResult(content=[], structuredContent={"status": "moved", "direction": "up", "degrees": degrees})

                    case "look_down":
                        if self._camera is None:
                            return [TextContent(type="text", text="カメラ未接続")]
                        degrees = arguments.get("degrees", 20)
                        await self._camera.tilt_down(degrees)
                        return CallToolResult(content=[], structuredContent={"status": "moved", "direction": "down", "degrees": degrees})

                    case "look_around":
                        if self._camera is None:
                            return [TextContent(type="text", text="カメラ未接続")]
                        captures = await self._camera.look_around()
                        contents = []
                        directions = ["Center", "Left", "Right", "Up"]
                        for i, capture in enumerate(captures):
                            direction = directions[i] if i < len(directions) else f"Angle {i}"
                            contents.append(TextContent(type="text", text=f"--- {direction} ---"))
                            contents.append(ImageContent(type="image", data=capture.image_base64, mimeType="image/jpeg"))
                        contents.append(TextContent(type="text", text=f"{len(captures)}枚撮影完了"))
                        return contents

                    case "camera_info":
                        if self._camera is None:
                            return [TextContent(type="text", text="カメラ未接続")]
                        info = await self._camera.get_device_info()
                        return [TextContent(type="text", text=f"カメラ情報:\n{json.dumps(info, indent=2, ensure_ascii=False)}")]

                    case "camera_presets":
                        if self._camera is None:
                            return [TextContent(type="text", text="カメラ未接続")]
                        presets = await self._camera.get_presets()
                        return [TextContent(type="text", text=f"プリセット:\n{json.dumps(presets, indent=2, ensure_ascii=False)}")]

                    case "camera_go_to_preset":
                        if self._camera is None:
                            return [TextContent(type="text", text="カメラ未接続")]
                        preset_id = arguments.get("preset_id", "")
                        await self._camera.go_to_preset(preset_id)
                        return CallToolResult(content=[], structuredContent={"status": "moved", "preset_id": preset_id})

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
            config = CameraConfig.from_env()
            self._camera = TapoCamera(config, self._server_config.capture_dir)
            await self._camera.connect()
        except Exception as e:
            logger.warning("Camera connection failed: %s", e)
            self._camera = None
        try:
            async with stdio_server() as (read_stream, write_stream):
                await self._server.run(
                    read_stream,
                    write_stream,
                    self._server.create_initialization_options(),
                )
        finally:
            self._go2rtc.stop()
            if self._camera is not None:
                await self._camera.disconnect()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(CameraDaemonMCPServer().run())
