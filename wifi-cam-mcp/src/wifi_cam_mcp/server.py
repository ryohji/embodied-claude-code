"""MCP Server for WiFi Camera Control - thin wrapper over camera-daemon HTTP API."""

from __future__ import annotations

import asyncio
import logging
import os

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import ImageContent, TextContent, Tool

logger = logging.getLogger(__name__)

CAMERA_DAEMON_URL = os.environ.get("CAMERA_DAEMON_URL", "http://localhost:8080")


class WifiCamMCPServer:
    def __init__(self) -> None:
        self._server = Server("wifi-cam-mcp")
        self._client = httpx.AsyncClient(base_url=CAMERA_DAEMON_URL, timeout=30.0)
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        @self._server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="see",
                    description="カメラで今の映像を見る。",
                    inputSchema={"type": "object", "properties": {}, "required": []},
                ),
                Tool(
                    name="look_left",
                    description="カメラを左に向ける（パン）。",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "degrees": {"type": "integer", "description": "移動量（度）、デフォルト30", "default": 30, "minimum": 1, "maximum": 90}
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
                            "degrees": {"type": "integer", "description": "移動量（度）、デフォルト30", "default": 30, "minimum": 1, "maximum": 90}
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
                            "degrees": {"type": "integer", "description": "移動量（度）、デフォルト20", "default": 20, "minimum": 1, "maximum": 90}
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
                            "degrees": {"type": "integer", "description": "移動量（度）、デフォルト20", "default": 20, "minimum": 1, "maximum": 90}
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
            ]

        @self._server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list:
            try:
                match name:
                    case "see":
                        resp = await self._client.get("/image")
                        if resp.status_code == 503:
                            return [TextContent(type="text", text="カメラ未接続")]
                        resp.raise_for_status()
                        data = resp.json()
                        return [
                            ImageContent(type="image", data=data["image"], mimeType=data["mime_type"]),
                            TextContent(type="text", text=f"撮影: {data['timestamp']} ({data['width']}x{data['height']})"),
                        ]

                    case "look_left":
                        degrees = int(arguments.get("degrees", 30))
                        resp = await self._client.get(f"/image?pan=-{degrees}", timeout=30.0)
                        if resp.status_code == 503:
                            return [TextContent(type="text", text="カメラ未接続")]
                        resp.raise_for_status()
                        data = resp.json()
                        return [
                            ImageContent(type="image", data=data["image"], mimeType=data["mime_type"]),
                            TextContent(type="text", text=f"撮影: {data['timestamp']} ({data['width']}x{data['height']})"),
                        ]

                    case "look_right":
                        degrees = int(arguments.get("degrees", 30))
                        resp = await self._client.get(f"/image?pan={degrees}", timeout=30.0)
                        if resp.status_code == 503:
                            return [TextContent(type="text", text="カメラ未接続")]
                        resp.raise_for_status()
                        data = resp.json()
                        return [
                            ImageContent(type="image", data=data["image"], mimeType=data["mime_type"]),
                            TextContent(type="text", text=f"撮影: {data['timestamp']} ({data['width']}x{data['height']})"),
                        ]

                    case "look_up":
                        degrees = int(arguments.get("degrees", 20))
                        resp = await self._client.get(f"/image?tilt={degrees}", timeout=30.0)
                        if resp.status_code == 503:
                            return [TextContent(type="text", text="カメラ未接続")]
                        resp.raise_for_status()
                        data = resp.json()
                        return [
                            ImageContent(type="image", data=data["image"], mimeType=data["mime_type"]),
                            TextContent(type="text", text=f"撮影: {data['timestamp']} ({data['width']}x{data['height']})"),
                        ]

                    case "look_down":
                        degrees = int(arguments.get("degrees", 20))
                        resp = await self._client.get(f"/image?tilt=-{degrees}", timeout=30.0)
                        if resp.status_code == 503:
                            return [TextContent(type="text", text="カメラ未接続")]
                        resp.raise_for_status()
                        data = resp.json()
                        return [
                            ImageContent(type="image", data=data["image"], mimeType=data["mime_type"]),
                            TextContent(type="text", text=f"撮影: {data['timestamp']} ({data['width']}x{data['height']})"),
                        ]

                    case "look_around":
                        shots = [
                            ("左", "/image?pan=-45"),
                            ("中央", "/image"),
                            ("右", "/image?pan=45"),
                            ("上", "/image?tilt=20"),
                        ]
                        contents = []
                        for label, path in shots:
                            r = await self._client.get(path, timeout=30.0)
                            if r.status_code == 503:
                                return [TextContent(type="text", text="カメラ未接続")]
                            r.raise_for_status()
                            d = r.json()
                            contents.append(TextContent(type="text", text=f"--- {label} ---"))
                            contents.append(ImageContent(type="image", data=d["image"], mimeType=d["mime_type"]))
                        contents.append(TextContent(type="text", text=f"{len(shots)}枚撮影完了"))
                        return contents

                    case "camera_info":
                        resp = await self._client.get("/info")
                        if resp.status_code == 503:
                            return [TextContent(type="text", text="カメラ未接続")]
                        resp.raise_for_status()
                        import json
                        return [TextContent(type="text", text=f"カメラ情報:\n{json.dumps(resp.json(), indent=2, ensure_ascii=False)}")]

                    case _:
                        return [TextContent(type="text", text=f"不明なツール: {name}")]

            except httpx.ConnectError:
                return [TextContent(type="text", text=f"camera-daemon に接続できません ({CAMERA_DAEMON_URL})")]
            except Exception as e:
                logger.exception("Error in tool %s", name)
                return [TextContent(type="text", text=f"エラー: {e!s}")]

    async def run(self) -> None:
        await self._client.__aenter__()
        try:
            async with stdio_server() as (read_stream, write_stream):
                await self._server.run(
                    read_stream,
                    write_stream,
                    self._server.create_initialization_options(),
                )
        finally:
            await self._client.__aexit__(None, None, None)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(WifiCamMCPServer().run())
