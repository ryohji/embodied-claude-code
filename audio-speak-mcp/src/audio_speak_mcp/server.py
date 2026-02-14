"""MCP Server for text-to-speech output."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .config import SpeakConfig
from .tts import TTSEngine, create_engine

logger = logging.getLogger(__name__)


class AudioSpeakMCPServer:
    def __init__(self) -> None:
        self._server = Server("audio-speak-mcp")
        self._config = SpeakConfig.from_env()
        self._engine: TTSEngine | None = None
        self._setup_handlers()

    def _ensure_engine(self) -> TTSEngine:
        if self._engine is None:
            self._engine = create_engine(self._config)
        return self._engine

    def _setup_handlers(self) -> None:
        @self._server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="say",
                    description=(
                        "テキストを声に出して話す。ユーザーに音声で伝えたいときに使う。"
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
                                "description": "使用する音声名（省略時はデフォルト音声）",
                            },
                            "rate": {
                                "type": "integer",
                                "description": "発話速度（words per minute）。省略時はデフォルト速度",
                            },
                        },
                        "required": ["text"],
                    },
                ),
                Tool(
                    name="get_voices",
                    description="利用可能な音声の一覧を取得する。",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                ),
            ]

        @self._server.call_tool()
        async def call_tool(
            name: str, arguments: dict[str, Any]
        ) -> list[TextContent]:
            try:
                match name:
                    case "say":
                        text = arguments.get("text", "")
                        if not text:
                            return [TextContent(
                                type="text",
                                text="エラー: テキストが空です。",
                            )]
                        voice = arguments.get("voice")
                        rate = arguments.get("rate")
                        engine = self._ensure_engine()
                        result = await engine.say(text, voice=voice, rate=rate)
                        return [TextContent(type="text", text=result)]

                    case "get_voices":
                        engine = self._ensure_engine()
                        voices = await engine.list_voices()
                        if not voices:
                            return [TextContent(
                                type="text",
                                text="利用可能な音声が見つかりませんでした。",
                            )]
                        lines = [
                            f"利用可能な音声（エンジン: {self._config.tts_engine}）:",
                            "",
                        ]
                        for v in voices:
                            parts = [v.get("name", "")]
                            if v.get("language"):
                                parts.append(f"[{v['language']}]")
                            if v.get("voice_id"):
                                parts.append(f"(id: {v['voice_id']})")
                            if v.get("sample"):
                                parts.append(f"— {v['sample']}")
                            lines.append("  ".join(parts))
                        return [TextContent(type="text", text="\n".join(lines))]

                    case _:
                        return [TextContent(
                            type="text",
                            text=f"不明なツール: {name}",
                        )]
            except Exception as e:
                logger.exception("Error in tool %s", name)
                return [TextContent(type="text", text=f"エラー: {e!s}")]

    async def run(self) -> None:
        async with stdio_server() as (read_stream, write_stream):
            await self._server.run(
                read_stream,
                write_stream,
                self._server.create_initialization_options(),
            )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(AudioSpeakMCPServer().run())
