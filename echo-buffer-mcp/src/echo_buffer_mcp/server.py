"""MCP Server for echo buffer — resonant, decaying memory layer."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool

from .buffer import EchoBuffer

logger = logging.getLogger(__name__)


class EchoBufferMCPServer:
    def __init__(self) -> None:
        self._server = Server("echo-buffer-mcp")
        self._buffer = EchoBuffer()
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        @self._server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="echo_add",
                    description=(
                        "応答や思考をエコーバッファに追加する。"
                        "追加された内容は時間とともに減衰しながら次のセッションにも残響する。"
                        "応答の後に要約や印象的な断片を追加するとよい。"
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "バッファに追加するテキスト（応答の要約や思考の断片）",
                            },
                            "strength": {
                                "type": "number",
                                "description": "初期強度（0.0〜1.0、デフォルト1.0）",
                            },
                        },
                        "required": ["content"],
                    },
                ),
                Tool(
                    name="echo_read",
                    description=(
                        "現在のエコーバッファを読む。交換回数ベースの減衰を適用した強度順に返す。"
                        "セッション開始時に呼ぶことで過去の残響を確認できる。"
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "top_k": {
                                "type": "integer",
                                "description": "返すエコーの最大数（デフォルト5）",
                            },
                        },
                        "required": [],
                    },
                ),
                Tool(
                    name="echo_clear",
                    description=(
                        "エコーバッファを全消去する。"
                        "エコーチャンバー状態や思考の固着を断ち切るための緊急ブレーキ。"
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                ),
                Tool(
                    name="echo_freeze",
                    description=(
                        "エコーへの新規追加を一時停止/再開する。"
                        "エコーなしで純粋に対話したいときや、"
                        "特定の話題をエコーに残したくないときに使う。"
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "enabled": {
                                "type": "boolean",
                                "description": "true=凍結（追加停止）、false=再開",
                            },
                        },
                        "required": ["enabled"],
                    },
                ),
                Tool(
                    name="echo_status",
                    description=(
                        "エコーバッファの現在状態を表示する。"
                        "各エコーの強度・経過時間を確認できる。"
                    ),
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
                    case "echo_add":
                        content = arguments.get("content", "")
                        strength = float(arguments.get("strength", 1.0))
                        if not content:
                            return CallToolResult(
                                content=[],
                                structuredContent={"status": "error", "message": "content が空です"},
                                isError=True,
                            )
                        echo_id = self._buffer.add(content, strength)
                        if echo_id == "frozen":
                            return CallToolResult(
                                content=[],
                                structuredContent={"status": "frozen"},
                            )
                        return CallToolResult(
                            content=[],
                            structuredContent={"status": "added", "id": echo_id},
                        )

                    case "echo_read":
                        top_k = int(arguments.get("top_k", 5))
                        echoes = self._buffer.read(top_k)
                        if not echoes:
                            return [TextContent(type="text", text="バッファ空: 有効なエコーはありません。")]
                        lines = [f"エコー ({len(echoes)}件):"]
                        for i, e in enumerate(echoes, 1):
                            age = e['age_steps']
                            age_label = "最新" if age == 0 else f"{age}回前"
                            lines.append(
                                f"[{i}] 強度={e['strength']:.3f} ({age_label})\n"
                                f"    {e['content']}"
                            )
                        return [TextContent(type="text", text="\n".join(lines))]

                    case "echo_clear":
                        count = self._buffer.clear()
                        return CallToolResult(
                            content=[],
                            structuredContent={"status": "cleared", "count": count},
                        )

                    case "echo_freeze":
                        enabled = bool(arguments.get("enabled", True))
                        self._buffer.freeze(enabled)
                        return CallToolResult(
                            content=[],
                            structuredContent={"status": "frozen" if enabled else "resumed"},
                        )

                    case "echo_status":
                        status = self._buffer.status()
                        lines = [
                            "エコーバッファ状態:",
                            f"  総数: {status['total_stored']}件",
                            f"  有効: {status['active']}件 (強度≥0.05)",
                            f"  凍結: {'はい' if status['frozen'] else 'いいえ'}",
                            f"  半減期: {status['half_life_steps']}回交換",
                            "",
                        ]
                        if status["echoes"]:
                            lines.append("有効なエコー:")
                            for e in status["echoes"]:
                                age = e['age_steps']
                                age_label = "最新" if age == 0 else f"{age}回前"
                                lines.append(
                                    f"  [{e['strength']:.3f}] ({age_label}) {e['content']}"
                                )
                        else:
                            lines.append("(有効なエコーなし)")
                        return [TextContent(type="text", text="\n".join(lines))]

                    case _:
                        return [TextContent(type="text", text=f"不明なツール: {name}")]

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
    asyncio.run(EchoBufferMCPServer().run())
