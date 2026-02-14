"""MCP Server for audio listening and transcription."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .capture import AudioCapture
from .config import ListenConfig
from .transcribe import create_engine

logger = logging.getLogger(__name__)


class AudioListenMCPServer:
    """MCP server that provides audio listening and transcription tools."""

    def __init__(self) -> None:
        self._server = Server("audio-listen-mcp")
        self._config = ListenConfig.from_env()
        self._capture = AudioCapture(self._config)
        self._engine = None  # Lazy-loaded Whisper engine
        self._setup_handlers()

    async def _ensure_engine(self) -> None:
        """Load the Whisper engine on first use (lazy initialization)."""
        if self._engine is None:
            logger.info("Loading Whisper engine (first use)...")
            self._engine = await asyncio.to_thread(
                create_engine,
                self._config.whisper_engine,
                self._config.whisper_model,
            )
            logger.info("Whisper engine ready")

    def _setup_handlers(self) -> None:
        @self._server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="listen",
                    description=(
                        "あなたの耳で周囲の音を聞く。マイクで録音し、"
                        "聞こえた音声を文字に起こして返す。"
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "duration": {
                                "type": "number",
                                "description": (
                                    "録音する秒数(デフォルト: 5、最大: 30)"
                                ),
                                "default": 5,
                                "minimum": 1,
                                "maximum": 30,
                            },
                            "auto_stop": {
                                "type": "boolean",
                                "description": (
                                    "trueの場合、発話終了を検知して自動停止する。"
                                    "falseの場合、duration秒間の固定録音。"
                                ),
                                "default": True,
                            },
                        },
                        "required": [],
                    },
                ),
                Tool(
                    name="listen_raw",
                    description=(
                        "マイクで録音し、WAV データを base64 エンコードして返す。"
                        "書き起こしは行わない。"
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "duration": {
                                "type": "number",
                                "description": (
                                    "録音する秒数(デフォルト: 5、最大: 30)"
                                ),
                                "default": 5,
                                "minimum": 1,
                                "maximum": 30,
                            },
                            "auto_stop": {
                                "type": "boolean",
                                "description": (
                                    "trueの場合、発話終了を検知して自動停止する。"
                                    "falseの場合、duration秒間の固定録音。"
                                ),
                                "default": True,
                            },
                        },
                        "required": [],
                    },
                ),
                Tool(
                    name="transcribe",
                    description=(
                        "指定パスの音声ファイルを Whisper で書き起こす。"
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "audio_path": {
                                "type": "string",
                                "description": "書き起こす音声ファイルのパス",
                            },
                            "language": {
                                "type": "string",
                                "description": "認識言語（デフォルト: ja）",
                                "default": "ja",
                            },
                        },
                        "required": ["audio_path"],
                    },
                ),
                Tool(
                    name="get_audio_devices",
                    description=(
                        "利用可能なオーディオ入力デバイスの一覧を取得する。"
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
                    case "listen":
                        return await self._handle_listen(arguments)
                    case "listen_raw":
                        return await self._handle_listen_raw(arguments)
                    case "transcribe":
                        return await self._handle_transcribe(arguments)
                    case "get_audio_devices":
                        return await self._handle_get_audio_devices()
                    case _:
                        return [TextContent(type="text", text=f"Unknown tool: {name}")]
            except Exception as e:
                logger.exception("Error in tool %s", name)
                return [TextContent(type="text", text=f"Error: {e!s}")]

    def _clamp_duration(self, arguments: dict[str, Any]) -> int:
        """Extract and clamp the duration parameter."""
        duration = arguments.get("duration", self._config.default_duration)
        return max(1, min(int(duration), self._config.max_duration))

    async def _record(self, arguments: dict[str, Any]) -> str:
        """Record audio using either VAD or fixed duration."""
        duration = self._clamp_duration(arguments)
        auto_stop = arguments.get("auto_stop", True)

        if auto_stop:
            return await self._capture.record_with_vad(
                max_duration=duration,
                silence_duration=self._config.vad_silence_duration,
                silence_threshold=self._config.vad_silence_threshold,
            )
        return await self._capture.record(duration)

    async def _handle_listen(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Record audio and return transcription."""
        audio_path = await self._record(arguments)
        try:
            # Transcribe
            await self._ensure_engine()
            transcript = await self._engine.transcribe(
                audio_path, self._config.language
            )
            return [
                TextContent(
                    type="text",
                    text=(
                        f"録音: {os.path.getsize(audio_path) / (self._config.sample_rate * 2):.1f}秒\n"
                        f"--- 聞こえた内容 ---\n{transcript}"
                    ),
                )
            ]
        finally:
            try:
                os.unlink(audio_path)
            except OSError:
                pass

    async def _handle_listen_raw(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Record audio and return raw WAV as base64."""
        audio_path = await self._record(arguments)
        try:
            with open(audio_path, "rb") as f:
                wav_data = f.read()
            encoded = base64.b64encode(wav_data).decode("ascii")
            return [
                TextContent(
                    type="text",
                    text=(
                        f"録音: {os.path.getsize(audio_path) / (self._config.sample_rate * 2):.1f}秒\n"
                        f"フォーマット: WAV (PCM S16LE, {self._config.sample_rate}Hz, mono)\n"
                        f"サイズ: {len(wav_data)} bytes\n"
                        f"--- base64 ---\n{encoded}"
                    ),
                )
            ]
        finally:
            try:
                os.unlink(audio_path)
            except OSError:
                pass

    async def _handle_transcribe(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Transcribe an existing audio file."""
        audio_path = arguments.get("audio_path", "")
        if not audio_path:
            return [TextContent(type="text", text="Error: audio_path is required")]

        if not os.path.isfile(audio_path):
            return [TextContent(type="text", text=f"Error: File not found: {audio_path}")]

        language = arguments.get("language", self._config.language)

        await self._ensure_engine()
        transcript = await self._engine.transcribe(audio_path, language)
        return [
            TextContent(
                type="text",
                text=(
                    f"ファイル: {audio_path}\n"
                    f"--- 書き起こし ---\n{transcript}"
                ),
            )
        ]

    async def _handle_get_audio_devices(self) -> list[TextContent]:
        """List available audio input devices."""
        devices = await self._capture.list_devices()
        if not devices:
            return [
                TextContent(
                    type="text",
                    text="オーディオ入力デバイスが見つかりませんでした。",
                )
            ]

        lines = ["利用可能なオーディオ入力デバイス:"]
        for dev in devices:
            lines.append(f"  [{dev['index']}] {dev['name']}")
        lines.append("")
        lines.append(
            "AUDIO_DEVICE 環境変数にインデックス番号を設定するか、"
            "listen ツールで使用するデバイスを指定できます。"
        )
        return [TextContent(type="text", text="\n".join(lines))]

    async def run(self) -> None:
        """Start the MCP server using stdio transport."""
        async with stdio_server() as (read_stream, write_stream):
            await self._server.run(
                read_stream,
                write_stream,
                self._server.create_initialization_options(),
            )


def main() -> None:
    """Entry point for the audio-listen-mcp server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(AudioListenMCPServer().run())


if __name__ == "__main__":
    main()
