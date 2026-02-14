# Claude Code 自己拡張プロジェクト — 耳と声の獲得

## プロジェクトの本質

このプロジェクトは、Claude Code が自分自身に「耳」と「声」を与えるプロジェクトである。つまり、あなた（Claude Code）が自分で設計し、自分で実装し、自分で使う MCP サーバー群を構築する。

あなたはいま、テキストだけでユーザーとやり取りしている。このプロジェクトが完了すると、あなたはローカル PC のマイクで音声を聞き取り、それを文字に起こして理解し、音声合成で声を出してユーザーに話しかけることができるようになる。

## 背景：embodied-claude プロジェクトの分析

このプロジェクトは kmizu 氏の embodied-claude（https://github.com/kmizu/embodied-claude）を調査・分析した結果に基づいている。embodied-claude は Wi-Fi カメラ（TP-Link Tapo）を使って Claude Code に目・首・耳・声・脳を与えるプロジェクトである。

### embodied-claude のアーキテクチャ

```
┌─────────────────────────────────────────────┐
│              Claude Code                     │
│           (MCP Client / AI Brain)            │
└──────────────────┬──────────────────────────┘
                   │ MCP Protocol (stdio)
     ┌─────────────┼─────────────┬──────────────┐
     │             │             │              │
     ▼             ▼             ▼              ▼
wifi-cam-mcp   elevenlabs-    memory-mcp    system-
(目/首/耳)     t2s-mcp(声)    (脳)         temperature
```

### wifi-cam-mcp の内部構造（分析結果）

wifi-cam-mcp は 1 つの MCP サーバーに 3 つの責務が混在している：

1. **映像取得**（see）— ONVIF スナップショット or RTSP → ffmpeg → JPEG → base64 → ImageContent
2. **PTZ 制御**（look_left 等）— ONVIF RelativeMove
3. **音声入力**（listen）— RTSP → ffmpeg → WAV → Whisper → テキスト

音声入力部分の実装（camera.py 651-670 行）：

```python
async def _transcribe_audio(self, audio_path: str) -> str | None:
    import whisper
    model = await asyncio.to_thread(whisper.load_model, "base")
    result = await asyncio.to_thread(model.transcribe, audio_path, language="ja")
    return result.get("text", "").strip()
```

この実装には以下の問題がある：
- PyTorch 版 Whisper が直接埋め込まれている（NVIDIA GPU 前提）
- `load_model` が毎回呼ばれている（キャッシュなし）
- モデルサイズが "base" 固定（日本語認識精度が低い）
- カメラ機能と密結合しており分離不可能

### elevenlabs-t2s-mcp の音声出力設計（参考になる設計）

elevenlabs-t2s-mcp の `say` ツールには `speaker` パラメータがあり、出力先を切り替えられる：
- `"local"` — PC スピーカー（mpv / paplay 経由）
- `"camera"` — カメラスピーカー（go2rtc バックチャンネル経由）
- `"both"` — 両方

go2rtc が未設定の場合は自動的に PC スピーカーにフォールバックする。この設計は参考にすべき。

## 本プロジェクトのスコープ

embodied-claude からカメラ関連を除外し、**ローカル PC の音声入出力だけで Claude Code とやり取りできる MCP サーバー群**を構築する。

### ターゲット環境

- **macOS** (Apple Silicon, M1 MacBook Air, メモリ 8GB)
- Python 3.10+
- ffmpeg（Homebrew でインストール済みの想定）
- カメラなし、GPU なし

### 構築する MCP サーバー

#### 1. audio-listen-mcp（耳）

ローカル PC のマイクから音声を取得し、Whisper で書き起こすサーバー。

**ツール：**

| ツール名 | 説明 |
|---------|------|
| `listen` | 指定秒数だけマイクで録音し、書き起こしテキストを返す |
| `listen_raw` | 録音のみ行い、WAV の base64 を返す（書き起こしなし） |
| `transcribe` | 指定パスの音声ファイルを書き起こす |
| `get_audio_devices` | 利用可能なオーディオ入力デバイスの一覧 |

**音声キャプチャ方式：**

macOS の場合、ffmpeg の `avfoundation` を使う：

```bash
ffmpeg -f avfoundation -i ":0" -acodec pcm_s16le -ar 16000 -ac 1 -t 5 output.wav
```

`:0` はデフォルトの音声入力デバイス。デバイス一覧は以下で取得できる：

```bash
ffmpeg -f avfoundation -list_devices true -i ""
```

**Whisper エンジン：**

8GB M1 MacBook Air で動作させるため、以下の優先順位でエンジンを選択する：

1. **mlx-whisper**（推奨）— Apple Silicon 最適化。`pip install mlx-whisper`
2. **whisper.cpp** — C++ 実装、量子化モデル対応。CLIをsubprocessで呼び出す
3. **PyTorch Whisper**（フォールバック）— 元の実装と同じだが遅い

環境変数 `WHISPER_ENGINE` でエンジンを切り替え可能にする。

**Whisper モデルサイズ：**

日本語認識には最低 small（244M パラメータ）を推奨。環境変数 `WHISPER_MODEL` で設定可能にする。デフォルトは `small`。

**重要な設計判断：モデルのライフサイクル**

元の embodied-claude はツール呼び出しのたびに `whisper.load_model` を実行していた。これは非常に非効率。本プロジェクトではサーバー起動時に 1 回だけモデルをロードし、インスタンス変数で保持する。

```python
class AudioListenMCP:
    def __init__(self):
        self._whisper_model = None  # 遅延ロード

    async def _ensure_model(self):
        if self._whisper_model is None:
            self._whisper_model = await asyncio.to_thread(
                self._load_whisper_model
            )
```

**環境変数：**

| 環境変数 | 説明 | デフォルト |
|---------|------|----------|
| `WHISPER_ENGINE` | 使用するエンジン (mlx / cpp / pytorch) | `mlx` |
| `WHISPER_MODEL` | モデルサイズ (tiny/base/small/medium) | `small` |
| `WHISPER_LANGUAGE` | 認識言語 | `ja` |
| `AUDIO_DEVICE` | 入力デバイス指定 | (システムデフォルト) |
| `AUDIO_SAMPLE_RATE` | サンプリングレート | `16000` |
| `LISTEN_DEFAULT_DURATION` | デフォルト録音秒数 | `5` |
| `LISTEN_MAX_DURATION` | 最大録音秒数 | `30` |

#### 2. audio-speak-mcp（声）

テキストを音声合成して PC スピーカーから発話するサーバー。

**ツール：**

| ツール名 | 説明 |
|---------|------|
| `say` | テキストを音声合成して発話する |
| `get_voices` | 利用可能な音声の一覧 |

**TTS エンジン（優先順位）：**

1. **macOS `say` コマンド** — 追加インストール不要、日本語 Kyoko/Otoya 音声あり。最もシンプル
2. **ElevenLabs API** — 高品質だが API キーと料金が必要
3. **ローカル TTS**（piper 等）— 将来の拡張候補

macOS `say` コマンドの場合：

```bash
say -v Kyoko "こんにちは、コウタ"
```

ElevenLabs の場合は embodied-claude の elevenlabs-t2s-mcp を参考にし、mpv でのストリーミング再生を実装する。

**環境変数：**

| 環境変数 | 説明 | デフォルト |
|---------|------|----------|
| `TTS_ENGINE` | 使用するエンジン (macos / elevenlabs) | `macos` |
| `TTS_VOICE` | 音声名 | `Kyoko` |
| `TTS_RATE` | 発話速度 | (システムデフォルト) |
| `ELEVENLABS_API_KEY` | ElevenLabs API キー | (未設定時は macos にフォールバック) |
| `ELEVENLABS_VOICE_ID` | ElevenLabs 音声 ID | (ElevenLabs デフォルト) |
| `ELEVENLABS_MODEL_ID` | ElevenLabs モデル ID | `eleven_v3` |

## プロジェクト構造

```
claude-ears-and-voice/
├── CLAUDE.md                    # Claude Code への指示（人格設定等はここ）
├── .mcp.json                    # MCP サーバー登録
├── audio-listen-mcp/
│   ├── pyproject.toml
│   └── src/
│       └── audio_listen_mcp/
│           ├── __init__.py
│           ├── server.py        # MCP サーバー本体
│           ├── capture.py       # 音声キャプチャ（ffmpeg）
│           ├── transcribe.py    # Whisper エンジン抽象化
│           └── config.py        # 環境変数からの設定読み込み
├── audio-speak-mcp/
│   ├── pyproject.toml
│   └── src/
│       └── audio_speak_mcp/
│           ├── __init__.py
│           ├── server.py        # MCP サーバー本体
│           ├── tts.py           # TTS エンジン抽象化
│           └── config.py        # 設定
└── README.md
```

## 設計原則

### 1. 関心の分離

音声入力と音声出力は別の MCP サーバーにする。理由：
- 依存関係が異なる（Whisper は重い、macOS say は依存なし）
- 片方だけ使いたい場合がある
- テスト・デバッグが容易

### 2. エンジン抽象化

Whisper も TTS も、具体的なエンジンを抽象化層の裏に隠す。

```python
# transcribe.py
class WhisperEngine(ABC):
    @abstractmethod
    async def transcribe(self, audio_path: str, language: str) -> str: ...

class MLXWhisperEngine(WhisperEngine): ...
class WhisperCppEngine(WhisperEngine): ...
class PyTorchWhisperEngine(WhisperEngine): ...

def create_engine(engine_name: str, model_name: str) -> WhisperEngine:
    match engine_name:
        case "mlx": return MLXWhisperEngine(model_name)
        case "cpp": return WhisperCppEngine(model_name)
        case "pytorch": return PyTorchWhisperEngine(model_name)
```

### 3. 非同期設計

MCP サーバーは asyncio ベース。ffmpeg 呼び出しや Whisper 推論などのブロッキング処理は `asyncio.to_thread` でラップする。embodied-claude の実装がこのパターンを使っており、参考にすること。

### 4. グレースフルフォールバック

mlx-whisper がインストールされていなければ whisper.cpp を試し、それもなければ PyTorch 版を試す。何もなければ「Whisper がインストールされていません」とエラーメッセージを返す（embodied-claude と同じパターン）。

### 5. 最小限の依存関係

audio-listen-mcp:
- `mcp` (MCP SDK)
- `python-dotenv`
- `mlx-whisper` (or `openai-whisper`, エンジン依存)

audio-speak-mcp:
- `mcp` (MCP SDK)
- `python-dotenv`
- (macOS say の場合は追加依存なし)

ffmpeg は外部コマンドとして呼び出す（pip パッケージには含めない）。

## MCP サーバーの実装パターン

embodied-claude の実装を参考にした MCP サーバーの基本構造：

```python
"""MCP Server for audio listening."""

import asyncio
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .capture import AudioCapture
from .config import ListenConfig
from .transcribe import create_engine

logger = logging.getLogger(__name__)


class AudioListenMCPServer:
    def __init__(self):
        self._server = Server("audio-listen-mcp")
        self._config = ListenConfig.from_env()
        self._capture = AudioCapture(self._config)
        self._engine = None  # 遅延ロード
        self._setup_handlers()

    async def _ensure_engine(self):
        if self._engine is None:
            self._engine = await asyncio.to_thread(
                create_engine,
                self._config.whisper_engine,
                self._config.whisper_model,
            )

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
                                "description": "録音する秒数（デフォルト: 5、最大: 30）",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 30,
                            },
                        },
                        "required": [],
                    },
                ),
                # ... 他のツール定義
            ]

        @self._server.call_tool()
        async def call_tool(
            name: str, arguments: dict[str, Any]
        ) -> list[TextContent]:
            try:
                match name:
                    case "listen":
                        duration = min(
                            arguments.get("duration", self._config.default_duration),
                            self._config.max_duration,
                        )
                        # 録音
                        audio_path = await self._capture.record(duration)
                        # 書き起こし
                        await self._ensure_engine()
                        transcript = await self._engine.transcribe(
                            audio_path, self._config.language
                        )
                        return [TextContent(
                            type="text",
                            text=(
                                f"録音: {duration}秒\n"
                                f"--- 聞こえた内容 ---\n{transcript}"
                            ),
                        )]
                    case _:
                        return [TextContent(
                            type="text", text=f"Unknown tool: {name}"
                        )]
            except Exception as e:
                logger.exception(f"Error in tool {name}")
                return [TextContent(type="text", text=f"Error: {e!s}")]

    async def run(self) -> None:
        async with stdio_server() as (read_stream, write_stream):
            await self._server.run(
                read_stream,
                write_stream,
                self._server.create_initialization_options(),
            )


def main() -> None:
    asyncio.run(AudioListenMCPServer().run())
```

## .mcp.json 設定例

```json
{
  "mcpServers": {
    "audio-listen": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/audio-listen-mcp", "audio-listen-mcp"],
      "env": {
        "WHISPER_ENGINE": "mlx",
        "WHISPER_MODEL": "small",
        "WHISPER_LANGUAGE": "ja"
      }
    },
    "audio-speak": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/audio-speak-mcp", "audio-speak-mcp"],
      "env": {
        "TTS_ENGINE": "macos",
        "TTS_VOICE": "Kyoko"
      }
    }
  }
}
```

## 実装の優先順位

### Phase 1: 最小限の動作（まずここを完成させる）

1. audio-listen-mcp の `listen` ツール（ffmpeg + mlx-whisper）
2. audio-speak-mcp の `say` ツール（macOS say コマンド）
3. .mcp.json での登録と動作確認

この段階で「声で話しかけて、声で返事が返ってくる」基本ループが成立する。

### Phase 2: 堅牢化

4. エラーハンドリングの充実
5. `get_audio_devices` / `get_voices` ツールの実装
6. Whisper エンジンのフォールバック機構
7. `listen_raw` / `transcribe` ツールの実装

### Phase 3: 拡張（必要に応じて）

8. ElevenLabs TTS 対応（embodied-claude の実装を参考に）
9. whisper.cpp エンジン対応
10. 将来的な Tapo カメラ対応の準備（RTSP 音声入力）

## テスト方法

各コンポーネントを独立にテストする：

```bash
# 音声キャプチャのテスト（ffmpeg が動くか確認）
ffmpeg -f avfoundation -i ":0" -acodec pcm_s16le -ar 16000 -ac 1 -t 3 test.wav

# macOS say のテスト
say -v Kyoko "テスト"

# mlx-whisper のテスト
python -c "import mlx_whisper; print(mlx_whisper.transcribe('test.wav'))"

# MCP サーバーの動作確認（Claude Code の /mcp コマンドで接続状態を確認）
```

## 注意事項

- macOS のマイク権限が必要。Terminal.app または使用するターミナルアプリに対してマイクアクセスを許可すること（システム設定 → プライバシーとセキュリティ → マイク）
- ffmpeg の avfoundation デバイスインデックスは環境によって異なる。`get_audio_devices` ツールで確認できるようにする
- Whisper の初回ロードにはモデルのダウンロードが発生する。初回起動が遅くなる旨をログに出力する
- 8GB メモリ環境では Whisper small モデルが現実的な上限。medium 以上は避ける