# 身体性実験プロジェクト — Claude Code に耳・声・目を与える

Claude Code に感覚器官を与え、環境との結合から何が生じるかを観察する実験プロジェクト。

[kmizu/embodied-claude](https://github.com/kmizu/embodied-claude) をベースに、macOS ローカル環境向けに再設計・拡張したもの。

## 必要なもの

- macOS (Apple Silicon)
- [Homebrew](https://brew.sh/)

Python や各種依存パッケージは uv が自動でインストールするため、事前準備は不要。

## 構成

```
.
├── audio-listen-mcp/    # 耳：マイク録音 + Whisper 書き起こし
├── audio-speak-mcp/     # 声：テキスト音声合成（PC / Tapo カメラスピーカー）
├── wifi-cam-mcp/        # 目・首：Tapo カメラ映像取得 + パン・チルト制御
├── memory-mcp/          # 記憶：ChromaDB ベースの外部記憶
├── echo-buffer-mcp/     # 残響：セッション間を持続する内部エコーバッファ
├── sample.mcp.json      # MCP サーバー設定のテンプレート（認証情報なし）
└── CLAUDE.md            # Claude Code への設計指示書
```

> **Note**: `.mcp.json`（実際の設定ファイル）は認証情報を含むため git 管理対象外。
> `sample.mcp.json` をコピーして作成し、認証情報を埋める。

## セットアップ

### 1. ffmpeg と uv のインストール

```bash
brew install ffmpeg uv mpv
```

### 2. .mcp.json の作成

```bash
cp sample.mcp.json .mcp.json
```

`sample.mcp.json` はテンプレートであり、以下の箇所を自分の環境に合わせて書き換える:

- `--directory` の `/path/to/embodied-claude-code/` を実際のプロジェクトの絶対パスに置換（全サーバー分）
- `YOUR_TAPO_USERNAME` / `YOUR_TAPO_LOCAL_PASSWORD` / `YOUR_TAPO_CLOUD_PASSWORD` を実際の認証情報に置換

各認証情報の意味は後述の「環境変数」を参照。`.mcp.json` は認証情報を含むため git 管理対象外になっている（`.gitignore` に登録済み）。

### 3. マイク権限の許可

macOS のシステム設定 → プライバシーとセキュリティ → マイク で、使用するターミナルアプリにアクセスを許可する。

### 4. 動作確認

Claude Code を起動し、MCP サーバーの接続状態を確認する。

```
/mcp
```

接続済み（connected）と表示されれば準備完了。初回起動時は依存パッケージのインストールと Whisper モデル（small: 約 500MB）のダウンロードが行われる。

## 使い方

### 会話モード

Claude Code に「会話モード」と送ると、音声による対話ループに入る。

```
あなた: 会話モード
Claude: (スピーカー)「会話モードを開始します。」
Claude: (マイクで録音)
あなた: (声で) 今日はいい天気ですね
Claude: (スピーカー)「そうですね。」
```

「終わり」「おしまい」「ストップ」等で通常モードに戻る。

### カメラスピーカーに声を出す

`say` ツールの `output` パラメータで出力先を切り替える。

```
say("こんにちは", output="camera")   # Tapo カメラのスピーカーから出力
say("こんにちは", output="pc")       # PC スピーカーから出力（デフォルト）
```

カメラ出力には `TAPO_CAMERA_HOST` / `TAPO_USERNAME` / `TAPO_PASSWORD` / `TAPO_CLOUD_PASSWORD` の設定が必要。

## 利用可能なツール

### audio-listen-mcp（耳）

| ツール | 説明 |
|--------|------|
| `listen` | マイクで録音し、書き起こしテキストを返す（VAD で自動停止） |
| `listen_raw` | 録音のみ行い、WAV の base64 を返す |
| `transcribe` | 指定パスの音声ファイルを Whisper で書き起こす |
| `get_audio_devices` | 利用可能な入力デバイス一覧を取得 |

### audio-speak-mcp（声）

| ツール | 説明 |
|--------|------|
| `say` | テキストを音声合成して発話する。`output="camera"` で Tapo スピーカーにも出力可 |
| `get_voices` | 利用可能な音声の一覧を取得 |

### wifi-cam-mcp（目・首）

| ツール | 説明 |
|--------|------|
| `see` | カメラで静止画を撮影して返す |
| `look_left/right/up/down` | カメラをパン・チルト操作する |
| `look_around` | 複数方向に向けて画像を撮影する |
| `camera_presets` | プリセット位置一覧を取得 |
| `camera_go_to_preset` | プリセット位置に移動 |
| `listen` | カメラのマイクから音声を録音する |
| `camera_info` | カメラのデバイス情報を取得 |

## 環境変数

### audio-listen-mcp

| 変数 | 説明 | デフォルト |
|------|------|-----------|
| `WHISPER_ENGINE` | Whisper エンジン (`mlx` / `pytorch`) | `mlx` |
| `WHISPER_MODEL` | モデルサイズ (`tiny` / `base` / `small` / `medium`) | `small` |
| `WHISPER_LANGUAGE` | 認識言語 | `ja` |
| `AUDIO_DEVICE` | 入力デバイスインデックス | システムデフォルト |

### audio-speak-mcp

| 変数 | 説明 | デフォルト |
|------|------|-----------|
| `TTS_ENGINE` | TTS エンジン (`macos` / `kokoro` / `elevenlabs`) | `macos` |
| `TTS_VOICE` | macOS の音声名 | `Kyoko` |
| `TTS_RATE` | 発話速度 (wpm) | システムデフォルト |
| `KOKORO_VOICE` | Kokoro 音声プリセット | `jf_alpha` |
| `KOKORO_MODEL_ID` | Kokoro モデル ID | `mlx-community/Kokoro-82M-bf16` |
| `KOKORO_SPEED` | Kokoro 発話速度倍率 | `1.0` |
| `KOKORO_LANG_CODE` | Kokoro 言語コード (`j` / `a` / `b`) | `j` |
| `ELEVENLABS_API_KEY` | ElevenLabs API キー | 未設定時は macOS にフォールバック |
| `ELEVENLABS_VOICE_ID` | ElevenLabs 音声 ID | — |
| `ELEVENLABS_MODEL_ID` | ElevenLabs モデル ID | `eleven_v3` |
| `TAPO_CAMERA_HOST` | カメラの IP アドレス | — |
| `TAPO_USERNAME` | カメラのローカルアカウント名（RTSP / ONVIF 認証用） | — |
| `TAPO_PASSWORD` | カメラのローカルアカウントパスワード（RTSP / ONVIF 認証用） | — |
| `TAPO_CLOUD_PASSWORD` | TP-Link クラウドパスワード（`tapo://` バックチャンネル音声出力用） | — |
| `GO2RTC_API_URL` | go2rtc API URL | `http://localhost:1984` |
| `GO2RTC_STREAM_NAME` | go2rtc ストリーム名 | `camera` |

> **認証情報が 2 種類ある理由**: Tapo C210 はカメラスピーカーへの音声出力に
> `tapo://` という独自プロトコル（ポート 8800）を使う。このプロトコルは RTSP / ONVIF
> とは異なる認証を要求し、TP-Link クラウドパスワードが必要になる。
> RTSP / ONVIF はカメラのローカルアカウントで認証する。

### wifi-cam-mcp

| 変数 | 説明 | デフォルト |
|------|------|-----------|
| `CAMERA_HOST` | カメラの IP アドレス | — |
| `CAMERA_USERNAME` | カメラのローカルアカウント名 | — |
| `CAMERA_PASSWORD` | カメラのローカルアカウントパスワード | — |
| `CAMERA_ONVIF_PORT` | ONVIF ポート番号 | `2020` |
| `CAMERA_MOUNT_MODE` | 設置方向 (`desk` / `ceiling`) | `desk` |

#### TTS エンジン比較

| エンジン | 品質 | レイテンシ | 備考 |
|----------|------|-----------|------|
| macOS (`macos`) | 低 | 即時 | OS 標準の `say` コマンド |
| Kokoro (`kokoro`) | 中 | 2-3 秒 | Apple Silicon ローカル推論 |
| ElevenLabs (`elevenlabs`) | 高 | 10-30 秒 | クラウド API。API キーが必要 |

## トラブルシューティング

### マイクが認識されない

```bash
ffmpeg -f avfoundation -list_devices true -i ""
```

表示されたオーディオデバイスのインデックスを `AUDIO_DEVICE` に設定する。

### カメラスピーカーから音が出ない

go2rtc のログを確認する:

```bash
cat /tmp/go2rtc.log
```

`tapo://` の認証失敗が出ている場合は `TAPO_CLOUD_PASSWORD` が正しくない。TP-Link アプリのアカウントパスワードを設定する（カメラのローカルパスワードではない）。

### メモリ不足

8GB メモリ環境では `WHISPER_MODEL=tiny` または `WHISPER_MODEL=base` に変更すると軽量化できる。
