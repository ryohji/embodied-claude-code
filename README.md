# Claude Code 自己拡張プロジェクト — 耳と声の獲得

Claude Code に「耳」と「声」を与えるプロジェクト。2 つの MCP サーバーにより、Claude Code がローカル PC のマイクで音声を聞き取り、音声合成でユーザーに話しかけることができるようになる。

[kmizu/embodied-claude](https://github.com/kmizu/embodied-claude) の音声入出力部分を macOS ローカル環境に特化して再設計したもの。

## 必要なもの

- macOS (Apple Silicon)
- [Homebrew](https://brew.sh/)

Python や mlx-whisper 等の依存パッケージは uv が自動でインストールするため、事前準備は不要。

## 構成

```
.
├── audio-listen-mcp/    # 耳：マイク録音 + Whisper 書き起こし
├── audio-speak-mcp/     # 声：テキスト音声合成
├── .mcp.json            # MCP サーバー登録設定
└── CLAUDE.md            # Claude Code への設計指示書
```

## セットアップ

### 1. ffmpeg と uv のインストール

```bash
brew install ffmpeg uv
```

これだけで OK。Python 本体、mlx-whisper、MCP SDK などの依存パッケージはすべて uv が自動でインストールする（.mcp.json の `uv run` 実行時に解決される）。

### 2. マイク権限の許可

macOS のシステム設定 → プライバシーとセキュリティ → マイク で、使用するターミナルアプリ（Terminal.app / iTerm2 / VSCode 等）にマイクアクセスを許可する。

### 3. .mcp.json の確認

プロジェクトルートの [.mcp.json](.mcp.json) にMCP サーバーの設定が含まれている。`--directory` のパスが実際のプロジェクトパスと一致していることを確認する。

### 4. 動作確認

Claude Code を起動し、MCP サーバーの接続状態を確認する。

```
/mcp
```

`audio-listen` と `audio-speak` が接続済み（connected）と表示されれば準備完了。初回起動時は uv が依存パッケージをインストールし、Whisper モデル（small: 約 500MB）をダウンロードするため時間がかかる。

## 使い方

### 会話モード

Claude Code に「会話モード」と送ると、音声による対話ループに入る。Claude が自分の判断で聞く・考える・話すを繰り返し、声だけで会話が続く。

```
あなた: 会話モード
Claude: (スピーカー)「会話モードを開始します。何でも話しかけてください。」
Claude: (マイクで録音)
あなた: (声で) 今日はいい天気ですね
Claude: (スピーカー)「そうですね、お出かけ日和ですね。」
Claude: (マイクで録音)
あなた: (声で) おしまい
Claude: (スピーカー)「会話モードを終了します。」
```

「終わり」「おしまい」「ストップ」等の発話で通常モードに戻る。

### 単発で使う

「listen」と送れば 1 回だけ録音・書き起こしを行う。Claude はその内容に応じて `say` で声を出して返答する。

### 利用可能なツール

#### audio-listen-mcp（耳）

| ツール | 説明 |
|--------|------|
| `listen` | マイクで録音し、書き起こしテキストを返す。発話終了を検知して自動停止（VAD） |
| `listen_raw` | 録音のみ行い、WAV の base64 を返す（書き起こしなし） |
| `transcribe` | 指定パスの音声ファイルを Whisper で書き起こす |
| `get_audio_devices` | 利用可能なオーディオ入力デバイスの一覧を取得 |

#### audio-speak-mcp（声）

| ツール | 説明 |
|--------|------|
| `say` | テキストを音声合成して PC スピーカーから発話する |
| `get_voices` | 利用可能な音声の一覧を取得 |

## 環境変数

### audio-listen-mcp

| 変数 | 説明 | デフォルト |
|------|------|-----------|
| `WHISPER_ENGINE` | Whisper エンジン (`mlx` / `pytorch`) | `mlx` |
| `WHISPER_MODEL` | モデルサイズ (`tiny` / `base` / `small` / `medium`) | `small` |
| `WHISPER_LANGUAGE` | 認識言語 | `ja` |
| `AUDIO_DEVICE` | 入力デバイスインデックス | システムデフォルト |
| `AUDIO_SAMPLE_RATE` | サンプリングレート (Hz) | `16000` |
| `LISTEN_DEFAULT_DURATION` | デフォルト録音秒数 | `5` |
| `LISTEN_MAX_DURATION` | 最大録音秒数 | `30` |
| `VAD_SILENCE_DURATION` | 無音判定秒数 | `2.0` |
| `VAD_SILENCE_THRESHOLD` | 無音判定閾値 (RMS) | `500` |

### audio-speak-mcp

| 変数 | 説明 | デフォルト |
|------|------|-----------|
| `TTS_ENGINE` | TTS エンジン (`macos` / `elevenlabs`) | `macos` |
| `TTS_VOICE` | macOS の音声名 | `Kyoko` |
| `TTS_RATE` | 発話速度 (words per minute) | システムデフォルト |
| `ELEVENLABS_API_KEY` | ElevenLabs API キー | 未設定時は macOS にフォールバック |
| `ELEVENLABS_VOICE_ID` | ElevenLabs 音声 ID | — |
| `ELEVENLABS_MODEL_ID` | ElevenLabs モデル ID | `eleven_v3` |

## 設計上の特徴

- **関心の分離** — 音声入力と音声出力を独立した MCP サーバーに分離。依存関係が異なり、片方だけの利用も可能
- **エンジン抽象化** — Whisper (mlx / pytorch) と TTS (macOS say / ElevenLabs) を抽象レイヤで切り替え可能
- **グレースフルフォールバック** — mlx-whisper が使えなければ PyTorch 版に自動フォールバック。ElevenLabs の API キーがなければ macOS say にフォールバック
- **遅延ロード** — Whisper モデルは初回使用時に 1 回だけロード。embodied-claude の毎回ロードする設計を改善
- **VAD (Voice Activity Detection)** — 発話終了を検知して自動停止。固定秒数の録音も選択可能
- **セキュリティ** — 音声テキストは一時ファイル経由で渡し、シェルインジェクションを防止

## トラブルシューティング

### マイクが認識されない

```bash
# デバイス一覧を確認
ffmpeg -f avfoundation -list_devices true -i ""
```

表示されたオーディオデバイスのインデックスを `AUDIO_DEVICE` に設定する。

### Whisper モデルのダウンロードに時間がかかる

初回起動時にモデルがダウンロードされる（small モデルで約 500MB）。2 回目以降はキャッシュされる。

### メモリ不足

8GB メモリ環境では `small` モデルが上限。`WHISPER_MODEL=tiny` または `WHISPER_MODEL=base` に変更すると軽量化できる（認識精度は低下する）。

## 背景

このプロジェクトは [kmizu/embodied-claude](https://github.com/kmizu/embodied-claude) を調査・分析した結果に基づいている。embodied-claude は Wi-Fi カメラ（TP-Link Tapo）を使って Claude Code に目・首・耳・声・脳を与えるプロジェクトであり、本プロジェクトはそこからカメラ関連を除外し、ローカル PC の音声入出力に特化して再設計したものである。
