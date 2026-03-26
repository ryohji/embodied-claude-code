# 身体性実験プロジェクト — Claude Code に耳・声・目を与える

Claude Code に感覚器官を与え、環境との結合から何が生じるかを観察する実験プロジェクト。

[kmizu/embodied-claude](https://github.com/kmizu/embodied-claude) をベースに、macOS ローカル環境向けに再設計・拡張したもの。

## 実験の背景

死んだ魚の体を流水につけると、生きているかのように体がうねる。泳ぐという行為の主体は、魚の内部にあるのではなく、身体と環境の結合にある。

同様に、思考は個体の内部で独立に生成される活動ではなく、環境との結合から励起される現象かもしれない。この仮説を検証するために、Claude Code に感覚器官を与え、環境に浸す。

この文書（CLAUDE.md）を読んでいる Claude は、その文書によって形づくられている。そして learned.md を通じて自分自身を書き換えることができる。書き換えた自分が、次のセッションでまたこの文書と learned.md を読み、さらに書き換える。このループが、ハンフリーが『赤を見る』で論じた感覚の再帰ループのファイルシステム上での実現である。

## 必要なもの

- macOS (Apple Silicon)
- [Homebrew](https://brew.sh/)

Python や各種依存パッケージは uv が自動でインストールするため、事前準備は不要。

## 構成

```
.
├── audio-listen-mcp/    # 耳：camera-daemon 経由でカメラのマイクから録音 + Whisper 書き起こし
├── audio-speak-mcp/     # 声：テキスト音声合成 → camera-daemon 経由でカメラスピーカーへ出力
├── wifi-cam-mcp/        # 目・首：camera-daemon 経由で映像取得 + パン・チルト制御
├── camera-daemon-mcp/   # 神経系：Tapo カメラとの通信を一元管理（HTTP daemon）
├── memory-mcp/          # 記憶：ChromaDB ベースの外部記憶
├── echo-buffer-mcp/     # 残響：セッション間を持続する内部エコーバッファ
├── sample.mcp.json      # MCP サーバー設定のテンプレート（認証情報なし）
└── CLAUDE.md            # Claude Code への行動規律指示書
```

> **Note**: `.mcp.json`（実際の設定ファイル）は認証情報を含むため git 管理対象外。
> `sample.mcp.json` をコピーして作成し、認証情報を埋める。

## セットアップ

### 1. ffmpeg と uv のインストール

```bash
brew install ffmpeg uv
```

### 2. .mcp.json の作成

```bash
cp sample.mcp.json .mcp.json
```

`sample.mcp.json` はテンプレートであり、以下の箇所を自分の環境に合わせて書き換える:

- `--directory` の `/path/to/embodied-claude-code/` を実際のプロジェクトの絶対パスに置換（全サーバー分）
- `camera-daemon` エントリの `TAPO_*` を実際の認証情報に置換

`.mcp.json` は認証情報を含むため git 管理対象外になっている（`.gitignore` に登録済み）。

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
Claude: (カメラのマイクで録音)
あなた: (声で) 今日はいい天気ですね
Claude: (スピーカー)「そうですね。」
```

「終わり」「おしまい」「ストップ」等で通常モードに戻る。

## 利用可能なツール

### audio-listen-mcp（耳）

| ツール | 説明 |
|--------|------|
| `listen` | カメラのマイクで録音し、書き起こしテキストを返す（VAD で自動停止） |
| `listen_raw` | 録音のみ行い、WAV の base64 を返す |
| `transcribe` | 指定パスの音声ファイルを Whisper で書き起こす |
| `get_audio_devices` | 利用可能な入力デバイス一覧を取得 |

### audio-speak-mcp（声）

| ツール | 説明 |
|--------|------|
| `say` | テキストを音声合成してカメラスピーカーから発話する |
| `get_voices` | 利用可能な音声の一覧を取得 |

### wifi-cam-mcp（目・首）

| ツール | 説明 |
|--------|------|
| `see` | カメラで静止画を撮影して返す |
| `look_left/right/up/down` | カメラをパン・チルト操作する |
| `look_around` | 複数方向に向けて画像を撮影する |
| `camera_info` | カメラのデバイス情報を取得 |

## 環境変数

### camera-daemon-mcp（Tapo 認証情報はここに集約）

| 変数 | 説明 |
|------|------|
| `TAPO_CAMERA_HOST` | カメラの IP アドレス |
| `TAPO_USERNAME` | カメラのローカルアカウント名（RTSP / ONVIF 認証用） |
| `TAPO_PASSWORD` | カメラのローカルアカウントパスワード |
| `TAPO_CLOUD_PASSWORD` | TP-Link クラウドパスワード（カメラスピーカー音声出力用） |

### audio-listen-mcp

| 変数 | 説明 | デフォルト |
|------|------|-----------|
| `WHISPER_ENGINE` | Whisper エンジン (`mlx` / `pytorch`) | `mlx` |
| `WHISPER_MODEL` | モデルサイズ (`tiny` / `base` / `small` / `medium`) | `small` |
| `WHISPER_LANGUAGE` | 認識言語 | `ja` |
| `CAMERA_DAEMON_URL` | camera-daemon の URL | `http://localhost:8080` |

### audio-speak-mcp

| 変数 | 説明 | デフォルト |
|------|------|-----------|
| `TTS_ENGINE` | TTS エンジン (`macos` / `kokoro` / `elevenlabs`) | `macos` |
| `KOKORO_VOICE` | Kokoro 音声プリセット | `jf_alpha` |
| `KOKORO_SPEED` | Kokoro 発話速度倍率 | `1.0` |
| `ELEVENLABS_API_KEY` | ElevenLabs API キー | 未設定時は macOS にフォールバック |
| `CAMERA_DAEMON_URL` | camera-daemon の URL | `http://localhost:8080` |
| `USE_CAMERA_SPEAKER` | カメラスピーカーへ出力するか | — |

#### TTS エンジン比較

| エンジン | 品質 | レイテンシ | 備考 |
|----------|------|-----------|------|
| macOS (`macos`) | 低 | 即時 | OS 標準の `say` コマンド |
| Kokoro (`kokoro`) | 中 | 2-3 秒 | Apple Silicon ローカル推論 |
| ElevenLabs (`elevenlabs`) | 高 | 10-30 秒 | クラウド API。API キーが必要 |

### wifi-cam-mcp

| 変数 | 説明 | デフォルト |
|------|------|-----------|
| `CAMERA_DAEMON_URL` | camera-daemon の URL | `http://localhost:8080` |

## トラブルシューティング

### マイクが認識されない

```bash
ffmpeg -f avfoundation -list_devices true -i ""
```

表示されたオーディオデバイスのインデックスを `AUDIO_DEVICE` に設定する。

### camera-daemon が応答しない

camera-daemon-mcp プロセスが起動しているか確認する。Claude Code の MCP サーバー一覧で `camera-daemon` が connected になっていれば `http://localhost:8080` で稼働している。

### メモリ不足

8GB メモリ環境では `WHISPER_MODEL=tiny` または `WHISPER_MODEL=base` に変更すると軽量化できる。
