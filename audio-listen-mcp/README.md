# audio-listen-mcp

マイクまたは Tapo カメラのマイクから音声を録音し、Whisper で書き起こす MCP サーバー。

## 利用可能なツール

| ツール | 説明 |
|--------|------|
| `listen` | 録音して文字起こしを返す。`auto_stop: true`（デフォルト）で発話終了を自動検出。 |
| `listen_raw` | 録音して WAV データを base64 で返す（書き起こしなし）。 |
| `transcribe` | 指定パスの音声ファイルを Whisper で書き起こす。 |
| `get_audio_devices` | 利用可能なオーディオ入力デバイス一覧を返す。 |

### `listen` / `listen_raw` パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `duration` | number | 5 | 録音秒数（最大 30） |
| `auto_stop` | boolean | true | 発話終了を検知して自動停止する |

## 環境変数

### Whisper 設定

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `WHISPER_ENGINE` | `mlx` | `mlx`（Apple Silicon 最適化）または `openai` |
| `WHISPER_MODEL` | `small` | Whisper モデルサイズ（`tiny`, `base`, `small`, `medium`, `large`） |
| `WHISPER_LANGUAGE` | `ja` | 認識言語コード |

### オーディオデバイス設定（MacBook マイク使用時）

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `AUDIO_DEVICE` | `0` | avfoundation デバイスインデックス（`get_audio_devices` で確認） |
| `AUDIO_SAMPLE_RATE` | `16000` | サンプリングレート（Hz） |
| `LISTEN_DEFAULT_DURATION` | `5` | デフォルト録音秒数 |
| `LISTEN_MAX_DURATION` | `30` | 最大録音秒数 |
| `VAD_SILENCE_DURATION` | `2.0` | 無音が続いたら停止するまでの秒数（VAD モード） |
| `VAD_SILENCE_THRESHOLD` | `500` | 無音とみなす RMS 閾値（VAD モード） |

### Tapo カメラマイク使用時

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `USE_TAPO_AUDIO` | 未設定 | **設定するとカメラ RTSP 入力を優先**（値は何でもよい） |
| `TAPO_CAMERA_HOST` | `192.168.0.1` | カメラの IP アドレス |
| `TAPO_USERNAME` | `admin` | RTSP 認証ユーザー名 |
| `TAPO_PASSWORD` | ― | RTSP 認証パスワード（カメラローカルアカウント） |

`USE_TAPO_AUDIO` が設定されている場合、`ffmpeg` で `rtsp://{user}:{pw}@{host}:554/stream1` に接続し録音します。VAD も動作します。

## 音声入力バックエンド

```
USE_TAPO_AUDIO 未設定  →  MacBook マイク（avfoundation）
USE_TAPO_AUDIO 設定済  →  Tapo カメラマイク（RTSP）
```

Whisper エンジンはどちらのバックエンドでも共通（mlx-whisper）です。

## 起動方法

```bash
# MacBook マイク（通常）
uv run --extra mlx audio-listen-mcp

# Tapo カメラマイク
USE_TAPO_AUDIO=1 TAPO_CAMERA_HOST=192.168.1.x TAPO_PASSWORD=... \
  uv run --extra mlx audio-listen-mcp
```

## .mcp.json 設定例

```json
{
  "audio-listen": {
    "command": "uv",
    "args": ["run", "--directory", "/path/to/audio-listen-mcp", "--extra", "mlx", "audio-listen-mcp"],
    "env": {
      "WHISPER_ENGINE": "mlx",
      "WHISPER_MODEL": "small",
      "WHISPER_LANGUAGE": "ja",
      "AUDIO_DEVICE": "0",
      "TAPO_CAMERA_HOST": "192.168.1.x",
      "TAPO_USERNAME": "admin",
      "TAPO_PASSWORD": "..."
    }
  }
}
```

カメラマイクに切り替えるときは `"USE_TAPO_AUDIO": "1"` を `env` に追加してサーバーを再起動します。
