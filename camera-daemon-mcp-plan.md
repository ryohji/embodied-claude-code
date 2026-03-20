# camera-daemon-mcp 移行計画

## 背景と目的

wifi-cam-mcp / audio-listen-mcp / audio-speak-mcp が個別に Tapo C210 へ RTSP 接続を開いており、
同時接続でカメラのセッション上限に抵触しうる。またパスワードが接続のたびに平文で送出される。

C210 関連の認証情報・接続を camera-daemon に集約し、go2rtc をセッションスコープで管理する
単一の窓口とすることで、これらの問題を解消する。

---

## アーキテクチャ方針（2026-03-16 改訂）

### camera-daemon の役割

camera-daemon は **Claude Code が直接利用するサーバーではない**。
他の MCP サーバー（wifi-cam, audio-listen, audio-speak）が呼ぶバックエンドサービスである。

- **MCP サーバーとして .mcp.json に登録する**: Claude Code のセッションライフサイクルを借りるため。
  セッション開始時に起動、終了時に停止。システムレベルのデーモンとして常駐させない。
- **MCP ツールは公開しない**: Claude のツール一覧に現れない。
- **HTTP サーバーを localhost で公開する**: 他の MCP サーバーからの呼び出しを受け付ける。
- 認証情報（TAPO_USERNAME / TAPO_PASSWORD 等）は camera-daemon の env のみに置く。

### 各 MCP サーバーの役割

| MCP サーバー      | 担当                                 | カメラアクセス           |
|-------------------|--------------------------------------|--------------------------|
| camera-daemon-mcp | C210 接続管理 + go2rtc + HTTP API    | 直接（唯一の接続元）     |
| wifi-cam-mcp      | 目（PTZ + 画像取得）                 | camera-daemon HTTP 経由  |
| audio-listen-mcp  | 耳（マイク + カメラ音声入力）        | camera-daemon HTTP 経由  |
| audio-speak-mcp   | 口（TTS + カメラスピーカー出力）     | camera-daemon HTTP 経由  |

```
Claude Code
  ↓ MCP
wifi-cam-mcp / audio-listen-mcp / audio-speak-mcp
  ↓ HTTP (CAMERA_DAEMON_URL=http://localhost:8080)
camera-daemon-mcp（MCPツールなし・HTTPサーバーのみ）
  ↓ ONVIF / RTSP / Tapo プロトコル
Tapo C210
```

---

## フェーズ 1: camera-daemon-mcp の骨格（完了）

- `camera-daemon-mcp/` 新規作成
- go2rtc.py を audio-speak-mcp から移植
- MCP サーバー起動時に go2rtc を開始、終了時に停止
- .mcp.json に登録

---

## フェーズ 2: PTZ・画像取得の移植（完了）

- wifi-cam-mcp の ONVIF ベース PTZ 実装を camera-daemon に移植（camera.py）
- wifi-cam-mcp の画像取得（see）を camera-daemon に移植
- say_to_camera（バックチャンネル）を camera-daemon に移植
- 動作確認済み（2026-03-14〜16）

---

## フェーズ 3: camera-daemon を HTTP デーモンに再設計（完了）

camera-daemon の MCP ツールを HTTP エンドポイントに変換し、
Claude に対してツールを公開しない形に作り直した。

- aiohttp で HTTP サーバーを追加（完了）
- MCP ツールを削除（list_tools が [] を返す）（完了）
- wifi-cam-mcp を復活させ、camera-daemon HTTP API を呼ぶ薄いラッパーとして再実装（完了）
- `.mcp.json` に wifi-cam-mcp を再追加（完了）
- エンドポイント再設計・実装（完了 2026-03-19）

### エンドポイント設計（確定）

カメラが提供するローレベルの機能（画像取得・音声入力・音声出力）を軸に整理する。

**目標 API 設計:**

| エンドポイント | メソッド | 説明 |
|---|---|---|
| `/image` | GET | 現在の向きで画像取得（JPEG base64 JSON） |
| `/image?pan=-30` | GET | 左に30°移動してから画像取得（正=右、負=左） |
| `/image?tilt=20` | GET | 上に20°移動してから画像取得（正=上、負=下） |
| `/audio?duration=5` | GET | カメラマイクから固定時間録音（WAV バイト列） |
| `/audio?duration=30&vad` | GET | VAD 付き録音。`duration` が max_duration として使われる。`vad=30` と書いても同義 |
| `/audio?duration=30&vad&silence_duration=1.5&silence_threshold=500` | GET | VAD 付き録音（詳細パラメーター指定） |
| `/audio` | POST | WAV バイト列をカメラスピーカーに送出 |
| `/info` | GET | カメラデバイス情報 |
| `/stream_url` | GET | go2rtc ストリーム URL |

**設計方針:**

- `/see` と `/ptz` を統合して `/image` に。PTZ は画像取得前のオプション動作として `pan`・`tilt` クエリパラメーターで指定。符号で方向を示す（`pan=-30` = 左30°、`pan=30` = 右30°、`tilt=20` = 上20°）
- `/audio` は入出力を HTTP メソッドで区別（GET = 入力、POST = 出力）
- `GET /audio` の時間指定は `duration` に統一。`vad` が指定された場合は VAD モードとなり、`duration` が max_duration として使われる。`vad` 自体に値を指定した場合（`vad=30`）はそれが duration を上書きする
- `/preset` は廃止。プリセット操作は利用側（wifi-cam-mcp 等）が `/image?pan=X&tilt=Y` をラップして提供する
- TTS 変換（テキスト→音声）は camera-daemon では行わない。音声合成は audio-speak-mcp 側で実施し、生成した WAV バイト列を `POST /audio` で送る
- `/say` エンドポイントは廃止（テキスト → TTS → バックチャンネル再生の責務を camera-daemon から外す）

**変更対象（現状 → 変更後）:**

| 現状 | 変更後 |
|---|---|
| `GET /see` | `GET /image` |
| `POST /ptz` body: `{"direction": "left", "degrees": 30}` | `GET /image?pan=-30` |
| `POST /ptz` body: `{"direction": "right", "degrees": 30}` | `GET /image?pan=30` |
| `POST /ptz` body: `{"direction": "up", "degrees": 20}` | `GET /image?tilt=20` |
| `POST /ptz` body: `{"direction": "down", "degrees": 20}` | `GET /image?tilt=-20` |
| `GET /look_around` | 廃止 |
| `GET /presets` | 廃止 |
| `POST /preset/{preset_id}` | 廃止 |
| `POST /say` | 廃止 → `POST /audio`（WAV バイト列受付）で代替 |

変更に合わせて wifi-cam-mcp 側の HTTP 呼び出しも更新する。

---

## フェーズ 3.5: PTZ API の絶対値化・direction エンドポイント追加（未着手）

### 背景

現在の `/image?pan=X&tilt=Y` は **相対移動・度数指定**（例: `pan=-30` = 左30°）。
これを **絶対位置・ONVIF 正規化値 [-1, 1] 指定** に変更する。
あわせて現在向きの取得エンドポイント `GET /direction` を追加する。

### camera-daemon 側の変更

**エンドポイント設計（改訂後）:**

| エンドポイント | メソッド | 説明 |
|---|---|---|
| `/image` | GET | 現在の向きで画像取得（移動なし） |
| `/image?pan=X` | GET | pan のみ絶対指定。tilt は GetStatus で取得した現在値を維持 |
| `/image?tilt=Y` | GET | tilt のみ絶対指定。pan は GetStatus で取得した現在値を維持 |
| `/image?pan=X&tilt=Y` | GET | (X, Y) は ONVIF 正規化絶対値 [-1, 1]。AbsoluteMove してから画像取得 |
| `/direction` | GET | 現在の向きを `{"pan": X, "tilt": Y}` で返す（GetStatus 使用、失敗時は 503） |
| `/direction` | POST | body `{"pan": X, "tilt": Y}` で絶対位置に移動（撮影なし） |

**camera.py への追加:**

- `absolute_move(pan: float, tilt: float)` メソッドを追加
  - [-1, 1] にクリップしてから ONVIF AbsoluteMove を送出
  - ceiling モード時は両軸を反転
  - 移動完了待ち（既存の `_wait_for_move_complete` を流用）

### wifi-cam 側の変更

**状態管理:**

- `_cached_direction: dict | None = None` をサーバーインスタンスに追加
- 初回 look_* 呼び出し時に `GET /direction` で取得してキャッシュ
- 失敗時はエラーを返し、移動しない

**look_* の実装方針:**

```
look_left(degrees=30):
  pan_delta = degrees / 180.0   # 度 → ONVIF 正規化
  new_pan = clip(cached_pan - pan_delta, -1.0, 1.0)
  GET /image?pan={new_pan}&tilt={cached_tilt}
  cached_pan = new_pan

look_up(degrees=20):
  tilt_delta = degrees / 90.0
  new_tilt = clip(cached_tilt + tilt_delta, -1.0, 1.0)
  GET /image?pan={cached_pan}&tilt={new_tilt}
  cached_tilt = new_tilt
```

**look_around の実装方針:**

1. 現在位置 (p, t) を取得（キャッシュ or GET /direction）
2. 4 ショットの絶対座標を事前計算してクリップ:
   - left:  `(clip(p - 0.25, -1, 1), t)`
   - up:    `(p, clip(t + 0.333, -1, 1))`  ← 二等辺三角形の頂点（pan は元に戻る）
   - right: `(clip(p + 0.25, -1, 1), t)`
   - front: `(p, t)`  ← 元の位置（最後に必ず戻る）
3. 各座標へ `GET /image?pan=X&tilt=Y` で AbsoluteMove + 撮影
4. front ショットが最終位置なので `POST /direction` は不要

`POST /direction` は look_around の戻り先復帰（撮影なし）にも使える。

---

## フェーズ 4: audio-listen / audio-speak の移行（未着手）

### 概要

- audio-listen-mcp の TapoAudioCapture を `GET /audio` 経由に変更
- audio-speak-mcp の Tapo バックチャンネル出力を `POST /audio` 経由に変更
- 各 MCP サーバーから Tapo 認証情報を削除

### camera-daemon 側の実装

フェーズ 3 で追加する `GET /audio` と `POST /audio` がそのまま使われる。

**音声入力（audio-listen 向け）:**

`GET /audio` の実装は `TapoAudioCapture` の `record` / `record_with_vad` を camera-daemon に移植したもの。
レスポンスは `Content-Type: audio/wav` で WAV バイト列を直接返す。
audio-listen-mcp は WAV を受け取り、既存の Whisper 転写処理に渡す。

**音声出力（audio-speak 向け）:**

`POST /audio` は `Content-Type: audio/wav` のリクエストボディ（WAV バイト列）を受け取り、
go2rtc バックチャンネル経由でカメラスピーカーに送出する。
audio-speak-mcp は Kokoro/ElevenLabs/say で音声合成 → WAV ファイル生成 → `POST /audio` に送信。

### MCP サーバー側の変更

- audio-listen-mcp の env から `TAPO_CAMERA_HOST / TAPO_USERNAME / TAPO_PASSWORD` を削除
- audio-listen-mcp に `CAMERA_DAEMON_URL` と `USE_CAMERA_MIC` 環境変数を追加
- audio-speak-mcp に `CAMERA_DAEMON_URL` と `USE_CAMERA_SPEAKER` 環境変数を追加

呼び出し例:
- `GET /audio?duration=5` → 固定5秒録音
- `GET /audio?duration=30&vad` または `GET /audio?vad=30` → VAD付き、最大30秒
- `GET /audio?vad=30&silence_duration=1.5&silence_threshold=500` → 詳細 VAD パラメーター付き

### 移行後の .mcp.json

- audio-listen-mcp の env: `TAPO_*` を削除し `CAMERA_DAEMON_URL` + `USE_CAMERA_MIC` を追加（任意）
- audio-speak-mcp の env: `CAMERA_DAEMON_URL` + `USE_CAMERA_SPEAKER` を追加（任意）

---

## フェーズ 5: pytapo による Tapo プロトコル一本化の検証（未着手）

- pytapo で RTSP URL 取得・PTZ 制御が動くか確認
- go2rtc の `tapo://` source が映像・音声「入力」にも使えるか確認
- 暗号化ストリームが実現できるなら RTSP から Tapo プロトコルへの一本化を設計
