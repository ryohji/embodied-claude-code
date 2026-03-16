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

## フェーズ 3: camera-daemon を HTTP デーモンに再設計（未着手）

camera-daemon の MCP ツールを HTTP エンドポイントに変換し、
Claude に対してツールを公開しない形に作り直す。

- FastAPI（または aiohttp）で HTTP サーバーを追加
- エンドポイント例: `GET /see`, `POST /ptz/left`, `POST /ptz/right` など
- MCP ツールを削除（または空にする）
- wifi-cam-mcp を復活させ、camera-daemon HTTP API を呼ぶ薄いラッパーとして再実装
- `.mcp.json` に wifi-cam-mcp を再追加

---

## フェーズ 4: audio-listen / audio-speak の移行（未着手）

- audio-listen-mcp の TapoAudioCapture を camera-daemon HTTP 経由に変更
- audio-speak-mcp の Tapo バックチャンネルを camera-daemon HTTP 経由に変更
- 各 MCP サーバーから Tapo 認証情報を削除

---

## フェーズ 5: pytapo による Tapo プロトコル一本化の検証（未着手）

- pytapo で RTSP URL 取得・PTZ 制御が動くか確認
- go2rtc の `tapo://` source が映像・音声「入力」にも使えるか確認
- 暗号化ストリームが実現できるなら RTSP から Tapo プロトコルへの一本化を設計
