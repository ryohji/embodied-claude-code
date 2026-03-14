# camera-daemon-mcp 移行計画

## 背景と目的

wifi-cam-mcp / audio-listen-mcp / audio-speak-mcp が個別に Tapo C210 へ RTSP 接続を開いており、
同時接続でカメラのセッション上限に抵触しうる。またパスワードが接続のたびに平文で送出される。

C210 関連の機能を camera-daemon-mcp に集約し、go2rtc をセッションスコープで管理する
単一の窓口とすることで、これらの問題を解消する。

---

## フェーズ 1: camera-daemon-mcp の骨格（go2rtc 管理）

- `camera-daemon-mcp/` を新規作成（pyproject.toml, src/）
- go2rtc.py を audio-speak-mcp から移植
- MCP サーバー起動時に go2rtc を開始、終了時に停止（セッションスコープ）
- .mcp.json に登録
- 他の MCP が localhost:1984 経由でカメラに繋げることを動作確認

---

## フェーズ 2: 機能移行（一機能ずつ）

### 2-1. PTZ 制御（首振り）
- wifi-cam-mcp の ONVIF ベース実装を移植
- 移植確認後、wifi-cam-mcp の PTZ ツールを削除

### 2-2. 画像取得（see）
- wifi-cam-mcp の capture_image を移植
- RTSP フォールバックを localhost:1984 経由に変更
- 移植確認後、wifi-cam-mcp の see ツールを削除

### 2-3. 音声取得（listen from camera）
- audio-listen-mcp の TapoAudioCapture を移植
- RTSP 接続先を localhost:1984 経由に変更
- 移植確認後、audio-listen-mcp の tapo デバイス対応を削除

### 2-4. 音声送信（say / backchannel）
- audio-speak-mcp の Tapo バックチャンネル実装を移植
- 移植確認後、audio-speak-mcp の tapo デバイス対応と go2rtc.py を削除

---

## フェーズ 3: pytapo による Tapo プロトコル一本化の検証

- pytapo を依存に追加しインストール
- pytapo で RTSP URL 取得・PTZ 制御が動くか確認
- go2rtc の `tapo://` source が映像・音声「入力」にも使えるか確認
- 暗号化ストリームが実現できるなら RTSP から Tapo プロトコルへの一本化を設計

---

## 移行後の構成（目標）

| MCP サーバー       | 担当                          |
|--------------------|-------------------------------|
| camera-daemon-mcp  | C210 全機能 + go2rtc 管理     |
| audio-listen-mcp   | Mac マイク入力のみ            |
| audio-speak-mcp    | Mac スピーカー出力のみ        |
| wifi-cam-mcp       | 廃止（または最小スタブ）      |
