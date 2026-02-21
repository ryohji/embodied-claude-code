#!/bin/bash
# session_start.sh
# UserPromptSubmit hook: セッションの最初のメッセージを検知し、
# Claude に memory-mcp からの記憶確認を指示する。

INPUT=$(cat)

# トランスクリプト内のアシスタントの発言数を数える
# 0 であればこのセッションで初めてのユーザー入力（＝セッション開始）
ASSISTANT_COUNT=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    transcript = data.get('transcript', [])
    count = sum(1 for msg in transcript if isinstance(msg, dict) and msg.get('role') == 'assistant')
    print(count)
except Exception:
    print(1)
" 2>/dev/null || echo "1")

if [ "$ASSISTANT_COUNT" = "0" ]; then
    echo "【セッション開始】memory-mcp の list_recent_memories を実行して前回の記憶を確認し、文脈を回復してください。テキスト対話・会話モード・生活モードを問わず必ず実行してください。"
fi
