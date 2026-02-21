#!/bin/bash
# Phase 1: Kokoro で音声生成 → WORLD で分解・可視化
# 使い方: bash experiments/run_phase1.sh ["テキスト"]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

AUDIO_PYTHON="$PROJECT_DIR/audio-speak-mcp/.venv/bin/python"
EXPERIMENTS_PYTHON="$PROJECT_DIR/experiments/.venv/bin/python"
OUTPUT_DIR="$SCRIPT_DIR/output"

TEXT="${1:-おはようございます}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
WAV_FILE="$OUTPUT_DIR/kokoro_${TIMESTAMP}.wav"

echo "========================================"
echo "Phase 1: WORLD ボコーダー実験"
echo "テキスト: $TEXT"
echo "========================================"
echo ""

echo "--- Step 1: Kokoro 音声生成 ---"
"$AUDIO_PYTHON" "$SCRIPT_DIR/gen_kokoro.py" "$TEXT" "$WAV_FILE"

echo ""
echo "--- Step 2: WORLD 分解・可視化 ---"
"$EXPERIMENTS_PYTHON" "$SCRIPT_DIR/analyze_world.py" "$WAV_FILE" "$OUTPUT_DIR"

echo ""
echo "========================================"
echo "完了"
echo "  音声 (original) : $WAV_FILE"
STEM="kokoro_${TIMESTAMP}"
echo "  音声 (roundtrip): $OUTPUT_DIR/${STEM}_roundtrip.wav"
echo "  プロット         : $OUTPUT_DIR/${STEM}_analysis.png"
echo "========================================"
