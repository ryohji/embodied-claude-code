"""Kokoro TTS で音声を生成して WAV ファイルに保存する。

実行方法:
  audio-speak-mcp/.venv/bin/python experiments/gen_kokoro.py <テキスト> <出力.wav>
  audio-speak-mcp/.venv/bin/python experiments/gen_kokoro.py "おはようございます" experiments/output/test.wav
"""
import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="Kokoro TTS → WAV")
parser.add_argument("text", help="合成するテキスト")
parser.add_argument("output", help="出力 WAV ファイルパス")
parser.add_argument("--voice", default="jf_alpha", help="音声名 (default: jf_alpha)")
parser.add_argument("--speed", type=float, default=1.0, help="速度 (default: 1.0)")
parser.add_argument("--model", default="mlx-community/Kokoro-82M-bf16", help="モデル ID")
args = parser.parse_args()

print(f"Kokoro モデルを読み込み中: {args.model}")
try:
    from mlx_audio.tts.utils import load_model
except ImportError:
    print("ERROR: mlx_audio が見つかりません。audio-speak-mcp/.venv/bin/python で実行してください。", file=sys.stderr)
    sys.exit(1)

import numpy as np
import soundfile as sf

model = load_model(args.model)
print(f"生成中: {args.text!r}  voice={args.voice}  speed={args.speed}")

results = list(model.generate(args.text, voice=args.voice, speed=args.speed, lang_code="j"))
if not results:
    print("ERROR: 音声が生成されませんでした。", file=sys.stderr)
    sys.exit(1)

chunks = [np.array(r.audio) for r in results]
audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
audio = audio.astype(np.float32)

output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)
sf.write(str(output_path), audio, 24000)

duration = len(audio) / 24000
print(f"保存完了: {output_path}  ({len(audio)} サンプル, {duration:.2f}s, 24000 Hz)")
