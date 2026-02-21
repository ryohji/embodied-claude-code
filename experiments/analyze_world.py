"""WORLD ボコーダーで音声を分解・可視化・再合成する。

実行方法:
  experiments/.venv/bin/python experiments/analyze_world.py <入力.wav> [出力ディレクトリ]

出力:
  <stem>_analysis.png   — F0 / SP / AP / 波形比較の 4 段プロット
  <stem>_roundtrip.wav  — 分解→再合成した音声
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # GUI なし環境でも動くように
import matplotlib.pyplot as plt
import numpy as np
import pyworld as pw
import soundfile as sf
import scipy.io.wavfile as wavfile

parser = argparse.ArgumentParser(description="WORLD analysis + visualize")
parser.add_argument("input_wav", help="入力 WAV ファイル")
parser.add_argument("output_dir", nargs="?", default="experiments/output",
                    help="出力ディレクトリ (default: experiments/output)")
args = parser.parse_args()

input_path = Path(args.input_wav)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# --- 音声読み込み ---
print(f"読み込み: {input_path}")
sr, audio_raw = wavfile.read(str(input_path))

# モノラル化・float64 変換（WORLD が要求する形式）
if audio_raw.ndim > 1:
    audio_raw = audio_raw.mean(axis=1)
if np.issubdtype(audio_raw.dtype, np.integer):
    audio = audio_raw.astype(np.float64) / np.iinfo(audio_raw.dtype).max
else:
    audio = audio_raw.astype(np.float64)

print(f"音声: {len(audio)} サンプル, {sr} Hz, {len(audio)/sr:.2f}s")

# --- WORLD 分解 ---
print("WORLD 分析中...")
f0, sp, ap = pw.wav2world(audio, sr)

t = np.arange(len(f0)) * pw.default_frame_period / 1000  # 秒

voiced_f0 = f0[f0 > 0]
print(f"F0: {len(f0)} フレーム  有声={len(voiced_f0)}  無声={len(f0)-len(voiced_f0)}")
if len(voiced_f0) > 0:
    print(f"F0 範囲: {voiced_f0.min():.1f}–{voiced_f0.max():.1f} Hz  "
          f"平均={voiced_f0.mean():.1f} Hz")
print(f"SP shape: {sp.shape}  AP shape: {ap.shape}")

# --- ラウンドトリップ再合成 ---
print("再合成中 (roundtrip)...")
audio_synth = pw.synthesize(f0, sp, ap, sr).astype(np.float32)
synth_path = output_dir / (input_path.stem + "_roundtrip.wav")
sf.write(str(synth_path), audio_synth, sr)
print(f"再合成保存: {synth_path}")

# --- プロット ---
fig, axes = plt.subplots(4, 1, figsize=(14, 10))
fig.suptitle(f"WORLD Analysis: {input_path.name}", fontsize=12)

# F0
axes[0].plot(t, f0, color="steelblue", linewidth=1.2)
axes[0].set_ylabel("Hz")
axes[0].set_title("F0（基本周波数・ピッチ）— 無声区間は 0")
axes[0].set_xlim([0, t[-1]])
axes[0].grid(True, alpha=0.3)

# SP（スペクトル包絡）— dB スケール
sp_db = 10 * np.log10(sp.T + 1e-10)
im1 = axes[1].imshow(
    sp_db, aspect="auto", origin="lower",
    extent=[0, t[-1], 0, sr / 2 / 1000],
    cmap="viridis",
)
axes[1].set_ylabel("kHz")
axes[1].set_title("SP（スペクトル包絡・音色）")
plt.colorbar(im1, ax=axes[1], label="dB", fraction=0.02, pad=0.01)

# AP（非周期成分）
im2 = axes[2].imshow(
    ap.T, aspect="auto", origin="lower",
    extent=[0, t[-1], 0, sr / 2 / 1000],
    cmap="hot", vmin=0, vmax=1,
)
axes[2].set_ylabel("kHz")
axes[2].set_title("AP（非周期成分・息の混じり）")
plt.colorbar(im2, ax=axes[2], fraction=0.02, pad=0.01)

# 波形比較
orig_t = np.linspace(0, len(audio) / sr, len(audio))
synth_t = np.linspace(0, len(audio_synth) / sr, len(audio_synth))
axes[3].plot(orig_t, audio, alpha=0.7, label="original", linewidth=0.4, color="steelblue")
axes[3].plot(synth_t, audio_synth, alpha=0.7, label="roundtrip", linewidth=0.4, color="orange")
axes[3].set_ylabel("振幅")
axes[3].set_xlabel("秒")
axes[3].set_title("波形比較（原音 vs 再合成）")
axes[3].legend(loc="upper right", fontsize=8)
axes[3].set_xlim([0, max(orig_t[-1], synth_t[-1])])
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = output_dir / (input_path.stem + "_analysis.png")
plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
plt.close()
print(f"プロット保存: {plot_path}")
print("完了。")
