"""Phase 2: 母音バンクの構築

各母音（あいうえお）を Kokoro で発声させ、WORLD で SP を抽出してバンクファイルに保存する。

実行方法（2ステップ）:

  # Step 1: 母音 WAV を生成（audio-speak-mcp の venv を使う）
  GEN=../audio-speak-mcp/.venv/bin/python
  for v in a i u e o; do
    $GEN experiments/gen_kokoro.py "$(python3 -c "print({'a':'あー','i':'いー','u':'うー','e':'えー','o':'おー'}['$v'])")" experiments/output/vowel_$v.wav
  done

  # Step 2: バンク抽出（experiments の venv を使う）
  experiments/.venv/bin/python experiments/phase2_vowel_bank.py

出力:
  experiments/output/vowel_bank.npz  — 母音ごとの SP ベクトル（1D, float64）
  experiments/output/vowel_bank.png  — スペクトル包絡の比較プロット
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyworld as pw
import soundfile as sf

VOWELS = {
    "a": ("あー", "experiments/output/vowel_a.wav"),
    "i": ("いー", "experiments/output/vowel_i.wav"),
    "u": ("うー", "experiments/output/vowel_u.wav"),
    "e": ("えー", "experiments/output/vowel_e.wav"),
    "o": ("おー", "experiments/output/vowel_o.wav"),
}

OUTPUT_DIR = Path("experiments/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_steady_sp(wav_path: str, vowel_label: str) -> np.ndarray | None:
    """WAV から WORLD 分析を行い、有声定常部分の SP を平均して返す。"""
    path = Path(wav_path)
    if not path.exists():
        print(f"  [SKIP] {path} が見つかりません。gen_kokoro.py で先に生成してください。")
        return None

    audio, sr = sf.read(str(path), dtype="float64")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    print(f"  分析中: {path.name}  ({len(audio)/sr:.2f}s, {sr} Hz)")
    f0, sp, ap = pw.wav2world(audio, sr)

    # 有声フレームのインデックスを取得
    voiced_idx = np.where(f0 > 0)[0]
    if len(voiced_idx) < 4:
        print(f"  [WARN] 有声フレームが少なすぎます ({len(voiced_idx)} フレーム)。")
        return None

    # 前半クリーン部分（onset 後 10% 〜 45%）を抽出する。
    # jf_alpha は孤立母音を上昇イントネーションで生成し、後半に off-glide が乗る。
    # 前半の方が音色が安定しているため、中央ではなく前寄りの窓を使う。
    n = len(voiced_idx)
    start = max(1, int(n * 0.10))
    end   = max(start + 2, int(n * 0.45))
    steady_idx = voiced_idx[start:end]

    sp_steady = sp[steady_idx]          # shape: (frames, freq_bins)
    sp_mean = sp_steady.mean(axis=0)    # shape: (freq_bins,)

    print(f"  有声フレーム: {len(voiced_idx)}  抽出窓 (10-45%): {len(steady_idx)} フレーム")
    print(f"  SP shape: {sp_mean.shape}")
    return sp_mean


def main():
    print("=== Phase 2: 母音バンク構築 ===\n")

    bank: dict[str, np.ndarray] = {}
    for key, (label, wav_path) in VOWELS.items():
        print(f"[{label}]")
        sp = extract_steady_sp(wav_path, key)
        if sp is not None:
            bank[key] = sp
        print()

    if not bank:
        print("ERROR: バンクに追加できた母音がありません。WAV を先に生成してください。")
        return

    # バンク保存
    bank_path = OUTPUT_DIR / "vowel_bank.npz"
    np.savez(str(bank_path), **bank)
    print(f"バンク保存: {bank_path}  ({list(bank.keys())} の SP を格納)")

    # 比較プロット
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {"a": "red", "i": "orange", "u": "green", "e": "blue", "o": "purple"}
    labels = {"a": "あ", "i": "い", "u": "う", "e": "え", "o": "お"}

    # SP を dB に変換して描画（周波数軸は便宜上インデックス）
    for key, sp_vec in bank.items():
        freq_bins = len(sp_vec)
        freqs = np.arange(freq_bins)
        sp_db = 10 * np.log10(sp_vec + 1e-10)
        ax.plot(freqs, sp_db, color=colors[key], label=labels[key], linewidth=1.5)

    ax.set_xlabel("周波数ビン")
    ax.set_ylabel("dB")
    ax.set_title("母音バンク: スペクトル包絡比較（SP 定常平均）")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = OUTPUT_DIR / "vowel_bank.png"
    plt.tight_layout()
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"プロット保存: {plot_path}")
    print("\n完了。vowel_bank.npz を Phase 3 で使用します。")


if __name__ == "__main__":
    main()
