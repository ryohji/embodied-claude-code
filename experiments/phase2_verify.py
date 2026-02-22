"""Phase 2 検証: バンク SP で再合成してスペクトログラム比較

各母音について:
  1. 元の WAV を WORLD 分解 → f0, sp_orig, ap を取得
  2. バンクの SP (sp_bank) を元の f0 と ap に組み合わせて再合成
  3. 元音 vs バンク再合成の SP カーブ + スペクトログラムを並べて可視化

実行方法:
  experiments/.venv/bin/python experiments/phase2_verify.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyworld as pw
import soundfile as sf

VOWELS = {
    "a": ("a", "experiments/output/vowel_a.wav"),
    "i": ("i", "experiments/output/vowel_i.wav"),
    "u": ("u", "experiments/output/vowel_u.wav"),
    "e": ("e", "experiments/output/vowel_e.wav"),
    "o": ("o", "experiments/output/vowel_o.wav"),
}
LABELS = {"a": "a (あ)", "i": "i (い)", "u": "u (う)", "e": "e (え)", "o": "o (お)"}

BANK_PATH = "experiments/output/vowel_bank.npz"
OUTPUT_DIR = Path("experiments/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_wav_float64(path: str):
    audio, sr = sf.read(path, dtype="float64")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def analyze(audio, sr):
    f0, sp, ap = pw.wav2world(audio, sr)
    return f0, sp, ap


def resynth_with_bank_sp(f0, sp_bank_vec, ap, sr) -> np.ndarray:
    """バンクの SP ベクトル（1D）を全フレームに適用して再合成する。"""
    n_frames = len(f0)
    sp_tiled = np.tile(sp_bank_vec, (n_frames, 1))  # (frames, freq_bins)
    audio = pw.synthesize(f0, sp_tiled, ap, sr)
    return audio.astype(np.float32)


def sp_to_db(sp):
    return 10 * np.log10(np.abs(sp) + 1e-10)


def main():
    bank = np.load(BANK_PATH)
    print(f"バンク読み込み: {list(bank.files)}\n")

    # 5 母音 × 3 列（SP カーブ比較 / 元音スペクトログラム / 再合成スペクトログラム）
    fig, axes = plt.subplots(5, 3, figsize=(18, 20))
    fig.suptitle("Phase 2 Verify: Original vs Bank-Resynth", fontsize=13)

    for row, (key, (label, wav_path)) in enumerate(VOWELS.items()):
        path = Path(wav_path)
        if not path.exists():
            print(f"[SKIP] {path} が見つかりません。")
            continue

        audio, sr = load_wav_float64(str(path))
        f0, sp_orig, ap = analyze(audio, sr)
        sp_bank_vec = bank[key]  # (513,)

        # 有声部の平均 SP
        voiced_idx = np.where(f0 > 0)[0]
        n = len(voiced_idx)
        start = max(1, int(n * 0.10))
        end   = max(start + 2, int(n * 0.45))
        sp_orig_mean = sp_orig[voiced_idx[start:end]].mean(axis=0)

        # バンク SP で再合成
        audio_bank = resynth_with_bank_sp(f0, sp_bank_vec, ap, sr)
        bank_wav_path = OUTPUT_DIR / f"vowel_{key}_bank_resynth.wav"
        sf.write(str(bank_wav_path), audio_bank, sr)

        # --- 列 0: SP カーブ比較 ---
        ax0 = axes[row, 0]
        freq_bins = len(sp_bank_vec)
        freqs = np.arange(freq_bins)
        ax0.plot(freqs, sp_to_db(sp_orig_mean), color="steelblue",
                 linewidth=1.5, label="orig (mean)")
        ax0.plot(freqs, sp_to_db(sp_bank_vec), color="tomato",
                 linewidth=1.5, linestyle="--", label="bank")
        ax0.set_xlim(0, freq_bins)
        ax0.set_ylim(-80, 0)
        ax0.set_ylabel("dB")
        ax0.set_title(f"{LABELS[key]}  SP curve")
        ax0.legend(fontsize=8)
        ax0.grid(True, alpha=0.3)

        # --- 列 1: 元音のスペクトログラム（WORLD SP） ---
        ax1 = axes[row, 1]
        t = np.arange(len(f0)) * pw.default_frame_period / 1000
        sp_db = sp_to_db(sp_orig.T)
        im1 = ax1.imshow(
            sp_db, aspect="auto", origin="lower",
            extent=[0, t[-1], 0, sr / 2 / 1000],
            cmap="viridis", vmin=-80, vmax=0,
        )
        ax1.set_ylabel("kHz")
        ax1.set_title(f"{LABELS[key]}  original SP")
        plt.colorbar(im1, ax=ax1, label="dB", fraction=0.03, pad=0.01)

        # --- 列 2: バンク再合成のスペクトログラム（flat SP） ---
        ax2 = axes[row, 2]
        n_frames = len(f0)
        sp_tiled = np.tile(sp_bank_vec, (n_frames, 1))
        sp_tiled_db = sp_to_db(sp_tiled.T)
        im2 = ax2.imshow(
            sp_tiled_db, aspect="auto", origin="lower",
            extent=[0, t[-1], 0, sr / 2 / 1000],
            cmap="viridis", vmin=-80, vmax=0,
        )
        ax2.set_ylabel("kHz")
        ax2.set_title(f"{LABELS[key]}  bank SP")
        plt.colorbar(im2, ax=ax2, label="dB", fraction=0.03, pad=0.01)

        voiced_ratio = len(voiced_idx) / len(f0) * 100
        f0_mean = f0[voiced_idx].mean() if len(voiced_idx) else 0
        print(f"[{key}] {len(audio)/sr:.2f}s  F0 mean={f0_mean:.0f} Hz  voiced={voiced_ratio:.0f}%"
              f"  bank_resynth -> {bank_wav_path.name}")

    # x 軸ラベルは最下行のみ
    for col in range(3):
        axes[-1, col].set_xlabel("sec" if col > 0 else "freq bin")

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "phase2_verify.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nプロット保存: {plot_path}")
    print("再合成 WAV: experiments/output/vowel_*_bank_resynth.wav")
    print("\nPhase 2 検証完了。")


if __name__ == "__main__":
    main()
