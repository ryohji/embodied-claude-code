"""Phase 4: 声のクロス合成

亮志さんの「おはよう」を WORLD 分解し、
F0・AP・タイミングはそのままで SP だけをバンク SP に置き換えて再合成する。

目的:
  - 「聴く（WORLD 分析）」と「話す（バンク合成）」が同じパラメータ空間でつながる
  - バンク SP が実声のタイミング・F0 と合わせて動くかを確認
  - Whisper が認識できれば「バンク SP が正しい」の裏付けになる

実行方法:
  experiments/.venv/bin/python experiments/phase4_voice_cross.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyworld as pw
import soundfile as sf

BANK_PATH   = "experiments/output/vowel_bank.npz"
INPUT_WAV   = "experiments/output/ryohji_ohayo.wav"
OUTPUT_DIR  = Path("experiments/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 「おはよう」の音素と対応するバンクキー（AP が高い「h」部分はバンクを使わずそのまま通す）
# voiced フレームを音素ごとに比率で割り当てる
# h フェーズ（is_unvoiced_consonant=True）: AP をそのまま通し SP は bank["a"] × 0.15
PHONEME_PLAN = [
    # (label,  vowel_key,  voiced_ratio,  is_h_phase)
    ("o",      "o",        0.22,          False),
    ("h",      "a",        0.10,          True ),   # は の h 部分
    ("a",      "a",        0.18,          False),
    ("y→o",    "o",        0.25,          False),   # よ（y グライドも o として扱う）
    ("u",      "u",        0.25,          False),
]


def load_wav_float64(path, target_sr=None):
    audio, sr = sf.read(path, dtype="float64")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if target_sr and sr != target_sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // g, sr // g)
        print(f"  リサンプル: {sr} Hz → {target_sr} Hz")
        sr = target_sr
    return audio, sr


def main():
    bank_data = np.load(BANK_PATH)
    bank = {k: bank_data[k] for k in bank_data.files}
    print(f"バンク読み込み: {list(bank.keys())}")

    audio, sr = load_wav_float64(INPUT_WAV, target_sr=24000)
    print(f"\n入力: {INPUT_WAV}  {len(audio)/sr:.2f}s @ {sr}Hz")

    # WORLD 分析
    print("WORLD 分析中...")
    f0, sp, ap = pw.wav2world(audio, sr)
    t = np.arange(len(f0)) * pw.default_frame_period / 1000

    voiced_idx = np.where(f0 > 0)[0]
    print(f"有声フレーム: {len(voiced_idx)}  全フレーム: {len(f0)}")

    if len(voiced_idx) == 0:
        print("ERROR: 有声フレームがありません。")
        return

    # 有声フレームを PHONEME_PLAN の比率で切り分ける
    ratios = [p[2] for p in PHONEME_PLAN]
    total_ratio = sum(ratios)
    n_voiced = len(voiced_idx)

    sp_cross = sp.copy()  # SP のみ置き換える; F0・AP はそのまま

    cursor = 0
    segment_info = []
    for label, key, ratio, is_h in PHONEME_PLAN:
        n_frames = max(1, round(n_voiced * ratio / total_ratio))
        seg_idx = voiced_idx[cursor : cursor + n_frames]

        if is_h:
            # h フェーズ: フラットスペクトル（AP はそのまま高いはず）
            sp_h = np.full(bank["a"].shape, bank["a"].mean() * 0.15)
            sp_cross[seg_idx] = sp_h
        else:
            sp_cross[seg_idx] = bank[key]

        t_start = seg_idx[0]  * pw.default_frame_period / 1000 if len(seg_idx) > 0 else 0
        t_end   = seg_idx[-1] * pw.default_frame_period / 1000 if len(seg_idx) > 0 else 0
        print(f"  [{label:5s}] key={key}  frames={len(seg_idx):3d}"
              f"  t={t_start:.3f}–{t_end:.3f}s")
        segment_info.append((label, seg_idx))
        cursor += n_frames

    # バンク SP を使う範囲外（無声区間）は SP をそのまま通す
    sp_cross = np.clip(sp_cross, 1e-16, None)

    # 再合成
    print("\nWORLD 再合成中...")
    audio_cross = pw.synthesize(f0, sp_cross, ap, sr).astype(np.float32)
    out_wav = OUTPUT_DIR / "phase4_cross.wav"
    sf.write(str(out_wav, ) if isinstance(out_wav, str) else str(out_wav), audio_cross, sr)
    print(f"保存: {out_wav}  ({len(audio_cross)/sr:.2f}s)")

    # SP 比較プロット（実声 vs バンク置換）
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle("Phase 4: Voice cross synthesis  (Ryohji F0/AP + Bank SP)", fontsize=13)

    sp_db_orig  = 10 * np.log10(sp.T + 1e-10)
    sp_db_cross = 10 * np.log10(sp_cross.T + 1e-10)
    vmin, vmax = -80, 0

    im0 = axes[0, 0].imshow(sp_db_orig, aspect="auto", origin="lower",
                             extent=[0, t[-1], 0, sr/2/1000],
                             cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Original SP (Ryohji)")
    axes[0, 0].set_ylabel("kHz")
    plt.colorbar(im0, ax=axes[0, 0], label="dB", fraction=0.03)

    im1 = axes[0, 1].imshow(sp_db_cross, aspect="auto", origin="lower",
                             extent=[0, t[-1], 0, sr/2/1000],
                             cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("Cross SP (Bank SP replacing vowels)")
    axes[0, 1].set_ylabel("kHz")
    plt.colorbar(im1, ax=axes[0, 1], label="dB", fraction=0.03)

    # F0 曲線
    voiced_mask = f0 > 0
    axes[1, 0].scatter(t[voiced_mask], f0[voiced_mask], s=4, color="steelblue")
    axes[1, 0].scatter(t[~voiced_mask], np.zeros(np.sum(~voiced_mask)), s=3, color="lightgray")
    axes[1, 0].set_ylabel("Hz")
    axes[1, 0].set_title(f"F0 (Ryohji)  mean={f0[voiced_mask].mean():.0f} Hz")
    axes[1, 0].set_xlabel("sec")
    axes[1, 0].grid(True, alpha=0.3)

    # AP
    im3 = axes[1, 1].imshow(ap.T, aspect="auto", origin="lower",
                              extent=[0, t[-1], 0, sr/2/1000],
                              cmap="hot", vmin=0, vmax=1)
    axes[1, 1].set_title("AP (Ryohji, unchanged)")
    axes[1, 1].set_ylabel("kHz")
    axes[1, 1].set_xlabel("sec")
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.03)

    # 音素境界線
    for label, seg_idx in segment_info:
        if len(seg_idx) == 0:
            continue
        boundary = seg_idx[0] * pw.default_frame_period / 1000
        for ax_row in axes:
            for ax in ax_row:
                ax.axvline(boundary, color="red", linewidth=0.8, alpha=0.6)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "phase4_cross.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"プロット保存: {plot_path}")

    # 元音との SP カーブ比較（有声区間の平均）
    fig2, axes2 = plt.subplots(1, len(PHONEME_PLAN), figsize=(20, 4))
    fig2.suptitle("SP curve: Ryohji original vs Bank", fontsize=12)
    freq_bins = sp.shape[1]
    freqs = np.arange(freq_bins)

    for ax, (label, seg_idx) in zip(axes2, segment_info):
        if len(seg_idx) < 2:
            ax.set_title(f"{label} (skip)")
            continue
        sp_orig_mean = sp[seg_idx].mean(axis=0)
        sp_bank = sp_cross[seg_idx[0]]  # バンク SP（定数）
        ax.plot(freqs, 10*np.log10(sp_orig_mean+1e-10),
                color="steelblue", linewidth=1.5, label="orig")
        ax.plot(freqs, 10*np.log10(sp_bank+1e-10),
                color="tomato", linestyle="--", linewidth=1.5, label="bank")
        ax.set_title(label)
        ax.set_ylim(-80, 0)
        ax.set_xlabel("freq bin")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    axes2[0].set_ylabel("dB")
    plt.tight_layout()
    plot2_path = OUTPUT_DIR / "phase4_sp_compare.png"
    plt.savefig(str(plot2_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SP 比較プロット: {plot2_path}")
    print("\nPhase 4 完了。")


if __name__ == "__main__":
    main()
