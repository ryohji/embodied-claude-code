"""Phase 3 比較: Kokoro 実音声 vs バンク合成のパラメータ比較

kokoro_ohayo.wav（Kokoro 実音声）と phase3_ohayo.wav（バンク合成）を
WORLD 分析して F0・SP・AP を並べて比較する。

実行方法:
  experiments/.venv/bin/python experiments/phase3_compare.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyworld as pw
import soundfile as sf

OUTPUT_DIR = Path("experiments/output")
KOKORO_WAV  = str(OUTPUT_DIR / "kokoro_ohayo.wav")
SYNTH_WAV   = str(OUTPUT_DIR / "phase3_ohayo.wav")


def load_wav_float64(path: str):
    audio, sr = sf.read(path, dtype="float64")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def analyze(audio, sr):
    f0, sp, ap = pw.wav2world(audio, sr)
    t = np.arange(len(f0)) * pw.default_frame_period / 1000
    return f0, sp, ap, t


def sp_db(sp):
    return 10 * np.log10(sp.T + 1e-10)


def mora_stats(f0, sr):
    """有声フレームの F0 統計を返す。"""
    voiced = f0[f0 > 0]
    if len(voiced) == 0:
        return 0, 0, 0
    return voiced.mean(), voiced.min(), voiced.max()


def main():
    print("=== Phase 3 比較: Kokoro 実音声 vs バンク合成 ===\n")

    # 読み込みと分析
    audio_k, sr_k = load_wav_float64(KOKORO_WAV)
    audio_s, sr_s = load_wav_float64(SYNTH_WAV)

    print(f"Kokoro 実音声 : {len(audio_k)/sr_k:.3f}s @ {sr_k} Hz")
    print(f"バンク合成    : {len(audio_s)/sr_s:.3f}s @ {sr_s} Hz")

    f0_k, sp_k, ap_k, t_k = analyze(audio_k, sr_k)
    f0_s, sp_s, ap_s, t_s = analyze(audio_s, sr_s)

    voiced_k = f0_k > 0
    voiced_s = f0_s > 0

    mean_k, min_k, max_k = mora_stats(f0_k, sr_k)
    mean_s, min_s, max_s = mora_stats(f0_s, sr_s)

    print(f"\nF0 統計:")
    print(f"  Kokoro  mean={mean_k:.0f}  min={min_k:.0f}  max={max_k:.0f} Hz"
          f"  voiced={voiced_k.sum()}/{len(f0_k)} フレーム ({voiced_k.mean()*100:.0f}%)")
    print(f"  合成    mean={mean_s:.0f}  min={min_s:.0f}  max={max_s:.0f} Hz"
          f"  voiced={voiced_s.sum()}/{len(f0_s)} フレーム ({voiced_s.mean()*100:.0f}%)")

    print(f"\n継続時間比較:")
    print(f"  Kokoro  : {len(audio_k)/sr_k*1000:.0f} ms  ({len(f0_k)} フレーム)")
    print(f"  合成    : {len(audio_s)/sr_s*1000:.0f} ms  ({len(f0_s)} フレーム)")

    # ── プロット ──────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(18, 12))
    fig.suptitle("Phase 3 比較: Kokoro 実音声 (左) vs バンク合成 (右)", fontsize=13)

    vmin, vmax = -80, 0

    # F0
    for col, (t, f0, v_mask, label) in enumerate([
        (t_k, f0_k, voiced_k, f"Kokoro  mean={mean_k:.0f} Hz"),
        (t_s, f0_s, voiced_s, f"合成    mean={mean_s:.0f} Hz"),
    ]):
        ax = axes[0, col]
        ax.scatter(t[v_mask],  f0[v_mask],  s=5, color="steelblue", label="voiced")
        ax.scatter(t[~v_mask], np.zeros(np.sum(~v_mask)), s=3, color="lightgray", label="unvoiced")
        ax.set_ylabel("Hz")
        ax.set_title(f"F0  {label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 400)

    # SP
    for col, (t, sp, label) in enumerate([
        (t_k, sp_k, "Kokoro"),
        (t_s, sp_s, "合成"),
    ]):
        ax = axes[1, col]
        sr = sr_k if col == 0 else sr_s
        im = ax.imshow(sp_db(sp), aspect="auto", origin="lower",
                       extent=[0, t[-1], 0, sr/2/1000],
                       cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_ylabel("kHz")
        ax.set_title(f"SP (spectral envelope)  {label}")
        plt.colorbar(im, ax=ax, label="dB", fraction=0.02, pad=0.01)

    # AP
    for col, (t, ap, label) in enumerate([
        (t_k, ap_k, "Kokoro"),
        (t_s, ap_s, "合成"),
    ]):
        ax = axes[2, col]
        sr = sr_k if col == 0 else sr_s
        im = ax.imshow(ap.T, aspect="auto", origin="lower",
                       extent=[0, t[-1], 0, sr/2/1000],
                       cmap="hot", vmin=0, vmax=1)
        ax.set_ylabel("kHz")
        ax.set_xlabel("sec")
        ax.set_title(f"AP (aperiodicity)  {label}")
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "phase3_compare.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nプロット保存: {plot_path}")

    # ── モーラ境界の推定（無声→有声、有声→無声 の切り替わりを使う）──────
    print("\n=== Kokoro 実音声のイベント推定 ===")
    _print_voice_events(f0_k, t_k, "Kokoro")
    print("\n=== バンク合成のイベント推定 ===")
    _print_voice_events(f0_s, t_s, "合成")

    print("\nPhase 3 比較完了。")


def _print_voice_events(f0, t, label):
    """有声/無声の遷移点を列挙する。"""
    voiced = (f0 > 0).astype(int)
    transitions = np.diff(voiced)
    on_idx  = np.where(transitions ==  1)[0] + 1   # 無声→有声
    off_idx = np.where(transitions == -1)[0] + 1   # 有声→無声

    print(f"  有声開始フレーム: {on_idx.tolist()}")
    print(f"  有声終了フレーム: {off_idx.tolist()}")
    for i, (on, off) in enumerate(zip(on_idx, off_idx)):
        dur_ms = (off - on) * pw.default_frame_period
        f0_seg = f0[on:off]
        print(f"  セグメント {i+1}: {t[on]:.3f}–{t[off-1]:.3f}s  "
              f"({dur_ms:.0f} ms)  F0 mean={f0_seg.mean():.0f} Hz")


if __name__ == "__main__":
    main()
