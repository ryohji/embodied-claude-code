"""Phase 3: パラメータ制御による合成

母音バンクの SP と指定した F0 曲線・継続時間から「おはよう」を合成する。

合成仕様:
  utterance = {
      "phonemes":  ["o", "ha", "yo", "u"],
      "durations": [100,  80,  150,  200],   # ms
      "f0_curve":  [260, 270,  280,  240],   # Hz（各モーラ中央のキーフレーム）
  }

子音の扱い:
  h (は): 最初の 30% を有気音（高 AP, SP は a の 10%）→ 残り 70% を bank["a"]
  y (よ): 最初の 25% を bank["i"] → bank["o"] へ線形ブレンド（硬口蓋接近音）

実行方法:
  experiments/.venv/bin/python experiments/phase3_synth.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyworld as pw
import soundfile as sf

BANK_PATH = "experiments/output/vowel_bank.npz"
OUTPUT_DIR = Path("experiments/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_PERIOD_MS = pw.default_frame_period  # 5.0 ms
SR = 24000

# ── 合成パラメータ ──────────────────────────────────────────
utterance = {
    "phonemes":  ["o",  "ha",  "yo",  "u"],
    "durations": [100,  110,   180,   200],   # ms（Kokoro実測値に合わせて短縮: 590ms）
    "f0_curve":  [255,  295,   325,   235],   # Hz（Kokoro実測: mean=293, max=331 Hz）
}
# ────────────────────────────────────────────────────────────


def ms_to_frames(ms: float) -> int:
    return max(1, round(ms / FRAME_PERIOD_MS))


def build_sp_segment(phoneme: str, n_frames: int, bank: dict) -> np.ndarray:
    """音素とフレーム数から SP 行列（n_frames × freq_bins）を生成する。"""
    freq_bins = len(next(iter(bank.values())))

    if phoneme == "o":
        return np.tile(bank["o"], (n_frames, 1))

    elif phoneme == "u":
        return np.tile(bank["u"], (n_frames, 1))

    elif phoneme == "a":
        return np.tile(bank["a"], (n_frames, 1))

    elif phoneme == "ha":
        # h 部分: 1 フレーム固定（~5ms）
        # SP は低エネルギーフラット（bank["a"] のフルエネルギーは「ごつん」になる）
        h_frames = 1
        sp = np.zeros((n_frames, freq_bins))
        sp_a = bank["a"]
        sp_h = np.full(freq_bins, sp_a.mean() * 0.15)
        sp[0] = sp_h
        sp[h_frames:] = sp_a
        return sp

    elif phoneme == "yo":
        # y 部分: 「い」を保持してから急速に「お」へ移行（硬口蓋接近音）
        # 構造: 20% 純粋「い」 → 30% 急速ブレンド → 50% 純粋「お」
        pure_i  = max(1, int(n_frames * 0.20))
        blend   = max(1, int(n_frames * 0.30))
        sp_i = bank["i"]
        sp_o = bank["o"]
        sp = np.zeros((n_frames, freq_bins))
        sp[:pure_i] = sp_i
        for i in range(blend):
            alpha = (i / max(blend - 1, 1)) ** 0.5   # 非線形: 最初は「い」寄り
            sp[pure_i + i] = sp_i * (1 - alpha) + sp_o * alpha
        sp[pure_i + blend:] = sp_o
        return sp

    else:
        raise ValueError(f"未対応の音素: {phoneme!r}")


def build_ap_segment(phoneme: str, n_frames: int) -> np.ndarray:
    """音素に応じた非周期成分（AP）を生成する。"""
    freq_bins = 513  # 24000 Hz の WORLD デフォルト

    if phoneme == "ha":
        h_frames = 1
        ap = np.zeros((n_frames, freq_bins))
        # h 部分: AP=0.40（白色ノイズではなく息混じりの「あ」に）
        ap[:h_frames] = 0.40
        # a 部分: 2 フレームで低 AP へ落とす
        drop = min(2, n_frames - h_frames)
        for i in range(drop):
            alpha = (i + 1) / (drop + 1)
            ap[h_frames + i] = 0.40 * (1 - alpha) + 0.01 * alpha
        if n_frames - h_frames - drop > 0:
            ap[h_frames + drop:] = 0.01
        return ap
    else:
        return np.full((n_frames, freq_bins), 0.01)


def build_f0_curve(phonemes: list, durations_ms: list, f0_keys: list) -> np.ndarray:
    """モーラごとの F0 キーフレームを線形補間して F0 配列を作る。"""
    # 各モーラの中央フレームに F0 キーフレームを置き、全体を線形補間
    total_frames = sum(ms_to_frames(d) for d in durations_ms)
    f0 = np.zeros(total_frames)

    # キーフレームの時刻（フレーム番号）を各モーラ中央に設定
    keyframe_positions = []
    cursor = 0
    for d in durations_ms:
        n = ms_to_frames(d)
        keyframe_positions.append(cursor + n // 2)
        cursor += n

    # 両端を外挿して補間
    xp = keyframe_positions
    fp = f0_keys
    # 全体を線形補間
    f0_interp = np.interp(np.arange(total_frames), xp, fp)

    # ha の h 部分（無声）は F0 = 0 にする
    cursor = 0
    for phoneme, d in zip(phonemes, durations_ms):
        n = ms_to_frames(d)
        # h フレームの F0=0 無声化を廃止。
        # F0 不連続（無声→有声）がクリックノイズの原因だった。
        # h は F0 を維持したまま AP=0.40 で有気音をシミュレートする。
        if False:  # noqa: disabled
            pass
        cursor += n

    return f0_interp


def smooth_sp_boundaries(sp: np.ndarray, window: int = 2) -> np.ndarray:
    """SP 行列の時間方向をガウシアン平滑化してセグメント境界のノイズを減らす。"""
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(sp, size=window, axis=0)


def main():
    bank_data = np.load(BANK_PATH)
    bank = {k: bank_data[k] for k in bank_data.files}
    print(f"バンク読み込み: {list(bank.keys())}\n")

    phonemes  = utterance["phonemes"]
    durations = utterance["durations"]
    f0_keys   = utterance["f0_curve"]

    print("=== 合成パラメータ ===")
    for p, d, f in zip(phonemes, durations, f0_keys):
        print(f"  {p:4s}  {d:3d} ms  F0={f} Hz  ({ms_to_frames(d)} フレーム)")

    # SP / AP 構築
    sp_segments = [build_sp_segment(p, ms_to_frames(d), bank)
                   for p, d in zip(phonemes, durations)]
    ap_segments = [build_ap_segment(p, ms_to_frames(d))
                   for p, d in zip(phonemes, durations)]

    sp_full = np.vstack(sp_segments)
    ap_full = np.vstack(ap_segments)

    # SP を母音セグメント内のみ平滑化する（子音境界を跨がない）
    # モーラ境界インデックスを計算して、各セグメントを個別に平滑化
    smoothed_segments = []
    for seg in sp_segments:
        if len(seg) >= 3:
            from scipy.ndimage import uniform_filter1d
            smoothed_segments.append(uniform_filter1d(seg, size=3, axis=0))
        else:
            smoothed_segments.append(seg)
    sp_full = np.vstack(smoothed_segments)
    sp_full = np.clip(sp_full, 1e-16, None)  # WORLD は正値を要求

    # F0 曲線
    f0_full = build_f0_curve(phonemes, durations, f0_keys)

    total_frames = len(f0_full)
    total_ms = total_frames * FRAME_PERIOD_MS
    voiced = np.sum(f0_full > 0)
    print(f"\n合計フレーム: {total_frames}  ({total_ms:.0f} ms)")
    print(f"有声フレーム: {voiced}  無声フレーム: {total_frames - voiced}")
    print(f"SP shape: {sp_full.shape}  AP shape: {ap_full.shape}")

    # 合成
    print("\nWORLD 合成中...")
    audio = pw.synthesize(f0_full, sp_full, ap_full, SR).astype(np.float32)

    out_wav = OUTPUT_DIR / "phase3_ohayo.wav"
    sf.write(str(out_wav), audio, SR)
    print(f"保存: {out_wav}  ({len(audio)/SR:.2f}s)")

    # ── プロット ──────────────────────────────────────────────
    t_frames = np.arange(total_frames) * FRAME_PERIOD_MS / 1000
    t_audio  = np.linspace(0, len(audio) / SR, len(audio))

    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    fig.suptitle("Phase 3: ohayo synthesis (WORLD parameter control)", fontsize=12)

    # F0
    voiced_mask = f0_full > 0
    axes[0].plot(t_frames[voiced_mask], f0_full[voiced_mask],
                 color="steelblue", linewidth=1.5, label="F0 (voiced)")
    axes[0].scatter(t_frames[~voiced_mask], np.zeros(np.sum(~voiced_mask)),
                    s=3, color="lightgray", label="unvoiced")
    axes[0].set_ylabel("Hz")
    axes[0].set_title("F0 curve")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # SP
    sp_db = 10 * np.log10(sp_full.T + 1e-10)
    im1 = axes[1].imshow(sp_db, aspect="auto", origin="lower",
                         extent=[0, t_frames[-1], 0, SR / 2 / 1000],
                         cmap="viridis", vmin=-80, vmax=0)
    axes[1].set_ylabel("kHz")
    axes[1].set_title("SP (spectral envelope)")
    plt.colorbar(im1, ax=axes[1], label="dB", fraction=0.02, pad=0.01)

    # AP
    im2 = axes[2].imshow(ap_full.T, aspect="auto", origin="lower",
                         extent=[0, t_frames[-1], 0, SR / 2 / 1000],
                         cmap="hot", vmin=0, vmax=1)
    axes[2].set_ylabel("kHz")
    axes[2].set_title("AP (aperiodicity)")
    plt.colorbar(im2, ax=axes[2], fraction=0.02, pad=0.01)

    # 波形
    axes[3].plot(t_audio, audio, linewidth=0.4, color="steelblue")
    axes[3].set_ylabel("amplitude")
    axes[3].set_xlabel("sec")
    axes[3].set_title("waveform")
    axes[3].grid(True, alpha=0.3)

    # モーラ境界線
    cursor = 0
    for phoneme, d in zip(phonemes, durations):
        n = ms_to_frames(d)
        boundary_t = cursor * FRAME_PERIOD_MS / 1000
        for ax in axes:
            ax.axvline(boundary_t, color="red", linewidth=0.8, alpha=0.5)
        cursor += n

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "phase3_ohayo.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"プロット保存: {plot_path}")
    print("\nPhase 3 合成完了。")


if __name__ == "__main__":
    main()
