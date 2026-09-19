import sys
import argparse
from pathlib import Path
import numpy as np
import scipy.signal as dsp
import matplotlib.pyplot as plt

current = Path(__file__).resolve().parent
root_dir = None
for p in [current] + list(current.parents):
    if (p / "jpack").exists():
        root_dir = p
        break
if root_dir is None:
    root_dir = Path("c:/TFG")

for p_add in [root_dir / "jpack", root_dir / "analysis", root_dir / "analysis" / "common", current]:
    if str(p_add) not in sys.path:
        sys.path.append(str(p_add))

from mhd_common import extract_instantaneous_frequency
from comparison_common import (
    DEFAULT_SHOT, DEFAULT_OUT_DIR,
    T_W1_START, T_W1_END, T_W2_START, T_W2_END,
    load_signals, safe_savefig
)


def run_self_coupling_comparison(shot=DEFAULT_SHOT, out_dir=None, verbose=False):
    if out_dir is None:
        out_dir = DEFAULT_OUT_DIR
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Self-Coupling Coherence Comparison (Shot {shot}) ---")

    time_ms, dt_s, fs, signals_dict, _ = load_signals(shot=shot)
    if "MP1" not in signals_dict or "MP3" not in signals_dict or "MP4" not in signals_dict:
        print("Error: Required Mirnov probes not available.")
        return

    windows = [
        ("ECH + NBI Phase", T_W1_START, T_W1_END, 80000.0, 120000.0, "mhd_self_coupling_coherence_shot_88653_ECH_NBI.png", 89.0),
        ("Just NBI Phase", T_W2_START, T_W2_END, 75000.0, 105000.0, "mhd_self_coupling_coherence_shot_88653_NBI_only.png", 86.0),
    ]

    probes = ["MP1", "MP3", "MP4"]
    colors = {"MP1": "tab:blue", "MP3": "tab:purple", "MP4": "tab:green"}
    styles = {"MP1": "-", "MP3": "-.", "MP4": "--"}

    for title_win, tw_start, tw_end, fl_hz, fu_hz, out_png, mode_f in windows:
        print(f"  Processing Self-Coupling: {title_win} ({tw_start:.1f} - {tw_end:.1f} ms)...")
        idx_win = np.where((time_ms >= tw_start) & (time_ms <= tw_end))[0]
        n_samples = len(idx_win)

        results = {}
        for p in probes:
            ys = signals_dict[p]
            env, _, _, _ = extract_instantaneous_frequency(ys, fs, fl_hz, fu_hz, 4, 325)
            ys_win = ys[idx_win]
            env_win = env[idx_win]

            nperseg = min(2048, n_samples // 2 if n_samples > 1000 else n_samples)
            noverlap = int(nperseg * 0.75)
            nfft = max(2048, nperseg * 2)

            f, Pxy = dsp.csd(ys_win, env_win, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, nfft=nfft)
            f, coh2 = dsp.coherence(ys_win, env_win, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, nfft=nfft)

            f_khz = f / 1000.0
            phase_deg = np.degrees(np.angle(Pxy))
            n_seg = int(np.floor((len(ys_win) - nperseg) / (nperseg - noverlap))) + 1

            results[p] = {"f_khz": f_khz, "coh2": coh2, "phase_deg": phase_deg, "n_seg": n_seg}

        n_seg_use = list(results.values())[0]["n_seg"]
        sig_floor = 1.0 - (0.05) ** (1.0 / max(1, n_seg_use - 1))

        plt.rcdefaults()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.5), sharex=True, facecolor="white")
        fig.suptitle(
            f"Heliotron J #{shot} — Self-Coupling Coherence: $\\tilde{{B}}$ vs. Mode Envelope\n"
            f"({title_win}: Active Window {tw_start:.1f} – {tw_end:.1f} ms, Mode ~ {mode_f:.1f} kHz)",
            fontsize=12, fontweight="bold", y=0.97
        )

        for p in probes:
            r = results[p]
            ax1.plot(r["f_khz"], r["coh2"], color=colors[p], ls=styles[p], lw=1.8, label=f"{p} (Self: $\\tilde{{B}}_{{{p}}}$ vs. $\\mathrm{{Env}}_{{{p}}}$)")

        ax1.axhline(sig_floor, color="dimgray", ls=":", lw=1.4, label=rf"95% Confidence Noise Floor ($\gamma^2 = {sig_floor:.2f}$)")
        ax1.text(4.92, sig_floor + 0.008, f"95% Floor ({sig_floor:.2f})", color="dimgray", fontsize=8.5, fontweight="bold", ha="right", va="bottom")
        ax1.axvspan(0.8, 2.5, color="gray", alpha=0.08, label="Coupling band ($0.8 - 2.5$ kHz)")

        m_band = (results["MP1"]["f_khz"] >= 0.5) & (results["MP1"]["f_khz"] <= 3.0)
        idx_pk_mod = np.argmax(results["MP1"]["coh2"][m_band])
        f_pk_mod = results["MP1"]["f_khz"][m_band][idx_pk_mod]
        coh_pk_mod = results["MP1"]["coh2"][m_band][idx_pk_mod]
        ax1.scatter([f_pk_mod], [coh_pk_mod], color="tab:blue", s=50, zorder=5, edgecolors="white", lw=1.0)
        ax1.annotate(f"{f_pk_mod:.2f} kHz (MP1)", xy=(f_pk_mod, coh_pk_mod), xytext=(f_pk_mod, coh_pk_mod + 0.05),
                     ha="center", fontsize=8.5, fontweight="bold", color="tab:blue", arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.2))

        ax1.set_xlim(0.0, 5.0)
        ax1.set_ylim(0.0, max(0.35, coh_pk_mod * 1.3))
        ax1.set_ylabel(r"Coherence $\gamma^2$", fontsize=10.5)
        ax1.set_title(r"(a) Self-Coherence $\gamma^2(\tilde{B}, \mathrm{Envelope})$", fontsize=11, fontweight="bold", pad=6)
        ax1.grid(True, ls=":", alpha=0.4)
        ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.95)

        for p in probes:
            r = results[p]
            ax2.plot(r["f_khz"], r["phase_deg"], color=colors[p], ls=styles[p], lw=1.8, label=f"{p}")

        ax2.axvspan(0.8, 2.5, color="gray", alpha=0.08)
        ax2.axvline(f_pk_mod, color="tab:blue", ls="--", lw=1.0, alpha=0.6)
        ax2.set_xlim(0.0, 5.0)
        ax2.set_ylim(-185.0, 185.0)
        ax2.set_yticks([-180, -90, 0, 90, 180])
        ax2.set_xlabel("Modulation Frequency (kHz)", fontsize=10.5)
        ax2.set_ylabel("Phase (deg)", fontsize=10.5)
        ax2.set_title(r"(b) Cross-Spectral Phase $\Delta\phi$", fontsize=11, fontweight="bold", pad=6)
        ax2.grid(True, ls=":", alpha=0.4)
        ax2.legend(loc="lower right", fontsize=8.5, framealpha=0.95)

        plt.tight_layout()
        out_path = out_dir / out_png
        safe_savefig(fig, out_path, dpi=150)
        plt.close(fig)
        print(f"    Saved figure: {out_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Coupling Coherence Comparison")
    parser.add_argument("-s", "--shots", type=int, default=DEFAULT_SHOT, help="Shot number to analyze")
    parser.add_argument("-o", "--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    run_self_coupling_comparison(shot=args.shots, out_dir=args.out_dir, verbose=args.verbose)
