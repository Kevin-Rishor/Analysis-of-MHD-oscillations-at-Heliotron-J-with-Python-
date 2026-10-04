"""Secondary-mode toroidal mode number (n) for the ECH+NBI and NBI-only windows.

Companion to mhd_compare_toroidal_and_radial.py (primary mode). Uses the *identical*
cross-phase fit (MP1/MP3/MP4, n in [-6, 6], RMSE of wrapped phase residuals) but targets
the secondary-band peak frequencies found by the poloidal decomposition
(mhd_compare_mode_structure.py): 48.8 kHz (ECH+NBI) and 41.0 kHz (NBI-only), so that the
m and n reported for the secondary mode refer to the same spectral line.

With three toroidal probes at separations of 90/180/270 deg, n is only determined modulo 4.

Outputs (single-panel figures in the same style as the primary-mode panel 2, plus JSON):
    Images/heating_phase_comparison/panels/
        mhd_secondary_mode_toroidal_n_shot_88653_<WIN>_panel2_measured_phase_vs_theoretical_toroidal_p.png
    Images/heating_phase_comparison/secondary_mode_toroidal_n_<WIN>.json
"""
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import scipy.signal as dsp
import matplotlib
matplotlib.use("Agg")
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

from comparison_common import (
    DEFAULT_SHOT, T_W1_START, T_W1_END, T_W2_START, T_W2_END,
    PROBE_ANGLES_TOR, load_signals, safe_savefig
)

DEFAULT_IMG_DIR = root_dir / "Images" / "heating_phase_comparison"


def run_secondary_toroidal(shot=DEFAULT_SHOT, img_dir=None):
    img_dir = Path(img_dir) if img_dir else DEFAULT_IMG_DIR
    panel_dir = img_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Secondary Mode Toroidal Number (n) Comparison (Shot {shot}) ---")
    time_ms, dt_s, fs, signals_dict, _ = load_signals(shot=shot)

    windows = [
        ("ECH + NBI Phase", "ECH_NBI", T_W1_START, T_W1_END, 48.8),
        ("Just NBI Phase", "NBI_only", T_W2_START, T_W2_END, 41.0),
    ]
    pairs = [("MP1", "MP3"), ("MP1", "MP4"), ("MP3", "MP4")]

    for title_win, tag, tw_start, tw_end, mode_freq in windows:
        idx_win = np.where((time_ms >= tw_start) & (time_ms <= tw_end))[0]
        n_samples = len(idx_win)
        nperseg = min(2048, n_samples // 2 if n_samples > 1000 else n_samples)
        noverlap = nperseg // 2

        measured = {}
        for p1, p2 in pairs:
            x1 = signals_dict[p1][idx_win]
            x2 = signals_dict[p2][idx_win]
            f_c, Pxy = dsp.csd(x1, x2, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
            _, coh = dsp.coherence(x1, x2, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
            f_khz = f_c / 1000.0
            i_f = int(np.argmin(np.abs(f_khz - mode_freq)))
            measured[(p1, p2)] = {
                "d_tor_deg": PROBE_ANGLES_TOR[p2] - PROBE_ANGLES_TOR[p1],
                "actual_f_khz": float(f_khz[i_f]),
                "gamma2": float(coh[i_f]),
                "phase_rad": float(np.angle(Pxy[i_f])),
                "phase_deg": float(np.degrees(np.angle(Pxy[i_f]))),
            }

        n_candidates = list(range(-6, 7))
        fits = {}
        for n in n_candidates:
            sq = 0.0
            for pdict in measured.values():
                theo = (n * np.radians(pdict["d_tor_deg"]) + np.pi) % (2 * np.pi) - np.pi
                diff = (pdict["phase_rad"] - theo + np.pi) % (2 * np.pi) - np.pi
                sq += diff ** 2
            fits[n] = float(np.degrees(np.sqrt(sq / len(pairs))))
        order = sorted(fits, key=fits.get)
        best_n, second_n = order[0], order[1]
        print(f"  {title_win}: f = {mode_freq:.1f} kHz -> best n = {best_n:+d} / {second_n:+d}, "
              f"RMSE = {fits[best_n]:.1f} deg; gamma2 = "
              + ", ".join(f"{a}-{b}: {measured[(a, b)]['gamma2']:.3f}" for a, b in pairs))

        # Single panel, same layout as panel 2 of the primary-mode toroidal figure
        plt.rcdefaults()
        fig, ax = plt.subplots(1, 1, figsize=(10, 3.67))
        d_angles_plot = np.linspace(-180, 270, 300)
        for n_test, col, ls in [(-1, "crimson", "-"), (+3, "navy", "--"), (+1, "gray", ":"), (+2, "darkorange", "-.")]:
            theo_line = [(n_test * np.radians(a) + np.pi) % (2 * np.pi) - np.pi for a in d_angles_plot]
            ax.plot(d_angles_plot, np.degrees(theo_line), color=col, ls=ls, lw=1.5,
                    label=f"Theory n = {n_test:+d} (RMSE={fits[n_test]:.1f} deg)")
        names = {("MP1", "MP3"): "MP1-MP3 (+180 deg)", ("MP1", "MP4"): "MP1-MP4 (+270 deg)",
                 ("MP3", "MP4"): "MP3-MP4 (+90 deg)"}
        for pr, pdict in measured.items():
            a, ph = pdict["d_tor_deg"], pdict["phase_deg"]
            ax.scatter([a], [ph], color="blue", s=90, zorder=5, edgecolors="black", lw=1.5)
            ax.annotate(f"{names[pr]}\n({ph:+.1f} deg)", (a, ph), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8.5, fontweight="bold")
        ax.set_xlim(-10, 290)
        ax.set_ylim(-195, 195)
        ax.set_xlabel(r"Toroidal Probe Separation $\Delta\varphi$ (deg)")
        ax.set_ylabel(r"Cross-Spectral Phase $\Delta\phi$ (deg)")
        ax.set_title(rf"Secondary Mode ({mode_freq:.1f} kHz): Measured vs. Theoretical Toroidal Phase "
                     r"$\Delta\phi = n \Delta\varphi$", fontsize=10)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_png = panel_dir / (f"mhd_secondary_mode_toroidal_n_shot_{shot}_{tag}"
                               "_panel2_measured_phase_vs_theoretical_toroidal_p.png")
        safe_savefig(fig, out_png, dpi=150)
        plt.close(fig)
        print(f"    Saved figure: {out_png.name}")

        out_json = img_dir / f"secondary_mode_toroidal_n_{tag}.json"
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump({
                "shot": shot, "window_ms": [tw_start, tw_end], "mode_freq_khz": mode_freq,
                "best_n": best_n, "second_best_n": second_n, "best_rmse_deg": fits[best_n],
                "rmse_by_n_deg": {str(k): v for k, v in fits.items()},
                "pairs": {f"{a}-{b}": {k: v for k, v in measured[(a, b)].items()} for a, b in pairs},
                "note": "Three toroidal probes: n is determined only modulo 4.",
            }, fh, indent=2)
        print(f"    Saved JSON: {out_json.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Secondary-mode toroidal n comparison (ECH+NBI vs NBI-only)")
    parser.add_argument("-s", "--shots", type=int, default=DEFAULT_SHOT, help="Shot number to analyze")
    parser.add_argument("-o", "--img-dir", type=str, default=str(DEFAULT_IMG_DIR), help="Image output directory")
    args = parser.parse_args()
    run_secondary_toroidal(shot=args.shots, img_dir=args.img_dir)
