import sys
import argparse
from pathlib import Path
import numpy as np
import scipy.signal as dsp
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from scipy.ndimage import maximum_filter

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

import libana_signal as LAS
from comparison_common import (
    DEFAULT_SHOT, DEFAULT_OUT_DIR,
    T_W1_START, T_W1_END, T_W2_START, T_W2_END,
    load_signals, safe_savefig
)


def run_bicoherence_comparison(shot=DEFAULT_SHOT, out_dir=None, verbose=False):
    if out_dir is None:
        out_dir = DEFAULT_OUT_DIR
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Objective 3 Bicoherence Matrix Comparison (Shot {shot}) ---")

    time_ms, dt_s, fs, signals_dict, _ = load_signals(shot=shot)
    if "MP1" not in signals_dict or time_ms is None:
        print("Error: MP1 signal not available.")
        return

    ys_mp1 = signals_dict["MP1"]

    windows = [
        ("ECH + NBI Phase", T_W1_START, T_W1_END, "mhd_bicoherence_objective3_88653_ECH_NBI.png", 30, 89.0),
        ("Just NBI Phase", T_W2_START, T_W2_END, "mhd_bicoherence_objective3_88653_NBI_only.png", 30, 86.0)
    ]

    for title_win, tw_start, tw_end, out_png, n_ens, f_pump in windows:
        print(f"  Processing Bicoherence: {title_win} ({tw_start:.1f} - {tw_end:.1f} ms)...")
        idx_win = np.where((time_ms >= tw_start) & (time_ms <= tw_end))[0]
        y_sub = ys_mp1[idx_win]
        t_sub_s = (time_ms[idx_win] - time_ms[idx_win][0]) / 1000.0

        nfft = 512
        noverlap = int(nfft * 0.5)
        f1, f2, bicoh2 = LAS.abicoh2(y_sub, t_sub_s, dt=dt_s, nfft=nfft, noverlap=noverlap, nensemble=n_ens, detrend='constant')

        ff1_k, ff2_k = np.meshgrid(f1 / 1000.0, f2 / 1000.0, indexing='ij')

        b2_thresh_95 = -np.log(0.05) / n_ens

        mask_diff = (ff1_k > 0) & (ff2_k < 0) & (ff1_k > -ff2_k) & (ff1_k >= 5.0) & (-ff2_k >= 5.0) & ((ff1_k + ff2_k) >= 5.0) & (ff1_k < 150.0) & (-ff2_k < 150.0)

        df_khz = (f1[1] - f1[0]) / 1000.0
        filter_size_bins = max(3, int(round(5.0 / df_khz)))
        if filter_size_bins % 2 == 0:
            filter_size_bins += 1

        local_max = (bicoh2 == maximum_filter(bicoh2, size=filter_size_bins)) & mask_diff
        coords = np.argwhere(local_max)
        vals = bicoh2[local_max]
        order = np.argsort(vals)[::-1]

        f_spec, tave_spec, S1 = LAS.periodogram(y_sub, t_sub_s, dt=dt_s, nfft=nfft, noverlap=noverlap, detrend='constant', fshift=True)
        f1_c, f2_c, bispct_complex = LAS.xbispct(S1, S1, f_spec, normalize=True, fshifted=True)

        diff_peaks = []
        for p_idx in order[:10]:
            r, c = coords[p_idx]
            val = float(bicoh2[r, c])
            pf1 = float(f1[r] / 1000.0)
            pf2 = float(-f2[c] / 1000.0)
            pf3 = float(pf1 - pf2)
            is_sig = val > b2_thresh_95

            r_c = np.argmin(np.abs(f1_c / 1000.0 - pf1))
            c_c = np.argmin(np.abs(f2_c / 1000.0 - (-pf2)))
            b_val = bispct_complex[r_c, c_c]
            im_b = np.imag(b_val)
            im_b_sum = -im_b
            direction = "UPWARD CASCADE" if im_b_sum > 0 else "DECAY"

            diff_peaks.append((val, pf1, pf2, pf3, is_sig, direction))
            if verbose:
                print(f"    Diff Triad ({pf1:5.1f} - {pf2:5.1f} = {pf3:5.1f} kHz) : b^2 = {val:.4f} [{'SIGNIFICANT' if is_sig else 'NOISE'}] -> [{direction}]")

        ff3_k = np.abs(ff1_k + ff2_k)
        valid_domain = (ff1_k > 0) & (ff1_k > np.abs(ff2_k)) & (bicoh2 > 0) & ~np.isnan(bicoh2)

        mask_40_80 = valid_domain & (ff3_k >= 40.0) & (ff3_k <= 80.0)
        mask_80_120 = valid_domain & (ff3_k >= 80.0) & (ff3_k <= 120.0)

        total_bicoh_40_80 = np.sum(bicoh2[mask_40_80])
        total_bicoh_80_120 = np.sum(bicoh2[mask_80_120])

        PLOT_FMAX_KHZ = 150.0
        display_mask = mask_diff
        if np.any(display_mask):
            color_vmax = float(np.nanmax(bicoh2[display_mask]))
        else:
            color_vmax = float(np.nanmax(bicoh2))
        color_vmax = max(color_vmax * 1.05, b2_thresh_95, 1e-6)
        color_levels = np.linspace(0.0, color_vmax, 101)

        # 2-Panel Figure
        plt.rcdefaults()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Panel 1: Squared Auto-Bicoherence b^2
        c = ax1.contourf(ff1_k, ff2_k, bicoh2, cmap='inferno', levels=color_levels, extend='max')
        ax1.set_title(f"Squared Auto-Bicoherence $b^2$ - Leading Channel: MP1 (nensemble={n_ens}, {title_win})", fontsize=11.5, fontweight='bold')
        ax1.set_xlabel(r"$f_1$ (kHz)", fontsize=10.5)
        ax1.set_ylabel(r"$f_2$ (kHz)", fontsize=10.5)
        ax1.grid(True, alpha=0.3, linestyle=':')
        ax1.set_xlim(0, PLOT_FMAX_KHZ)
        ax1.set_ylim(-PLOT_FMAX_KHZ, 0.0)

        # Reference lines
        ax1.plot([0, PLOT_FMAX_KHZ], [0, -0.5 * PLOT_FMAX_KHZ], color='cyan', ls='-.', lw=1.6, alpha=0.95,
                 label=r"Subharmonic Decay $f_2 = -0.5 f_1$ (Slope $-0.5$)")
        if "NBI" in title_win and "ECH" not in title_win:
            f1_casc = np.linspace(41.6, PLOT_FMAX_KHZ, 100)
            f2_casc = -0.5 * (f1_casc - 41.6)
            ax1.plot(f1_casc, f2_casc, color='magenta', ls='--', lw=1.6, alpha=0.95,
                     label=r"Secondary Cascade $f_2 = -0.5(f_1 - 41.6\ \mathrm{kHz})$")
        ax1.axvline(f_pump, color='gold', ls='--', lw=1.2, alpha=0.75, label=rf"Pump $f_1 = {f_pump:.0f}$ kHz")
        ax1.legend(loc='lower left', fontsize=8.0, framealpha=0.92, facecolor='white', edgecolor='gray')

        cbar = fig.colorbar(c, ax=ax1, label=r"Squared Bicoherence $b^2$")
        cbar.ax.axhline(b2_thresh_95, color='cyan', ls=':', lw=1.5)

        offsets_map = {
            1: (8, 6), 2: (-22, -10), 3: (8, 6), 4: (-20, 10), 5: (6, 8),
            6: (8, 6), 7: (-20, 8), 8: (8, -12), 9: (-22, 8), 10: (8, -12)
        }
        for idx, p in enumerate(diff_peaks[:10]):
            y_coord = -p[2]
            off_x, off_y = offsets_map.get(idx + 1, (8, 6))
            if p[1] > 125.0:
                off_x = -22.0
            elif p[1] < 18.0:
                off_x = 8.0
            if y_coord > -18.0:
                off_y = -14.0
            elif y_coord < -135.0:
                off_y = 12.0

            target_x = float(np.clip(p[1] + off_x, 6.0, 140.0))
            target_y = float(np.clip(y_coord + off_y, -144.0, -6.0))

            ax1.annotate(f"#{idx+1}", xy=(p[1], y_coord), xytext=(target_x, target_y),
                         color='white', fontsize=9.0, fontweight='bold', zorder=6,
                         arrowprops=dict(arrowstyle="->", color='white', lw=1.1, alpha=0.85, shrinkA=2, shrinkB=3),
                         path_effects=[path_effects.withStroke(linewidth=2.5, foreground="black")])

        subh_peaks = [f"#{idx+1}" for idx, p in enumerate(diff_peaks[:10]) if abs(p[2] / p[1] - 0.5) < 0.05]
        pump_peaks = [f"#{idx+1} ({p[1]:.1f} - {p[2]:.1f} = {p[3]:.1f} kHz)" for idx, p in enumerate(diff_peaks[:10]) if abs(p[1] - f_pump) < 6.0]
        pump_str = "\n  ".join(pump_peaks[:2]) if pump_peaks else "None"
        extra_badge_sec = ""
        if "NBI" in title_win and "ECH" not in title_win:
            sec_casc_peaks = [f"#{idx+1}" for idx, p in enumerate(diff_peaks[:10]) if abs(p[2] - 0.5 * (p[1] - 41.6)) < 8.0]
            extra_badge_sec = f"\n\u2022 Sec. Cascade (f1-int = 41.6 kHz):\n  Peaks {', '.join(sec_casc_peaks)}"

        badge_text = (
            f"Leading: f1={diff_peaks[0][1]:.1f}, f2={diff_peaks[0][2]:.1f}, f3={diff_peaks[0][3]:.1f} kHz | b\u00b2={diff_peaks[0][0]:.3f}\n"
            f"95% Stat. Floor: b\u00b2 = {b2_thresh_95:.3f}\n\n"
            "Non-Linear Triad Clustering:\n"
            "\u2022 Subharmonic Line (Slope -0.5):\n"
            f"  Peaks {', '.join(subh_peaks) if subh_peaks else 'None'}\n"
            f"\u2022 Pump Column (f1 = {f_pump:.0f} kHz):\n"
            f"  {pump_str}"
            f"{extra_badge_sec}"
        )
        ax1.text(0.98, 0.03, badge_text, transform=ax1.transAxes, fontsize=8.0,
                 va="bottom", ha="right", bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray", alpha=0.9))

        # Panel 2: Linear PSD
        f_psd, Pxx = LAS.psd(y_sub, t_sub_s, dt=dt_s, nfft=nfft, noverlap=noverlap, nensemble=n_ens, detrend='linear')
        f_psd_k = f_psd / 1000.0
        mask_psd = (f_psd_k >= 0) & (f_psd_k <= 150.0)

        ax2.semilogy(f_psd_k[mask_psd], Pxx[mask_psd], color='tab:blue', linewidth=1.8, label=f"PSD MP1 ({title_win})")
        ax2.set_title("Power Spectral Density (PSD) of Leading Channel: MP1", fontsize=11, fontweight="bold", pad=8)
        ax2.set_xlabel("Frequency (kHz)", fontsize=10)
        ax2.set_ylabel(r"Power (V$^2$/Hz)", fontsize=10)
        ax2.grid(True, alpha=0.3, which='both', linestyle=':')
        ax2.set_xlim(0, 150.0)

        p1 = diff_peaks[0]
        ax2.axvline(x=p1[1], color='#d62728', linestyle='--', linewidth=1.4, label=f"Peak #1: $f_1$={p1[1]:.1f} kHz ($b^2$={p1[0]:.2f})")
        ax2.axvline(x=p1[2], color='#d62728', linestyle=':', linewidth=1.4, label=f"Peak #1: $f_2$={p1[2]:.1f} kHz")

        prim_peak = next((p for p in diff_peaks if abs(p[1] - f_pump) <= 6.0 or abs(p[2] - f_pump) <= 6.0), None)
        if prim_peak and prim_peak != p1:
            ax2.axvline(x=prim_peak[1], color='darkorange', linestyle='--', linewidth=1.6, label=f"Primary Triad: $f_1$={prim_peak[1]:.1f} kHz ($b^2$={prim_peak[0]:.2f})")
            ax2.axvline(x=prim_peak[2], color='darkorange', linestyle=':', linewidth=1.6, label=f"Primary Triad: $f_2$={prim_peak[2]:.1f} kHz")
            ax2.axvline(x=prim_peak[3], color='darkorange', linestyle='-.', linewidth=1.2, alpha=0.7, label=f"Primary Triad: $f_3$={prim_peak[3]:.1f} kHz")
        elif len(diff_peaks) > 1:
            p2 = diff_peaks[1]
            ax2.axvline(x=p2[1], color='#2ca02c', linestyle='--', linewidth=1.3, label=f"Peak #2: $f_1$={p2[1]:.1f} kHz ($b^2$={p2[0]:.2f})")
            ax2.axvline(x=p2[2], color='#2ca02c', linestyle=':', linewidth=1.3, label=f"Peak #2: $f_2$={p2[2]:.1f} kHz")

        ax2.legend(loc='upper right', fontsize=8.5, framealpha=0.92)

        plt.tight_layout()
        out_path = out_dir / out_png
        safe_savefig(fig, out_path, dpi=150)
        plt.close(fig)
        print(f"    Saved figure: {out_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bicoherence Matrix Comparison (ECH+NBI vs Just NBI)")
    parser.add_argument("-s", "--shots", type=int, default=DEFAULT_SHOT, help="Shot number to analyze")
    parser.add_argument("-o", "--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print triad listings")
    args = parser.parse_args()

    run_bicoherence_comparison(shot=args.shots, out_dir=args.out_dir, verbose=args.verbose)
