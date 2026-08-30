"""
mhd_plot_bicoherence_lines.py
High-contrast publication bicoherence plot for Heliotron J #88653 (MP1).
Features:
  1. High-contrast colormap ('inferno') for optimal dynamic range.
  2. Analyzes the diagonal line f1 + f2 = f3 (slope = -1).
  3. Analyzes other important lines:
     - Harmonic generation line: f1 = f2 (slope = +1).
     - Primary pump mode column: f1 = 89.0 kHz.
     - Secondary coherent mode row: f2 = 41.0 kHz.
     - Low-frequency modulation row: f2 ~ 1-2 kHz.
  4. Eliminates obstructing circles: peak pixels are completely visible.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from scipy.ndimage import maximum_filter

import sys
sys.path.append(str(Path(__file__).parent.parent / "jpack"))
sys.path.append(str(Path(__file__).parent))
import turnelib as TE
import libana_signal as LAS
from mhd_analysis_obj3 import load_edf_signal, slice_window, statistical_b2_threshold

def generate_bicoherence_lines_plot(shot=88653, out_dir="."):
    out_dir = Path(out_dir)
    data_path = Path(r"c:\TFG\data") / f"hj{shot}" / f"MP1@{shot}.edf"

    t_start, t_end = 259.1, 275.0  # Stationary flattop sub-window
    t_sec, ys, dt, fs = load_edf_signal(data_path)
    t_sec_win, ys_win = slice_window(t_sec, ys, t_start, t_end)

    nfft = 1024
    noverlap = 512
    nensemble = 30
    b2_thresh_95 = statistical_b2_threshold(nensemble, alpha=0.05)

    f1, f2, bicoh2 = LAS.abicoh2(ys_win, t_sec_win, dt=dt, nfft=nfft, noverlap=noverlap,
                                 nensemble=nensemble, detrend='linear')
    f_psd, Pxx = LAS.psd(ys_win, t_sec_win, dt=dt, nfft=nfft, noverlap=noverlap,
                         nensemble=nensemble, detrend='linear')

    f1_khz = f1 / 1000.0
    f2_khz = f2 / 1000.0
    f_psd_khz = f_psd / 1000.0

    # Restrict to Sum domain (0 to 150 kHz)
    fmax_plot = 150.0
    idx1 = np.where((f1_khz >= 0.0) & (f1_khz <= fmax_plot))[0]
    idx2 = np.where((f2_khz >= 0.0) & (f2_khz <= fmax_plot))[0]

    sub_f1 = f1_khz[idx1]
    sub_f2 = f2_khz[idx2]
    sub_b2 = bicoh2[np.ix_(idx1, idx2)]

    # Non-redundant mask: f1 >= f2, f1+f2 <= fmax_plot, f >= 5 kHz
    F1, F2 = np.meshgrid(sub_f1, sub_f2, indexing='ij')
    search_mask = (F1 >= 5.0) & (F2 >= 5.0) & (F1 >= F2) & (F1 + F2 <= fmax_plot)

    # Find 2D local peaks (3x3 filter)
    df_khz = sub_f1[1] - sub_f1[0]
    filt_size = 5
    local_max = (sub_b2 == maximum_filter(sub_b2, size=filt_size)) & search_mask
    peak_coords = np.argwhere(local_max)
    peak_vals = sub_b2[local_max]
    sorted_idx = np.argsort(peak_vals)[::-1]

    top_peaks = []
    for p_i in sorted_idx[:6]:
        c = peak_coords[p_i]
        val = sub_b2[c[0], c[1]]
        p_f1 = sub_f1[c[0]]
        p_f2 = sub_f2[c[1]]
        top_peaks.append((val, p_f1, p_f2, p_f1 + p_f2))

    # Reset rcParams to standard sans-serif
    plt.rcdefaults()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.8), facecolor="white", gridspec_kw={'width_ratios': [1.15, 1.0]})
    fig.suptitle(
        f"Heliotron J #{shot} — Nonlinear Auto-Bicoherence ($b^2$) & Mode Triad Geometry\n"
        f"Mirnov Coil MP1 (Stationary Window: {t_start:.1f} – {t_end:.1f} ms, $N_\\mathrm{{ens}} = {nensemble}$, $b^2_{{95}} = {b2_thresh_95:.3f}$)",
        fontsize=12.5, fontweight="bold", y=0.98
    )

    # -------------------------------------------------------------------------
    # PANEL (a): 2D BICOHERENCE MAP (SUM DOMAIN) WITH CHARACTERISTIC LINES
    # -------------------------------------------------------------------------
    # High-contrast 'inferno' colormap
    vmax = 0.45
    levels = np.linspace(0.0, vmax, 120)
    cax = ax1.contourf(F1, F2, sub_b2, levels=levels, cmap="inferno", extend="max")
    cbar = fig.colorbar(cax, ax=ax1, label=r"Squared Auto-Bicoherence $b^2(f_1, f_2)$", pad=0.02)
    cbar.ax.axhline(b2_thresh_95, color="cyan", ls=":", lw=1.5, label="95% Stat. Floor")

    # 1. Diagonal Line 1: Upper-left to down-right border line (f1 + f2 = 150 kHz, slope = -1)
    ax1.plot([0, fmax_plot], [fmax_plot, 0], color="white", ls="--", lw=1.5, alpha=0.8,
             label=r"Diagonal $f_1 + f_2 = 150$ kHz (Slope $-1$)")

    # 2. Diagonal Line 2: Primary Mode Sum Resonance (f1 + f2 = 89.0 kHz, slope = -1)
    f_mode = 89.0
    ax1.plot([0, f_mode], [f_mode, 0], color="cyan", ls="-.", lw=1.6, alpha=0.9,
             label=rf"Primary Mode Sum $f_1 + f_2 = {f_mode:.0f}$ kHz (Slope $-1$)")

    # 3. Harmonic Generation Line (f1 = f2, slope = +1)
    ax1.plot([0, fmax_plot / 2.0], [0, fmax_plot / 2.0], color="lime", ls=":", lw=1.8, alpha=0.9,
             label=r"Harmonic Doubling $f_1 = f_2$ (Slope $+1$)")

    # 4. Vertical Line: Primary Pump Mode (f1 = 89.0 kHz)
    ax1.axvline(f_mode, color="gold", ls="--", lw=1.3, alpha=0.75, label=rf"Pump Column $f_1 = {f_mode:.0f}$ kHz")

    # 5. Horizontal Line: Secondary Mode (f2 = 41.0 kHz)
    f_sec = 41.0
    ax1.axhline(f_sec, color="magenta", ls=":", lw=1.3, alpha=0.75, label=rf"Secondary Row $f_2 = {f_sec:.0f}$ kHz")

    # Annotate Top Peaks WITHOUT obscuring circles (pure pointer arrows)
    offsets = [(8, 8), (-16, 12), (12, -12), (10, 10), (-16, -14), (8, -16)]
    for idx, (val, pf1, pf2, pf3) in enumerate(top_peaks[:4]):
        off_x, off_y = offsets[idx % len(offsets)]
        # Unobtrusive text with arrow pointing near the peak
        ax1.annotate(
            f"#{idx+1}\n($b^2$={val:.2f})",
            xy=(pf1, pf2), xytext=(pf1 + off_x, pf2 + off_y),
            color="white", fontsize=8.5, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="white", lw=1.2, shrinkA=2, shrinkB=3),
            path_effects=[path_effects.withStroke(linewidth=2.5, foreground="black")]
        )

    ax1.set_xlim(0, fmax_plot)
    ax1.set_ylim(0, fmax_plot)
    ax1.set_xlabel(r"$f_1$ (kHz)", fontsize=10.5)
    ax1.set_ylabel(r"$f_2$ (kHz)", fontsize=10.5)
    ax1.set_title(r"(a) Auto-Bicoherence Map $b^2(f_1, f_2)$ & Characteristic Lines", fontsize=11, fontweight="bold", pad=8)
    ax1.grid(True, ls=":", alpha=0.35, color="gray")
    ax1.legend(loc="upper right", fontsize=8.0, framealpha=0.92, facecolor="black", labelcolor="white")

    # -------------------------------------------------------------------------
    # PANEL (b): POWER SPECTRAL DENSITY (PSD) AND TRIAD ALIGNMENT
    # -------------------------------------------------------------------------
    mask_psd = (f_psd_khz >= 0) & (f_psd_khz <= fmax_plot)
    ax2.semilogy(f_psd_khz[mask_psd], Pxx[mask_psd], color="tab:blue", lw=1.8, label="Linear PSD (MP1)")

    # Highlight Physical Mode Triad
    ax2.axvline(f_mode, color="gold", ls="--", lw=1.6, label=rf"Primary Mode $f_1 \approx {f_mode:.1f}$ kHz")
    ax2.axvline(f_sec, color="magenta", ls=":", lw=1.6, label=rf"Secondary Mode $f_2 \approx {f_sec:.1f}$ kHz")
    f_diff = f_mode - f_sec
    ax2.axvline(f_diff, color="tab:green", ls="-.", lw=1.4, label=rf"Difference Wave $f_3 = f_1 - f_2 \approx {f_diff:.1f}$ kHz")

    # Also mark Top Peak #1 frequencies
    top1 = top_peaks[0]
    ax2.axvline(top1[1], color="red", ls=":", lw=1.2, alpha=0.7, label=rf"Top Sum Peak $f_1={top1[1]:.1f}, f_2={top1[2]:.1f}$ kHz")
    ax2.axvline(top1[2], color="red", ls=":", lw=1.2, alpha=0.7)

    ax2.set_xlim(0, fmax_plot)
    ax2.set_xlabel("Frequency (kHz)", fontsize=10.5)
    ax2.set_ylabel(r"Spectral Power (V$^2$/Hz)", fontsize=10.5)
    ax2.set_title(r"(b) Corresponding Plasma Spectral Peaks (Linear PSD)", fontsize=11, fontweight="bold", pad=8)
    ax2.grid(True, which="both", ls=":", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    plt.tight_layout()
    out_png = out_dir / f"mhd_bicoherence_characteristic_lines_shot_{shot}.png"
    plt.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"High-contrast bicoherence lines figure saved to: '{out_png}'")
    return str(out_png)

if __name__ == "__main__":
    generate_bicoherence_lines_plot(88653, out_dir=r"c:\TFG")
