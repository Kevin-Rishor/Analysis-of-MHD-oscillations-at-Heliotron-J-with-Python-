import sys
from pathlib import Path
import numpy as np
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

for p_add in [root_dir / "jpack", root_dir / "analysis", root_dir / "analysis" / "common"]:
    if str(p_add) not in sys.path:
        sys.path.append(str(p_add))

import turnelib as TE
import libana_signal as LAS
from mhd_analysis_obj3 import load_edf_signal, slice_window


def generate_top10_plot(shot=88653, out_dir=None):
    if out_dir is None:
        out_dir = root_dir
    else:
        out_dir = Path(out_dir)

    data_path = root_dir / "data" / f"hj{shot}" / f"MP1@{shot}.edf"

    t_start, t_end = 259.1, 275.0
    t_sec, ys, dt, fs = load_edf_signal(data_path)
    t_sec_win, ys_win = slice_window(t_sec, ys, t_start, t_end)

    nfft = 1024
    noverlap = 512
    nensemble = 30
    b2_thresh_95 = 0.100

    f1, f2, bicoh2 = LAS.abicoh2(ys_win, t_sec_win, dt=dt, nfft=nfft, noverlap=noverlap,
                                 nensemble=nensemble, detrend='linear')
    f_psd, Pxx = LAS.psd(ys_win, t_sec_win, dt=dt, nfft=nfft, noverlap=noverlap,
                         nensemble=nensemble, detrend='linear')

    f1_khz = f1 / 1000.0
    f2_khz = f2 / 1000.0
    f_psd_khz = f_psd / 1000.0

    idx1 = np.where((f1_khz >= 0.0) & (f1_khz <= 150.0))[0]
    idx2 = np.where((f2_khz >= -150.0) & (f2_khz <= 0.0))[0]

    sub_f1 = f1_khz[idx1]
    sub_f2 = f2_khz[idx2]
    sub_b2 = bicoh2[np.ix_(idx1, idx2)]

    F1, F2 = np.meshgrid(sub_f1, sub_f2, indexing='ij')
    mask_diff = (F1 >= 5.0) & (-F2 >= 5.0) & (F1 > -F2) & ((F1 + F2) >= 5.0)

    local_max = (sub_b2 == maximum_filter(sub_b2, size=5)) & mask_diff
    coords = np.argwhere(local_max)
    vals = sub_b2[local_max]
    order = np.argsort(vals)[::-1]

    plt.rcdefaults()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.2, 7.0), facecolor='white', gridspec_kw={'width_ratios': [1.18, 1.0]})
    fig.suptitle(
        f"Heliotron J #{shot} - 10 Leading Non-Linear Couplings & Dual-Regime Geometry (MP1)\n"
        f"Stationary Window: {t_start:.1f} - {t_end:.1f} ms (N_ens = {nensemble}, 95% Confidence Floor b^2_95 = {b2_thresh_95:.3f})",
        fontsize=12.5, fontweight='bold', y=0.98
    )

    levels = np.linspace(0.0, 0.46, 120)
    c = ax1.contourf(F1, F2, sub_b2, levels=levels, cmap='inferno', extend='max')
    cbar = fig.colorbar(c, ax=ax1, label=r"Squared Auto-Bicoherence $b^2(f_1, f_2)$", pad=0.02)
    cbar.ax.axhline(b2_thresh_95, color='cyan', ls=':', lw=1.5, label="95% Stat. Floor")

    ax1.plot([0, 150], [0, -150], color='white', ls='--', lw=1.3, alpha=0.7, label=r"Boundary $f_1 - f_2' = 0$ (Slope $-1$)")
    ax1.plot([0, 150], [0, -75], color='cyan', ls='-.', lw=1.8, alpha=0.95,
             label=r"Subharmonic Line $f_2 = -0.5 f_1$ (7 of Top 10 Peaks)")
    ax1.axvline(89.0, color='gold', ls='--', lw=1.4, alpha=0.85, label=r"Primary Pump Column $f_1 = 89$ kHz (#3, #7)")

    offsets_map = {
        1: (8, 6),
        2: (-22, -10),
        3: (8, 6),
        4: (-20, 10),
        5: (6, 8),
        6: (8, 6),
        7: (-20, 8),
        8: (8, -12),
        9: (-22, 8),
        10: (8, -12)
    }

    for rank, idx in enumerate(order[:10], 1):
        c_pt = coords[idx]
        pf1 = sub_f1[c_pt[0]]
        pf2 = sub_f2[c_pt[1]]
        off_x, off_y = offsets_map.get(rank, (8, 6))

        ax1.annotate(
            f"#{rank}", xy=(pf1, pf2), xytext=(pf1 + off_x, pf2 + off_y),
            color="white", fontsize=9.0, fontweight="bold", zorder=6,
            arrowprops=dict(arrowstyle="->", color="white", lw=1.1, alpha=0.85, shrinkA=2, shrinkB=3),
            path_effects=[path_effects.withStroke(linewidth=2.5, foreground="black")]
        )

    ax1.set_xlim(0, 150)
    ax1.set_ylim(-150, 0)
    ax1.set_xlabel(r"$f_1$ (kHz)", fontsize=10.5)
    ax1.set_ylabel(r"$f_2$ (kHz)", fontsize=10.5)
    ax1.set_title(r"(a) 2D Bicoherence Map: Top 10 Couplings", fontsize=11, fontweight="bold", pad=8)
    ax1.grid(True, ls=":", alpha=0.35, color="gray")
    ax1.legend(loc="lower left", fontsize=8.0, framealpha=0.9, facecolor="white", edgecolor="gray")

    box_text = (
        "Non-Linear Triad Clustering:\n"
        "• Subharmonic Line (Slope -0.5):\n"
        "  Peaks #1, #2, #4, #5, #8, #9, #10\n"
        "• Pump Column (f1 = 89 kHz):\n"
        "  Peak #3 (89.8 - 41.0 = 48.8 kHz)\n"
        "  Peak #7 (84.0 - 23.4 = 60.5 kHz)\n"
        "95% Confidence Floor: b² = 0.100"
    )
    ax1.text(0.98, 0.03, box_text, transform=ax1.transAxes, fontsize=8.2,
             va="bottom", ha="right", bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray", alpha=0.9))

    mask_psd = (f_psd_khz >= 0) & (f_psd_khz <= 150)
    ax2.semilogy(f_psd_khz[mask_psd], Pxx[mask_psd], color="tab:blue", lw=1.8, label="Linear PSD (MP1)")
    ax2.axvline(89.8, color="gold", ls="--", lw=1.5, label=r"Primary Pump $f_1 = 89.8$ kHz")
    ax2.axvline(41.0, color="magenta", ls=":", lw=1.5, label=r"Secondary Mode $f_2 = 41.0$ kHz")
    f_diff = 89.8 - 41.0
    ax2.axvline(f_diff, color="tab:green", ls="-.", lw=1.4, label=rf"Daughter Wave $f_3 = {f_diff:.1f}$ kHz")

    ax2.axvline(98.6, color="cyan", ls=":", lw=1.2, alpha=0.75, label="Subharmonic #1 ($f_1=98.6$ kHz)")
    ax2.axvline(53.7, color="cyan", ls="--", lw=1.2, alpha=0.75, label="Subharmonic #4 ($f_1=53.7$ kHz)")
    ax2.axvline(15.6, color="cyan", ls="-.", lw=1.2, alpha=0.75, label="Subharmonic #5 ($f_1=15.6$ kHz)")

    ax2.set_xlim(0, 150)
    ax2.set_xlabel("Frequency (kHz)", fontsize=10.5)
    ax2.set_ylabel(r"Spectral Power (V$^2$/Hz)", fontsize=10.5)
    ax2.set_title(r"(b) Corresponding Spectral Power (PSD)", fontsize=11, fontweight="bold", pad=8)
    ax2.grid(True, which="both", ls=":", alpha=0.35)
    ax2.legend(loc="upper right", fontsize=8.2, framealpha=0.92)

    plt.tight_layout()
    out_png = out_dir / f"mhd_bicoherence_top10_shot_{shot}.png"
    plt.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Top 10 bicoherence figure saved to: '{out_png}'")
    return str(out_png)


if __name__ == "__main__":
    generate_top10_plot(88653)

