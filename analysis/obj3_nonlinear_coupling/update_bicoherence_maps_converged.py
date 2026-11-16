#!/usr/bin/env python3
"""
update_bicoherence_maps_converged.py

Generates:
1. Images/obj3_nonlinear_coupling/fig_bicoherence_maps_both_shots.png (for Results.tex Figure 18)
   - 2x2 grid (shots #88653 & #88654, ECH+NBI & NBI-only, 10-125 kHz)
   - Numbered markers 1-4 for validated converged triads & noise control

2. Images/obj3_nonlinear_coupling/corrected_bicoherence_map_ECH_NBI.png and
   Images/obj3_nonlinear_coupling/corrected_bicoherence_map_NBI_only.png (for presentation_v3.tex Slide 16)
   - Difference plane bicoherence maps
   - Numbered markers 1-4 for validated converged triads & noise control
   - Updated information badge and legend
"""

import sys
from pathlib import Path
import numpy as np
import scipy.signal as dsp
from scipy.ndimage import maximum_filter
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.transforms as mtransforms

# Ensure common paths
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "analysis" / "common"))
from abicoh2_corrected import abicoh2_corrected

OUT_DIR = ROOT_DIR / "Images" / "obj3_nonlinear_coupling"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# PART 1: Generate Thesis Figure 18 (fig_bicoherence_maps_both_shots.png)
# =============================================================================
print("=" * 80)
print("PART 1: Generating Thesis Figure 18 (fig_bicoherence_maps_both_shots.png)")
print("=" * 80)

FS = 1.0e6
NFFT = 512
NOV = 256
FMIN = 10.0
FMAX = 125.0
WIN = [("ECH+NBI", 259.1, 267.0), ("NBI-only", 291.3, 299.2)]

df = FS / NFFT / 1e3
f = np.arange(256) * df
sel = (f >= FMIN) & (f <= FMAX)
fs_ = f[sel]

MAPS = {}
for s in ("88653", "88654"):
    c = np.load(ROOT_DIR / "cache" / f"shot_{s}_full_record_cache.npz")
    t = np.round(c["t_ms"].astype(float), 4)
    for nm, a, b in WIN:
        i0 = int(np.searchsorted(t, a))
        sig = c["mp1"][i0:i0 + 7936]
        _, b2 = abicoh2_corrected(sig, fs=FS, nfft=NFFT, noverlap=NOV)
        Z = b2[:256, :256][np.ix_(sel, sel)]
        MAPS[(s, nm)] = Z

vmax = max(float(Z.max()) for Z in MAPS.values())

fig, axs = plt.subplots(2, 2, figsize=(10.2, 10.2))
plt.subplots_adjust(hspace=0.28, wspace=0.32, top=0.95, bottom=0.22, left=0.09, right=0.86)

# Define triads for thesis figure (sum quadrant coordinates)
# (f1, f2, label, offset_x, offset_y, color)
TRIADS_53 = [
    (87.9, 87.9, "1", 12.0, 10.0, "#00e5ff", "Primary Sum Harmonic (88+88->176)"),
    (82.0, 33.2, "2", 12.0, -10.0, "#ffd600", "Difference Triad (115-33->82)"),
    (46.9, 80.1, "3", -12.0, 10.0, "#ff4081", "Cross-Mode Triad (47+80->127)"),
    (33.2, 80.1, "4", -14.0, -12.0, "#ffffff", "Noise Control (33+80->113)"),
]

TRIADS_54 = [
    (87.9, 87.9, "1", 12.0, 10.0, "#00e5ff", "Primary Sum Harmonic (88+88->176)"),
    (89.8, 89.8, "2", -14.0, 12.0, "#ffd600", "Difference Harmonic (180-90->90)"),
    (19.5, 19.5, "3", 12.0, 12.0, "#ff4081", "Sub-Harmonic Triad (20+20->39)"),
    (72.3, 76.2, "4", -14.0, -14.0, "#ffffff", "Noise Control (72+76->148)"),
]

for i, s in enumerate(("88653", "88654")):
    triads = TRIADS_53 if s == "88653" else TRIADS_54
    for j, (nm, a, b) in enumerate(WIN):
        ax = axs[i, j]
        im = ax.imshow(
            MAPS[(s, nm)].T, origin="lower",
            extent=[fs_[0], fs_[-1], fs_[0], fs_[-1]],
            aspect="equal", vmin=0, vmax=vmax, cmap="viridis"
        )
        ax.set_title(f"#{s} {nm} ({a:.1f}–{b:.1f} ms)", fontsize=11.6, fontweight="bold", pad=7)
        ax.set_xlabel(r"Primary Frequency $f_1$ [kHz]", fontsize=12.5)
        ax.set_ylabel(r"Secondary Frequency $f_2$ [kHz]", fontsize=12.5)
        ax.tick_params(labelsize=11.5)
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.set_xlim(FMIN, FMAX)
        ax.set_ylim(FMIN, FMAX)

        # Plot triads
        is_ech = (nm == "ECH+NBI")
        for f1_pt, f2_pt, num_str, off_x, off_y, col, desc in triads:
            ls_style = "-" if is_ech else "--"
            alpha_val = 0.95 if is_ech else 0.75
            ax.scatter(
                [f1_pt], [f2_pt], s=68, facecolors='none', edgecolors=col,
                linewidths=2.0, linestyle=ls_style, zorder=5, alpha=alpha_val
            )
            ax.scatter(
                [f1_pt], [f2_pt], s=12, color=col, zorder=5, alpha=alpha_val
            )
            # Label callout
            tx = f1_pt + off_x
            ty = f2_pt + off_y
            ax.annotate(
                num_str, xy=(f1_pt, f2_pt), xytext=(tx, ty),
                color="black", fontsize=11.0, fontweight="bold", zorder=6,
                bbox=dict(boxstyle="circle,pad=0.22", fc=col, ec="black", lw=1.0, alpha=0.94),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.2, shrinkA=2, shrinkB=4)
            )

# Shared colorbar
cbar_ax = fig.add_axes([0.885, 0.30, 0.024, 0.54])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Squared Auto-Bicoherence $b^2$ [-]", fontsize=12.5)
cbar.ax.tick_params(labelsize=11.5)

# Two-line per shot triad legend strips below figure (large, crisp, centered)
leg_text_53 = (
    "Shot #88653 Triads:  [1] Sum Harm. (88+88 → 176 kHz)   |   [2] Diff. Triad (115−33 → 82 kHz)\n"
    "[3] Cross-Mode (47+80 → 127 kHz)   |   [4] Noise Control (33+80 → 113 kHz)"
)
leg_text_54 = (
    "Shot #88654 Triads:  [1] Sum Harm. (88+88 → 176 kHz)   |   [2] Diff. Harm. (180−90 → 90 kHz)\n"
    "[3] Sub-Harm. (20+20 ↔ 39 kHz, conj.)   |   [4] Noise Control (72+76 → 148 kHz)"
)
fig.text(0.49, 0.118, leg_text_53, ha="center", va="center", fontsize=11.4, fontweight="bold",
         multialignment="center", linespacing=1.35,
         bbox=dict(boxstyle="round,pad=0.38", fc="#f8f9fa", ec="#aaaaaa", lw=0.9))
fig.text(0.49, 0.036, leg_text_54, ha="center", va="center", fontsize=11.4, fontweight="bold",
         multialignment="center", linespacing=1.35,
         bbox=dict(boxstyle="round,pad=0.38", fc="#f8f9fa", ec="#aaaaaa", lw=0.9))

out_thesis = OUT_DIR / "fig_bicoherence_maps_both_shots.png"
plt.savefig(out_thesis, dpi=300, bbox_inches="tight", pad_inches=0.10)
plt.close(fig)
print(f"Saved: {out_thesis} ({out_thesis.stat().st_size} bytes)")


# =============================================================================
# PART 2: Generate Presentation Slide 16 Maps (Shot #88653 difference plane)
# =============================================================================
print("\n" + "=" * 80)
print("PART 2: Generating Presentation Slide 16 Maps")
print("=" * 80)

c53 = np.load(ROOT_DIR / "cache" / "shot_88653_full_record_cache.npz")
t_ms_53 = np.round(c53["t_ms"].astype(float), 4)
ys_mp1_53 = c53["mp1"]
fs_53 = FS
dt_53 = 1.0 / fs_53
sig_len = 7936
n_ens = 30
nhalf = 256

map_windows = [
    ("ECH + NBI Phase", 259.1, "corrected_bicoherence_map_ECH_NBI.png", 89.0),
    ("Just NBI Phase", 291.3, "corrected_bicoherence_map_NBI_only.png", 86.0),
]

PLOT_F1_MAX = 190.0
PLOT_F2_MIN = -150.0

# Converged triads in difference coordinates (f1, f2, label, text_offset_x, text_offset_y, color, full_label)
# Notice in difference plane: f1 > 0, f2 < 0. Coordinates: (f1, f2)
DIFF_TRIADS = [
    (179.7, -89.8, "1", -22.0, 10.0, "#00e5ff", "Diff. Harmonic / Steepening: 180 - 90 -> 90 kHz (b0 = +0.144 - +0.546)"),
    (115.2, -33.2, "2", 12.0, 10.0, "#ffd600", "Difference Triad: 115 - 33 -> 82 kHz (b0 = +0.146, p = 0.014)"),
    (127.0, -46.9, "3", 12.0, -12.0, "#ff4081", "Cross-Mode Interaction: 127 - 47 -> 80 kHz (b0 = +0.092, p = 0.044)"),
    (113.3, -80.1, "4", -14.0, -14.0, "#ffffff", "Noise Contrast Null: 113 - 80 -> 33 kHz (b0 = +0.017, p = 0.315)"),
]

for title_win, tw_start, out_png, f_pump in map_windows:
    idx0 = int(np.where(t_ms_53 >= tw_start)[0][0])
    w_start = float(t_ms_53[idx0])
    w_end = float(t_ms_53[idx0] + sig_len * dt_53 * 1000.0)
    y_sub = ys_mp1_53[idx0 : idx0 + sig_len]
    t_sub_s = (t_ms_53[idx0 : idx0 + sig_len] - t_ms_53[idx0]) / 1000.0

    _, _, spec = dsp.spectrogram(
        y_sub, fs=fs_53, window="hann", nperseg=NFFT, noverlap=NOV, nfft=NFFT,
        detrend="linear", return_onesided=False, scaling="density", axis=0, mode="complex"
    )
    spec = np.transpose(spec, [1, 0])[:n_ens, :]
    _, b2_raw = abicoh2_corrected(spec)
    bicoh2 = np.transpose(np.fft.fftshift(b2_raw, axes=0), [1, 0])

    f1 = np.arange(nhalf) * (fs_53 / NFFT) / 1000.0
    f2 = (np.arange(NFFT) - nhalf) * (fs_53 / NFFT) / 1000.0
    ff1_k, ff2_k = np.meshgrid(f1, f2, indexing='ij')

    b2_thresh_95 = -np.log(0.05) / n_ens

    mask_diff = (
        (ff1_k > 0)
        & (ff2_k < 0)
        & (ff1_k > -ff2_k)
        & (ff1_k >= 5.0)
        & (-ff2_k >= 5.0)
        & ((ff1_k + ff2_k) >= 5.0)
        & (ff1_k <= PLOT_F1_MAX)
        & (-ff2_k <= -PLOT_F2_MIN)
    )

    color_vmax = float(np.nanmax(bicoh2[mask_diff]))
    color_vmax = max(color_vmax * 1.05, b2_thresh_95, 1e-6)
    color_levels = np.linspace(0.0, color_vmax, 101)

    plt.rcdefaults()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Panel 1: Contourf
    c = ax1.contourf(ff1_k, ff2_k, bicoh2, cmap='inferno', levels=color_levels, extend='max')

    ax1.set_title(
        f"Squared Auto-Bicoherence $b^2$ - Leading Channel: MP1 (nensemble={n_ens}, {title_win}, {w_start:.1f} - {w_end:.1f} ms)",
        fontsize=11.5, fontweight='bold'
    )
    ax1.set_xlabel(r"$f_1$ (kHz)", fontsize=10.5)
    ax1.set_ylabel(r"$f_2$ (kHz)", fontsize=10.5)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.set_xlim(0, PLOT_F1_MAX)
    ax1.set_ylim(PLOT_F2_MIN, 0.0)

    # Reference guide lines
    ax1.plot(
        [0, PLOT_F1_MAX], [0, -0.5 * PLOT_F1_MAX], color='cyan', ls='-.', lw=1.6, alpha=0.95,
        label=r"Self-product line $f_2 = -0.5 f_1$ ($f/2+f/2 \leftrightarrow f$)"
    )
    ax1.axvline(f_pump, color='gold', ls='--', lw=1.2, alpha=0.75, label=rf"Pump $f_1 = {f_pump:.0f}$ kHz")
    ax1.legend(loc='lower left', fontsize=8.0, framealpha=0.92, facecolor='white', edgecolor='gray')

    cbar = fig.colorbar(c, ax=ax1, label=r"Squared Bicoherence $b^2$")
    cbar.ax.axhline(b2_thresh_95, color='cyan', ls=':', lw=1.5)

    # Plot converged triads
    for p_f1, p_f2, num_str, off_x, off_y, col, full_label in DIFF_TRIADS:
        ax1.scatter([p_f1], [p_f2], s=80, facecolors='none', edgecolors=col, linewidths=2.0, zorder=6)
        ax1.scatter([p_f1], [p_f2], s=12, color=col, zorder=6)

        tx = float(np.clip(p_f1 + off_x, 6.0, PLOT_F1_MAX - 10.0))
        ty = float(np.clip(p_f2 + off_y, PLOT_F2_MIN + 6.0, -6.0))

        ax1.annotate(
            f"#{num_str}", xy=(p_f1, p_f2), xytext=(tx, ty),
            color="black", fontsize=9.5, fontweight="bold", zorder=7,
            bbox=dict(boxstyle="circle,pad=0.25", fc=col, ec="black", lw=1.1, alpha=0.95),
            arrowprops=dict(arrowstyle="->", color=col, lw=1.3, shrinkA=2, shrinkB=4)
        )

    # Clean badge describing the converged triads
    badge_text = (
        "Asymptotically Converged Triads (1/K <= 0.055):\n"
        "  #1 Harmonic Steepening: 180 - 90 -> 90 kHz (b0 = +0.144 - +0.546, p <= 0.016)\n"
        "  #2 Difference Triad: 115 - 33 -> 82 kHz (b0 = +0.146, p = 0.014)\n"
        "  #3 Cross-Mode Coupling: 127 - 47 -> 80 kHz (b0 = +0.092, p = 0.044)\n"
        "  #4 Off-Diagonal Noise: 113 - 80 -> 33 kHz (b0 = +0.017, p = 0.315)\n\n"
        f"95% Surrogate Noise Floor: b2 = {b2_thresh_95:.3f}"
    )
    ax1.text(
        0.98, 0.03, badge_text, transform=ax1.transAxes, fontsize=8.0,
        va="bottom", ha="right", bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.92)
    )

    # Panel 2: Linear PSD matching presentation
    f_psd, Pxx = dsp.welch(y_sub, fs=fs_53, window='hann', nperseg=NFFT, noverlap=NOV, detrend='linear')
    f_psd_k = f_psd / 1000.0
    mask_psd = (f_psd_k >= 0) & (f_psd_k <= PLOT_F1_MAX)

    ax2.semilogy(f_psd_k[mask_psd], Pxx[mask_psd], color='tab:blue', linewidth=1.8, label=f"PSD MP1 ({title_win})")
    ax2.set_title("Power Spectral Density (PSD) of Leading Channel: MP1", fontsize=11, fontweight="bold", pad=8)
    ax2.set_xlabel("Frequency (kHz)", fontsize=10)
    ax2.set_ylabel(r"Power (V$^2$/Hz)", fontsize=10)
    ax2.grid(True, alpha=0.3, which='both', linestyle=':')
    ax2.set_xlim(0, PLOT_F1_MAX)

    # Mark the key frequencies on PSD
    ax2.axvline(x=88.0, color='#00cccc', linestyle='--', linewidth=1.4, label=r"Primary Mode: $88$ kHz")
    ax2.axvline(x=39.0, color='#ff9900', linestyle='--', linewidth=1.4, label=r"Secondary Mode: $39$ kHz")
    ax2.axvline(x=176.0, color='#00ffff', linestyle=':', linewidth=1.3, label=r"Primary 2nd Harmonic: $176$ kHz")
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    plt.tight_layout()

    # Union-bbox save for Panel 1 only (preserves exact presentation aspect ratio)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pad = 0.08
    bb_ax = ax1.get_tightbbox(renderer)
    bb_cb = cbar.ax.get_tightbbox(renderer)
    bb_union = mtransforms.Bbox.union([bb_ax, bb_cb])
    bbi = bb_union.transformed(fig.dpi_scale_trans.inverted())
    bbp = mtransforms.Bbox.from_extents(bbi.x0 - pad, bbi.y0 - pad, bbi.x1 + pad, bbi.y1 + pad)

    out_file = OUT_DIR / out_png
    fig.savefig(str(out_file), dpi=150, bbox_inches=bbp, facecolor=fig.get_facecolor(), edgecolor="none", pad_inches=0)
    plt.close(fig)

    print(f"Saved: {out_file} ({out_file.stat().st_size} bytes)")

print("\nAll bicoherence map images updated successfully!")
