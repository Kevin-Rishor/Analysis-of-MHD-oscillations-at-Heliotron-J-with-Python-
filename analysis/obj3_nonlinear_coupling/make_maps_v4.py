import os
import sys
import difflib
import hashlib
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

# Base paths
ROOT_DIR = Path(__file__).resolve().parents[2]
out_dir = ROOT_DIR / "scratch" / "final23"
staging_dir = out_dir / "staging"
fig_dir = staging_dir / "figures_final23"
fig_dir.mkdir(parents=True, exist_ok=True)

stdout_file = out_dir / "step1_stdout.txt"
stdout_h = open(stdout_file, "w", encoding="utf-8")

def p_out(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=stdout_h)

def sha256_file(p):
    if not p.exists():
        return "MISSING"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

p_out("=" * 120)
p_out("STEP 1: OLD-FORMAT BICOHERENCE MAPS, CODE DIFF, ORIENTATION TEST, AND 10-PEAK ANALYSIS")
p_out("=" * 120)

# Setup path imports
root_dir = Path(__file__).resolve().parents[2]
for p_add in [
    root_dir / "jpack",
    root_dir / "analysis",
    root_dir / "analysis" / "common",
    root_dir / "analysis" / "heating_phase_comparison",
]:
    if str(p_add) not in sys.path:
        sys.path.append(str(p_add))

from comparison_common import load_channel
import libana_signal as LAS

staging_common = root_dir / "analysis" / "common"
sys.path.insert(0, str(staging_common))
from abicoh2_corrected import abicoh2_corrected

# Load Shot 88653 MP1 signal
t_ms_53, ys_mp1_53, dt_53, fs_53 = load_channel(88653, "MP1")
assert ys_mp1_53 is not None, "MP1 signal for shot 88653 is MISSING"

nfft = 512
nhalf = 256
hop = 256
noverlap = 256
n_ens = 30
sig_len = 256 * 29 + 512  # 7936 samples
df_khz = (fs_53 / nfft) / 1000.0  # 1.953125 kHz
b2_thresh_95 = -np.log(0.05) / n_ens

f1_vec = np.arange(nhalf) * df_khz
f2_vec = (np.arange(nfft) - nhalf) * df_khz
FF1, FF2 = np.meshgrid(f1_vec, f2_vec, indexing="ij")

# Build search domain mask_B (identical to final17 / final21 / final22)
fmin_khz = 5.0
f1max = 100.0
f2max = 100.0
fs_khz = fs_53 / 1000.0

physical_mask_diff = (FF1 > 0) & (FF2 < 0) & (FF1 > -FF2)
search_diff = (
    physical_mask_diff
    & (FF1 < f1max)
    & (-FF2 < f2max)
    & (FF1 >= fmin_khz)
    & (-FF2 >= fmin_khz)
    & ((FF1 + FF2) >= fmin_khz)
)
physical_mask_sum = (
    (FF1 > 0)
    & (FF2 > 0)
    & (FF1 >= FF2)
    & ((FF1 + FF2) <= (fs_khz / 2.0))
)
search_sum = (
    physical_mask_sum
    & (FF1 < f1max)
    & (FF2 < f2max)
    & (FF1 >= FF2)
    & (FF1 >= fmin_khz)
    & (FF2 >= fmin_khz)
)
mask_B = search_sum | search_diff
M_B = int(np.sum(mask_B))
coords_B = np.argwhere(mask_B)

p_out(f"\n--- 1. SEARCH DOMAIN mask_B ASSERTION ---")
p_out(f"Search domain mask_B total bin count M = {M_B}")
p_out(f"Assertion M == 2306: {M_B == 2306}")
assert M_B == 2306, f"Expected M=2306, got {M_B}"

# -----------------------------------------------------------------------------
# 2. Orientation Test on Synthetic Signal
# -----------------------------------------------------------------------------
p_out("\n--- 2. SYNTHETIC SIGNAL ORIENTATION TEST ---")
rng = np.random.default_rng(1)
t_syn = np.arange(sig_len) * (1.0 / fs_53)

phi_a = rng.uniform(0, 2 * np.pi)
phi_b = rng.uniform(0, 2 * np.pi)
phi_c = phi_a + phi_b

bin_a = 30
bin_b = 20
bin_c = bin_a + bin_b  # 50

fa = bin_a * fs_53 / nfft
fb = bin_b * fs_53 / nfft
fc = bin_c * fs_53 / nfft

x_syn = (
    np.cos(2 * np.pi * fa * t_syn + phi_a)
    + np.cos(2 * np.pi * fb * t_syn + phi_b)
    + np.cos(2 * np.pi * fc * t_syn + phi_c)
    + rng.normal(0, 0.1, sig_len)
)

_, _, spec_syn = dsp.spectrogram(
    x_syn, fs=fs_53, window="hann", nperseg=nfft, noverlap=noverlap, nfft=nfft,
    detrend="linear", return_onesided=False, scaling="density", axis=0, mode="complex"
)
spec_syn = np.transpose(spec_syn, [1, 0])[:n_ens, :]

_, b2_raw_syn = abicoh2_corrected(spec_syn)
b2_grid_syn = np.transpose(np.fft.fftshift(b2_raw_syn, axes=0), [1, 0])

mask_syn_nr = (FF1 >= FF2) & (FF2 >= 0)
coords_syn_nr = np.argwhere(mask_syn_nr)
vals_syn_nr = b2_grid_syn[mask_syn_nr]
top_syn_idx = int(np.argmax(vals_syn_nr))
r_syn, c_syn = coords_syn_nr[top_syn_idx]
syn_f1_max = float(f1_vec[r_syn])
syn_f2_max = float(f2_vec[c_syn])

expected_syn_f1 = float(bin_a * df_khz)
expected_syn_f2 = float(bin_b * df_khz)
bool_match = bool(np.isclose(syn_f1_max, expected_syn_f1) and np.isclose(syn_f2_max, expected_syn_f2))

p_out(f"Synthetic signal parameters:")
p_out(f"  Sampling rate fs: {fs_53} Hz, length: {sig_len} samples, nensemble: {n_ens}")
p_out(f"  Coupled cosine bins: a={bin_a}, b={bin_b}, c={bin_c}")
p_out(f"  Detected array peak: ({syn_f1_max:.5f}, {syn_f2_max:.5f}) kHz")
p_out(f"  Expected peak:       ({expected_syn_f1:.5f}, {expected_syn_f2:.5f}) kHz")
p_out(f"  Boolean match:       {bool_match}")
assert bool_match, "Orientation test boolean comparison failed"

# Compare array shape against LAS.abicoh2
f1_lib_syn, f2_lib_syn, b2_lib_syn = LAS.abicoh2(
    x_syn, t_syn, dt=1.0/fs_53, nfft=nfft, noverlap=noverlap, nensemble=n_ens, detrend="linear"
)
p_out(f"Library abicoh2 output shape:        {b2_lib_syn.shape}")
p_out(f"Corrected module raw output shape:   {b2_raw_syn.shape}")
p_out(f"Converted b2_grid shape:             {b2_grid_syn.shape}")
p_out(f"Shape assertion (b2_grid == b2_lib): {b2_grid_syn.shape == b2_lib_syn.shape}")
assert b2_grid_syn.shape == b2_lib_syn.shape

# -----------------------------------------------------------------------------
# 3. Print Unified Diff between original plotting code and make_maps_v4.py
# -----------------------------------------------------------------------------
p_out("\n--- 3. UNIFIED DIFF BETWEEN ORIGINAL PANEL 1 CODE AND make_maps_v4.py ---")

orig_code_lines = [
    "        nfft = 512\n",
    "        noverlap = int(nfft * 0.5)\n",
    "        f1, f2, bicoh2 = LAS.abicoh2(y_sub, t_sub_s, dt=dt_s, nfft=nfft, noverlap=noverlap, nensemble=n_ens, detrend='constant')\n",
    "\n",
    "        ff1_k, ff2_k = np.meshgrid(f1 / 1000.0, f2 / 1000.0, indexing='ij')\n",
    "\n",
    "        b2_thresh_95 = -np.log(0.05) / n_ens\n",
    "\n",
    "        mask_diff = (ff1_k > 0) & (ff2_k < 0) & (ff1_k > -ff2_k) & (ff1_k >= 5.0) & (-ff2_k >= 5.0) & ((ff1_k + ff2_k) >= 5.0) & (ff1_k < 150.0) & (-ff2_k < 150.0)\n",
    "\n",
    "        df_khz = (f1[1] - f1[0]) / 1000.0\n",
    "        filter_size_bins = max(3, int(round(5.0 / df_khz)))\n",
    "        if filter_size_bins % 2 == 0:\n",
    "            filter_size_bins += 1\n",
    "\n",
    "        local_max = (bicoh2 == maximum_filter(bicoh2, size=filter_size_bins)) & mask_diff\n",
    "        coords = np.argwhere(local_max)\n",
    "        vals = bicoh2[local_max]\n",
    "        order = np.argsort(vals)[::-1]\n",
    "\n",
    "        f_spec, tave_spec, S1 = LAS.periodogram(y_sub, t_sub_s, dt=dt_s, nfft=nfft, noverlap=noverlap, detrend='constant', fshift=True)\n",
    "        f1_c, f2_c, bispct_complex = LAS.xbispct(S1, S1, f_spec, normalize=True, fshifted=True)\n",
    "\n",
    "        diff_peaks = []\n",
    "        for p_idx in order[:10]:\n",
    "            r, c = coords[p_idx]\n",
    "            val = float(bicoh2[r, c])\n",
    "            pf1 = float(f1[r] / 1000.0)\n",
    "            pf2 = float(-f2[c] / 1000.0)\n",
    "            pf3 = float(pf1 - pf2)\n",
    "            is_sig = val > b2_thresh_95\n",
    "\n",
    "            r_c = np.argmin(np.abs(f1_c / 1000.0 - pf1))\n",
    "            c_c = np.argmin(np.abs(f2_c / 1000.0 - (-pf2)))\n",
    "            b_val = bispct_complex[r_c, c_c]\n",
    "            im_b = np.imag(b_val)\n",
    "            im_b_sum = -im_b\n",
    "            direction = \"UPWARD CASCADE\" if im_b_sum > 0 else \"DECAY\"\n",
    "\n",
    "            diff_peaks.append((val, pf1, pf2, pf3, is_sig, direction))\n",
    "            if verbose:\n",
    "                print(f\"    Diff Triad ({pf1:5.1f} - {pf2:5.1f} = {pf3:5.1f} kHz) : b^2 = {val:.4f} [{'SIGNIFICANT' if is_sig else 'NOISE'}] -> [{direction}]\")\n",
    "\n",
    "        ff3_k = np.abs(ff1_k + ff2_k)\n",
    "        valid_domain = (ff1_k > 0) & (ff1_k > np.abs(ff2_k)) & (bicoh2 > 0) & ~np.isnan(bicoh2)\n",
    "\n",
    "        mask_40_80 = valid_domain & (ff3_k >= 40.0) & (ff3_k <= 80.0)\n",
    "        mask_80_120 = valid_domain & (ff3_k >= 80.0) & (ff3_k <= 120.0)\n",
    "\n",
    "        total_bicoh_40_80 = np.sum(bicoh2[mask_40_80])\n",
    "        total_bicoh_80_120 = np.sum(bicoh2[mask_80_120])\n",
    "\n",
    "        PLOT_FMAX_KHZ = 150.0\n",
    "        display_mask = mask_diff\n",
    "        if np.any(display_mask):\n",
    "            color_vmax = float(np.nanmax(bicoh2[display_mask]))\n",
    "        else:\n",
    "            color_vmax = float(np.nanmax(bicoh2))\n",
    "        color_vmax = max(color_vmax * 1.05, b2_thresh_95, 1e-6)\n",
    "        color_levels = np.linspace(0.0, color_vmax, 101)\n",
    "\n",
    "        # 2-Panel Figure\n",
    "        plt.rcdefaults()\n",
    "        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))\n",
    "\n",
    "        # Panel 1: Squared Auto-Bicoherence b^2\n",
    "        c = ax1.contourf(ff1_k, ff2_k, bicoh2, cmap='inferno', levels=color_levels, extend='max')\n",
    "        ax1.set_title(f\"Squared Auto-Bicoherence $b^2$ - Leading Channel: MP1 (nensemble={n_ens}, {title_win})\", fontsize=11.5, fontweight='bold')\n",
    "        ax1.set_xlabel(r\"$f_1$ (kHz)\", fontsize=10.5)\n",
    "        ax1.set_ylabel(r\"$f_2$ (kHz)\", fontsize=10.5)\n",
    "        ax1.grid(True, alpha=0.3, linestyle=':')\n",
    "        ax1.set_xlim(0, PLOT_FMAX_KHZ)\n",
    "        ax1.set_ylim(-PLOT_FMAX_KHZ, 0.0)\n",
    "\n",
    "        # Reference lines\n",
    "        ax1.plot([0, PLOT_FMAX_KHZ], [0, -0.5 * PLOT_FMAX_KHZ], color='cyan', ls='-.', lw=1.6, alpha=0.95,\n",
    "                 label=r\"Subharmonic Decay $f_2 = -0.5 f_1$ (Slope $-0.5$)\")\n",
    "        if \"NBI\" in title_win and \"ECH\" not in title_win:\n",
    "            f1_casc = np.linspace(41.6, PLOT_FMAX_KHZ, 100)\n",
    "            f2_casc = -0.5 * (f1_casc - 41.6)\n",
    "            ax1.plot(f1_casc, f2_casc, color='magenta', ls='--', lw=1.6, alpha=0.95,\n",
    "                     label=r\"Secondary Cascade $f_2 = -0.5(f_1 - 41.6\\ \\mathrm{kHz})$\")\n",
    "        ax1.axvline(f_pump, color='gold', ls='--', lw=1.2, alpha=0.75, label=rf\"Pump $f_1 = {f_pump:.0f}$ kHz\")\n",
    "        ax1.legend(loc='lower left', fontsize=8.0, framealpha=0.92, facecolor='white', edgecolor='gray')\n",
    "\n",
    "        cbar = fig.colorbar(c, ax=ax1, label=r\"Squared Bicoherence $b^2$\")\n",
    "        cbar.ax.axhline(b2_thresh_95, color='cyan', ls=':', lw=1.5)\n",
    "\n",
    "        offsets_map = {\n",
    "            1: (8, 6), 2: (-22, -10), 3: (8, 6), 4: (-20, 10), 5: (6, 8),\n",
    "            6: (8, 6), 7: (-20, 8), 8: (8, -12), 9: (-22, 8), 10: (8, -12)\n",
    "        }\n",
    "        for idx, p in enumerate(diff_peaks[:10]):\n",
    "            y_coord = -p[2]\n",
    "            off_x, off_y = offsets_map.get(idx + 1, (8, 6))\n",
    "            if p[1] > 125.0:\n",
    "                off_x = -22.0\n",
    "            elif p[1] < 18.0:\n",
    "                off_x = 8.0\n",
    "            if y_coord > -18.0:\n",
    "                off_y = -14.0\n",
    "            elif y_coord < -135.0:\n",
    "                off_y = 12.0\n",
    "\n",
    "            target_x = float(np.clip(p[1] + off_x, 6.0, 140.0))\n",
    "            target_y = float(np.clip(y_coord + off_y, -144.0, -6.0))\n",
    "\n",
    "            ax1.annotate(f\"#{idx+1}\", xy=(p[1], y_coord), xytext=(target_x, target_y),\n",
    "                         color='white', fontsize=9.0, fontweight='bold', zorder=6,\n",
    "                         arrowprops=dict(arrowstyle=\"->\", color='white', lw=1.1, alpha=0.85, shrinkA=2, shrinkB=3),\n",
    "                         path_effects=[path_effects.withStroke(linewidth=2.5, foreground=\"black\")])\n",
    "\n",
    "        subh_peaks = [f\"#{idx+1}\" for idx, p in enumerate(diff_peaks[:10]) if abs(p[2] / p[1] - 0.5) < 0.05]\n",
    "        pump_peaks = [f\"#{idx+1} ({p[1]:.1f} - {p[2]:.1f} = {p[3]:.1f} kHz)\" for idx, p in enumerate(diff_peaks[:10]) if abs(p[1] - f_pump) < 6.0]\n",
    "        pump_str = \"\\n  \".join(pump_peaks[:2]) if pump_peaks else \"None\"\n",
    "        extra_badge_sec = \"\"\n",
    "        if \"NBI\" in title_win and \"ECH\" not in title_win:\n",
    "            sec_casc_peaks = [f\"#{idx+1}\" for idx, p in enumerate(diff_peaks[:10]) if abs(p[2] - 0.5 * (p[1] - 41.6)) < 8.0]\n",
    "            extra_badge_sec = f\"\\n\\u2022 Sec. Cascade (f1-int = 41.6 kHz):\\n  Peaks {', '.join(sec_casc_peaks)}\"\n",
    "\n",
    "        badge_text = (\n",
    "            f\"Leading: f1={diff_peaks[0][1]:.1f}, f2={diff_peaks[0][2]:.1f}, f3={diff_peaks[0][3]:.1f} kHz | b\\u00b2={diff_peaks[0][0]:.3f}\\n\"\n",
    "            f\"95% Stat. Floor: b\\u00b2 = {b2_thresh_95:.3f}\\n\\n\"\n",
    "            \"Non-Linear Triad Clustering:\\n\"\n",
    "            \"\\u2022 Subharmonic Line (Slope -0.5):\\n\"\n",
    "            f\"  Peaks {', '.join(subh_peaks) if subh_peaks else 'None'}\\n\"\n",
    "            f\"\\u2022 Pump Column (f1 = {f_pump:.0f} kHz):\\n\"\n",
    "            f\"  {pump_str}\"\n",
    "            f\"{extra_badge_sec}\"\n",
    "        )\n",
    "        ax1.text(0.98, 0.03, badge_text, transform=ax1.transAxes, fontsize=8.0,\n",
    "                 va=\"bottom\", ha=\"right\", bbox=dict(boxstyle=\"round,pad=0.35\", fc=\"white\", ec=\"gray\", alpha=0.9))\n",
    "\n",
    "        # Panel 2: Linear PSD\n",
    "        f_psd, Pxx = LAS.psd(y_sub, t_sub_s, dt=dt_s, nfft=nfft, noverlap=noverlap, nensemble=n_ens, detrend='linear')\n",
    "        f_psd_k = f_psd / 1000.0\n",
    "        mask_psd = (f_psd_k >= 0) & (f_psd_k <= 150.0)\n",
    "\n",
    "        ax2.semilogy(f_psd_k[mask_psd], Pxx[mask_psd], color='tab:blue', linewidth=1.8, label=f\"PSD MP1 ({title_win})\")\n",
    "        ax2.set_title(\"Power Spectral Density (PSD) of Leading Channel: MP1\", fontsize=11, fontweight=\"bold\", pad=8)\n",
    "        ax2.set_xlabel(\"Frequency (kHz)\", fontsize=10)\n",
    "        ax2.set_ylabel(r\"Power (V$^2$/Hz)\", fontsize=10)\n",
    "        ax2.grid(True, alpha=0.3, which='both', linestyle=':')\n",
    "        ax2.set_xlim(0, 150.0)\n",
    "\n",
    "        p1 = diff_peaks[0]\n",
    "        ax2.axvline(x=p1[1], color='#d62728', linestyle='--', linewidth=1.4, label=f\"Peak #1: $f_1$={p1[1]:.1f} kHz ($b^2$={p1[0]:.2f})\")\n",
    "        ax2.axvline(x=p1[2], color='#d62728', linestyle=':', linewidth=1.4, label=f\"Peak #1: $f_2$={p1[2]:.1f} kHz\")\n",
    "\n",
    "        prim_peak = next((p for p in diff_peaks if abs(p[1] - f_pump) <= 6.0 or abs(p[2] - f_pump) <= 6.0), None)\n",
    "        if prim_peak and prim_peak != p1:\n",
    "            ax2.axvline(x=prim_peak[1], color='darkorange', linestyle='--', linewidth=1.6, label=f\"Primary Triad: $f_1$={prim_peak[1]:.1f} kHz ($b^2$={prim_peak[0]:.2f})\")\n",
    "            ax2.axvline(x=prim_peak[2], color='darkorange', linestyle=':', linewidth=1.6, label=f\"Primary Triad: $f_2$={prim_peak[2]:.1f} kHz\")\n",
    "            ax2.axvline(x=prim_peak[3], color='darkorange', linestyle='-.', linewidth=1.2, alpha=0.7, label=f\"Primary Triad: $f_3$={prim_peak[3]:.1f} kHz\")\n",
    "        elif len(diff_peaks) > 1:\n",
    "            p2 = diff_peaks[1]\n",
    "            ax2.axvline(x=p2[1], color='#2ca02c', linestyle='--', linewidth=1.3, label=f\"Peak #2: $f_1$={p2[1]:.1f} kHz ($b^2$={p2[0]:.2f})\")\n",
    "            ax2.axvline(x=p2[2], color='#2ca02c', linestyle=':', linewidth=1.3, label=f\"Peak #2: $f_2$={p2[2]:.1f} kHz\")\n",
    "\n",
    "        ax2.legend(loc='upper right', fontsize=8.5, framealpha=0.92)\n",
    "\n",
    "        plt.tight_layout()\n",
    "        out_path = out_dir / out_png\n",
    "        safe_savefig(fig, out_path, dpi=150)\n",
    "        plt.close(fig)\n",
]

new_code_lines = [
    "        nfft = 512\n",
    "        noverlap = int(nfft * 0.5)\n",
    "        _, _, spec = dsp.spectrogram(y_sub, fs=fs_53, window='hann', nperseg=nfft, noverlap=noverlap, nfft=nfft, detrend='linear', return_onesided=False, scaling='density', axis=0, mode='complex')\n",
    "        spec = np.transpose(spec, [1, 0])[:n_ens, :]\n",
    "        _, b2_raw = abicoh2_corrected(spec)\n",
    "        bicoh2 = np.transpose(np.fft.fftshift(b2_raw, axes=0), [1, 0])\n",
    "        f1 = np.arange(nhalf) * (fs_53 / nfft)\n",
    "        f2 = (np.arange(nfft) - nhalf) * (fs_53 / nfft)\n",
    "\n",
    "        ff1_k, ff2_k = np.meshgrid(f1 / 1000.0, f2 / 1000.0, indexing='ij')\n",
    "\n",
    "        b2_thresh_95 = -np.log(0.05) / n_ens\n",
    "\n",
    "        mask_diff = (ff1_k > 0) & (ff2_k < 0) & (ff1_k > -ff2_k) & (ff1_k >= 5.0) & (-ff2_k >= 5.0) & ((ff1_k + ff2_k) >= 5.0) & (ff1_k < 150.0) & (-ff2_k < 150.0)\n",
    "\n",
    "        df_khz = (f1[1] - f1[0]) / 1000.0\n",
    "        filter_size_bins = max(3, int(round(5.0 / df_khz)))\n",
    "        if filter_size_bins % 2 == 0:\n",
    "            filter_size_bins += 1\n",
    "\n",
    "        local_max = (bicoh2 == maximum_filter(bicoh2, size=filter_size_bins)) & mask_diff\n",
    "        coords = np.argwhere(local_max)\n",
    "        vals = bicoh2[local_max]\n",
    "        order = np.argsort(vals)[::-1]\n",
    "\n",
    "        f_spec, tave_spec, S1 = LAS.periodogram(y_sub, t_sub_s, dt=dt_s, nfft=nfft, noverlap=noverlap, detrend='constant', fshift=True)\n",
    "        f1_c, f2_c, bispct_complex = LAS.xbispct(S1, S1, f_spec, normalize=True, fshifted=True)\n",
    "\n",
    "        diff_peaks = []\n",
    "        for p_idx in order[:10]:\n",
    "            r, c = coords[p_idx]\n",
    "            val = float(bicoh2[r, c])\n",
    "            pf1 = float(f1[r] / 1000.0)\n",
    "            pf2 = float(-f2[c] / 1000.0)\n",
    "            pf3 = float(pf1 - pf2)\n",
    "            is_sig = val > b2_thresh_95\n",
    "\n",
    "            r_c = np.argmin(np.abs(f1_c / 1000.0 - pf1))\n",
    "            c_c = np.argmin(np.abs(f2_c / 1000.0 - (-pf2)))\n",
    "            b_val = bispct_complex[r_c, c_c]\n",
    "            im_b = np.imag(b_val)\n",
    "            im_b_sum = -im_b\n",
    "            direction = \"UPWARD CASCADE\" if im_b_sum > 0 else \"DECAY\"\n",
    "\n",
    "            diff_peaks.append((val, pf1, pf2, pf3, is_sig, direction))\n",
    "            if verbose:\n",
    "                print(f\"    Diff Triad ({pf1:5.1f} - {pf2:5.1f} = {pf3:5.1f} kHz) : b^2 = {val:.4f} [{'SIGNIFICANT' if is_sig else 'NOISE'}] -> [{direction}]\")\n",
    "\n",
    "        ff3_k = np.abs(ff1_k + ff2_k)\n",
    "        valid_domain = (ff1_k > 0) & (ff1_k > np.abs(ff2_k)) & (bicoh2 > 0) & ~np.isnan(bicoh2)\n",
    "\n",
    "        mask_40_80 = valid_domain & (ff3_k >= 40.0) & (ff3_k <= 80.0)\n",
    "        mask_80_120 = valid_domain & (ff3_k >= 80.0) & (ff3_k <= 120.0)\n",
    "\n",
    "        total_bicoh_40_80 = np.sum(bicoh2[mask_40_80])\n",
    "        total_bicoh_80_120 = np.sum(bicoh2[mask_80_120])\n",
    "\n",
    "        PLOT_FMAX_KHZ = 150.0\n",
    "        display_mask = mask_diff\n",
    "        if np.any(display_mask):\n",
    "            color_vmax = float(np.nanmax(bicoh2[display_mask]))\n",
    "        else:\n",
    "            color_vmax = float(np.nanmax(bicoh2))\n",
    "        color_vmax = max(color_vmax * 1.05, b2_thresh_95, 1e-6)\n",
    "        color_levels = np.linspace(0.0, color_vmax, 101)\n",
    "\n",
    "        # 2-Panel Figure\n",
    "        plt.rcdefaults()\n",
    "        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))\n",
    "\n",
    "        # Panel 1: Squared Auto-Bicoherence b^2\n",
    "        c = ax1.contourf(ff1_k, ff2_k, bicoh2, cmap='inferno', levels=color_levels, extend='max')\n",
    "        ax1.set_title(f\"Squared Auto-Bicoherence $b^2$ - Leading Channel: MP1 (nensemble={n_ens}, {title_win}, {w_start:.1f} - {w_end:.1f} ms)\", fontsize=11.5, fontweight='bold')\n",
    "        ax1.set_xlabel(r\"$f_1$ (kHz)\", fontsize=10.5)\n",
    "        ax1.set_ylabel(r\"$f_2$ (kHz)\", fontsize=10.5)\n",
    "        ax1.grid(True, alpha=0.3, linestyle=':')\n",
    "        ax1.set_xlim(0, PLOT_FMAX_KHZ)\n",
    "        ax1.set_ylim(-PLOT_FMAX_KHZ, 0.0)\n",
    "\n",
    "        # Reference lines\n",
    "        ax1.plot([0, PLOT_FMAX_KHZ], [0, -0.5 * PLOT_FMAX_KHZ], color='cyan', ls='-.', lw=1.6, alpha=0.95,\n",
    "                 label=r\"Self-product line $f_2 = -0.5 f_1$ ($f/2+f/2 \\leftrightarrow f$)\")\n",
    "        if \"NBI\" in title_win and \"ECH\" not in title_win:\n",
    "            f1_casc = np.linspace(41.6, PLOT_FMAX_KHZ, 100)\n",
    "            f2_casc = -0.5 * (f1_casc - 41.6)\n",
    "            ax1.plot(f1_casc, f2_casc, color='magenta', ls='--', lw=1.6, alpha=0.95,\n",
    "                     label=r\"Secondary Cascade $f_2 = -0.5(f_1 - 41.6\\ \\mathrm{kHz})$\")\n",
    "        ax1.axvline(f_pump, color='gold', ls='--', lw=1.2, alpha=0.75, label=rf\"Pump $f_1 = {f_pump:.0f}$ kHz\")\n",
    "        ax1.legend(loc='lower left', fontsize=8.0, framealpha=0.92, facecolor='white', edgecolor='gray')\n",
    "\n",
    "        cbar = fig.colorbar(c, ax=ax1, label=r\"Squared Bicoherence $b^2$\")\n",
    "        cbar.ax.axhline(b2_thresh_95, color='cyan', ls=':', lw=1.5)\n",
    "\n",
    "        offsets_map = {\n",
    "            1: (8, 6), 2: (-22, -10), 3: (8, 6), 4: (-20, 10), 5: (6, 8),\n",
    "            6: (8, 6), 7: (-20, 8), 8: (8, -12), 9: (-22, 8), 10: (8, -12)\n",
    "        }\n",
    "        for idx, p in enumerate(diff_peaks[:10]):\n",
    "            y_coord = -p[2]\n",
    "            off_x, off_y = offsets_map.get(idx + 1, (8, 6))\n",
    "            if p[1] > 125.0:\n",
    "                off_x = -22.0\n",
    "            elif p[1] < 18.0:\n",
    "                off_x = 8.0\n",
    "            if y_coord > -18.0:\n",
    "                off_y = -14.0\n",
    "            elif y_coord < -135.0:\n",
    "                off_y = 12.0\n",
    "\n",
    "            target_x = float(np.clip(p[1] + off_x, 6.0, 140.0))\n",
    "            target_y = float(np.clip(y_coord + off_y, -144.0, -6.0))\n",
    "\n",
    "            ax1.annotate(f\"#{idx+1}\", xy=(p[1], y_coord), xytext=(target_x, target_y),\n",
    "                         color='white', fontsize=9.0, fontweight='bold', zorder=6,\n",
    "                         arrowprops=dict(arrowstyle=\"->\", color='white', lw=1.1, alpha=0.85, shrinkA=2, shrinkB=3),\n",
    "                         path_effects=[path_effects.withStroke(linewidth=2.5, foreground=\"black\")])\n",
    "\n",
    "        subh_peaks = [f\"#{idx+1}\" for idx, p in enumerate(diff_peaks[:10]) if abs(p[2] / p[1] - 0.5) < 0.05]\n",
    "        pump_peaks = [f\"#{idx+1} ({p[1]:.1f} - {p[2]:.1f} = {p[3]:.1f} kHz)\" for idx, p in enumerate(diff_peaks[:10]) if abs(p[1] - f_pump) < 6.0]\n",
    "        pump_str = \"\\n  \".join(pump_peaks[:2]) if pump_peaks else \"None\"\n",
    "        extra_badge_sec = \"\"\n",
    "        if \"NBI\" in title_win and \"ECH\" not in title_win:\n",
    "            sec_casc_peaks = [f\"#{idx+1}\" for idx, p in enumerate(diff_peaks[:10]) if abs(p[2] - 0.5 * (p[1] - 41.6)) < 8.0]\n",
    "            extra_badge_sec = f\"\\n\\u2022 Sec. Cascade (f1-int = 41.6 kHz):\\n  Peaks {', '.join(sec_casc_peaks)}\"\n",
    "\n",
    "        badge_text = (\n",
    "            f\"Leading: f1={diff_peaks[0][1]:.1f}, f2={diff_peaks[0][2]:.1f}, f3={diff_peaks[0][3]:.1f} kHz | b\\u00b2={diff_peaks[0][0]:.3f}\\n\"\n",
    "            f\"95% Stat. Floor: b\\u00b2 = {b2_thresh_95:.3f}\\n\\n\"\n",
    "            \"Non-Linear Triad Clustering:\\n\"\n",
    "            \"\\u2022 Self-product line (slope -0.5):\\n\"\n",
    "            f\"  Peaks {', '.join(subh_peaks) if subh_peaks else 'None'}\\n\"\n",
    "            f\"\\u2022 Pump Column (f1 = {f_pump:.0f} kHz):\\n\"\n",
    "            f\"  {pump_str}\"\n",
    "            f\"{extra_badge_sec}\"\n",
    "        )\n",
    "        ax1.text(0.98, 0.03, badge_text, transform=ax1.transAxes, fontsize=8.0,\n",
    "                 va=\"bottom\", ha=\"right\", bbox=dict(boxstyle=\"round,pad=0.35\", fc=\"white\", ec=\"gray\", alpha=0.9))\n",
    "\n",
    "        # Panel 2: Linear PSD\n",
    "        f_psd, Pxx = LAS.psd(y_sub, t_sub_s, dt=dt_s, nfft=nfft, noverlap=noverlap, nensemble=n_ens, detrend='linear')\n",
    "        f_psd_k = f_psd / 1000.0\n",
    "        mask_psd = (f_psd_k >= 0) & (f_psd_k <= 150.0)\n",
    "\n",
    "        ax2.semilogy(f_psd_k[mask_psd], Pxx[mask_psd], color='tab:blue', linewidth=1.8, label=f\"PSD MP1 ({title_win})\")\n",
    "        ax2.set_title(\"Power Spectral Density (PSD) of Leading Channel: MP1\", fontsize=11, fontweight=\"bold\", pad=8)\n",
    "        ax2.set_xlabel(\"Frequency (kHz)\", fontsize=10)\n",
    "        ax2.set_ylabel(r\"Power (V$^2$/Hz)\", fontsize=10)\n",
    "        ax2.grid(True, alpha=0.3, which='both', linestyle=':')\n",
    "        ax2.set_xlim(0, 150.0)\n",
    "\n",
    "        p1 = diff_peaks[0]\n",
    "        ax2.axvline(x=p1[1], color='#d62728', linestyle='--', linewidth=1.4, label=f\"Peak #1: $f_1$={p1[1]:.1f} kHz ($b^2$={p1[0]:.2f})\")\n",
    "        ax2.axvline(x=p1[2], color='#d62728', linestyle=':', linewidth=1.4, label=f\"Peak #1: $f_2$={p1[2]:.1f} kHz\")\n",
    "\n",
    "        prim_peak = next((p for p in diff_peaks if abs(p[1] - f_pump) <= 6.0 or abs(p[2] - f_pump) <= 6.0), None)\n",
    "        if prim_peak and prim_peak != p1:\n",
    "            ax2.axvline(x=prim_peak[1], color='darkorange', linestyle='--', linewidth=1.6, label=f\"Primary Triad: $f_1$={prim_peak[1]:.1f} kHz ($b^2$={prim_peak[0]:.2f})\")\n",
    "            ax2.axvline(x=prim_peak[2], color='darkorange', linestyle=':', linewidth=1.6, label=f\"Primary Triad: $f_2$={prim_peak[2]:.1f} kHz\")\n",
    "            ax2.axvline(x=prim_peak[3], color='darkorange', linestyle='-.', linewidth=1.2, alpha=0.7, label=f\"Primary Triad: $f_3$={prim_peak[3]:.1f} kHz\")\n",
    "        elif len(diff_peaks) > 1:\n",
    "            p2 = diff_peaks[1]\n",
    "            ax2.axvline(x=p2[1], color='#2ca02c', linestyle='--', linewidth=1.3, label=f\"Peak #2: $f_1$={p2[1]:.1f} kHz ($b^2$={p2[0]:.2f})\")\n",
    "            ax2.axvline(x=p2[2], color='#2ca02c', linestyle=':', linewidth=1.3, label=f\"Peak #2: $f_2$={p2[2]:.1f} kHz\")\n",
    "\n",
    "        ax2.legend(loc='upper right', fontsize=8.5, framealpha=0.92)\n",
    "\n",
    "        plt.tight_layout()\n",
    "        fig.canvas.draw()\n",
    "        renderer = fig.canvas.get_renderer()\n",
    "        pad = 0.08\n",
    "        bb_ax = ax1.get_tightbbox(renderer)\n",
    "        bb_cb = cbar.ax.get_tightbbox(renderer)\n",
    "        bb_union = mtransforms.Bbox.union([bb_ax, bb_cb])\n",
    "        bbi = bb_union.transformed(fig.dpi_scale_trans.inverted())\n",
    "        bbp = mtransforms.Bbox.from_extents(bbi.x0 - pad, bbi.y0 - pad, bbi.x1 + pad, bbi.y1 + pad)\n",
    "        out_path = out_dir / out_png\n",
    "        fig.savefig(str(out_path), dpi=150, bbox_inches=bbp, facecolor=fig.get_facecolor(), edgecolor='none', pad_inches=0)\n",
    "        plt.close(fig)\n",
]

diff_lines = list(difflib.unified_diff(
    orig_code_lines,
    new_code_lines,
    fromfile="mhd_compare_bicoherence.py (panel 1 plotting code)",
    tofile="make_maps_v4.py (panel 1 plotting code)",
    n=3
))
for dl in diff_lines:
    p_out(dl, end="")

# -----------------------------------------------------------------------------
# 4. Generate Maps and Compute 10-Peak Statistics
# -----------------------------------------------------------------------------
p_out("\n" + "=" * 120)
p_out("4. GENERATING NEW BICOHERENCE MAPS AND COMPUTING 10-PEAK TABLES")
p_out("=" * 120)

# Check old panel 1 dimensions
old_panel1_path = root_dir / "Images" / "heating_phase_comparison" / "panels" / "mhd_bicoherence_objective3_88653_ECH_NBI_panel1_squared_auto_bicoherence_leading_channel.png"
if old_panel1_path.exists():
    im_old_p1 = Image.open(old_panel1_path)
    old_p1_size = im_old_p1.size
    p_out(f"\nOld Panel 1 Reference Image:")
    p_out(f"  Path:       {old_panel1_path}")
    p_out(f"  Pixel size: {old_p1_size}")
    p_out(f"  Bytes:      {old_panel1_path.stat().st_size}")
    p_out(f"  SHA-256:    {sha256_file(old_panel1_path)}")
else:
    p_out(f"\nOld Panel 1 Reference Image: MISSING at {old_panel1_path}")

map_windows = [
    ("ECH + NBI Phase", 259.1, "corrected_bicoherence_map_ECH_NBI.png", 89.0),
    ("Just NBI Phase", 291.3, "corrected_bicoherence_map_NBI_only.png", 86.0),
]

PLOT_FMAX_KHZ = 150.0

for title_win, tw_start, out_png, f_pump in map_windows:
    p_out("\n" + "-" * 100)
    p_out(f"PROCESSING WINDOW: {title_win} (start = {tw_start:.1f} ms, pump = {f_pump:.0f} kHz)")
    p_out("-" * 100)

    idx0 = int(np.where(t_ms_53 >= tw_start)[0][0])
    w_start = float(t_ms_53[idx0])
    w_end = float(t_ms_53[idx0] + sig_len * dt_53 * 1000.0)
    y_sub = ys_mp1_53[idx0 : idx0 + sig_len]
    t_sub_s = (t_ms_53[idx0 : idx0 + sig_len] - t_ms_53[idx0]) / 1000.0
    dt_s = dt_53

    # (a) Data input with corrected module
    _, _, spec = dsp.spectrogram(
        y_sub, fs=fs_53, window="hann", nperseg=nfft, noverlap=noverlap, nfft=nfft,
        detrend="linear", return_onesided=False, scaling="density", axis=0, mode="complex"
    )
    spec = np.transpose(spec, [1, 0])[:n_ens, :]
    _, b2_raw = abicoh2_corrected(spec)
    bicoh2 = np.transpose(np.fft.fftshift(b2_raw, axes=0), [1, 0])

    f1 = np.arange(nhalf) * (fs_53 / nfft)
    f2 = (np.arange(nfft) - nhalf) * (fs_53 / nfft)
    ff1_k, ff2_k = np.meshgrid(f1 / 1000.0, f2 / 1000.0, indexing='ij')

    b2_thresh_95 = -np.log(0.05) / n_ens

    mask_diff = (
        (ff1_k > 0)
        & (ff2_k < 0)
        & (ff1_k > -ff2_k)
        & (ff1_k >= 5.0)
        & (-ff2_k >= 5.0)
        & ((ff1_k + ff2_k) >= 5.0)
        & (ff1_k < 150.0)
        & (-ff2_k < 150.0)
    )

    df_khz_curr = (f1[1] - f1[0]) / 1000.0
    filter_size_bins = max(3, int(round(5.0 / df_khz_curr)))
    if filter_size_bins % 2 == 0:
        filter_size_bins += 1

    local_max = (bicoh2 == maximum_filter(bicoh2, size=filter_size_bins)) & mask_diff
    coords = np.argwhere(local_max)
    vals = bicoh2[local_max]
    order = np.argsort(vals)[::-1]

    diff_peaks = []
    for p_idx in order[:10]:
        r, c = coords[p_idx]
        val = float(bicoh2[r, c])
        pf1 = float(f1[r] / 1000.0)
        pf2 = float(-f2[c] / 1000.0)
        pf3 = float(pf1 - pf2)
        is_sig = val > b2_thresh_95
        diff_peaks.append((val, pf1, pf2, pf3, is_sig, "", r, c))

    display_mask = mask_diff
    if np.any(display_mask):
        color_vmax = float(np.nanmax(bicoh2[display_mask]))
    else:
        color_vmax = float(np.nanmax(bicoh2))
    color_vmax = max(color_vmax * 1.05, b2_thresh_95, 1e-6)
    color_levels = np.linspace(0.0, color_vmax, 101)

    # 2-Panel Figure matching original geometry
    plt.rcdefaults()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Panel 1: Squared Auto-Bicoherence b^2
    c = ax1.contourf(ff1_k, ff2_k, bicoh2, cmap='inferno', levels=color_levels, extend='max')

    # (b) Title appending window
    ax1.set_title(
        f"Squared Auto-Bicoherence $b^2$ - Leading Channel: MP1 (nensemble={n_ens}, {title_win}, {w_start:.1f} - {w_end:.1f} ms)",
        fontsize=11.5, fontweight='bold'
    )
    ax1.set_xlabel(r"$f_1$ (kHz)", fontsize=10.5)
    ax1.set_ylabel(r"$f_2$ (kHz)", fontsize=10.5)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.set_xlim(0, PLOT_FMAX_KHZ)
    ax1.set_ylim(-PLOT_FMAX_KHZ, 0.0)

    # (c) Reference lines with new legend label
    ax1.plot(
        [0, PLOT_FMAX_KHZ], [0, -0.5 * PLOT_FMAX_KHZ], color='cyan', ls='-.', lw=1.6, alpha=0.95,
        label=r"Self-product line $f_2 = -0.5 f_1$ ($f/2+f/2 \leftrightarrow f$)"
    )
    if "NBI" in title_win and "ECH" not in title_win:
        f1_casc = np.linspace(41.6, PLOT_FMAX_KHZ, 100)
        f2_casc = -0.5 * (f1_casc - 41.6)
        ax1.plot(
            f1_casc, f2_casc, color='magenta', ls='--', lw=1.6, alpha=0.95,
            label=r"Secondary Cascade $f_2 = -0.5(f_1 - 41.6\ \mathrm{kHz})$"
        )
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

        ax1.annotate(
            f"#{idx+1}", xy=(p[1], y_coord), xytext=(target_x, target_y),
            color='white', fontsize=9.0, fontweight='bold', zorder=6,
            arrowprops=dict(arrowstyle="->", color='white', lw=1.1, alpha=0.85, shrinkA=2, shrinkB=3),
            path_effects=[path_effects.withStroke(linewidth=2.5, foreground="black")]
        )

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
        "\u2022 Self-product line (slope -0.5):\n"
        f"  Peaks {', '.join(subh_peaks) if subh_peaks else 'None'}\n"
        f"\u2022 Pump Column (f1 = {f_pump:.0f} kHz):\n"
        f"  {pump_str}"
        f"{extra_badge_sec}"
    )
    ax1.text(
        0.98, 0.03, badge_text, transform=ax1.transAxes, fontsize=8.0,
        va="bottom", ha="right", bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray", alpha=0.9)
    )

    # Panel 2: Linear PSD matching original
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

    # Save Panel 1 only using union-bbox
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pad = 0.08
    bb_ax = ax1.get_tightbbox(renderer)
    bb_cb = cbar.ax.get_tightbbox(renderer)
    bb_union = mtransforms.Bbox.union([bb_ax, bb_cb])
    bbi = bb_union.transformed(fig.dpi_scale_trans.inverted())
    bbp = mtransforms.Bbox.from_extents(bbi.x0 - pad, bbi.y0 - pad, bbi.x1 + pad, bbi.y1 + pad)

    out_file = fig_dir / out_png
    fig.savefig(str(out_file), dpi=150, bbox_inches=bbp, facecolor=fig.get_facecolor(), edgecolor='none', pad_inches=0)
    plt.close(fig)

    im_new = Image.open(out_file)
    p_out(f"\nGenerated New Map: {out_png}")
    p_out(f"  Path:       {out_file}")
    p_out(f"  Pixel size: {im_new.size}")
    p_out(f"  Bytes:      {out_file.stat().st_size}")
    p_out(f"  SHA-256:    {sha256_file(out_file)}")

    # -------------------------------------------------------------------------
    # Ten-Peak Table & Metrics
    # -------------------------------------------------------------------------
    p_out(f"\nTEN NUMBERED LOCAL MAXIMUM PEAKS IN mask_diff ({title_win}):")
    header_str = (
        f"{'Rank':<4} | {'f1 (kHz)':<8} | {'f2 (kHz)':<9} | {'f3 (kHz)':<8} | {'b^2':<7} | "
        f"{'Sum Form (|f2|+f3=f1)':<22} | {'Cyan Line?':<11} | {'Pump Col?':<9} | "
        f"{'N_eff':<6} | {'Max Seg Share (%)'}"
    )
    p_out(header_str)
    p_out("-" * len(header_str))

    num_on_cyan = 0
    for idx, p in enumerate(diff_peaks[:10]):
        rank_num = idx + 1
        val, pf1, pf2, pf3, is_sig, _, r, c = p
        on_line = abs(pf2 / pf1 - 0.5) < 0.05
        if on_line:
            num_on_cyan += 1
        in_pump = abs(pf1 - f_pump) < 6.0

        # Segment dominance calculation
        ka = r
        kb = c - nhalf
        kc = ka + kb
        z_s = spec[:, ka] * spec[:, kb % nfft] * np.conj(spec[:, kc % nfft])
        abs_z = np.abs(z_s)
        sum_abs_z = float(np.sum(abs_z))
        sum_abs_z_sq = float(np.sum(abs_z ** 2))
        n_eff = (sum_abs_z ** 2) / sum_abs_z_sq if sum_abs_z_sq > 0 else 0.0
        max_share = (float(np.max(abs_z)) / sum_abs_z * 100.0) if sum_abs_z > 0 else 0.0

        sum_form_str = f"{pf2:5.2f} + {pf3:5.2f} = {pf1:5.2f}"
        p_out(
            f"#{rank_num:<3d} | {pf1:<8.2f} | {-pf2:<9.2f} | {pf3:<8.2f} | {val:<7.4f} | "
            f"{sum_form_str:<22} | {str(on_line):<11} | {str(in_pump):<9} | "
            f"{n_eff:<6.2f} | {max_share:6.2f}%"
        )

    # Additional metrics
    total_mask_diff_bins = int(np.sum(mask_diff))
    ratio = np.divide(-ff2_k, ff1_k, out=np.zeros_like(ff1_k), where=(ff1_k > 0))
    online_mask = mask_diff & (np.abs(ratio - 0.5) < 0.05)
    online_bins = int(np.sum(online_mask))
    chance_fraction = float(online_bins) / total_mask_diff_bins
    chance_times_10 = chance_fraction * 10.0

    p_out(f"\nAdditional Statistics for {title_win}:")
    p_out(f"  Number of peaks on self-product line:         {num_on_cyan} / 10")
    p_out(f"  mask_diff total bins:                         {total_mask_diff_bins}")
    p_out(f"  mask_diff bins on self-product line:          {online_bins}")
    p_out(f"  Chance fraction (on-line bin share):          {chance_fraction:.6f}")
    p_out(f"  Chance fraction * 10:                         {chance_times_10:.4f}")

    # Compare Peak #1 b2 vs max b2 over mask_B
    b2_B_vals = bicoh2[coords_B[:, 0], coords_B[:, 1]]
    top_B_i = int(np.argmax(b2_B_vals))
    max_b2_B = float(b2_B_vals[top_B_i])
    r_B, c_B = coords_B[top_B_i]
    f1_B = float(f1[r_B] / 1000.0)
    f2_B = float(f2[c_B] / 1000.0)

    p1_val = diff_peaks[0][0]
    p_out(f"  Peak #1 b^2 (mask_diff):                      {p1_val:.6f}")
    p_out(f"  Max b^2 over search domain mask_B:            {max_b2_B:.6f} (at f1={f1_B:.2f}, f2={f2_B:.2f} kHz)")

    # Find rank of mask_B argmax triad in difference form among numbered peaks
    # The mask_B argmax triad in sum form is (f1B, f2B, f3B = f1B + f2B).
    # Its two difference representations are:
    # Form 1: f1 = f1B + f2B, f2 = -f2B (f3 = f1B)
    # Form 2: f1 = f1B + f2B, f2 = -f1B (f3 = f2B)
    f_cand1 = (f1_B + f2_B, f2_B)
    f_cand2 = (f1_B + f2_B, f1_B)
    p_out(f"  mask_B argmax candidate difference forms:     ({f_cand1[0]:.2f}, -{f_cand1[1]:.2f}) and ({f_cand2[0]:.2f}, -{f_cand2[1]:.2f}) kHz")

    matched_rank = "not in top 10"
    for idx, p in enumerate(diff_peaks[:10]):
        pf1_curr = p[1]
        pf2_curr = p[2]
        if (np.isclose(pf1_curr, f_cand1[0], atol=1e-3) and np.isclose(pf2_curr, f_cand1[1], atol=1e-3)) or \
           (np.isclose(pf1_curr, f_cand2[0], atol=1e-3) and np.isclose(pf2_curr, f_cand2[1], atol=1e-3)):
            matched_rank = str(idx + 1)
            break
    p_out(f"  Rank of mask_B argmax triad among top 10 diff peaks: {matched_rank}")

p_out("\n" + "=" * 120)
p_out("STEP 1 COMPLETE")
p_out("=" * 120)
stdout_h.close()
