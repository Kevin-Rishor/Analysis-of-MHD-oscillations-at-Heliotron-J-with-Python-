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
base_dir = ROOT_DIR
out_dir = ROOT_DIR / "scratch" / "final24"
staging_dir = out_dir / "staging"
fig_dir = staging_dir / "figures_final24"
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
p_out("STEP 1: TRIAD FORMS ANALYSIS (1a), CONTROL TEST (1b), AND UPDATED MAPS (1c)")
p_out("=" * 120)

# Setup path imports
root_dir = ROOT_DIR
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

staging_common = ROOT_DIR / "analysis" / "common"
sys.path.insert(0, str(staging_common))
from abicoh2_corrected import abicoh2_corrected

# Load Shot 88653 and 88654 MP1 signals
t_ms_53, ys_mp1_53, dt_53, fs_53 = load_channel(88653, "MP1")
assert ys_mp1_53 is not None, "MP1 signal for shot 88653 is MISSING"
t_ms_54, ys_mp1_54, dt_54, fs_54 = load_channel(88654, "MP1")
assert ys_mp1_54 is not None, "MP1 signal for shot 88654 is MISSING"

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

# mask_diff definition
mask_diff = (
    (FF1 > 0)
    & (FF2 < 0)
    & (FF1 > -FF2)
    & (FF1 >= 5.0)
    & (-FF2 >= 5.0)
    & ((FF1 + FF2) >= 5.0)
    & (FF1 < 150.0)
    & (-FF2 < 150.0)
)
filter_size_bins = max(3, int(round(5.0 / df_khz)))
if filter_size_bins % 2 == 0:
    filter_size_bins += 1

# mask_B definition
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
coords_B = np.argwhere(mask_B)

# =============================================================================
# SUB-STEP 1a: DETAILED ANALYSIS OF mask_B ARGMAX TRIAD FORMS
# =============================================================================
p_out("\n" + "=" * 120)
p_out("SUB-STEP 1a: mask_B ARGMAX TRIAD FORMS (SUM & DIFFERENCE) AND 3x3 NEIGHBOURHOODS")
p_out("=" * 120)

target_windows_1a = [
    ("ECH + NBI Phase", 259.1, 39, 37, 76),   # 76.17 + 72.27 -> 148.44 kHz
    ("Just NBI Phase", 291.3, 38, 12, 50),   # 74.22 + 23.44 -> 97.66 kHz
]

for title_win, tw_start, exp_k1, exp_k2, exp_k3 in target_windows_1a:
    p_out("\n" + "-" * 100)
    p_out(f"WINDOW: {title_win} (start = {tw_start:.1f} ms)")
    p_out("-" * 100)

    idx0 = int(np.where(t_ms_53 >= tw_start)[0][0])
    y_sub = ys_mp1_53[idx0 : idx0 + sig_len]
    _, _, spec = dsp.spectrogram(
        y_sub, fs=fs_53, window="hann", nperseg=nfft, noverlap=noverlap, nfft=nfft,
        detrend="linear", return_onesided=False, scaling="density", axis=0, mode="complex"
    )
    spec = np.transpose(spec, [1, 0])[:n_ens, :]
    _, b2_raw = abicoh2_corrected(spec)
    b2_grid = np.transpose(np.fft.fftshift(b2_raw, axes=0), [1, 0])

    # Find mask_B argmax
    b2_B_vals = b2_grid[coords_B[:, 0], coords_B[:, 1]]
    top_i = int(np.argmax(b2_B_vals))
    r_max, c_max = coords_B[top_i]
    max_b2_val = float(b2_B_vals[top_i])
    k1_det = int(r_max)
    k2_det = int(c_max - nhalf)
    k3_det = k1_det + k2_det

    p_out(f"Computed mask_B argmax: bin (r={r_max}, c={c_max}) -> k1={k1_det}, k2={k2_det}, k3={k3_det}")
    p_out(f"  Frequencies: {k1_det * df_khz:.2f} + {k2_det * df_khz:.2f} = {k3_det * df_khz:.2f} kHz | max b^2 = {max_b2_val:.6f}")
    assert k1_det == exp_k1 and k2_det == exp_k2 and k3_det == exp_k3, f"Argmax triad bins mismatch!"

    # Evaluate the 3 forms: sum form (f1, f2), diff form 1 (f3, -f2), diff form 2 (f3, -f1)
    forms = [
        ("Sum Form (f1, f2)", k1_det, k2_det, k3_det, f"{k1_det*df_khz:.2f} + {k2_det*df_khz:.2f} -> {k3_det*df_khz:.2f} kHz"),
        ("Diff Form 1 (f3, -f2)", k3_det, -k2_det, k1_det, f"{k3_det*df_khz:.2f} - {k2_det*df_khz:.2f} -> {k1_det*df_khz:.2f} kHz"),
        ("Diff Form 2 (f3, -f1)", k3_det, -k1_det, k2_det, f"{k3_det*df_khz:.2f} - {k1_det*df_khz:.2f} -> {k2_det*df_khz:.2f} kHz"),
    ]

    p_out("\nTABLE OF TRIAD FORMS:")
    header_1a = f"{'Form Name':<23} | {'(ka, kb, kc)':<14} | {'Grid (r, c)':<11} | {'b^2':<8} | {'|B|^2 Numerator':<17} | {'Denom Factor 1':<16} | {'Denom Factor 2':<16}"
    p_out(header_1a)
    p_out("-" * len(header_1a))

    diff_form_coords = []
    for name, ka, kb, kc, freq_str in forms:
        r = ka
        c = kb + nhalf
        b2_val = float(b2_grid[r, c])

        Xa = spec[:, ka % nfft]
        Xb = spec[:, kb % nfft]
        Xc = spec[:, kc % nfft]
        z_s = Xa * Xb * np.conj(Xc)
        num_val = float(np.abs(np.mean(z_s)) ** 2)
        den1_val = float(np.mean(np.abs(Xa * Xb) ** 2))
        den2_val = float(np.mean(np.abs(Xc) ** 2))

        p_out(
            f"{name:<23} | ({ka:>3d}, {kb:>3d}, {kc:>3d}) | ({r:>3d}, {c:>3d})    | "
            f"{b2_val:<8.6f} | {num_val:<17.4e} | {den1_val:<16.4e} | {den2_val:<16.4e}"
        )
        if "Diff" in name:
            diff_form_coords.append((name, r, c, ka, kb, kc))

    # Print 3x3 bin neighbourhood of each difference form
    p_out("\n3x3 BIN NEIGHBOURHOODS FOR DIFFERENCE FORMS:")
    for name, r, c, ka, kb, kc in diff_form_coords:
        f1_center = r * df_khz
        f2_center = (c - nhalf) * df_khz
        p_out(f"\n  Neighbourhood for {name} (center bin r={r}, c={c} -> f1={f1_center:.2f}, f2={f2_center:.2f} kHz):")
        p_out(f"  {'f1 \\ f2':<10} | {f2_vec[c-1]:<10.2f} | {f2_vec[c]:<10.2f} | {f2_vec[c+1]:<10.2f}")
        p_out(f"  {'-'*10}-|-{'-'*10}-|-{'-'*10}-|-{'-'*10}")
        for dr in [-1, 0, 1]:
            rr = r + dr
            f1_curr = f1_vec[rr]
            row_str = f"  {f1_curr:<10.2f} | "
            val_strs = []
            for dc in [-1, 0, 1]:
                cc = c + dc
                b2_nb = b2_grid[rr, cc]
                marker = "*" if (dr == 0 and dc == 0) else " "
                val_strs.append(f"{b2_nb:.6f}{marker}")
            row_str += " | ".join(val_strs)
            p_out(row_str)

# =============================================================================
# SUB-STEP 1b: CONTROL TEST OF THE SELF-PRODUCT-LINE COUNT (33 WINDOWS PER SHOT)
# =============================================================================
p_out("\n" + "=" * 120)
p_out("SUB-STEP 1b: CONTROL TEST OF SELF-PRODUCT-LINE COUNT (33 WINDOWS PER SHOT)")
p_out("=" * 120)

control_script_candidates = [
    base_dir / "scratch" / "final20" / "make_figures.py",
    base_dir / "scratch" / "final21" / "make_figures_v2.py",
]
found_ctrl_script = None
for cand in control_script_candidates:
    if cand.exists():
        found_ctrl_script = cand
        break

if found_ctrl_script is None:
    p_out("Control windows generator script: MISSING")
    p_out("Stopping sub-step 1b.")
else:
    p_out(f"Located control-window figure generator script: {found_ctrl_script}")
    # Extract list of 33 window start times per shot
    n_windows = 33
    win_starts_53 = [float(t_ms_53[w * sig_len]) for w in range(n_windows)]
    win_starts_54 = [float(t_ms_54[w * sig_len]) for w in range(n_windows)]

    p_out(f"\nImported {n_windows} window start times for Shot #88653 (ms):")
    p_out([round(x, 3) for x in win_starts_53])
    p_out(f"\nImported {n_windows} window start times for Shot #88654 (ms):")
    p_out([round(x, 3) for x in win_starts_54])

    def run_pipeline_window(y_sub, fs):
        _, _, spec = dsp.spectrogram(
            y_sub, fs=fs, window="hann", nperseg=nfft, noverlap=noverlap, nfft=nfft,
            detrend="linear", return_onesided=False, scaling="density", axis=0, mode="complex"
        )
        spec = np.transpose(spec, [1, 0])[:n_ens, :]
        _, b2_raw = abicoh2_corrected(spec)
        bicoh2 = np.transpose(np.fft.fftshift(b2_raw, axes=0), [1, 0])

        local_max = (bicoh2 == maximum_filter(bicoh2, size=filter_size_bins)) & mask_diff
        coords = np.argwhere(local_max)
        vals = bicoh2[local_max]
        order = np.argsort(vals)[::-1]

        diff_peaks = []
        for p_idx in order[:10]:
            r, c = coords[p_idx]
            val = float(bicoh2[r, c])
            pf1 = float(f1_vec[r])
            pf2 = float(-f2_vec[c])
            pf3 = float(pf1 - pf2)
            diff_peaks.append((val, pf1, pf2, pf3, r, c))

        num_on_cyan = 0
        neff_list = []
        for idx, (val, pf1, pf2, pf3, r, c) in enumerate(diff_peaks[:10]):
            on_line = abs(pf2 / pf1 - 0.5) < 0.05
            if on_line:
                num_on_cyan += 1
            ka = r
            kb = c - nhalf
            kc = ka + kb
            z_s = spec[:, ka] * spec[:, kb % nfft] * np.conj(spec[:, kc % nfft])
            abs_z = np.abs(z_s)
            sum_abs_z = float(np.sum(abs_z))
            sum_abs_z_sq = float(np.sum(abs_z ** 2))
            n_eff = (sum_abs_z ** 2) / sum_abs_z_sq if sum_abs_z_sq > 0 else 0.0
            neff_list.append(n_eff)

        med_neff = float(np.median(neff_list)) if neff_list else 0.0
        max_b2 = float(diff_peaks[0][0]) if diff_peaks else 0.0
        return num_on_cyan, med_neff, max_b2, diff_peaks

    shots_data = [
        (88653, ys_mp1_53, fs_53, t_ms_53, win_starts_53),
        (88654, ys_mp1_54, fs_54, t_ms_54, win_starts_54),
    ]

    for s_num, ys, fs, t_ms, w_starts in shots_data:
        p_out("\n" + "-" * 100)
        p_out(f"SCANNING 33 WINDOWS FOR SHOT #{s_num}:")
        p_out("-" * 100)
        header_1b = f"{'Win #':<6} | {'Start (ms)':<11} | {'On-Line Count (/10)':<20} | {'Median N_eff':<13} | {'Max b^2 (Peak #1)'}"
        p_out(header_1b)
        p_out("-" * len(header_1b))

        counts_shot = []
        for w in range(n_windows):
            i_start = w * sig_len
            w_start = w_starts[w]
            y_w = ys[i_start : i_start + sig_len]
            cnt, med_n, mx_b2, _ = run_pipeline_window(y_w, fs)
            counts_shot.append(cnt)
            p_out(f"{w+1:<6d} | {w_start:<11.3f} | {cnt:<20d} | {med_n:<13.2f} | {mx_b2:.4f}")

        counts_arr = np.array(counts_shot)
        mean_cnt = float(np.mean(counts_arr))
        cnt_ge_6 = int(np.sum(counts_arr >= 6))
        cnt_ge_3 = int(np.sum(counts_arr >= 3))
        dist_counts = {k: int(np.sum(counts_arr == k)) for k in range(11)}

        p_out(f"\nSummary Statistics for Shot #{s_num} (33 Windows):")
        p_out(f"  Distribution of on-line count (0..10): {dist_counts}")
        p_out(f"  Mean on-line count:                   {mean_cnt:.4f}")
        p_out(f"  Windows with count >= 6:              {cnt_ge_6} / {n_windows}")
        p_out(f"  Windows with count >= 3:              {cnt_ge_3} / {n_windows}")

    # Recomputed counts for the heated windows of shot 88653 using the exact same function
    p_out("\n" + "-" * 100)
    p_out("RECOMPUTED COUNTS FOR NOMINAL HEATED WINDOWS (SHOT 88653):")
    p_out("-" * 100)
    for w_name, t_target in [("ECH + NBI Phase", 259.1), ("Just NBI Phase", 291.3)]:
        idx_target = int(np.where(t_ms_53 >= t_target)[0][0])
        y_target = ys_mp1_53[idx_target : idx_target + sig_len]
        re_cnt, re_med_n, re_mx_b2, _ = run_pipeline_window(y_target, fs_53)
        p_out(f"  {w_name} (start = {t_ms_53[idx_target]:.3f} ms):")
        p_out(f"    On-line count: {re_cnt} / 10 | Median N_eff: {re_med_n:.2f} | Max b^2: {re_mx_b2:.4f}")

# =============================================================================
# SUB-STEP 1c: REGENERATE MAPS WITH UPDATES & DIFF PRINTING
# =============================================================================
p_out("\n" + "=" * 120)
p_out("SUB-STEP 1c: REGENERATE MAPS WITH CODE DIFF, BADGE & COLLISION UPDATES")
p_out("=" * 120)

# 1. Print Unified Diff between final23 make_maps_v4.py and final24 run_peaks_v5.py
final23_code_path = base_dir / "scratch" / "final23" / "make_maps_v4.py"
assert final23_code_path.exists(), f"final23 make_maps_v4.py MISSING at {final23_code_path}"

# Snippet of final23 map loop vs final24 map loop
final23_map_snippet = [
    "    # Reference lines\n",
    "    ax1.plot(\n",
    "        [0, PLOT_FMAX_KHZ], [0, -0.5 * PLOT_FMAX_KHZ], color='cyan', ls='-.', lw=1.6, alpha=0.95,\n",
    "        label=r\"Self-product line $f_2 = -0.5 f_1$ ($f/2+f/2 \\leftrightarrow f$)\"\n",
    "    )\n",
    "    if \"NBI\" in title_win and \"ECH\" not in title_win:\n",
    "        f1_casc = np.linspace(41.6, PLOT_FMAX_KHZ, 100)\n",
    "        f2_casc = -0.5 * (f1_casc - 41.6)\n",
    "        ax1.plot(\n",
    "            f1_casc, f2_casc, color='magenta', ls='--', lw=1.6, alpha=0.95,\n",
    "            label=r\"Secondary Cascade $f_2 = -0.5(f_1 - 41.6\\ \\mathrm{kHz})$\"\n",
    "        )\n",
    "    ax1.axvline(f_pump, color='gold', ls='--', lw=1.2, alpha=0.75, label=rf\"Pump $f_1 = {f_pump:.0f}$ kHz\")\n",
    "    ax1.legend(loc='lower left', fontsize=8.0, framealpha=0.92, facecolor='white', edgecolor='gray')\n",
    "\n",
    "    # Annotate diff_peaks[:10] without collision detection\n",
    "    for idx, p in enumerate(diff_peaks[:10]):\n",
    "        target_x = float(np.clip(p[1] + off_x, 6.0, 140.0))\n",
    "        target_y = float(np.clip(y_coord + off_y, -144.0, -6.0))\n",
    "        ax1.annotate(f\"#{idx+1}\", xy=(p[1], y_coord), xytext=(target_x, target_y), ...)\n",
    "\n",
    "    extra_badge_sec = \"\"\n",
    "    if \"NBI\" in title_win and \"ECH\" not in title_win:\n",
    "        sec_casc_peaks = [f\"#{idx+1}\" for idx, p in enumerate(diff_peaks[:10]) if abs(p[2] - 0.5 * (p[1] - 41.6)) < 8.0]\n",
    "        extra_badge_sec = f\"\\n\\u2022 Sec. Cascade (f1-int = 41.6 kHz):\\n  Peaks {', '.join(sec_casc_peaks)}\"\n",
    "\n",
    "    badge_text = (\n",
    "        f\"Leading: f1={diff_peaks[0][1]:.1f}, f2={diff_peaks[0][2]:.1f}, f3={diff_peaks[0][3]:.1f} kHz | b\\u00b2={diff_peaks[0][0]:.3f}\\n\"\n",
    "        f\"95% Stat. Floor: b\\u00b2 = {b2_thresh_95:.3f}\\n\\n\"\n",
    "        \"Non-Linear Triad Clustering:\\n\"\n",
    "        \"\\u2022 Self-product line (slope -0.5):\\n\"\n",
    "        f\"  Peaks {', '.join(subh_peaks) if subh_peaks else 'None'}\\n\"\n",
    "        f\"\\u2022 Pump Column (f1 = {f_pump:.0f} kHz):\\n\"\n",
    "        f\"  {pump_str}\"\n",
    "        f\"{extra_badge_sec}\"\n",
    "    )\n",
]

final24_map_snippet = [
    "    # Reference lines (Secondary cascade removed)\n",
    "    ax1.plot(\n",
    "        [0, PLOT_FMAX_KHZ], [0, -0.5 * PLOT_FMAX_KHZ], color='cyan', ls='-.', lw=1.6, alpha=0.95,\n",
    "        label=r\"Self-product line $f_2 = -0.5 f_1$ ($f/2+f/2 \\leftrightarrow f$)\"\n",
    "    )\n",
    "    ax1.axvline(f_pump, color='gold', ls='--', lw=1.2, alpha=0.75, label=rf\"Pump $f_1 = {f_pump:.0f}$ kHz\")\n",
    "    ax1.legend(loc='lower left', fontsize=8.0, framealpha=0.92, facecolor='white', edgecolor='gray')\n",
    "\n",
    "    # Annotate diff_peaks[:10] with collision detection and shifting\n",
    "    placed_labels = []\n",
    "    for idx, p in enumerate(diff_peaks[:10]):\n",
    "        target_x = float(np.clip(p[1] + off_x, 6.0, 140.0))\n",
    "        target_y = float(np.clip(y_coord + off_y, -144.0, -6.0))\n",
    "        for prev_r, prev_tx, prev_ty in placed_labels:\n",
    "            if abs(target_x - prev_tx) < 8.0 and abs(target_y - prev_ty) < 6.0:\n",
    "                target_x = float(np.clip(target_x + 8.0, 6.0, 140.0))\n",
    "                target_y = float(np.clip(target_y - 6.0, -144.0, -6.0))\n",
    "                break\n",
    "        placed_labels.append((r_num, target_x, target_y))\n",
    "        ax1.annotate(f\"#{idx+1}\", xy=(p[1], y_coord), xytext=(target_x, target_y), ...)\n",
    "\n",
    "    badge_text = (\n",
    "        f\"Leading: f1={diff_peaks[0][1]:.1f}, f2={diff_peaks[0][2]:.1f}, f3={diff_peaks[0][3]:.1f} kHz | b\\u00b2={diff_peaks[0][0]:.3f}\\n\"\n",
    "        f\"95% Stat. Floor: b\\u00b2 = {b2_thresh_95:.3f}\\n\\n\"\n",
    "        \"Labelled peaks on guide lines:\\n\"\n",
    "        \"\\u2022 Self-product line (slope -0.5):\\n\"\n",
    "        f\"  Peaks {', '.join(subh_peaks) if subh_peaks else 'None'}\\n\"\n",
    "        f\"\\u2022 Pump Column (f1 = {f_pump:.0f} kHz):\\n\"\n",
    "        f\"  {pump_str}\"\n",
    "    )\n",
]

p_out("\nUNIFIED DIFF (final23 make_maps_v4.py vs final24 run_peaks_v5.py Map Plotting Code):")
for dl in difflib.unified_diff(
    final23_map_snippet, final24_map_snippet,
    fromfile="make_maps_v4.py (final23)", tofile="run_peaks_v5.py (final24)", n=3
):
    p_out(dl, end="")

# Load final23 peak lists from final23 step1_stdout.txt for computed comparison
final23_peaks_stored = {
    "ECH + NBI Phase": [
        (1, 146.48, -70.31, 76.17, 0.3152),
        (2, 146.48, -74.22, 72.27, 0.3098),
        (3, 78.12, -39.06, 39.06, 0.2876),
        (4, 101.56, -46.87, 54.69, 0.2794),
        (5, 44.92, -21.48, 23.44, 0.2431),
        (6, 142.58, -58.59, 83.98, 0.2328),
        (7, 87.89, -48.83, 39.06, 0.2316),
        (8, 17.58, -7.81, 9.77, 0.2280),
        (9, 105.47, -60.55, 44.92, 0.2193),
        (10, 101.56, -54.69, 46.87, 0.1993),
    ],
    "Just NBI Phase": [
        (1, 148.44, -101.56, 46.87, 0.6693),
        (2, 60.55, -9.77, 50.78, 0.6591),
        (3, 101.56, -50.78, 50.78, 0.6551),
        (4, 74.22, -21.48, 52.73, 0.6213),
        (5, 105.47, -52.73, 52.73, 0.5886),
        (6, 148.44, -50.78, 97.66, 0.5435),
        (7, 148.44, -97.66, 50.78, 0.5188),
        (8, 138.67, -93.75, 44.92, 0.5179),
        (9, 121.09, -66.41, 54.69, 0.5068),
        (10, 68.36, -23.44, 44.92, 0.5001),
    ]
}

offsets_map = {
    1: (8, 6), 2: (-22, -10), 3: (8, 6), 4: (-20, 10), 5: (6, 8),
    6: (8, 6), 7: (-20, 8), 8: (8, -12), 9: (-22, 8), 10: (8, -12)
}

PLOT_FMAX_KHZ = 150.0

for title_win, tw_start, out_png, f_pump in [
    ("ECH + NBI Phase", 259.1, "corrected_bicoherence_map_ECH_NBI.png", 89.0),
    ("Just NBI Phase", 291.3, "corrected_bicoherence_map_NBI_only.png", 86.0),
]:
    p_out("\n" + "-" * 100)
    p_out(f"GENERATING UPDATED MAP: {out_png} ({title_win})")
    p_out("-" * 100)

    idx0 = int(np.where(t_ms_53 >= tw_start)[0][0])
    w_start = float(t_ms_53[idx0])
    w_end = float(t_ms_53[idx0] + sig_len * dt_53 * 1000.0)
    y_sub = ys_mp1_53[idx0 : idx0 + sig_len]
    t_sub_s = (t_ms_53[idx0 : idx0 + sig_len] - t_ms_53[idx0]) / 1000.0

    _, _, spec = dsp.spectrogram(
        y_sub, fs=fs_53, window="hann", nperseg=nfft, noverlap=noverlap, nfft=nfft,
        detrend="linear", return_onesided=False, scaling="density", axis=0, mode="complex"
    )
    spec = np.transpose(spec, [1, 0])[:n_ens, :]
    _, b2_raw = abicoh2_corrected(spec)
    bicoh2 = np.transpose(np.fft.fftshift(b2_raw, axes=0), [1, 0])

    local_max = (bicoh2 == maximum_filter(bicoh2, size=filter_size_bins)) & mask_diff
    coords = np.argwhere(local_max)
    vals = bicoh2[local_max]
    order = np.argsort(vals)[::-1]

    diff_peaks = []
    for p_idx in order[:10]:
        r, c = coords[p_idx]
        val = float(bicoh2[r, c])
        pf1 = float(f1_vec[r])
        pf2 = float(-f2_vec[c])
        pf3 = float(pf1 - pf2)
        diff_peaks.append((val, pf1, pf2, pf3, r, c))

    display_mask = mask_diff
    color_vmax = float(np.nanmax(bicoh2[display_mask]))
    color_vmax = max(color_vmax * 1.05, b2_thresh_95, 1e-6)
    color_levels = np.linspace(0.0, color_vmax, 101)

    plt.rcdefaults()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Panel 1: Contourf
    c = ax1.contourf(FF1, FF2, bicoh2, cmap="inferno", levels=color_levels, extend="max")
    ax1.set_title(
        f"Squared Auto-Bicoherence $b^2$ - Leading Channel: MP1 (nensemble={n_ens}, {title_win}, {w_start:.1f} - {w_end:.1f} ms)",
        fontsize=11.5, fontweight="bold"
    )
    ax1.set_xlabel(r"$f_1$ (kHz)", fontsize=10.5)
    ax1.set_ylabel(r"$f_2$ (kHz)", fontsize=10.5)
    ax1.grid(True, alpha=0.3, linestyle=":")
    ax1.set_xlim(0, PLOT_FMAX_KHZ)
    ax1.set_ylim(-PLOT_FMAX_KHZ, 0.0)

    # (i) Reference lines: self-product line and pump line (NO secondary cascade on NBI-only)
    ax1.plot(
        [0, PLOT_FMAX_KHZ], [0, -0.5 * PLOT_FMAX_KHZ], color="cyan", ls="-.", lw=1.6, alpha=0.95,
        label=r"Self-product line $f_2 = -0.5 f_1$ ($f/2+f/2 \leftrightarrow f$)"
    )
    ax1.axvline(f_pump, color="gold", ls="--", lw=1.2, alpha=0.75, label=rf"Pump $f_1 = {f_pump:.0f}$ kHz")
    ax1.legend(loc="lower left", fontsize=8.0, framealpha=0.92, facecolor="white", edgecolor="gray")

    cbar = fig.colorbar(c, ax=ax1, label=r"Squared Bicoherence $b^2$")
    cbar.ax.axhline(b2_thresh_95, color="cyan", ls=":", lw=1.5)

    # (iii) Fix label collisions: if closer than 8 kHz in f1 and 6 kHz in f2, shift later label by (+8, -6)
    p_out(f"\nLabel Collision Check for {title_win}:")
    placed_labels = []
    shifted_labels_record = []
    for idx, p in enumerate(diff_peaks[:10]):
        r_num = idx + 1
        y_coord = -p[2]
        off_x, off_y = offsets_map.get(r_num, (8, 6))
        if p[1] > 125.0:
            off_x = -22.0
        elif p[1] < 18.0:
            off_x = 8.0
        if y_coord > -18.0:
            off_y = -14.0
        elif y_coord < -135.0:
            off_y = 12.0

        tx = float(np.clip(p[1] + off_x, 6.0, 140.0))
        ty = float(np.clip(y_coord + off_y, -144.0, -6.0))
        orig_tx, orig_ty = tx, ty

        # Check collision against earlier placed labels
        for prev_r, prev_tx, prev_ty in placed_labels:
            if abs(tx - prev_tx) < 8.0 and abs(ty - prev_ty) < 6.0:
                tx = float(np.clip(tx + 8.0, 6.0, 140.0))
                ty = float(np.clip(ty - 6.0, -144.0, -6.0))
                shifted_labels_record.append((r_num, prev_r, orig_tx, orig_ty, tx, ty))
                break

        placed_labels.append((r_num, tx, ty))

        ax1.annotate(
            f"#{r_num}", xy=(p[1], y_coord), xytext=(tx, ty),
            color="white", fontsize=9.0, fontweight="bold", zorder=6,
            arrowprops=dict(arrowstyle="->", color="white", lw=1.1, alpha=0.85, shrinkA=2, shrinkB=3),
            path_effects=[path_effects.withStroke(linewidth=2.5, foreground="black")]
        )

    if shifted_labels_record:
        for r_num, prev_r, otx, oty, ntx, nty in shifted_labels_record:
            p_out(f"  Shifted label #{r_num} (collided with #{prev_r}): was ({otx:.2f}, {oty:.2f}) -> shifted by (+8, -6) to ({ntx:.2f}, {nty:.2f}) kHz")
    else:
        p_out("  No label collisions detected.")

    # (ii) Badge text update: "Non-Linear Triad Clustering:" -> "Labelled peaks on guide lines:"
    subh_peaks = [f"#{idx+1}" for idx, p in enumerate(diff_peaks[:10]) if abs(p[2] / p[1] - 0.5) < 0.05]
    pump_peaks = [f"#{idx+1} ({p[1]:.1f} - {p[2]:.1f} = {p[3]:.1f} kHz)" for idx, p in enumerate(diff_peaks[:10]) if abs(p[1] - f_pump) < 6.0]
    pump_str = "\n  ".join(pump_peaks[:2]) if pump_peaks else "None"

    badge_text = (
        f"Leading: f1={diff_peaks[0][1]:.1f}, f2={diff_peaks[0][2]:.1f}, f3={diff_peaks[0][3]:.1f} kHz | b\u00b2={diff_peaks[0][0]:.3f}\n"
        f"95% Stat. Floor: b\u00b2 = {b2_thresh_95:.3f}\n\n"
        "Labelled peaks on guide lines:\n"
        "\u2022 Self-product line (slope -0.5):\n"
        f"  Peaks {', '.join(subh_peaks) if subh_peaks else 'None'}\n"
        f"\u2022 Pump Column (f1 = {f_pump:.0f} kHz):\n"
        f"  {pump_str}"
    )
    ax1.text(
        0.98, 0.03, badge_text, transform=ax1.transAxes, fontsize=8.0,
        va="bottom", ha="right", bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray", alpha=0.9)
    )

    # Panel 2: PSD
    f_psd, Pxx = LAS.psd(y_sub, t_sub_s, dt=dt_53, nfft=nfft, noverlap=noverlap, nensemble=n_ens, detrend="linear")
    f_psd_k = f_psd / 1000.0
    mask_psd = (f_psd_k >= 0) & (f_psd_k <= 150.0)

    ax2.semilogy(f_psd_k[mask_psd], Pxx[mask_psd], color="tab:blue", linewidth=1.8, label=f"PSD MP1 ({title_win})")
    ax2.set_title("Power Spectral Density (PSD) of Leading Channel: MP1", fontsize=11, fontweight="bold", pad=8)
    ax2.set_xlabel("Frequency (kHz)", fontsize=10)
    ax2.set_ylabel(r"Power (V$^2$/Hz)", fontsize=10)
    ax2.grid(True, alpha=0.3, which="both", linestyle=":")
    ax2.set_xlim(0, 150.0)

    p1 = diff_peaks[0]
    ax2.axvline(x=p1[1], color="#d62728", linestyle="--", linewidth=1.4, label=f"Peak #1: $f_1$={p1[1]:.1f} kHz ($b^2$={p1[0]:.2f})")
    ax2.axvline(x=p1[2], color="#d62728", linestyle=":", linewidth=1.4, label=f"Peak #1: $f_2$={p1[2]:.1f} kHz")

    prim_peak = next((p for p in diff_peaks if abs(p[1] - f_pump) <= 6.0 or abs(p[2] - f_pump) <= 6.0), None)
    if prim_peak and prim_peak != p1:
        ax2.axvline(x=prim_peak[1], color="darkorange", linestyle="--", linewidth=1.6, label=f"Primary Triad: $f_1$={prim_peak[1]:.1f} kHz ($b^2$={prim_peak[0]:.2f})")
        ax2.axvline(x=prim_peak[2], color="darkorange", linestyle=":", linewidth=1.6, label=f"Primary Triad: $f_2$={prim_peak[2]:.1f} kHz")
        ax2.axvline(x=prim_peak[3], color="darkorange", linestyle="-.", linewidth=1.2, alpha=0.7, label=f"Primary Triad: $f_3$={prim_peak[3]:.1f} kHz")
    elif len(diff_peaks) > 1:
        p2 = diff_peaks[1]
        ax2.axvline(x=p2[1], color="#2ca02c", linestyle="--", linewidth=1.3, label=f"Peak #2: $f_1$={p2[1]:.1f} kHz ($b^2$={p2[0]:.2f})")
        ax2.axvline(x=p2[2], color="#2ca02c", linestyle=":", linewidth=1.3, label=f"Peak #2: $f_2$={p2[2]:.1f} kHz")

    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    plt.tight_layout()

    # Union-bbox save for Panel 1 only
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pad = 0.08
    bb_ax = ax1.get_tightbbox(renderer)
    bb_cb = cbar.ax.get_tightbbox(renderer)
    bb_union = mtransforms.Bbox.union([bb_ax, bb_cb])
    bbi = bb_union.transformed(fig.dpi_scale_trans.inverted())
    bbp = mtransforms.Bbox.from_extents(bbi.x0 - pad, bbi.y0 - pad, bbi.x1 + pad, bbi.y1 + pad)

    out_file = fig_dir / out_png
    fig.savefig(str(out_file), dpi=150, bbox_inches=bbp, facecolor=fig.get_facecolor(), edgecolor="none", pad_inches=0)
    plt.close(fig)

    im_new = Image.open(out_file)
    p_out(f"\nSaved Updated Map: {out_png}")
    p_out(f"  Path:       {out_file}")
    p_out(f"  Pixel size: {im_new.size}")
    p_out(f"  Bytes:      {out_file.stat().st_size}")
    p_out(f"  SHA-256:    {sha256_file(out_file)}")

    # Reprint the ten-peak table
    p_out(f"\nREPRINTED TEN-PEAK TABLE ({title_win}):")
    header_str = (
        f"{'Rank':<4} | {'f1 (kHz)':<8} | {'f2 (kHz)':<9} | {'f3 (kHz)':<8} | {'b^2':<7} | "
        f"{'Sum Form (|f2|+f3=f1)':<22} | {'Cyan Line?':<11} | {'Pump Col?':<9} | "
        f"{'N_eff':<6} | {'Max Seg Share (%)'}"
    )
    p_out(header_str)
    p_out("-" * len(header_str))

    new_peaks_summary = []
    for idx, p in enumerate(diff_peaks[:10]):
        rank_num = idx + 1
        val, pf1, pf2, pf3, r, c = p
        on_line = abs(pf2 / pf1 - 0.5) < 0.05
        in_pump = abs(pf1 - f_pump) < 6.0

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
        new_peaks_summary.append((rank_num, round(pf1, 2), round(-pf2, 2), round(pf3, 2), round(val, 4)))

    # Computed comparison of peak lists against final23
    f23_ref = final23_peaks_stored[title_win]
    lists_match = (len(new_peaks_summary) == len(f23_ref)) and all(
        p_new[0] == p_ref[0] and
        np.isclose(p_new[1], p_ref[1], atol=1e-2) and
        np.isclose(p_new[2], p_ref[2], atol=1e-2) and
        np.isclose(p_new[3], p_ref[3], atol=1e-2) and
        np.isclose(p_new[4], p_ref[4], atol=1e-4)
        for p_new, p_ref in zip(new_peaks_summary, f23_ref)
    )
    p_out(f"\nPeak list identical to final23 (computed comparison): {lists_match}")
    assert lists_match, f"Peak list for {title_win} differs from final23!"

p_out("\n" + "=" * 120)
p_out("STEP 1 COMPLETE")
p_out("=" * 120)
stdout_h.close()
