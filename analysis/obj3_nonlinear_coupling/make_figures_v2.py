import sys
import time
import hashlib
import traceback
import difflib
from pathlib import Path
import numpy as np
import scipy.signal as dsp
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t_start_wall = time.time()

base_dir = Path(r"C:\Users\kevin\.gemini\antigravity\brain\5a23ddc3-ffcc-4a9b-a7f0-d135446e46c3")
out_dir = base_dir / "scratch" / "final21"
fig_dir = out_dir / "staging" / "figures_final20"
fig_dir.mkdir(parents=True, exist_ok=True)
stdout_file = out_dir / "step1_stdout.txt"
stdout_h = open(stdout_file, "w", encoding="utf-8")

def p_out(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=stdout_h)

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

p_out("========================================================================================================================")
p_out("TASK FINAL21 - STEP 1: FIGURE 3 FIX AND COPIED MAPS VERIFICATION")
p_out("========================================================================================================================")

try:
    # -------------------------------------------------------------------------
    # Reusing data loading, mask_B, abicoh2_corrected, and state classification
    # -------------------------------------------------------------------------
    root_dir = Path("C:/TFG")
    for p_add in [
        root_dir / "jpack",
        root_dir / "analysis",
        root_dir / "analysis" / "common",
        root_dir / "analysis" / "heating_phase_comparison",
    ]:
        if str(p_add) not in sys.path:
            sys.path.append(str(p_add))

    from comparison_common import load_channel
    staging_common = base_dir / "scratch" / "final11" / "staging" / "analysis" / "common"
    sys.path.insert(0, str(staging_common))
    from abicoh2_corrected import abicoh2_corrected

    t_ms_53, ys_mp1_53, dt_53, fs_53 = load_channel(88653, "MP1")
    t_ms_54, ys_mp1_54, dt_54, fs_54 = load_channel(88654, "MP1")

    nfft = 512
    nhalf = 256
    hop = 256
    noverlap = 256
    n_ens = 30
    sig_len = 256 * 29 + 512  # 7936 samples
    df_khz = (fs_53 / nfft) / 1e3  # 1.953125 kHz

    f1_vec = np.arange(nhalf) * df_khz
    f2_vec = (np.arange(nfft) - nhalf) * df_khz
    FF1, FF2 = np.meshgrid(f1_vec, f2_vec, indexing="ij")

    # Construct mask_B exactly as in final17 run_scan.py
    fmin_khz = 5.0
    f1max = 100.0
    f2max = 100.0
    fs_khz = fs_53 / 1e3
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
    ka_B = coords_B[:, 0]
    kb_B = coords_B[:, 1] - nhalf
    M_B = len(coords_B)

    p_out(f"\nMask B total bin count M = {M_B}")
    assert M_B == 2306, f"Mask B bin count mismatch: got {M_B}, expected 2306"
    p_out("Assertion M == 2306 passed: True")

    # State classifier
    ECH_on, ECH_off = 171.2, 291.32
    NBI_on, NBI_off = 222.4, 300.4

    def classify_state(start_ms, end_ms):
        if end_ms < NBI_on:
            return "pre-NBI"
        elif start_ms >= NBI_on and end_ms <= ECH_on:
            return "NBI-only (pre-ECH)"
        elif start_ms >= ECH_on and end_ms <= ECH_off:
            return "ECH+NBI"
        elif start_ms >= ECH_off and end_ms <= NBI_off:
            return "NBI-only (post-ECH)"
        elif start_ms >= NBI_off:
            return "post-NBI"
        else:
            return "transition"

    # Set matplotlib global style
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    # =========================================================================
    # FIGURES 1 & 2: Copy from final20 unchanged (do not regenerate)
    # =========================================================================
    p_out("\n" + "=" * 120)
    p_out("FIGURES 1 & 2: COPYING FROM FINAL20 UNCHANGED AND ASSERTING SHA-256")
    p_out("=" * 120)

    src_fig_dir = base_dir / "scratch" / "final20" / "figures"
    map_files = [
        "corrected_bicoherence_map_ECH_NBI.png",
        "corrected_bicoherence_map_NBI_only.png",
    ]

    p_out(f"{'Figure File':<42} | {'Original SHA-256':<64} | {'Copied SHA-256':<64} | {'Bytes':<8} | {'Assert'}")
    p_out("-" * 195)

    for fname in map_files:
        src_p = src_fig_dir / fname
        dst_p = fig_dir / fname
        assert src_p.exists(), f"Source figure missing: {src_p}"
        raw_bytes = src_p.read_bytes()
        dst_p.write_bytes(raw_bytes)

        sha_src = sha256_file(src_p)
        sha_dst = sha256_file(dst_p)
        assert sha_src == sha_dst, f"SHA mismatch for {fname}: src={sha_src}, dst={sha_dst}"
        p_out(f"{fname:<42} | {sha_src:<64} | {sha_dst:<64} | {len(raw_bytes):<8d} | True")

    # =========================================================================
    # FIGURE 3: Control-Window Plot (Shots 88653 and 88654)
    # =========================================================================
    p_out("\n" + "=" * 120)
    p_out("GENERATING FIGURE 3: CONTROL-WINDOW PLOT (SHOTS 88653 & 88654) WITH FIGURE-LEVEL LEGEND")
    p_out("=" * 120)

    # Compute max b2 over mask_B for all 33 windows for both shots
    def scan_shot_tiling1(ys, fs, t_ms):
        n_windows = 33
        win_starts = []
        max_b2_list = []
        states_list = []
        for w in range(n_windows):
            i_start = w * sig_len
            w_start = float(t_ms[i_start])
            w_end = float(t_ms[i_start] + sig_len * dt_53 * 1000.0)
            y_w = ys[i_start : i_start + sig_len]
            _, _, spec = dsp.spectrogram(
                y_w, fs=fs, window="hann", nperseg=nfft, noverlap=noverlap, nfft=nfft,
                detrend="linear", return_onesided=False, scaling="density", axis=0, mode="complex"
            )
            spec = np.transpose(spec, [1, 0])[:n_ens, :]
            _, b2_raw = abicoh2_corrected(spec)
            b2_grid = np.transpose(np.fft.fftshift(b2_raw, axes=0), [1, 0])
            b2_obs = b2_grid[coords_B[:, 0], coords_B[:, 1]]
            max_val = float(np.max(b2_obs))
            st = classify_state(w_start, w_end)
            win_starts.append(w_start)
            max_b2_list.append(max_val)
            states_list.append(st)
        return np.array(win_starts), np.array(max_b2_list), states_list

    # Nominal window baseline max b2 for dashed lines
    def get_nominal_max_b2(ys, fs, t_ms, t_target):
        idx0 = int(np.where(t_ms >= t_target)[0][0])
        y_w = ys[idx0 : idx0 + sig_len]
        _, _, spec = dsp.spectrogram(
            y_w, fs=fs, window="hann", nperseg=nfft, noverlap=noverlap, nfft=nfft,
            detrend="linear", return_onesided=False, scaling="density", axis=0, mode="complex"
        )
        spec = np.transpose(spec, [1, 0])[:n_ens, :]
        _, b2_raw = abicoh2_corrected(spec)
        b2_grid = np.transpose(np.fft.fftshift(b2_raw, axes=0), [1, 0])
        b2_obs = b2_grid[coords_B[:, 0], coords_B[:, 1]]
        top_i = int(np.argmax(b2_obs))
        max_val = float(b2_obs[top_i])
        ka_t = int(coords_B[top_i, 0])
        kb_t = int(coords_B[top_i, 1] - nhalf)
        return max_val, ka_t, kb_t

    ws_53, maxs_53, states_53 = scan_shot_tiling1(ys_mp1_53, fs_53, t_ms_53)
    ws_54, maxs_54, states_54 = scan_shot_tiling1(ys_mp1_54, fs_54, t_ms_54)

    nom_ech_53, ka_ech_53, kb_ech_53 = get_nominal_max_b2(ys_mp1_53, fs_53, t_ms_53, 259.1)
    nom_nbi_53, ka_nbi_53, kb_nbi_53 = get_nominal_max_b2(ys_mp1_53, fs_53, t_ms_53, 291.3)

    nom_ech_54, ka_ech_54, kb_ech_54 = get_nominal_max_b2(ys_mp1_54, fs_54, t_ms_54, 259.1)
    nom_nbi_54, ka_nbi_54, kb_nbi_54 = get_nominal_max_b2(ys_mp1_54, fs_54, t_ms_54, 291.3)

    p_out("\nNominal window reference lines:")
    p_out(f"  Shot 88653 nominal 259.1 ms window: max b2 = {nom_ech_53:.4f} (triad {ka_ech_53*df_khz:.1f}, {kb_ech_53*df_khz:.1f} kHz)")
    p_out(f"  Shot 88653 nominal 291.3 ms window: max b2 = {nom_nbi_53:.4f} (triad {ka_nbi_53*df_khz:.1f}, {kb_nbi_53*df_khz:.1f} kHz)")
    p_out(f"  Shot 88654 nominal 259.1 ms window: max b2 = {nom_ech_54:.4f} (triad {ka_ech_54*df_khz:.1f}, {kb_ech_54*df_khz:.1f} kHz)")
    p_out(f"  Shot 88654 nominal 291.3 ms window: max b2 = {nom_nbi_54:.4f} (triad {ka_nbi_54*df_khz:.1f}, {kb_nbi_54*df_khz:.1f} kHz)")

    # Plot figure 3
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True, dpi=200)

    # State marker styling
    state_style = {
        "pre-NBI": {"marker": "o", "color": "#1f77b4", "label": "pre-NBI"},
        "NBI-only (pre-ECH)": {"marker": "^", "color": "#ff7f0e", "label": "NBI-only / transition"},
        "transition": {"marker": "^", "color": "#ff7f0e", "label": "NBI-only / transition"},
        "ECH+NBI": {"marker": "s", "color": "#d62728", "label": "ECH+NBI"},
        "post-NBI": {"marker": "D", "color": "#9467bd", "label": "post-NBI"},
    }

    def plot_control_panel(ax, ws, maxs, states, nom_ech, nom_nbi, shot_title):
        # Shaded heating spans
        ax.axvspan(222.4, 300.4, color="#d9f0d3", alpha=0.55, label="NBI (222.4-300.4 ms)", zorder=0)
        ax.axvspan(171.2, 291.32, color="#fee0b6", alpha=0.65, label="ECH (171.2-291.32 ms)", zorder=1)

        # Dashed horizontal reference lines
        ax.axhline(
            nom_ech, color="#d62728", linestyle="--", linewidth=1.4,
            label=f"259.1 ms window ({nom_ech:.4f})", zorder=2
        )
        ax.axhline(
            nom_nbi, color="#1f77b4", linestyle="--", linewidth=1.4,
            label=f"291.3 ms window ({nom_nbi:.4f})", zorder=2
        )

        # Plot data points by state category
        plotted_categories = set()
        for x, y, st in zip(ws, maxs, states):
            st_key = st if st in state_style else "transition"
            cfg = state_style[st_key]
            cat_label = cfg["label"]
            lbl = cat_label if cat_label not in plotted_categories else None
            plotted_categories.add(cat_label)
            ax.plot(
                x, y,
                marker=cfg["marker"], color=cfg["color"],
                markersize=6, linestyle="None", markeredgecolor="black",
                markeredgewidth=0.6, label=lbl, zorder=4
            )

        # Subtle connecting line
        ax.plot(ws, maxs, color="#7f7f7f", linestyle=":", linewidth=0.8, alpha=0.7, zorder=3)

        ax.set_ylabel(r"max b$^2$ over search domain")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(shot_title, fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4)

    plot_control_panel(ax1, ws_53, maxs_53, states_53, nom_ech_53, nom_nbi_53, "Shot #88653 MP1 (Control Windows)")
    plot_control_panel(ax2, ws_54, maxs_54, states_54, nom_ech_54, nom_nbi_54, "Shot #88654 MP1 (Control Windows)")

    ax2.set_xlabel("window start (ms)")
    ax2.set_xlim(145.0, 415.0)

    # Figure-level legend placed below the two panels (ncol=3, fontsize=10, frameon=True)
    # with subplots_adjust(bottom=0.22) leaving room so nothing overlaps data
    handles1, labels1 = ax1.get_legend_handles_labels()
    by_label1 = dict(zip(labels1, handles1))

    fig.subplots_adjust(bottom=0.22, top=0.95, left=0.12, right=0.95, hspace=0.28)
    fig.legend(
        by_label1.values(),
        by_label1.keys(),
        loc="lower center",
        ncol=3,
        fontsize=10,
        frameon=True
    )

    fig3_path = fig_dir / "bicoherence_control_windows_88653_88654.png"
    plt.savefig(fig3_path, dpi=200, facecolor="white")
    plt.close(fig)

    p_out(f"\nSaved bicoherence_control_windows_88653_88654.png successfully to {fig3_path}.")

    # =========================================================================
    # FIGURE METADATA & RE-READ VERIFICATION
    # =========================================================================
    p_out("\n" + "=" * 120)
    p_out("SAVED FIGURES SUMMARY AND FILE METADATA IN STAGING")
    p_out("=" * 120)

    fig_files = [
        fig_dir / "corrected_bicoherence_map_ECH_NBI.png",
        fig_dir / "corrected_bicoherence_map_NBI_only.png",
        fig_dir / "bicoherence_control_windows_88653_88654.png",
    ]

    p_out(f"{'Figure File':<45} | {'Dimensions (px)':<16} | {'Bytes':<10} | {'SHA-256'}")
    p_out("-" * 142)

    for fp in fig_files:
        assert fp.exists(), f"Figure file missing: {fp}"
        raw = fp.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        size_bytes = len(raw)
        with Image.open(fp) as im:
            dims = f"{im.size[0]} x {im.size[1]}"
        p_out(f"{str(fp):<45} | {dims:<16} | {size_bytes:<10d} | {sha}")

    # =========================================================================
    # UNIFIED DIFF: make_figures.py vs make_figures_v2.py
    # =========================================================================
    p_out("\n" + "=" * 120)
    p_out("UNIFIED DIFF: make_figures.py vs make_figures_v2.py")
    p_out("=" * 120)

    src_py_path = base_dir / "scratch" / "final20" / "make_figures.py"
    dst_py_path = base_dir / "scratch" / "final21" / "make_figures_v2.py"

    src_lines = src_py_path.read_text(encoding="utf-8").splitlines(keepends=True)
    dst_lines = dst_py_path.read_text(encoding="utf-8").splitlines(keepends=True)

    diff_lines = list(difflib.unified_diff(
        src_lines,
        dst_lines,
        fromfile="scratch/final20/make_figures.py",
        tofile="scratch/final21/make_figures_v2.py",
        n=3
    ))

    diff_text = "".join(diff_lines)
    p_out(diff_text)

    wall_elapsed = time.time() - t_start_wall
    p_out("\n" + "=" * 120)
    p_out(f"Total make_figures_v2.py wall-clock execution time: {wall_elapsed:.2f} s")
    p_out("=" * 120)

except Exception as ex:
    p_out(f"\nERROR: Exception occurred during make_figures_v2.py execution:")
    p_out(traceback.format_exc())
    stdout_h.close()
    sys.exit(1)

stdout_h.close()
raw_bytes = stdout_file.read_bytes()
out_sha = hashlib.sha256(raw_bytes).hexdigest()
print(f"Saved stdout to step1_stdout.txt ({len(raw_bytes)} bytes, SHA-256: {out_sha})")
