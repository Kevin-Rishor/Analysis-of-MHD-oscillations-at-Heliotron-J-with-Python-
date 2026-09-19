import sys
from pathlib import Path
import numpy as np
import scipy.signal as dsp
import matplotlib.pyplot as plt

root_dir = Path(r"c:\TFG")
for p_add in [root_dir / "jpack", root_dir / "analysis", root_dir / "analysis" / "common"]:
    if str(p_add) not in sys.path:
        sys.path.append(str(p_add))

import turnelib as TE
from mhd_common import extract_instantaneous_frequency

def generate_nbi_comparison_envelope_ip_plot(shot=88653, out_dir=None):
    if out_dir is None:
        out_dir = root_dir / "NBI vs NBI and ECH"
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = root_dir / "data" / f"hj{shot}"
    edf = TE.edf()

    # Load Mirnov MP1
    d_mp1 = edf.load(str(data_dir / f"MP1@{shot}.edf"))
    t_raw = d_mp1[:, 0]
    t_ms = t_raw if edf.DimUnit[0] == "ms" else t_raw * 1000.0
    dt = (t_ms[1] - t_ms[0]) / 1000.0
    fs = 1.0 / dt
    ys_mp1 = d_mp1[:, 1]

    # Load Plasma Current Ip15
    d_ip = edf.load(str(data_dir / f"Ip15@{shot}.edf"))
    t_ip_raw = d_ip[:, 0]
    t_ip_ms = t_ip_raw if edf.DimUnit[0] == "ms" else t_ip_raw * 1000.0
    ip_raw = d_ip[:, 1]
    ip_interp = np.interp(t_ms, t_ip_ms, ip_raw)

    # Extract envelopes
    # Mode 1: Primary Mode (80-120 kHz band, f0 ~ 89.8 kHz, m = 3)
    env_m1, _, _, _ = extract_instantaneous_frequency(ys_mp1, fs, 80000.0, 120000.0, 4, 325)
    # Mode 2: Secondary Mode (40-80 kHz band, f0 ~ 43.0 kHz, m = 2)
    env_m2, _, _, _ = extract_instantaneous_frequency(ys_mp1, fs, 40000.0, 80000.0, 4, 325)

    # Comparison focus window (255.0 to 302.0 ms)
    t_start, t_end = 255.0, 302.0
    idx = np.where((t_ms >= t_start) & (t_ms <= t_end))[0]
    t_w = t_ms[idx]
    ip_w = ip_interp[idx]
    env_m1_w = env_m1[idx]
    env_m2_w = env_m2[idx]

    # 200 Hz lowpass smoothing
    b_lp, a_lp = dsp.butter(3, 200.0 / (fs / 2.0), btype="low")
    env_m1_sm = dsp.filtfilt(b_lp, a_lp, env_m1_w)
    env_m2_sm = dsp.filtfilt(b_lp, a_lp, env_m2_w)
    ip_sm = dsp.filtfilt(b_lp, a_lp, ip_w)

    # Phase boundaries
    T_W1_START = 259.1  # ECH + NBI start
    T_W1_END = 291.3    # Measured ECH turn-off
    T_W2_START = 291.3  # Just NBI start
    T_W2_END = 300.0    # Just NBI end

    # Styling
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.0,
        "lines.linewidth": 1.8,
        "axes.linewidth": 0.9,
    })

    fig, (ax_env, ax_ip) = plt.subplots(
        2, 1, figsize=(11.5, 8.5), sharex=True, facecolor="white",
        gridspec_kw={"height_ratios": [1.3, 1.0], "hspace": 0.10}
    )

    fig.suptitle(
        f"Heliotron J #{shot} — Mode Envelopes ($I_\\mathrm{{env}}$) & Plasma Current ($I_p$)\n"
        r"Phase Comparison: ECH + NBI (259.1 – 291.3 ms) vs. Just NBI (291.3 – 300.0 ms)",
        fontsize=12.5, fontweight="bold", y=0.97
    )

    c_m1 = "#d62728"   # Crimson red for Mode 1 (m=3)
    c_m2 = "#1f77b4"   # Steel blue for Mode 2 (m=2)
    c_ip = "#2ca02c"   # Forest green for Ip

    # Shading the two comparison phases
    for ax in (ax_env, ax_ip):
        ax.axvspan(T_W1_START, T_W1_END, color="#3498db", alpha=0.08)
        ax.axvspan(T_W2_START, T_W2_END, color="#e67e22", alpha=0.10)
        ax.axvline(T_W1_START, color="#2980b9", ls="--", lw=1.2, alpha=0.85)
        ax.axvline(T_W1_END, color="#d35400", ls="-", lw=1.8, alpha=0.95)
        ax.axvline(T_W2_END, color="#7f8c8d", ls="--", lw=1.2, alpha=0.85)

    # -------------------------------------------------------------
    # Top Panel: Mode Envelopes
    # -------------------------------------------------------------
    ax_env.plot(t_w, env_m1_w, color=c_m1, lw=0.5, alpha=0.20)
    ax_env.plot(t_w, env_m2_w, color=c_m2, lw=0.5, alpha=0.20)

    ax_env.plot(t_w, env_m1_sm, color=c_m1, lw=2.4, label=r"Mode 1 ($m = 3$, $f \approx 89.8$ kHz, $80-120$ kHz)")
    ax_env.plot(t_w, env_m2_sm, color=c_m2, lw=2.2, label=r"Mode 2 ($m = 2$, $f \approx 43.0$ kHz, $40-80$ kHz)")

    ax_env.set_ylabel(r"$I_\mathrm{env}$ (Mode Amplitude, V)", fontsize=11.5, fontweight="bold")
    ax_env.set_ylim(-0.01, 0.42)
    ax_env.grid(True, ls=":", alpha=0.45)

    # Annotate mode behaviors
    ax_env.annotate(
        r"$\leftarrow$ Mode 1 ($m = 3$)" + "\nPeak at ECH turn-off",
        xy=(291.5, 0.32), xytext=(293.0, 0.355),
        fontsize=9.5, fontweight="bold", color=c_m1,
        arrowprops=dict(arrowstyle="->", color=c_m1, lw=1.3)
    )

    ax_env.annotate(
        r"$\leftarrow$ Mode 2 ($m = 2$)" + "\nQuenches when ECH drops",
        xy=(293.0, 0.08), xytext=(294.0, 0.17),
        fontsize=9.5, fontweight="bold", color=c_m2,
        arrowprops=dict(arrowstyle="->", color=c_m2, lw=1.3)
    )

    # Phase banners placed in clean positions
    ax_env.text(278.0, 0.38, "PHASE 1: ECH + NBI (259.1–291.3 ms)\nBoth Modes Active, Mode 1 Ramps",
                ha="center", va="center", fontsize=9.0, fontweight="bold", color="#1b4f72",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#ebf5fb", edgecolor="#3498db", alpha=0.92))

    ax_env.text(295.6, 0.26, "PHASE 2: JUST NBI\n(291.3–300.0 ms)\nMode 1 Persists\nMode 2 Decays",
                ha="center", va="center", fontsize=8.5, fontweight="bold", color="#78281f",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#fef5e7", edgecolor="#e67e22", alpha=0.92))

    # ECH Off marker text
    ax_env.text(T_W1_END - 0.4, 0.03, "ECH Cut-off (291.3 ms)", rotation=90, va="bottom", ha="right",
                fontsize=8.5, fontweight="bold", color="#d35400",
                bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="#d35400", alpha=0.85))

    ax_env.legend(loc="upper left", framealpha=0.92, edgecolor="#cccccc")

    # -------------------------------------------------------------
    # Bottom Panel: Plasma Current Ip
    # -------------------------------------------------------------
    ax_ip.plot(t_w, ip_w, color=c_ip, lw=0.8, alpha=0.30, label=r"Raw $I_p$")
    ax_ip.plot(t_w, ip_sm, color=c_ip, lw=2.4, label=r"Plasma Current $I_p(t)$")
    ax_ip.set_ylabel(r"$I_p$ (Plasma Current, kA)", fontsize=11.5, fontweight="bold")
    ax_ip.set_xlabel("Time (ms)", fontsize=11.5, fontweight="bold")
    ax_ip.set_xlim(t_start, t_end)
    ax_ip.set_ylim(0.4, 1.55)
    ax_ip.grid(True, ls=":", alpha=0.45)

    # Annotations on Ip panel
    idx_pk_ip = np.argmax(ip_sm)
    t_pk_ip = t_w[idx_pk_ip]
    pk_val_ip = ip_sm[idx_pk_ip]
    ax_ip.annotate(
        f"Peak $I_p = {pk_val_ip:.2f}$ kA\n($t = {t_pk_ip:.1f}$ ms in Phase 1)",
        xy=(t_pk_ip, pk_val_ip),
        xytext=(t_pk_ip - 14.0, pk_val_ip - 0.18),
        fontsize=9.0, fontweight="bold", color=c_ip,
        arrowprops=dict(arrowstyle="->", color=c_ip, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=c_ip, alpha=0.9)
    )

    ax_ip.text(295.6, 0.60, "Ip Rollover & Decay\n(Ip: 1.23 -> 0.89 kA)",
               ha="center", va="center", fontsize=9.0, fontweight="bold", color="#27ae60",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="#eafaf1", edgecolor="#2ecc71", alpha=0.9))

    ax_ip.legend(loc="lower left", framealpha=0.92, edgecolor="#cccccc")

    out_file = out_dir / "mhd_mode_envelopes_and_ip_ECH_NBI_comparison.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved NBI comparison plot to {out_file}")
    return out_file

if __name__ == "__main__":
    generate_nbi_comparison_envelope_ip_plot()
