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

def generate_mode_envelopes_and_ip_plot(shot=88653, out_path=None, t_start=245.0, t_end=310.0):
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

    # Extract envelopes:
    # Mode 1: Primary Mode (80-120 kHz band, f0 ~ 89.8 kHz, m = 3)
    env_m1, _, _, _ = extract_instantaneous_frequency(ys_mp1, fs, 80000.0, 120000.0, 4, 325)
    # Mode 2: Secondary Mode (40-80 kHz band, f0 ~ 43.0 kHz, m = 2)
    env_m2, _, _, _ = extract_instantaneous_frequency(ys_mp1, fs, 40000.0, 80000.0, 4, 325)

    # Time window masking
    idx = np.where((t_ms >= t_start) & (t_ms <= t_end))[0]
    t_w = t_ms[idx]
    ip_w = ip_interp[idx]
    env_m1_w = env_m1[idx]
    env_m2_w = env_m2[idx]

    # Lowpass filter for smooth envelopes (200 Hz butterworth)
    b_lp, a_lp = dsp.butter(3, 200.0 / (fs / 2.0), btype="low")
    env_m1_sm = dsp.filtfilt(b_lp, a_lp, env_m1_w)
    env_m2_sm = dsp.filtfilt(b_lp, a_lp, env_m2_w)
    ip_sm = dsp.filtfilt(b_lp, a_lp, ip_w)

    # Styling
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "lines.linewidth": 1.8,
        "axes.linewidth": 0.9,
    })

    fig, (ax_env, ax_ip) = plt.subplots(
        2, 1, figsize=(11.0, 8.0), sharex=True, facecolor="white",
        gridspec_kw={"height_ratios": [1.3, 1.0], "hspace": 0.10}
    )

    fig.suptitle(
        f"Heliotron J #{shot} — Mode Envelopes ($I_\\mathrm{{env}}$) and Plasma Current ($I_p$) Time Evolution",
        fontsize=13, fontweight="bold", y=0.965
    )

    # Color scheme
    c_m1 = "#d62728"   # Crimson red for Mode 1 (m=3)
    c_m2 = "#1f77b4"   # Steel blue for Mode 2 (m=2)
    c_ip = "#2ca02c"   # Forest green for Ip

    # Key transition timestamps
    t_burst_start = 259.1  # Mode activity / ECH+NBI onset
    t_ech_off = 291.3      # ECH turn-off
    t_burst_end = 300.0    # End of NBI flat-top / mode quench

    # -------------------------------------------------------------
    # Top Panel: Mode Envelopes I_env
    # -------------------------------------------------------------
    # Fast envelope fluctuations
    ax_env.plot(t_w, env_m1_w, color=c_m1, lw=0.5, alpha=0.20)
    ax_env.plot(t_w, env_m2_w, color=c_m2, lw=0.5, alpha=0.20)

    # Smoothed envelopes
    line_m1, = ax_env.plot(t_w, env_m1_sm, color=c_m1, lw=2.4, label=r"Mode 1 ($m = 3$, $f \approx 89.8$ kHz, $80-120$ kHz)")
    line_m2, = ax_env.plot(t_w, env_m2_sm, color=c_m2, lw=2.2, label=r"Mode 2 ($m = 2$, $f \approx 43.0$ kHz, $40-80$ kHz)")

    ax_env.set_ylabel(r"$I_\mathrm{env}$ (Mode Amplitude, V)", fontsize=11.5, fontweight="bold")
    ax_env.set_ylim(-0.01, 0.40)
    ax_env.grid(True, ls=":", alpha=0.45)

    # Annotations matching the user's sketch pointers
    idx_pk_m1 = np.argmax(env_m1_sm)
    t_pk_m1 = t_w[idx_pk_m1]
    pk_val_m1 = env_m1_sm[idx_pk_m1]
    ax_env.annotate(
        r"$\leftarrow$ Mode 1 ($m = 3$)",
        xy=(t_pk_m1, pk_val_m1),
        xytext=(t_pk_m1 + 2.0, pk_val_m1 + 0.02),
        fontsize=10.5, fontweight="bold", color=c_m1,
        arrowprops=dict(arrowstyle="->", color=c_m1, lw=1.3)
    )

    idx_pk_m2 = np.argmax(env_m2_sm)
    t_pk_m2 = t_w[idx_pk_m2]
    pk_val_m2 = env_m2_sm[idx_pk_m2]
    ax_env.annotate(
        r"$\leftarrow$ Mode 2 ($m = 2$)",
        xy=(t_pk_m2, pk_val_m2),
        xytext=(t_pk_m2 + 2.5, pk_val_m2 + 0.03),
        fontsize=10.5, fontweight="bold", color=c_m2,
        arrowprops=dict(arrowstyle="->", color=c_m2, lw=1.3)
    )

    # Vertical dashed lines across both panels
    for ax in (ax_env, ax_ip):
        ax.axvline(t_burst_start, color="#555555", ls="--", lw=1.3, alpha=0.85)
        ax.axvline(t_ech_off, color="#e67e22", ls="--", lw=1.4, alpha=0.95)
        ax.axvline(t_burst_end, color="#555555", ls="--", lw=1.3, alpha=0.85)

    # Labels for vertical lines
    ax_env.text(t_burst_start - 0.6, 0.33, "Mode Onset\n(259.1 ms)",
                rotation=90, va="top", ha="right", fontsize=8.5, color="#444444", fontweight="bold")
    ax_env.text(t_ech_off - 0.6, 0.33, "ECH Off\n(291.3 ms)",
                rotation=90, va="top", ha="right", fontsize=8.5, color="#b95e00", fontweight="bold")
    ax_env.text(t_burst_end + 0.8, 0.33, "Mode Quench\n(300.0 ms)",
                rotation=90, va="top", ha="left", fontsize=8.5, color="#444444", fontweight="bold")

    ax_env.legend(loc="upper left", framealpha=0.95, edgecolor="#cccccc")

    # -------------------------------------------------------------
    # Bottom Panel: Plasma Current Ip
    # -------------------------------------------------------------
    ax_ip.plot(t_w, ip_w, color=c_ip, lw=0.8, alpha=0.30, label=r"Raw $I_p$")
    ax_ip.plot(t_w, ip_sm, color=c_ip, lw=2.4, label=r"Plasma Current $I_p(t)$")
    ax_ip.set_ylabel(r"$I_p$ (Plasma Current, kA)", fontsize=11.5, fontweight="bold")
    ax_ip.set_xlabel("Time (ms)", fontsize=11.5, fontweight="bold")
    ax_ip.set_xlim(t_start, t_end)
    ax_ip.set_ylim(0.0, 1.55)
    ax_ip.grid(True, ls=":", alpha=0.45)

    # Annotate Ip peak
    idx_pk_ip = np.argmax(ip_sm)
    t_pk_ip = t_w[idx_pk_ip]
    pk_val_ip = ip_sm[idx_pk_ip]
    ax_ip.annotate(
        f"Peak $I_p = {pk_val_ip:.2f}$ kA\n($t = {t_pk_ip:.1f}$ ms)",
        xy=(t_pk_ip, pk_val_ip),
        xytext=(t_pk_ip - 15.0, pk_val_ip - 0.25),
        fontsize=9.5, fontweight="bold", color=c_ip,
        arrowprops=dict(arrowstyle="->", color=c_ip, lw=1.3),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=c_ip, alpha=0.9)
    )

    ax_ip.legend(loc="upper left", framealpha=0.95, edgecolor="#cccccc")

    if out_path is None:
        out_path = root_dir / f"mhd_mode_envelopes_and_ip_evolution_shot_{shot}.png"
    else:
        out_path = Path(out_path)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")
    return out_path

if __name__ == "__main__":
    generate_mode_envelopes_and_ip_plot()
