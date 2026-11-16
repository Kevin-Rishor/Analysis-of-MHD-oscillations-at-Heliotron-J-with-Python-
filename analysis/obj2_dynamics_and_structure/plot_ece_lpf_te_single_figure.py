"""
plot_ece_lpf_te_single_figure.py
================================
Generates a clean, publication-quality, wide horizontal 2-panel figure for:
Verification of Relative Te Proxy via Low-Pass Filtered ECE:
  Panel (a): Dynamic Tracking Across Heating Phases (Normalized ECE LPF vs Wp, ne, Wp/ne)
  Panel (b): Direct Cross-Correlation & Linear Regression (ECE LPF vs Wp and Wp/ne)

Output:
  - Images/obj2_dynamics_and_structure/mhd_ece_lpf_te_verification_88653.png
  - overleaf_upload_20261017/Images/obj2_dynamics_and_structure/mhd_ece_lpf_te_verification_88653.png
"""

import sys
import shutil
from pathlib import Path
import numpy as np
import scipy.signal as dsp
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root_dir = Path(__file__).resolve().parents[2]
sys.path.extend([
    str(root_dir / "jpack"),
    str(root_dir / "analysis"),
    str(root_dir / "analysis" / "common"),
    str(root_dir / "analysis" / "obj2_dynamics_and_structure"),
])

import turnelib as TE
from mhd_analysis_obj2 import estimate_acf, conservative_p_value

def generate_single_figure(shot=88653, lpf_cutoff_hz=30.0):
    data_dir = root_dir / "data" / f"hj{shot}"
    edf = TE.edf()

    # Load timebase from MP1
    dat_mp1 = edf.load(str(data_dir / f"MP1@{shot}.edf"))
    t_raw = dat_mp1[:, 0]
    t_ms = t_raw if edf.DimUnit[0] == "ms" else t_raw * 1000.0
    dt = (t_ms[100] - t_ms[0]) / 100.0 / 1000.0
    fs = 1.0 / dt

    # Load Stored Energy Wp
    dat_wp = edf.load(str(data_dir / f"Wp@{shot}.edf"))
    t_wp = dat_wp[:, 0] if edf.DimUnit[0] == "ms" else dat_wp[:, 0] * 1000.0
    wp = np.interp(t_ms, t_wp, dat_wp[:, 1])

    # Load Line-Averaged Density ne
    dat_ne = edf.load(str(data_dir / f"nave@{shot}.edf"))
    t_ne = dat_ne[:, 0] if edf.DimUnit[0] == "ms" else dat_ne[:, 0] * 1000.0
    raw_ne = dsp.medfilt(np.clip(dat_ne[:, 1], 0.01, None), 5)
    ne = np.interp(t_ms, t_ne, raw_ne)
    glitch = (t_ms >= 170.0) & (t_ms <= 190.0)
    if np.any(glitch):
        pre_ne = ne[t_ms < 170.0][-1]
        post_ne = ne[t_ms > 190.0][0]
        ne[glitch] = np.linspace(pre_ne, post_ne, np.sum(glitch))
    ne_clean = np.clip(ne, 0.05, None)

    # Diamagnetic temperature proxy Te_diamag proportional to Wp / ne
    te_diamag = wp / ne_clean

    # Design 4th-order zero-phase Butterworth LPF
    sos_lpf = dsp.butter(4, lpf_cutoff_hz / (fs / 2.0), btype="low", output="sos")

    # Load ECE9FAST (58.5 GHz, outboard/chord rho ~ 1.05) and ECE12FAST (64.5 GHz, core resonant rho = 0.465)
    def load_lpf_ece(ch):
        dat_e = edf.load(str(data_dir / f"ECE{ch}FAST@{shot}.edf"))
        t_e = dat_e[:, 0] if edf.DimUnit[0] == "ms" else dat_e[:, 0] * 1000.0
        y_e = np.interp(t_ms, t_e, dat_e[:, 1])
        base = np.median(y_e[t_ms < 170.0])
        return np.clip(dsp.sosfiltfilt(sos_lpf, y_e - base), 1e-4, None)

    lpf9 = load_lpf_ece(9)
    lpf12 = load_lpf_ece(12)

    # Normalization over active burst window [248.3, 300.9] ms
    m_burst = (t_ms >= 248.3) & (t_ms <= 300.9)
    def norm01(arr):
        sub = arr[m_burst]
        mn, mx = np.min(sub), np.max(sub)
        return (arr - mn) / max(mx - mn, 1e-6)

    norm_wp = norm01(wp)
    norm_ne = norm01(ne_clean)
    norm_tedia = norm01(te_diamag)
    norm_ece9 = norm01(lpf9)
    norm_ece12 = norm01(lpf12)

    # Subsampled arrays for regression / cross-correlation
    step = 100
    wp_sub = norm_wp[m_burst][::step]
    tedia_sub = norm_tedia[m_burst][::step]
    ece9_sub = norm_ece9[m_burst][::step]
    ece12_sub = norm_ece12[m_burst][::step]

    # Calculate statistics
    acf_b = estimate_acf(ece9_sub, nlags=min(50, len(ece9_sub) - 2))
    N_eff_b = max(3.0, min(float(len(ece9_sub)), len(ece9_sub) / (1.0 + 2.0 * np.sum(acf_b[1:]))))

    acf_12 = estimate_acf(ece12_sub, nlags=min(50, len(ece12_sub) - 2))
    N_eff_12 = max(3.0, min(float(len(ece12_sub)), len(ece12_sub) / (1.0 + 2.0 * np.sum(acf_12[1:]))))

    r_wp9, p_wp9_raw = stats.pearsonr(ece9_sub, wp_sub)
    p_wp9 = conservative_p_value(r_wp9, N_eff_b)

    r_ted9, p_ted9_raw = stats.pearsonr(ece9_sub, tedia_sub)
    p_ted9 = conservative_p_value(r_ted9, N_eff_b)

    r_wp12, p_wp12_raw = stats.pearsonr(ece12_sub, wp_sub)
    p_wp12 = conservative_p_value(r_wp12, N_eff_12)

    r_ted12, p_ted12_raw = stats.pearsonr(ece12_sub, tedia_sub)
    p_ted12 = conservative_p_value(r_ted12, N_eff_12)

    # Setup 2-panel horizontal figure with publication-legible typography
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "font.size": 12.5,
        "axes.labelsize": 13.5,
        "axes.titlesize": 13.5,
        "xtick.labelsize": 12.0,
        "ytick.labelsize": 12.0,
        "legend.fontsize": 10.5,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 5.2), gridspec_kw={"width_ratios": [1.08, 1.05]})

    # -------------------------------------------------------------
    # Panel (a): Normalized Dynamic Tracking Across Heating Phases
    # -------------------------------------------------------------
    m_plot = (t_ms >= 240.0) & (t_ms <= 305.0)
    t_p = t_ms[m_plot]

    ax1.plot(t_p, norm_ece9[m_plot], color="#1f77b4", lw=2.5, zorder=5,
             label=r"Outboard/Chord LPF ECE09 ($\rho \approx 1.05$, 58.5 GHz)")
    ax1.plot(t_p, norm_ece12[m_plot], color="#d62728", lw=2.2, linestyle="--", zorder=4,
             label=r"Core Resonant LPF ECE12 ($\rho = 0.465$, 64.5 GHz)")
    ax1.plot(t_p, norm_wp[m_plot], color="#2ca02c", lw=2.1, linestyle="-.", zorder=3,
             label=r"Stored Energy $W_p(t)$")
    ax1.plot(t_p, norm_tedia[m_plot], color="#ff7f0e", lw=2.1, linestyle=":", zorder=3,
             label=r"Diamagnetic $T_e$ proxy ($W_p/n_e$)")
    ax1.plot(t_p, norm_ne[m_plot], color="#9467bd", lw=1.5, alpha=0.80, zorder=2,
             label=r"Line Density $n_e(t)$")

    # Shaded pump-out region & ECH line
    ax1.axvspan(270.0, 291.2, color="#ffe8cc", alpha=0.55, zorder=1)
    ax1.axvline(291.2, color="black", linestyle=":", lw=1.6, zorder=6)

    # Text placed cleanly in open gaps
    ax1.text(280.6, 0.015, "Density Pump-Out\n(270.0–291.2 ms)", fontsize=10.2,
             color="#b35900", ha="center", va="bottom", fontstyle="italic",
             bbox=dict(boxstyle="round,pad=0.22", facecolor="#fff9f2", edgecolor="#ffcc99", alpha=0.92))
    ax1.text(292.1, 0.44, "ECH Off\n(291.2 ms)", fontsize=10.2, color="black", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc", alpha=0.88))

    ax1.set_xlim(240.0, 305.0)
    # Set y-limits to provide generous header space above y=1.02 for clear legend placement
    ax1.set_ylim(-0.05, 1.92)
    ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_xlabel(r"Time $t$ [ms]", fontsize=13.0, fontweight="bold")
    ax1.set_ylabel(r"Normalized Signal Amplitude $A_{\mathrm{norm}}$ [a.u.]", fontsize=13.0, fontweight="bold")
    ax1.set_title(r"(a) Dynamic Tracking vs. $W_p$, $n_e$, and $W_p/n_e$",
                  fontsize=12.8, fontweight="bold", pad=8)
    ax1.grid(True, alpha=0.3, linestyle="--")

    # Clean legend floating in clear top space
    ax1.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), ncol=1, fontsize=10.0,
               labelspacing=0.25, framealpha=0.95, edgecolor="#cccccc")

    # -------------------------------------------------------------
    # Panel (b): Direct Cross-Correlation & Linear Fits
    # -------------------------------------------------------------
    def _fmt_p(pv):
        return f"$p = {pv:.3f}$" if pv >= 0.001 else r"$p < 0.001$"

    # 1. ECE9 vs Wp
    slope_wp9, int_wp9 = np.polyfit(ece9_sub, wp_sub, 1)
    x_fit = np.linspace(0.0, 1.0, 100)
    ax2.scatter(ece9_sub, wp_sub, color="#2ca02c", alpha=0.60, s=30, edgecolors="none",
                label=(rf"Outboard/Chord LPF ECE09 ($\rho \approx 1.05$, 58.5 GHz)"
                       + "\n" + rf"vs. $W_p$: $r = \mathbf{{+{r_wp9:.3f}}}$ ({_fmt_p(p_wp9)})"))
    ax2.plot(x_fit, slope_wp9 * x_fit + int_wp9, color="#2ca02c", lw=2.3, linestyle="-")

    # 2. ECE9 vs Wp/ne (diamagnetic Te)
    slope_ted9, int_ted9 = np.polyfit(ece9_sub, tedia_sub, 1)
    ax2.scatter(ece9_sub, tedia_sub, color="#ff7f0e", alpha=0.60, s=30, marker="^", edgecolors="none",
                label=(rf"Outboard/Chord LPF ECE09 ($\rho \approx 1.05$, 58.5 GHz)"
                       + "\n" + rf"vs. $W_p/n_e$: $r = \mathbf{{+{r_ted9:.3f}}}$ ({_fmt_p(p_ted9)})"))
    ax2.plot(x_fit, slope_ted9 * x_fit + int_ted9, color="#ff7f0e", lw=2.3, linestyle=":")

    # 3. ECE12 vs Wp/ne (resonant check)
    slope_ted12, int_ted12 = np.polyfit(ece12_sub, tedia_sub, 1)
    ax2.scatter(ece12_sub, tedia_sub, color="#d62728", alpha=0.48, s=26, marker="s", edgecolors="none",
                label=(rf"Core Resonant LPF ECE12 ($\rho = 0.465$, 64.5 GHz)"
                       + "\n" + rf"vs. $W_p/n_e$: $r = \mathbf{{+{r_ted12:.3f}}}$ ({_fmt_p(p_ted12)})"))
    ax2.plot(x_fit, slope_ted12 * x_fit + int_ted12, color="#d62728", lw=2.0, linestyle="--")

    # Reference unity line
    ax2.plot([0, 1], [0, 1], color="#888888", linestyle="-.", lw=1.2, alpha=0.55, label="Ideal 1:1 Identity")

    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.92)
    ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_xlabel(r"Normalized 30 Hz Low-Pass ECE $\tilde{T}_{e,\mathrm{LPF}}$ [a.u.]", fontsize=13.0, fontweight="bold")
    ax2.set_ylabel(r"Normalized Target $W_p$ or $W_p/n_e$ [a.u.]", fontsize=13.0, fontweight="bold")
    ax2.set_title(r"(b) Cross-Correlation ($t \in [248.3, 300.9]$ ms)",
                  fontsize=12.8, fontweight="bold", pad=8)
    ax2.grid(True, alpha=0.3, linestyle="--")

    # Legend in clear top-left area
    ax2.legend(loc="upper left", bbox_to_anchor=(0.015, 0.99), fontsize=9.6,
               labelspacing=0.25, framealpha=0.94, edgecolor="#cccccc")

    # Note in open bottom-right corner
    note_txt = (
        r"$\mathbf{Burst\ Window\ (248.3–300.9\ ms):}$" + "\n"
        + rf"$\bullet\ N_{{\mathrm{{eff}}}} \approx {N_eff_b:.1f}\ \mathrm{{(decorr.\ ACF)}}$" + "\n"
        + r"$\bullet\ \mathrm{Filter:\ 30\ Hz\ SOS\ Butterworth}$"
    )
    print(f"ECE9xWp: r={r_wp9:+.3f}, p={p_wp9:.4f}; ECE9xTe_dia: r={r_ted9:+.3f}, p={p_ted9:.4f}; ECE12xTe_dia: r={r_ted12:+.3f}, p={p_ted12:.4f}; N_eff={N_eff_b:.2f}")
    ax2.text(0.97, 0.04, note_txt, transform=ax2.transAxes, fontsize=9.8,
             ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f9fa", edgecolor="#dddddd", alpha=0.92))

    plt.tight_layout()

    out_dir = root_dir / "Images" / "obj2_dynamics_and_structure"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mhd_ece_lpf_te_verification_{shot}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated clean verification plot: {out_path}")

    print(f"Single-panel verification plot saved to: {out_path}")

if __name__ == "__main__":
    generate_single_figure(88653)
