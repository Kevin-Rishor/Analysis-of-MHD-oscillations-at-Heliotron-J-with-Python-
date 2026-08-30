"""
mhd_plot_b_tilde_vs_envelope.py
Analysis of fast carrier oscillation B_tilde vs Hilbert envelope A(t) for Shot 88653.
Investigates wave-packet beating, envelope modulation spectrum, and candidate
three-wave coupling / parametric decay instability (PDI) signatures.
Typography matched to project standard (sans-serif, bold titles).
"""

from pathlib import Path
import numpy as np
import scipy.signal as dsp
import matplotlib.pyplot as plt

import sys
current = Path(__file__).resolve().parent
ROOT_DIR = None
for p in [current] + list(current.parents):
    if (p / "jpack").exists():
        ROOT_DIR = p
        break
if ROOT_DIR is None:
    ROOT_DIR = Path("c:/TFG")

for p_add in [ROOT_DIR / "jpack", ROOT_DIR / "analysis", ROOT_DIR / "analysis" / "common"]:
    if str(p_add) not in sys.path:
        sys.path.append(str(p_add))

import turnelib as TE
from mhd_common import extract_instantaneous_frequency

def analyze_b_tilde_vs_envelope(shot=88653, out_dir="."):
    out_dir = Path(out_dir)
    data_dir = Path(r"c:\TFG\data") / f"hj{shot}"
    mp1_file = data_dir / f"MP1@{shot}.edf"

    if not mp1_file.exists():
        raise FileNotFoundError(f"Missing Mirnov probe file: {mp1_file}")

    # Load MP1 data
    edf = TE.edf()
    dat = edf.load(str(mp1_file))
    t_raw = dat[:, 0]
    t_ms = t_raw if edf.DimUnit[0] == "ms" else t_raw * 1000.0
    dt_sec = (t_ms[1] - t_ms[0]) / 1000.0
    fs = 1.0 / dt_sec
    ys = dat[:, 1]

    # Filter parameters matching Objective 2 (80 - 120 kHz)
    fl_hz = 80000.0
    fu_hz = 120000.0
    order = 4
    smoothing = 325

    envelope, phase, b_tilde, ifreq_hz = extract_instantaneous_frequency(
        ys, fs, fl_hz, fu_hz, order, smoothing
    )
    ifreq_khz = ifreq_hz / 1000.0

    # Active mode burst window
    t_start, t_end = 259.1, 275.0
    idx_win = np.where((t_ms >= t_start) & (t_ms <= t_end))[0]
    t_win = t_ms[idx_win]
    b_win = b_tilde[idx_win]
    env_win = envelope[idx_win]

    # Zoom window for detailed wave-packet visualization
    t_zoom_start, t_zoom_end = 264.5, 267.0
    idx_zoom = np.where((t_ms >= t_zoom_start) & (t_ms <= t_zoom_end))[0]

    # 1. Envelope Modulation Spectrum (Welch PSD of detrended envelope)
    env_detrend = env_win - np.mean(env_win)
    nperseg = min(4096, len(env_win))
    f_env, P_env = dsp.welch(env_detrend, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2)
    f_env_khz = f_env / 1000.0

    # Identify dominant low-frequency modulation peaks (f < 8 kHz)
    mask_low = (f_env_khz >= 0.5) & (f_env_khz <= 8.0)
    top_indices = np.argsort(P_env[mask_low])[::-1][:2]
    top_f_mod = f_env_khz[mask_low][top_indices]
    top_P_mod = P_env[mask_low][top_indices]

    print(f"Top Envelope Modulation Frequencies for Shot {shot}:")
    for fp, pp in zip(top_f_mod, top_P_mod):
        print(f"  f_mod = {fp:.2f} kHz | PSD = {pp:.3e}")

    # Set up publication figure
    plt.rcdefaults()
    fig = plt.figure(figsize=(13.5, 9.0), facecolor="white")
    fig.suptitle(
        f"Heliotron J #{shot} — Nonlinear Wave-Envelope Dynamics & Three-Wave Coupling Diagnostics\n"
        rf"Mirnov Coil MP1 ($f_\mathrm{{carrier}} \approx 89.0$ kHz, Mode Window: {t_start:.1f} - {t_end:.1f} ms)",
        fontsize=12.5, fontweight="bold", y=0.97
    )

    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28, top=0.90, bottom=0.08, left=0.08, right=0.95)
    ax_time = fig.add_subplot(gs[0, :])
    ax_spec = fig.add_subplot(gs[1, 0])
    ax_scat = fig.add_subplot(gs[1, 1])

    # =========================================================================
    # PANEL (a): TIME-DOMAIN CARRIER & ENVELOPE DYNAMICS
    # =========================================================================
    ax_time.plot(t_ms[idx_zoom], b_tilde[idx_zoom], color="tab:blue", lw=1.0, alpha=0.85, label=r"Fast Carrier $\tilde{B}(t)$ ($80-120$ kHz)")
    ax_time.plot(t_ms[idx_zoom], envelope[idx_zoom], color="crimson", lw=2.0, label=r"Hilbert Envelope $+A(t)$")
    ax_time.plot(t_ms[idx_zoom], -envelope[idx_zoom], color="crimson", ls="--", lw=1.5, label=r"Lower Bound $-A(t)$")

    ax_time.set_xlim(t_zoom_start, t_zoom_end)
    ax_time.set_xlabel("Time (ms)", fontsize=10)
    ax_time.set_ylabel(r"Fluctuation Voltage $\tilde{B}_\theta$ (V)", fontsize=10)
    ax_time.set_title(r"(a) Carrier Oscillation $\tilde{B}(t)$ vs. Hilbert Envelope $A(t)$ (Wave-Packet Beating)", fontsize=11, fontweight="bold", pad=8)
    ax_time.grid(True, ls=":", alpha=0.4)
    ax_time.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # Highlight a wave packet beat period
    ax_time.annotate(
        r"Beating Envelope $\Delta t \approx 0.7$ ms" + "\n" + r"($f_\mathrm{mod} \approx 1.5$ kHz)",
        xy=(265.60, 0.16), xytext=(265.15, 0.14),
        arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5),
        fontsize=9, fontweight="bold", color="darkred",
        bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", ec="crimson", alpha=0.9)
    )

    # =========================================================================
    # PANEL (b): ENVELOPE MODULATION POWER SPECTRUM
    # =========================================================================
    ax_spec.plot(f_env_khz, P_env, color="tab:purple", lw=1.8, label=r"Envelope PSD")
    # Annotate primary peak
    p_primary_f, p_primary_p = top_f_mod[0], top_P_mod[0]
    ax_spec.plot(p_primary_f, p_primary_p, "o", color="crimson", ms=6)
    ax_spec.annotate(
        f"Peak $f_2 \\approx {p_primary_f:.2f}$ kHz", xy=(p_primary_f, p_primary_p), xytext=(p_primary_f + 0.4, p_primary_p * 1.4),
        fontsize=8.5, fontweight="bold", color="darkred",
        arrowprops=dict(arrowstyle="->", color="darkred", lw=1.0)
    )

    ax_spec.set_xlim(0.0, 8.0)
    ax_spec.set_yscale("log")
    ax_spec.set_xlabel("Modulation Frequency (kHz)", fontsize=10)
    ax_spec.set_ylabel(r"PSD $(\mathrm{V}^2/\mathrm{Hz})$", fontsize=10)
    ax_spec.set_title(r"(b) Envelope Modulation Spectrum (Candidate $f_2$)", fontsize=11, fontweight="bold", pad=8)
    ax_spec.grid(True, ls=":", alpha=0.4)

    # Inset textbox for Three-Wave Matching physics
    text_pdi = (
        r"Three-Wave Coupling Hypothesis:" + "\n"
        r"$\bullet$ Pump wave: $f_0 \approx 89.0$ kHz" + "\n"
        r"$\bullet$ Daughter wave 1: $f_1 \approx 87.5$ kHz" + "\n"
        r"$\bullet$ Daughter wave 2 (acoustic/GAM):" + "\n"
        rf"  $f_2 = f_0 - f_1 = f_\mathrm{{mod}} \approx {top_f_mod[0]:.2f}$ kHz" + "\n"
        r"$\bullet$ Resonance: $f_0 = f_1 + f_2$"
    )
    ax_spec.text(
        0.38, 0.62, text_pdi, transform=ax_spec.transAxes,
        fontsize=8.5, va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", fc="aliceblue", ec="tab:blue", alpha=0.95)
    )

    # =========================================================================
    # PANEL (c): PHASE-SPACE: B_TILDE VS ENVELOPE
    # =========================================================================
    skip = 2
    sc = ax_scat.scatter(
        b_win[::skip], env_win[::skip],
        c=t_win[::skip], cmap="viridis", s=6, alpha=0.5, rasterized=True
    )
    cbar = plt.colorbar(sc, ax=ax_scat, fraction=0.046, pad=0.04)
    cbar.set_label("Time (ms)", fontsize=9.5)

    # Draw theoretical cone boundaries B = +A and B = -A
    a_max = np.max(env_win) * 1.05
    ax_scat.plot([0, a_max], [0, a_max], "r--", lw=1.5, label=r"Boundary $\tilde{B} = +A$")
    ax_scat.plot([0, -a_max], [0, a_max], "r--", lw=1.5, label=r"Boundary $\tilde{B} = -A$")

    ax_scat.set_xlim(-0.55, 0.55)
    ax_scat.set_ylim(0.0, 0.55)
    ax_scat.set_xlabel(r"Fast Fluctuation $\tilde{B}(t)$ (V)", fontsize=10)
    ax_scat.set_ylabel(r"Hilbert Envelope $A(t)$ (V)", fontsize=10)
    ax_scat.set_title(r"(c) Fast $\tilde{B}$ vs. Envelope $A(t)$ Phase Space", fontsize=11, fontweight="bold", pad=8)
    ax_scat.grid(True, ls=":", alpha=0.4)
    ax_scat.legend(loc="upper center", fontsize=8.5, framealpha=0.9)

    out_png = out_dir / f"mhd_b_tilde_vs_envelope_shot_{shot}.png"
    plt.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Clean B_tilde vs Envelope diagnostic saved to: '{out_png}'")
    return str(out_png)

if __name__ == "__main__":
    analyze_b_tilde_vs_envelope(88653, out_dir=r"c:\TFG")
