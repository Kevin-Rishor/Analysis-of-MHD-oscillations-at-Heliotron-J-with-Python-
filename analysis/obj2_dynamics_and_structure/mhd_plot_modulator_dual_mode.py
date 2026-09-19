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


def generate_modulator_dual_mode_plot(shot=88653, out_dir=None):
    if out_dir is None:
        out_dir = root_dir
    else:
        out_dir = Path(out_dir)

    data_dir = root_dir / "data" / f"hj{shot}"
    t_start, t_end = 259.1, 275.0
    edf = TE.edf()

    # 1. Mirnov MP1
    mp1_file = data_dir / f"MP1@{shot}.edf"
    dat = edf.load(str(mp1_file))
    t_raw = dat[:, 0]
    t_ms = t_raw if edf.DimUnit[0] == "ms" else t_raw * 1000.0
    dt = (t_ms[1] - t_ms[0]) / 1000.0
    fs = 1.0 / dt
    ys_mp1 = dat[:, 1]

    # 2. ECE13FAST
    ece_file = data_dir / f"ECE13FAST@{shot}.edf"
    dat_ece = edf.load(str(ece_file))
    ys_ece = dat_ece[:, 1]

    # 3. nave (Interferometer density)
    nave_file = data_dir / f"nave@{shot}.edf"
    dat_nave = edf.load(str(nave_file))
    ys_nave = dat_nave[:, 1]

    # 4. HAFAST7.5
    ha_file = data_dir / f"HAFAST7.5@{shot}.edf"
    dat_ha = edf.load(str(ha_file))
    ys_ha = dat_ha[:, 1]

    # Extract envelopes (order 4, smoothing 325 as in thesis standard)
    env_p, _, b_p, _ = extract_instantaneous_frequency(ys_mp1, fs, 80000.0, 120000.0, 4, 325)
    env_s, _, b_s, _ = extract_instantaneous_frequency(ys_mp1, fs, 40000.0, 80000.0, 4, 325)

    idx_win = np.where((t_ms >= t_start) & (t_ms <= t_end))[0]
    ys_win = ys_mp1[idx_win]
    env_p_win = env_p[idx_win] - np.mean(env_p[idx_win])
    env_s_win = env_s[idx_win] - np.mean(env_s[idx_win])

    nperseg = 4096
    noverlap = 3072
    nfft = 8192

    # Diagnostic PSDs in low frequency
    f, p_mp1 = dsp.welch(ys_win - np.mean(ys_win), fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    _, p_ece = dsp.welch(ys_ece[idx_win] - np.mean(ys_ece[idx_win]), fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    _, p_nave = dsp.welch(ys_nave[idx_win] - np.mean(ys_nave[idx_win]), fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    _, p_ha = dsp.welch(ys_ha[idx_win] - np.mean(ys_ha[idx_win]), fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, nfft=nfft)

    # Coherence calculations
    # Cross-envelope coherence: Env_P vs Env_S
    f, coh_ps = dsp.coherence(env_p_win, env_s_win, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    f, pxy_ps = dsp.csd(env_p_win, env_s_win, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, nfft=nfft)

    # Self-coherence: Raw B vs Env_P, Raw B vs Env_S
    f, coh_self_p = dsp.coherence(ys_win, env_p_win, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    f, coh_self_s = dsp.coherence(ys_win, env_s_win, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, nfft=nfft)

    f_khz = f / 1000.0
    phase_ps_deg = np.degrees(np.angle(pxy_ps))

    n_seg = int(np.floor((len(ys_win) - nperseg) / (nperseg - noverlap))) + 1
    sig_floor = 1.0 - (0.05) ** (1.0 / max(1, n_seg - 1))

    # Normalize PSDs within the 0.5 - 5.0 kHz modulation frequency band
    m_lf = (f_khz >= 0.5) & (f_khz <= 5.0)
    norm_p_mp1 = p_mp1 / np.max(p_mp1[m_lf])
    norm_p_ece = p_ece / np.max(p_ece[m_lf])
    norm_p_nave = p_nave / np.max(p_nave[m_lf])
    norm_p_ha = p_ha / np.max(p_ha[m_lf])

    # Plot formatting exactly matching mhd_self_coupling_coherence.py
    plt.rcdefaults()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9.5, 9.6), sharex=True, facecolor="white")
    fig.suptitle(
        f"Heliotron J #{shot} — Low-Frequency Modulator ($1-3$ kHz) & Dual-Mode Envelope Coherence\n"
        f"(Primary Mode $f_0 \\approx 89.0$ kHz, Secondary Mode $f_s \\approx 42.0$ kHz, Active Window: {t_start:.1f} – {t_end:.1f} ms)",
        fontsize=12, fontweight="bold", y=0.98
    )

    # -------------------------------------------------------------
    # Panel 1: Modulator Identification across Diagnostics
    # -------------------------------------------------------------
    ax1.plot(f_khz[m_lf], norm_p_mp1[m_lf], color="tab:blue", linestyle="-", lw=1.7, label=r"Mirnov MP1 ($\tilde{B}_\theta$)")
    ax1.plot(f_khz[m_lf], norm_p_ece[m_lf], color="tab:red", linestyle="--", lw=1.7, label=r"Core $T_e$ Fluctuation (ECE13FAST)")
    ax1.plot(f_khz[m_lf], norm_p_nave[m_lf], color="tab:green", linestyle="-.", lw=1.5, label=r"Line Density $\tilde{n}_e$ (Interferometer)")
    ax1.plot(f_khz[m_lf], norm_p_ha[m_lf], color="tab:purple", linestyle=":", lw=1.6, label=r"Edge Emission (HAFAST7.5)")

    ax1.axvspan(0.8, 2.5, color="gray", alpha=0.08, label="Modulation band ($0.8 - 2.5$ kHz)")
    ax1.axvline(1.22, color="tab:blue", ls="--", lw=1.0, alpha=0.6)
    ax1.axvline(1.95, color="tab:purple", ls="--", lw=1.0, alpha=0.6)

    idx_mp_122 = np.argmin(np.abs(f_khz - 1.22))
    idx_mp_195 = np.argmin(np.abs(f_khz - 1.95))
    ax1.scatter([1.22], [norm_p_mp1[idx_mp_122]], color="tab:blue", s=50, zorder=5, edgecolors="white", lw=1.0)
    ax1.annotate(
        "1.22 kHz", xy=(1.22, norm_p_mp1[idx_mp_122]), xytext=(1.22, norm_p_mp1[idx_mp_122] + 0.16),
        ha="center", fontsize=8.5, fontweight="bold", color="tab:blue",
        arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.2)
    )

    ax1.scatter([1.95], [norm_p_mp1[idx_mp_195]], color="tab:purple", s=50, zorder=5, edgecolors="white", lw=1.0)
    ax1.annotate(
        "1.95 kHz", xy=(1.95, norm_p_mp1[idx_mp_195]), xytext=(1.95, 1.08),
        ha="center", fontsize=8.5, fontweight="bold", color="tab:purple",
        arrowprops=dict(arrowstyle="->", color="tab:purple", lw=1.2)
    )

    ax1.set_xlim(0.0, 5.0)
    ax1.set_ylim(0.0, 1.25)
    ax1.set_ylabel("Normalized PSD", fontsize=10.5)
    ax1.set_title(r"(a) Low-Frequency Modulator Signature Across Core & Edge Diagnostics", fontsize=11, fontweight="bold", pad=6)
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.95)

    # -------------------------------------------------------------
    # Panel 2: Cross-Envelope & Self-Coupling Coherence
    # -------------------------------------------------------------
    ax2.plot(
        f_khz, coh_ps,
        color="crimson", linestyle="-", lw=2.0,
        label=r"Envelope Cross-Coherence $\gamma^2(A_\mathrm{prim}, A_\mathrm{sec})$"
    )
    ax2.plot(
        f_khz, coh_self_p,
        color="tab:blue", linestyle="--", lw=1.6,
        label=r"Primary Self-Coherence $\gamma^2(\tilde{B}_\mathrm{MP1}, A_\mathrm{prim})$"
    )
    ax2.plot(
        f_khz, coh_self_s,
        color="tab:green", linestyle="-.", lw=1.6,
        label=r"Secondary Self-Coherence $\gamma^2(\tilde{B}_\mathrm{MP1}, A_\mathrm{sec})$"
    )

    ax2.axhline(sig_floor, color="dimgray", ls=":", lw=1.4, label=rf"95% Confidence Noise Floor ($\gamma^2 = {sig_floor:.2f}$)")
    ax2.text(4.92, sig_floor + 0.015, f"95% Floor ({sig_floor:.2f})", color="dimgray", fontsize=8.5, fontweight="bold", ha="right", va="bottom")

    ax2.axvspan(0.8, 2.5, color="gray", alpha=0.08)
    ax2.axvline(1.22, color="crimson", ls="--", lw=1.0, alpha=0.6)

    # Highlight 1.22 kHz peak
    idx_122 = np.argmin(np.abs(f_khz - 1.22))
    c_122 = coh_ps[idx_122]
    ax2.scatter([1.22], [c_122], color="crimson", s=50, zorder=5, edgecolors="white", lw=1.0)
    ax2.annotate(
        f"1.22 kHz ($\\gamma^2 = {c_122:.2f}$)", xy=(1.22, c_122), xytext=(1.22, c_122 + 0.12),
        ha="center", fontsize=8.5, fontweight="bold", color="crimson",
        arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2)
    )

    ax2.set_xlim(0.0, 5.0)
    ax2.set_ylim(0.0, 0.75)
    ax2.set_ylabel(r"Coherence $\gamma^2$", fontsize=10.5)
    ax2.set_title(r"(b) Dual-Mode Cross-Envelope Coherence & Self-Coupling", fontsize=11, fontweight="bold", pad=6)
    ax2.grid(True, ls=":", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.95)

    # -------------------------------------------------------------
    # Panel 3: Cross-Spectral Phase between Envelopes
    # -------------------------------------------------------------
    ax3.plot(
        f_khz, phase_ps_deg,
        color="crimson", linestyle="-", lw=1.8,
        label=r"Cross-Phase $\Delta\phi(A_\mathrm{prim}, A_\mathrm{sec})$"
    )

    ax3.axhline(0, color="dimgray", ls=":", lw=1.0)
    ax3.axvspan(0.8, 2.5, color="gray", alpha=0.08)
    ax3.axvline(1.22, color="crimson", ls="--", lw=1.0, alpha=0.6)

    phi_122 = phase_ps_deg[idx_122]
    ax3.scatter([1.22], [phi_122], color="crimson", s=50, zorder=5, edgecolors="white", lw=1.0)
    ax3.annotate(
        f"$\\Delta\\phi = {phi_122:+.1f}^\\circ$ (In-Phase Locking)", xy=(1.22, phi_122), xytext=(1.45, 32.0),
        ha="left", fontsize=8.5, fontweight="bold", color="crimson",
        arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2)
    )

    ax3.set_xlim(0.0, 5.0)
    ax3.set_ylim(-185.0, 185.0)
    ax3.set_yticks([-180, -90, 0, 90, 180])
    ax3.set_xlabel("Modulation Frequency (kHz)", fontsize=10.5)
    ax3.set_ylabel("Phase (deg)", fontsize=10.5)
    ax3.set_title(r"(c) Cross-Spectral Phase $\Delta\phi(A_\mathrm{prim}, A_\mathrm{sec})$", fontsize=11, fontweight="bold", pad=6)
    ax3.grid(True, ls=":", alpha=0.4)
    ax3.legend(loc="lower right", fontsize=8.5, framealpha=0.95)

    plt.tight_layout()
    out_png = out_dir / f"mhd_modulator_dual_mode_coherence_shot_{shot}.png"
    plt.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Figure saved to: '{out_png}'")

    return str(out_png)


if __name__ == "__main__":
    generate_modulator_dual_mode_plot()
