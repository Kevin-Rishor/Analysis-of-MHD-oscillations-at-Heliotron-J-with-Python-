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


def generate_hysteresis_plot(shot=88653, out_dir=None):
    if out_dir is None:
        out_dir = root_dir
    else:
        out_dir = Path(out_dir)

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

    # Extract Primary Mode Envelope (80 - 120 kHz, f0 ~ 89 kHz)
    envelope, _, _, _ = extract_instantaneous_frequency(ys_mp1, fs, 80000.0, 120000.0, 4, 325)

    # Active window
    t_start, t_end = 250.0, 305.0
    idx = np.where((t_ms >= t_start) & (t_ms <= t_end))[0]
    t_w = t_ms[idx]
    ip_w = ip_interp[idx]
    env_w = envelope[idx]

    # Smooth for phase-space trajectory
    b_lp, a_lp = dsp.butter(3, 200.0 / (fs / 2.0), btype="low")
    env_smooth = dsp.filtfilt(b_lp, a_lp, env_w)
    ip_smooth = dsp.filtfilt(b_lp, a_lp, ip_w)

    # Lagged cross-correlation
    lags = dsp.correlation_lags(len(t_w), len(t_w), mode="same") * (dt * 1000.0)
    corr = dsp.correlate(env_smooth - np.mean(env_smooth), ip_smooth - np.mean(ip_smooth), mode="same")
    corr /= (np.std(env_smooth) * np.std(ip_smooth) * len(t_w))
    idx_pk = np.argmax(corr)
    peak_lag_ms = lags[idx_pk]
    peak_corr = corr[idx_pk]

    # Signed circulation (Green's theorem): Area = 0.5 * sum(x_i * dy - y_i * dx)
    circulation = np.sum(ip_smooth[:-1] * np.diff(env_smooth) - env_smooth[:-1] * np.diff(ip_smooth))
    dir_label = "Counter-Clockwise (CCW)" if circulation > 0 else "Clockwise (CW)"

    # Plotting layout matching mhd_plot_b_tilde_vs_envelope.py exactly
    plt.rcdefaults()
    fig = plt.figure(figsize=(13.5, 9.0), facecolor="white")
    fig.suptitle(
        f"Heliotron J #{shot} — Plasma Current ($I_p$) vs. Mode Envelope Hysteresis Dynamics\n"
        f"Mirnov Coil MP1 ($f_0 \\approx 89.0$ kHz, Window: {t_start:.1f} – {t_end:.1f} ms)",
        fontsize=12.5, fontweight="bold", y=0.97
    )

    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28, top=0.90, bottom=0.08, left=0.08, right=0.95)
    ax_time = fig.add_subplot(gs[0, :])
    ax_corr = fig.add_subplot(gs[1, 0])
    ax_scat = fig.add_subplot(gs[1, 1])

    # -------------------------------------------------------------
    # (a) Time Evolution
    # -------------------------------------------------------------
    c_ip = "tab:blue"
    c_env = "crimson"

    ax_time.plot(t_w, ip_w, color=c_ip, lw=1.6, label=r"Plasma Current $I_p$ (Ip15)")
    ax_time.set_xlim(t_start, t_end)
    ax_time.set_xlabel("Time (ms)", fontsize=10)
    ax_time.set_ylabel(r"Plasma Current $I_p$ (kA)", color=c_ip, fontsize=10)
    ax_time.tick_params(axis="y", labelcolor=c_ip)
    ax_time.grid(True, ls=":", alpha=0.4)

    ax_env = ax_time.twinx()
    ax_env.plot(t_w, env_w, color=c_env, lw=0.8, alpha=0.4, label=r"Fast Envelope $A(t)$")
    ax_env.plot(t_w, env_smooth, color=c_env, lw=2.0, label=r"Smoothed Envelope $\langle A(t) \rangle$")
    ax_env.set_ylabel(r"Mode Envelope $A(t)$ (V)", color=c_env, fontsize=10)
    ax_env.tick_params(axis="y", labelcolor=c_env)

    # ECH turn-off vertical line
    ax_time.axvline(291.3, color="dimgray", ls="--", lw=1.2)
    ax_time.text(291.0, np.min(ip_w) + 0.05, "ECH Off (291.3 ms)", rotation=90, va="bottom", ha="right", fontsize=8.5, color="dimgray")

    ax_time.set_title(r"(a) Time Evolution: Plasma Current $I_p(t)$ vs. Hilbert Envelope $A(t)$", fontsize=11, fontweight="bold", pad=8)

    # Combined clean legend
    l1, b1 = ax_time.get_legend_handles_labels()
    l2, b2 = ax_env.get_legend_handles_labels()
    ax_time.legend(l1 + l2, b1 + b2, loc="upper left", fontsize=8.5, framealpha=0.9)

    # -------------------------------------------------------------
    # (b) Lagged Cross-Correlation
    # -------------------------------------------------------------
    m_lag = (lags >= -15.0) & (lags <= 15.0)
    ax_corr.plot(lags[m_lag], corr[m_lag], color="tab:purple", lw=1.8, label=r"Correlation $R(\tau)$")
    ax_corr.axvline(0, color="dimgray", ls=":", lw=1.2, label=r"Zero Lag ($\tau = 0$)")
    ax_corr.axvline(peak_lag_ms, color="crimson", ls="--", lw=1.2, label=rf"Peak $\tau = +{peak_lag_ms:.2f}$ ms")

    ax_corr.plot([peak_lag_ms], [peak_corr], "o", color="crimson", ms=6)
    ax_corr.annotate(
        f"Peak $\\tau = +{peak_lag_ms:.2f}$ ms\n($r = {peak_corr:.2f}$)",
        xy=(peak_lag_ms, peak_corr), xytext=(peak_lag_ms + 2.0, peak_corr - 0.12),
        fontsize=8.5, fontweight="bold", color="darkred",
        arrowprops=dict(arrowstyle="->", color="darkred", lw=1.0)
    )

    ax_corr.set_xlim(-15.0, 15.0)
    ax_corr.set_ylim(-0.3, 1.0)
    ax_corr.set_xlabel(r"Time Lag $\tau$ (ms)", fontsize=10)
    ax_corr.set_ylabel("Normalized Cross-Correlation", fontsize=10)
    ax_corr.set_title(r"(b) Lagged Cross-Correlation: $A(t)$ vs. $I_p(t)$", fontsize=11, fontweight="bold", pad=8)
    ax_corr.grid(True, ls=":", alpha=0.4)
    ax_corr.legend(loc="lower left", fontsize=8.5, framealpha=0.9)

    # -------------------------------------------------------------
    # (c) Hysteresis Phase Space
    # -------------------------------------------------------------
    skip = 2
    sc = ax_scat.scatter(
        ip_smooth[::skip], env_smooth[::skip],
        c=t_w[::skip], cmap="viridis", s=6, alpha=0.8, rasterized=True
    )
    cbar = plt.colorbar(sc, ax=ax_scat, fraction=0.046, pad=0.04)
    cbar.set_label("Time (ms)", fontsize=9.5)

    ax_scat.plot(ip_smooth, env_smooth, color="gray", lw=1.0, alpha=0.4)

    # 2 subtle directional arrows on the trajectory
    # Arrow on ramp-up (~265 ms)
    i1 = np.argmin(np.abs(t_w - 265.0))
    ax_scat.annotate(
        "", xy=(ip_smooth[i1 + 250], env_smooth[i1 + 250]), xytext=(ip_smooth[i1], env_smooth[i1]),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5)
    )

    # Arrow on return path (~296 ms)
    i2 = np.argmin(np.abs(t_w - 296.0))
    ax_scat.annotate(
        "", xy=(ip_smooth[i2 + 250], env_smooth[i2 + 250]), xytext=(ip_smooth[i2], env_smooth[i2]),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5)
    )

    ax_scat.set_xlim(0.40, 1.45)
    ax_scat.set_ylim(0.0, 0.45)
    ax_scat.set_xlabel(r"Plasma Current $I_p$ (kA)", fontsize=10)
    ax_scat.set_ylabel(r"Hilbert Envelope $A(t)$ (V)", fontsize=10)
    ax_scat.set_title(rf"(c) $I_p$ vs. Envelope $A(t)$ Phase Space [{dir_label}]", fontsize=11, fontweight="bold", pad=8)
    ax_scat.grid(True, ls=":", alpha=0.4)

    out_png = out_dir / f"mhd_hysteresis_ip_vs_envelope_shot_{shot}.png"
    plt.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Figure saved to: '{out_png}'")

    return str(out_png)


if __name__ == "__main__":
    generate_hysteresis_plot()
