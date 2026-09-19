import sys
from pathlib import Path
import shutil
import numpy as np
import scipy.signal as dsp
import matplotlib.pyplot as plt

root_dir = Path(r"c:\TFG")
for p_add in [root_dir / "jpack", root_dir / "analysis", root_dir / "analysis" / "common"]:
    if str(p_add) not in sys.path:
        sys.path.append(str(p_add))

import turnelib as TE
from mhd_common import extract_instantaneous_frequency

shot = 88653
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
env_p, _, _, _ = extract_instantaneous_frequency(ys_mp1, fs, 80000.0, 120000.0, 4, 325)
env_s, _, _, _ = extract_instantaneous_frequency(ys_mp1, fs, 40000.0, 80000.0, 4, 325)

# Active window (250 - 305 ms)
t_start, t_end = 250.0, 305.0
idx = np.where((t_ms >= t_start) & (t_ms <= t_end))[0]
t_w = t_ms[idx]
ip_w = ip_interp[idx]
env_p_w = env_p[idx]
env_s_w = env_s[idx]

# Smooth for hysteresis trajectory (200 Hz lowpass)
b_lp, a_lp = dsp.butter(3, 200.0 / (fs / 2.0), btype="low")
ip_sm = dsp.filtfilt(b_lp, a_lp, ip_w)
env_p_sm = dsp.filtfilt(b_lp, a_lp, env_p_w)
env_s_sm = dsp.filtfilt(b_lp, a_lp, env_s_w)

def calc_cross_stats(env_sm, ip_sm):
    lags = dsp.correlation_lags(len(t_w), len(t_w), mode="same") * (dt * 1000.0)
    corr = dsp.correlate(env_sm - np.mean(env_sm), ip_sm - np.mean(ip_sm), mode="same")
    corr /= (np.std(env_sm) * np.std(ip_sm) * len(t_w))
    m_lag = (lags >= -15.0) & (lags <= 15.0)
    idx_pk = np.where(m_lag)[0][np.argmax(corr[m_lag])]
    
    # Centered circulation around loop centroid
    xc = ip_sm - np.mean(ip_sm)
    yc = env_sm - np.mean(env_sm)
    circ = 0.5 * np.sum(xc[:-1] * np.diff(yc) - yc[:-1] * np.diff(xc))
    dir_label = "Counter-Clockwise (CCW)" if circ > 0 else "Clockwise (CW)"
    return lags[m_lag], corr[m_lag], lags[idx_pk], corr[idx_pk], circ, dir_label

lags_p, corr_p, pk_lag_p, pk_r_p, circ_p, dir_p = calc_cross_stats(env_p_sm, ip_sm)
lags_s, corr_s, pk_lag_s, pk_r_s, circ_s, dir_s = calc_cross_stats(env_s_sm, ip_sm)

def add_directional_arrow(ax, x, y, t_target, step=600, color="crimson", lw=2.2, mutation_scale=16):
    i = np.argmin(np.abs(t_w - t_target))
    x0, y0 = x[i], y[i]
    x1, y1 = x[i + step], y[i + step]
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=mutation_scale)
    )

# ----------------------------------------------------------------------------------------
# GENERATE UNIFIED SIDE-BY-SIDE DUAL-MODE COMPARISON FIGURE
# ----------------------------------------------------------------------------------------
plt.rcdefaults()
fig_both, axs_both = plt.subplots(2, 2, figsize=(13.0, 9.0), facecolor="white")
fig_both.suptitle(
    f"Heliotron J #{shot} — Mode Envelope vs. Plasma Current ($I_p$) Hysteresis Comparison\n"
    f"Primary Mode ($80-120$ kHz) vs. Secondary Mode ($40-80$ kHz), Window: {t_start:.1f} – {t_end:.1f} ms",
    fontsize=12, fontweight="bold", y=0.98
)
plt.subplots_adjust(top=0.90, bottom=0.08, left=0.08, right=0.94, hspace=0.32, wspace=0.25)

# (a) Primary Lagged Cross-Correlation
axs_both[0, 0].plot(lags_p, corr_p, color="tab:purple", lw=1.8, label=r"$R(\tau): A_\mathrm{prim}$ vs. $I_p$")
axs_both[0, 0].axvline(0, color="dimgray", ls=":", lw=1.0)
axs_both[0, 0].axvline(pk_lag_p, color="crimson", ls="--", lw=1.2)
axs_both[0, 0].plot([pk_lag_p], [pk_r_p], "o", color="crimson", ms=5)
axs_both[0, 0].annotate(
    f"Peak $r = {pk_r_p:.2f}$\n$\\tau = {pk_lag_p:+.2f}$ ms\n($I_p$ leads $A$)",
    xy=(pk_lag_p, pk_r_p), xytext=(pk_lag_p + 1.5, pk_r_p - 0.12),
    fontsize=8.5, fontweight="bold", color="darkred",
    arrowprops=dict(arrowstyle="->", color="darkred", lw=1.0),
    bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85)
)
axs_both[0, 0].set_xlim(-15, 15)
axs_both[0, 0].set_ylim(-0.25, 1.0)
axs_both[0, 0].set_xlabel(r"Time Lag $\tau$ (ms)", fontsize=10)
axs_both[0, 0].set_ylabel("Cross-Correlation", fontsize=10)
axs_both[0, 0].set_title(r"(a) Primary Mode ($80-120$ kHz): Lagged Cross-Correlation", fontsize=10.5, fontweight="bold")
axs_both[0, 0].grid(True, ls=":", alpha=0.4)
axs_both[0, 0].legend(loc="lower left", fontsize=8.5, framealpha=0.9)

# (b) Secondary Lagged Cross-Correlation
axs_both[0, 1].plot(lags_s, corr_s, color="tab:purple", lw=1.8, label=r"$R(\tau): A_\mathrm{sec}$ vs. $I_p$")
axs_both[0, 1].axvline(0, color="dimgray", ls=":", lw=1.0)
axs_both[0, 1].axvline(pk_lag_s, color="crimson", ls="--", lw=1.2)
axs_both[0, 1].plot([pk_lag_s], [pk_r_s], "o", color="crimson", ms=5)
axs_both[0, 1].annotate(
    f"Peak $r = {pk_r_s:.2f}$\n$\\tau = {pk_lag_s:+.2f}$ ms\n(In-phase)",
    xy=(pk_lag_s, pk_r_s), xytext=(pk_lag_s + 1.5, pk_r_s - 0.12),
    fontsize=8.5, fontweight="bold", color="darkred",
    arrowprops=dict(arrowstyle="->", color="darkred", lw=1.0),
    bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85)
)
axs_both[0, 1].set_xlim(-15, 15)
axs_both[0, 1].set_ylim(-0.25, 1.0)
axs_both[0, 1].set_xlabel(r"Time Lag $\tau$ (ms)", fontsize=10)
axs_both[0, 1].set_ylabel("Cross-Correlation", fontsize=10)
axs_both[0, 1].set_title(r"(b) Secondary Mode ($40-80$ kHz): Lagged Cross-Correlation", fontsize=10.5, fontweight="bold")
axs_both[0, 1].grid(True, ls=":", alpha=0.4)
axs_both[0, 1].legend(loc="lower left", fontsize=8.5, framealpha=0.9)

# (c) Primary Hysteresis
skip = 2
sc_c = axs_both[1, 0].scatter(ip_sm[::skip], env_p_sm[::skip], c=t_w[::skip], cmap="plasma", s=9, alpha=0.85, edgecolors="none")
cb_c = plt.colorbar(sc_c, ax=axs_both[1, 0], pad=0.02)
cb_c.set_label("Time (ms)", fontsize=9)

# Two clear directional arrows: forward ramp (rising Ip) and return ramp (falling Ip)
add_directional_arrow(axs_both[1, 0], ip_sm, env_p_sm, 265.0, step=600, color="crimson")
add_directional_arrow(axs_both[1, 0], ip_sm, env_p_sm, 294.0, step=600, color="crimson")

axs_both[1, 0].text(
    0.05, 0.90, f"Trajectory: {dir_p}\n(Circulation $\\Gamma > 0$)",
    transform=axs_both[1, 0].transAxes, fontsize=8.5, fontweight="bold", color="crimson",
    bbox=dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor="crimson", alpha=0.85)
)
axs_both[1, 0].set_xlabel(r"Plasma Current $I_p$ (kA)", fontsize=10)
axs_both[1, 0].set_ylabel(r"Primary Envelope $A_\mathrm{prim}$ (V)", fontsize=10)
axs_both[1, 0].set_title(r"(c) Primary Mode Hysteresis: $I_p$ vs. $A_\mathrm{prim}$", fontsize=10.5, fontweight="bold")
axs_both[1, 0].grid(True, ls=":", alpha=0.4)

# (d) Secondary Hysteresis
sc_d = axs_both[1, 1].scatter(ip_sm[::skip], env_s_sm[::skip], c=t_w[::skip], cmap="plasma", s=9, alpha=0.85, edgecolors="none")
cb_d = plt.colorbar(sc_d, ax=axs_both[1, 1], pad=0.02)
cb_d.set_label("Time (ms)", fontsize=9)

# Two clear directional arrows:
# 1. Forward ramp (rising Ip, lower purple branch, t ~ 265 ms, pointing right-up)
add_directional_arrow(axs_both[1, 1], ip_sm, env_s_sm, 265.0, step=600, color="crimson")
# 2. Return ramp (falling Ip, upper orange branch, t ~ 292 ms, pointing down-left)
add_directional_arrow(axs_both[1, 1], ip_sm, env_s_sm, 292.0, step=600, color="crimson")

axs_both[1, 1].text(
    0.05, 0.90, f"Trajectory: {dir_s}\n(Circulation $\\Gamma > 0$)",
    transform=axs_both[1, 1].transAxes, fontsize=8.5, fontweight="bold", color="crimson",
    bbox=dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor="crimson", alpha=0.85)
)
axs_both[1, 1].set_xlabel(r"Plasma Current $I_p$ (kA)", fontsize=10)
axs_both[1, 1].set_ylabel(r"Secondary Envelope $A_\mathrm{sec}$ (V)", fontsize=10)
axs_both[1, 1].set_title(r"(d) Secondary Mode Hysteresis: $I_p$ vs. $A_\mathrm{sec}$", fontsize=10.5, fontweight="bold")
axs_both[1, 1].grid(True, ls=":", alpha=0.4)

out_both = root_dir / "mhd_hysteresis_ip_vs_envelope_dual_modes_shot_88653.png"
out_tmp = root_dir / "mhd_hysteresis_ip_vs_envelope_dual_modes_shot_88653.tmp.png"
plt.savefig(str(out_tmp), dpi=300, bbox_inches="tight")
plt.close(fig_both)
shutil.move(str(out_tmp), str(out_both))
print(f"Saved dual-mode comparison hysteresis plot to {out_both}")
