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

# Active window (245 - 308 ms)
t_start, t_end = 245.0, 308.0
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
    
    # Centered circulation
    xc = ip_sm - np.mean(ip_sm)
    yc = env_sm - np.mean(env_sm)
    circ = 0.5 * np.sum(xc[:-1] * np.diff(yc) - yc[:-1] * np.diff(xc))
    dir_label = "Counter-Clockwise (CCW)" if circ > 0 else "Clockwise (CW)"
    return lags[m_lag], corr[m_lag], lags[idx_pk], corr[idx_pk], circ, dir_label

lags_p, corr_p, pk_lag_p, pk_r_p, circ_p, dir_p = calc_cross_stats(env_p_sm, ip_sm)
lags_s, corr_s, pk_lag_s, pk_r_s, circ_s, dir_s = calc_cross_stats(env_s_sm, ip_sm)

def add_directional_arrow(ax, x, y, t_target, step=600, color="crimson", lw=2.0, mutation_scale=15):
    i = np.argmin(np.abs(t_w - t_target))
    x0, y0 = x[i], y[i]
    x1, y1 = x[i + step], y[i + step]
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=mutation_scale)
    )

# ----------------------------------------------------------------------------------------
# 5-PANEL MASTER FIGURE
# ----------------------------------------------------------------------------------------
plt.rcdefaults()
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9.5,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "lines.linewidth": 1.6,
})

fig = plt.figure(figsize=(13.5, 12.5), facecolor="white")
fig.suptitle(
    f"Heliotron J #{shot} — Mode Envelopes ($I_\\mathrm{{env}}$) vs. Plasma Current ($I_p$) Hysteresis & Dynamics\n"
    f"Primary Mode ($m=3$, $80-120$ kHz) vs. Secondary Mode ($m=2$, $40-80$ kHz), Window: {t_start:.1f} – {t_end:.1f} ms",
    fontsize=12.5, fontweight="bold", y=0.98
)

gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 0.95, 1.05], hspace=0.34, wspace=0.25,
                      top=0.92, bottom=0.06, left=0.08, right=0.94)

# -------------------------------------------------------------
# Row 0: Full-width Time Evolution (Twin-axis)
# -------------------------------------------------------------
ax_time = fig.add_subplot(gs[0, :])
c_m1 = "#d62728"
c_m2 = "#1f77b4"
c_ip = "#2ca02c"

ax_time.plot(t_w, env_p_sm, color=c_m1, lw=2.2, label=r"Primary Mode $I_\mathrm{env}$ ($m=3$, $80-120$ kHz)")
ax_time.plot(t_w, env_s_sm, color=c_m2, lw=2.0, label=r"Secondary Mode $I_\mathrm{env}$ ($m=2$, $40-80$ kHz)")
ax_time.set_ylabel(r"Mode Amplitude $I_\mathrm{env}$ (V)", fontsize=11, fontweight="bold")
ax_time.set_xlabel("Time (ms)", fontsize=10.5, fontweight="bold")
ax_time.set_xlim(t_start, t_end)
ax_time.set_ylim(-0.01, 0.38)
ax_time.grid(True, ls=":", alpha=0.45)

# Twin axis for Ip
ax_ip_twin = ax_time.twinx()
ax_ip_twin.plot(t_w, ip_sm, color=c_ip, lw=2.2, ls="--", label=r"Plasma Current $I_p(t)$ (kA)")
ax_ip_twin.set_ylabel(r"Plasma Current $I_p$ (kA)", color=c_ip, fontsize=11, fontweight="bold")
ax_ip_twin.tick_params(axis="y", labelcolor=c_ip)
ax_ip_twin.set_ylim(0.0, 1.55)

# Annotate ECH cut-off
ax_time.axvline(291.3, color="#e67e22", ls="--", lw=1.4)
ax_time.text(291.0, 0.34, "ECH Cut-off (291.3 ms)", rotation=90, va="top", ha="right",
             fontsize=8.5, fontweight="bold", color="#b95e00")

# Annotations pointing to modes
ax_time.annotate(r"$\leftarrow$ Mode 1 ($m=3$)", xy=(291.5, 0.315), xytext=(293.0, 0.33),
                 fontsize=9.5, fontweight="bold", color=c_m1,
                 arrowprops=dict(arrowstyle="->", color=c_m1, lw=1.2))

ax_time.annotate(r"$\leftarrow$ Mode 2 ($m=2$)", xy=(284.0, 0.18), xytext=(285.5, 0.22),
                 fontsize=9.5, fontweight="bold", color=c_m2,
                 arrowprops=dict(arrowstyle="->", color=c_m2, lw=1.2))

ax_time.set_title(r"(a) Time Evolution: Mode Envelopes $I_\mathrm{env}(t)$ vs. Plasma Current $I_p(t)$",
                  fontsize=11, fontweight="bold", pad=8)

lines1, labels1 = ax_time.get_legend_handles_labels()
lines2, labels2 = ax_ip_twin.get_legend_handles_labels()
ax_time.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.92, edgecolor="#cccccc")

# -------------------------------------------------------------
# Row 1: Cross-correlations
# -------------------------------------------------------------
ax_corr_p = fig.add_subplot(gs[1, 0])
ax_corr_p.plot(lags_p, corr_p, color="tab:purple", lw=1.8, label=r"$R(\tau): A_\mathrm{prim}$ vs. $I_p$")
ax_corr_p.axvline(0, color="dimgray", ls=":", lw=1.0)
ax_corr_p.axvline(pk_lag_p, color="crimson", ls="--", lw=1.2)
ax_corr_p.plot([pk_lag_p], [pk_r_p], "o", color="crimson", ms=5)
ax_corr_p.annotate(
    f"Peak $r = {pk_r_p:.2f}$\n$\\tau = {pk_lag_p:+.2f}$ ms\n($I_p$ leads $A$)",
    xy=(pk_lag_p, pk_r_p), xytext=(pk_lag_p + 1.5, pk_r_p - 0.12),
    fontsize=8.5, fontweight="bold", color="darkred",
    arrowprops=dict(arrowstyle="->", color="darkred", lw=1.0),
    bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85)
)
ax_corr_p.set_xlim(-15, 15)
ax_corr_p.set_ylim(-0.25, 1.0)
ax_corr_p.set_xlabel(r"Time Lag $\tau$ (ms)", fontsize=10)
ax_corr_p.set_ylabel("Cross-Correlation", fontsize=10)
ax_corr_p.set_title(r"(b) Primary Mode ($80-120$ kHz): Lagged Cross-Correlation", fontsize=10.5, fontweight="bold")
ax_corr_p.grid(True, ls=":", alpha=0.4)
ax_corr_p.legend(loc="lower left", fontsize=8.5, framealpha=0.9)

ax_corr_s = fig.add_subplot(gs[1, 1])
ax_corr_s.plot(lags_s, corr_s, color="tab:purple", lw=1.8, label=r"$R(\tau): A_\mathrm{sec}$ vs. $I_p$")
ax_corr_s.axvline(0, color="dimgray", ls=":", lw=1.0)
ax_corr_s.axvline(pk_lag_s, color="crimson", ls="--", lw=1.2)
ax_corr_s.plot([pk_lag_s], [pk_r_s], "o", color="crimson", ms=5)
ax_corr_s.annotate(
    f"Peak $r = {pk_r_s:.2f}$\n$\\tau = {pk_lag_s:+.2f}$ ms\n(In-phase)",
    xy=(pk_lag_s, pk_r_s), xytext=(pk_lag_s + 1.5, pk_r_s - 0.12),
    fontsize=8.5, fontweight="bold", color="darkred",
    arrowprops=dict(arrowstyle="->", color="darkred", lw=1.0),
    bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85)
)
ax_corr_s.set_xlim(-15, 15)
ax_corr_s.set_ylim(-0.25, 1.0)
ax_corr_s.set_xlabel(r"Time Lag $\tau$ (ms)", fontsize=10)
ax_corr_s.set_ylabel("Cross-Correlation", fontsize=10)
ax_corr_s.set_title(r"(c) Secondary Mode ($40-80$ kHz): Lagged Cross-Correlation", fontsize=10.5, fontweight="bold")
ax_corr_s.grid(True, ls=":", alpha=0.4)
ax_corr_s.legend(loc="lower left", fontsize=8.5, framealpha=0.9)

# -------------------------------------------------------------
# Row 2: Hysteresis loops
# -------------------------------------------------------------
skip = 2
ax_hyst_p = fig.add_subplot(gs[2, 0])
sc_c = ax_hyst_p.scatter(ip_sm[::skip], env_p_sm[::skip], c=t_w[::skip], cmap="plasma", s=9, alpha=0.85, edgecolors="none")
cb_c = plt.colorbar(sc_c, ax=ax_hyst_p, pad=0.02)
cb_c.set_label("Time (ms)", fontsize=9)

add_directional_arrow(ax_hyst_p, ip_sm, env_p_sm, 265.0, step=600, color="crimson")
add_directional_arrow(ax_hyst_p, ip_sm, env_p_sm, 294.0, step=600, color="crimson")

ax_hyst_p.text(
    0.05, 0.90, f"Trajectory: {dir_p}\n(Circulation $\\Gamma > 0$)",
    transform=ax_hyst_p.transAxes, fontsize=8.5, fontweight="bold", color="crimson",
    bbox=dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor="crimson", alpha=0.85)
)
ax_hyst_p.set_xlabel(r"Plasma Current $I_p$ (kA)", fontsize=10)
ax_hyst_p.set_ylabel(r"Primary Envelope $A_\mathrm{prim}$ (V)", fontsize=10)
ax_hyst_p.set_title(r"(d) Primary Mode Hysteresis: $I_p$ vs. $A_\mathrm{prim}$", fontsize=10.5, fontweight="bold")
ax_hyst_p.grid(True, ls=":", alpha=0.4)

ax_hyst_s = fig.add_subplot(gs[2, 1])
sc_d = ax_hyst_s.scatter(ip_sm[::skip], env_s_sm[::skip], c=t_w[::skip], cmap="plasma", s=9, alpha=0.85, edgecolors="none")
cb_d = plt.colorbar(sc_d, ax=ax_hyst_s, pad=0.02)
cb_d.set_label("Time (ms)", fontsize=9)

add_directional_arrow(ax_hyst_s, ip_sm, env_s_sm, 265.0, step=600, color="crimson")
add_directional_arrow(ax_hyst_s, ip_sm, env_s_sm, 292.0, step=600, color="crimson")

ax_hyst_s.text(
    0.05, 0.90, f"Trajectory: {dir_s}\n(Circulation $\\Gamma > 0$)",
    transform=ax_hyst_s.transAxes, fontsize=8.5, fontweight="bold", color="crimson",
    bbox=dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor="crimson", alpha=0.85)
)
ax_hyst_s.set_xlabel(r"Plasma Current $I_p$ (kA)", fontsize=10)
ax_hyst_s.set_ylabel(r"Secondary Envelope $A_\mathrm{sec}$ (V)", fontsize=10)
ax_hyst_s.set_title(r"(e) Secondary Mode Hysteresis: $I_p$ vs. $A_\mathrm{sec}$", fontsize=10.5, fontweight="bold")
ax_hyst_s.grid(True, ls=":", alpha=0.4)

out_both = root_dir / "mhd_hysteresis_ip_vs_envelope_dual_modes_shot_88653.png"
out_tmp = root_dir / "mhd_hysteresis_ip_vs_envelope_dual_modes_shot_88653.tmp.png"
plt.savefig(str(out_tmp), dpi=300, bbox_inches="tight")
plt.close(fig)
shutil.move(str(out_tmp), str(out_both))
print(f"Saved 5-panel integrated hysteresis plot to {out_both}")

