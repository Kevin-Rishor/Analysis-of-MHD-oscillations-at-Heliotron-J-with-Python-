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

shot = 88653
data_dir = root_dir / "data" / f"hj{shot}"
edf = TE.edf()

# Load Mirnov MP1
dat_mp1 = edf.load(str(data_dir / f"MP1@{shot}.edf"))
t_raw = dat_mp1[:, 0]
t_ms = t_raw if edf.DimUnit[0] == "ms" else t_raw * 1000.0
dt = (t_ms[1] - t_ms[0]) / 1000.0
fs = 1.0 / dt
ys_b = dat_mp1[:, 1]

# Primary mode envelope (80 - 120 kHz, f0 ~ 89 kHz)
env_p, _, b_p, _ = extract_instantaneous_frequency(ys_b, fs, 80000.0, 120000.0, 4, 325)

# Windows
t_base_start, t_base_end = 150.0, 175.0
t_act_start, t_act_end = 259.1, 275.0

idx_base = np.where((t_ms >= t_base_start) & (t_ms <= t_base_end))[0]
idx_act = np.where((t_ms >= t_act_start) & (t_ms <= t_act_end))[0]

# Load HAFAST7.5
dat_ha = edf.load(str(data_dir / f"HAFAST7.5@{shot}.edf"))
t_ha = dat_ha[:, 0]
t_ha_ms = t_ha if edf.DimUnit[0] == "ms" else t_ha * 1000.0
ys_ha = np.interp(t_ms, t_ha_ms, dat_ha[:, 1])

# Active window signals
t_win = t_ms[idx_act]
b_raw_act = ys_b[idx_act]
b_p_act = b_p[idx_act]
env_act = env_p[idx_act]
ha_act = ys_ha[idx_act]

# 5 kHz lowpass filter (Butterworth order 4)
sos_5k = dsp.butter(4, 5000.0, btype='lowpass', fs=fs, output='sos')
b_raw_5k = dsp.sosfiltfilt(sos_5k, b_raw_act)
b_p_5k = dsp.sosfiltfilt(sos_5k, b_p_act)
env_5k = dsp.sosfiltfilt(sos_5k, env_act)
ha_5k = dsp.sosfiltfilt(sos_5k, ha_act)

# SNR computations
# 1. Bandpassed mode
rms_bp_base = np.sqrt(np.mean(b_p[idx_base]**2))
rms_bp_act = np.sqrt(np.mean(b_p_act**2))
snr_bp_pwr = (rms_bp_act**2) / (rms_bp_base**2)
snr_bp_db = 10.0 * np.log10(snr_bp_pwr)

# 2. Envelope
rms_env_base = np.sqrt(np.mean(env_p[idx_base]**2))
rms_env_act = np.sqrt(np.mean(env_act**2))
snr_env_pwr = (rms_env_act**2) / (rms_env_base**2)
snr_env_db = 10.0 * np.log10(snr_env_pwr)

# Cross-correlations (search window: +/- 5 ms)
max_lag_samp = int(0.005 * fs)
def calc_xcorr(x, y, max_samps):
    xc = x - np.mean(x)
    yc = y - np.mean(y)
    norm = np.sqrt(np.sum(xc**2) * np.sum(yc**2))
    corr = np.correlate(xc, yc, mode='full') / norm
    mid = len(x) - 1
    lags = np.arange(-max_samps, max_samps + 1)
    return corr[mid - max_samps : mid + max_samps + 1], lags

corr_env_ha, lags = calc_xcorr(env_5k, ha_5k, max_lag_samp)
corr_braw_ha, _ = calc_xcorr(b_raw_5k, ha_5k, max_lag_samp)
corr_bp_ha, _ = calc_xcorr(b_p_act, ha_act, max_lag_samp)
lags_ms = lags * dt * 1000.0

idx_peak = np.argmax(corr_env_ha)
peak_r = corr_env_ha[idx_peak]
peak_lag = lags_ms[idx_peak]

# Plot formatting matching authentic thesis figures (e.g. mhd_plot_hysteresis_ip_vs_envelope.py)
plt.rcdefaults()
fig = plt.figure(figsize=(12.5, 8.2), facecolor="white")
fig.suptitle(
    f"Heliotron J #{shot} — Mode Signal-to-Noise Ratio & Edge $H_\\alpha$ Correlation Dynamics\n"
    f"Mirnov Coil MP1 ($f_0 \\approx 89.0$ kHz) vs. Fast Edge Emission (HAFAST7.5, Bandwidth $\\leq 5.0$ kHz)",
    fontsize=12, fontweight="bold", y=0.97
)

gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25, top=0.89, bottom=0.09, left=0.08, right=0.94)
ax_time = fig.add_subplot(gs[0, :])
ax_snr = fig.add_subplot(gs[1, 0])
ax_corr = fig.add_subplot(gs[1, 1])

# -------------------------------------------------------------
# Panel (a): Time Evolution
# -------------------------------------------------------------
ax_time.plot(t_win, b_p_act, color="lightsteelblue", lw=0.7, alpha=0.6, label=r"Mode Carrier $\tilde{B}(t)$ ($80-120$ kHz)")
ax_time.plot(t_win, env_act, color="crimson", lw=1.6, label=r"Mode Envelope $A(t)$")
ax_time.set_xlim(t_act_start, t_act_end)
ax_time.set_xlabel("Time (ms)", fontsize=10)
ax_time.set_ylabel(r"Mirnov Amplitude $\tilde{B},\, A(t)$ (V)", fontsize=10)
ax_time.grid(True, ls=":", alpha=0.4)

ax_ha = ax_time.twinx()
ax_ha.plot(t_win, ha_5k, color="tab:green", lw=1.4, ls="--", label=r"Edge $I_{\mathrm{H}\alpha}$ (HAFAST7.5, $\leq 5$ kHz)")
ax_ha.set_ylabel(r"Edge $I_{\mathrm{H}\alpha}$ Emission (V)", color="tab:green", fontsize=10)
ax_ha.tick_params(axis="y", labelcolor="tab:green")

ax_time.set_title(r"(a) Time Evolution: Mode Envelope $A(t)$ vs. Edge $I_{\mathrm{H}\alpha}$ Emission", fontsize=11, fontweight="bold", pad=8)

# Combined clean legend in upper left
l1, b1 = ax_time.get_legend_handles_labels()
l2, b2 = ax_ha.get_legend_handles_labels()
ax_time.legend(l1 + l2, b1 + b2, loc="upper left", fontsize=8.5, framealpha=0.9)

# -------------------------------------------------------------
# Panel (b): SNR & Power Spectral Density
# -------------------------------------------------------------
f_psd, p_base = dsp.welch(ys_b[idx_base] - np.mean(ys_b[idx_base]), fs=fs, window="hann", nperseg=4096, noverlap=2048, nfft=8192)
_, p_act = dsp.welch(b_raw_act - np.mean(b_raw_act), fs=fs, window="hann", nperseg=4096, noverlap=2048, nfft=8192)

f_psd_khz = f_psd / 1000.0
m_spec = (f_psd_khz >= 0.0) & (f_psd_khz <= 150.0)

ax_snr.semilogy(f_psd_khz[m_spec], p_act[m_spec], color="tab:blue", lw=1.3, label="Active Mode Phase (259.1 – 275.0 ms)")
ax_snr.semilogy(f_psd_khz[m_spec], p_base[m_spec], color="gray", lw=1.1, ls="--", label="Baseline Noise (150.0 – 175.0 ms)")
ax_snr.axvspan(80.0, 120.0, color="crimson", alpha=0.10, label=r"Mode Band ($80-120$ kHz)")

ax_snr.set_xlim(0, 150)
ax_snr.set_ylim(1e-11, 2e-5)
ax_snr.set_xlabel("Frequency (kHz)", fontsize=10)
ax_snr.set_ylabel(r"PSD $(\mathrm{V}^2/\mathrm{Hz})$", fontsize=10)
ax_snr.set_title(r"(b) Mode & Envelope Signal-to-Noise Ratio (SNR)", fontsize=11, fontweight="bold", pad=8)
ax_snr.grid(True, ls=":", alpha=0.4)
ax_snr.legend(loc="upper right", fontsize=8.2, framealpha=0.9)

# Clean, authentic text annotation of SNR values (no bulleted boxes)
ax_snr.text(
    0.05, 0.08,
    f"SNR (Normal Mode $\\tilde{{B}}$) = +{snr_bp_db:.1f} dB ({snr_bp_pwr:.0f}$\\times$)\n"
    f"SNR (Envelope $A(t)$) = +{snr_env_db:.1f} dB ({snr_env_pwr:.0f}$\\times$)",
    transform=ax_snr.transAxes, fontsize=9, fontweight="bold",
    bbox=dict(boxstyle="square,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9)
)

# -------------------------------------------------------------
# Panel (c): Cross-Correlation against I_Halpha (5 kHz)
# -------------------------------------------------------------
ax_corr.plot(lags_ms, corr_env_ha, color="crimson", lw=1.8, label=r"Envelope $A(t)$ vs. $I_{\mathrm{H}\alpha}$ ($5$ kHz)")
ax_corr.plot(lags_ms, corr_braw_ha, color="tab:blue", lw=1.2, ls="-.", label=r"Mirnov $\tilde{B}_{\leq 5\mathrm{kHz}}$ vs. $I_{\mathrm{H}\alpha}$ ($5$ kHz)")
ax_corr.plot(lags_ms, corr_bp_ha, color="gray", lw=1.0, ls=":", label=r"Mode Carrier $\tilde{B}_{\mathrm{mode}}$ ($89$ kHz) vs. $I_{\mathrm{H}\alpha}$")

ax_corr.axhline(0, color="black", lw=0.6, alpha=0.5)
ax_corr.axvline(0, color="dimgray", ls=":", lw=1.0)
ax_corr.axvline(peak_lag, color="crimson", ls="--", lw=1.1)

# Annotate peak with subtle white background
ax_corr.plot([peak_lag], [peak_r], "o", color="crimson", ms=5)
ax_corr.annotate(
    f"Peak $r = +{peak_r:.2f}$\n$\\tau = +{peak_lag:.2f}$ ms\n($A$ leads $I_{{\\mathrm{{H}}\\alpha}}$)",
    xy=(peak_lag, peak_r),
    xytext=(peak_lag + 0.9, peak_r - 0.05),
    fontsize=8.5, fontweight="bold", color="darkred",
    arrowprops=dict(arrowstyle="->", color="darkred", lw=1.0),
    bbox=dict(boxstyle="square,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85)
)

ax_corr.set_xlim(-5.0, 5.0)
ax_corr.set_ylim(-0.25, 0.80)
ax_corr.set_xlabel(r"Lag $\tau$ (ms)  [Positive $\tau$: Signal leads $I_{\mathrm{H}\alpha}$]", fontsize=10)
ax_corr.set_ylabel("Normalized Cross-Correlation", fontsize=10)
ax_corr.set_title(r"(c) Cross-Correlation: $\tilde{B}$ and Envelope vs. $I_{\mathrm{H}\alpha}$ ($5$ kHz)", fontsize=11, fontweight="bold", pad=8)
ax_corr.grid(True, ls=":", alpha=0.4)
ax_corr.legend(loc="upper left", fontsize=8.2, framealpha=0.9)

out_file = str(root_dir / "mhd_snr_halpha_correlation_shot_88653.png")
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved figure successfully to {out_file}")
