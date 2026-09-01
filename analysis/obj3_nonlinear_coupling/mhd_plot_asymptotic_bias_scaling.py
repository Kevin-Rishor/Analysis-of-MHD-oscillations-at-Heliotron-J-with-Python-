import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

current = Path(__file__).resolve().parent
root_dir = None
for p in [current] + list(current.parents):
    if (p / "jpack").exists():
        root_dir = p
        break
if root_dir is None:
    root_dir = Path("c:/TFG")

for p_add in [root_dir / "jpack", root_dir / "analysis", root_dir / "analysis" / "common"]:
    if str(p_add) not in sys.path:
        sys.path.append(str(p_add))

import turnelib as TE
import libana_signal as LAS
from mhd_analysis_obj3 import load_edf_signal, slice_window


def generate_sober_asymptotic_plot(shot=88653, out_dir=None):
    if out_dir is None:
        out_dir = root_dir
    else:
        out_dir = Path(out_dir)

    data_path = root_dir / "data" / f"hj{shot}" / f"MP1@{shot}.edf"

    t_start, t_end = 259.1, 275.0
    t_sec, ys, dt, fs = load_edf_signal(data_path)
    t_sec_win, ys_win = slice_window(t_sec, ys, t_start, t_end)

    nfft = 1024
    step = 512
    M_max = (len(ys_win) - nfft) // step + 1

    window = np.hanning(nfft)
    W2 = np.sum(window**2)
    segments_fft = []
    for k in range(M_max):
        idx0 = k * step
        seg = ys_win[idx0:idx0 + nfft]
        seg = seg - np.polyval(np.polyfit(np.arange(nfft), seg, 1), np.arange(nfft))
        X = np.fft.fft(seg * window)
        segments_fft.append(X)
    segments_fft = np.array(segments_fft)

    freqs = np.fft.fftfreq(nfft, d=dt) / 1000.0

    i_89 = np.argmin(np.abs(freqs - 89.8))
    i_41 = np.argmin(np.abs(freqs - 41.0))
    i_49 = np.argmin(np.abs(freqs - 48.8))

    i_99 = np.argmin(np.abs(freqs - 98.6))
    i_48 = np.argmin(np.abs(freqs - 48.8))
    i_50 = np.argmin(np.abs(freqs - 49.8))

    i_135 = np.argmin(np.abs(freqs - 135.0))
    i_20 = np.argmin(np.abs(freqs - 20.0))
    i_115 = np.argmin(np.abs(freqs - 115.0))

    def calc_b2(X_sub, i1, i2, i3):
        num = np.abs(np.mean(X_sub[:, i1] * np.conj(X_sub[:, i2]) * np.conj(X_sub[:, i3])))**2
        den = np.mean(np.abs(X_sub[:, i1] * X_sub[:, i2])**2) * np.mean(np.abs(X_sub[:, i3])**2)
        return num / den if den > 0 else 0.0

    def calc_psd(X_sub, idx):
        return np.mean(np.abs(X_sub[:, idx])**2) * (2.0 / (fs * W2))

    N_list = [5, 8, 12, 16, 20, 25, 30]
    inv_N = np.array([1.0 / N for N in N_list])

    np.random.seed(42)
    b2_triad, b2_sub, b2_noi = [], [], []
    psd_89, psd_41, psd_noi = [], [], []

    for N in N_list:
        trials_t, trials_s, trials_n = [], [], []
        trials_p89, trials_p41, trials_pn = [], [], []
        n_trials = 60 if N < 30 else 1
        for _ in range(n_trials):
            idx_sample = np.random.choice(M_max, size=N, replace=False) if N < M_max else np.arange(M_max)
            X_s = segments_fft[idx_sample]
            trials_t.append(calc_b2(X_s, i_89, i_41, i_49))
            trials_s.append(calc_b2(X_s, i_99, i_48, i_50))
            trials_n.append(calc_b2(X_s, i_135, i_20, i_115))
            trials_p89.append(calc_psd(X_s, i_89))
            trials_p41.append(calc_psd(X_s, i_41))
            trials_pn.append(calc_psd(X_s, i_135))
        b2_triad.append(np.mean(trials_t))
        b2_sub.append(np.mean(trials_s))
        b2_noi.append(np.mean(trials_n))
        psd_89.append(np.mean(trials_p89))
        psd_41.append(np.mean(trials_p41))
        psd_noi.append(np.mean(trials_pn))

    b2_triad = np.array(b2_triad)
    b2_sub = np.array(b2_sub)
    b2_noi = np.array(b2_noi)
    psd_89 = np.array(psd_89)
    psd_41 = np.array(psd_41)
    psd_noi = np.array(psd_noi)

    m_t, b_t = np.polyfit(inv_N, b2_triad, 1)
    m_s, b_s = np.polyfit(inv_N, b2_sub, 1)
    m_n, b_n = np.polyfit(inv_N, b2_noi, 1)

    m_p89, b_p89 = np.polyfit(inv_N, psd_89, 1)
    m_p41, b_p41 = np.polyfit(inv_N, psd_41, 1)
    m_pn, b_pn = np.polyfit(inv_N, psd_noi, 1)

    plt.rcdefaults()
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 0.8

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.2), facecolor='white')
    fig.suptitle(
        f"Heliotron J #{shot} — Asymptotic Convergence Analysis ($1/N \\to 0$)\n"
        f"Mirnov Coil MP1 (Stationary Phase: {t_start:.1f} – {t_end:.1f} ms, $N_\\mathrm{{FFT}} = 1024$)",
        fontsize=11.5, fontweight='bold', y=0.98
    )

    x_line = np.linspace(0.0, 0.22, 100)

    ax1.plot(x_line, x_line, color='gray', linestyle=':', linewidth=1.2,
             label=r"Theoretical noise expectation $\mathbb{E}[b^2] = 1/N$")
    ax1.plot(x_line, m_s * x_line + b_s, color='#1f77b4', linestyle='-', linewidth=1.5)
    ax1.plot(inv_N, b2_sub, color='#1f77b4', marker='s', markersize=5.5, linestyle='None',
             label=rf"Subharmonic ($98.6 - 48.8 = 49.8$ kHz): $b^2(0) = {b_s:+.3f}$")
    ax1.plot(x_line, m_t * x_line + b_t, color='#ff7f0e', linestyle='-', linewidth=1.5)
    ax1.plot(inv_N, b2_triad, color='#ff7f0e', marker='o', markersize=5.5, linestyle='None',
             label=rf"Primary triad ($89.8 - 41.0 = 48.8$ kHz): $b^2(0) = {b_t:+.3f}$")
    ax1.plot(x_line, m_n * x_line + b_n, color='#d62728', linestyle='--', linewidth=1.4)
    ax1.plot(inv_N, b2_noi, color='#d62728', marker='^', markersize=5.5, linestyle='None',
             label=rf"Noise baseline ($135.0, -20.0$ kHz): $b^2(0) = {b_n:+.3f} \approx 0$")

    ax1.axvline(0.0, color='black', linewidth=0.8, linestyle='-')
    ax1.set_xlim(-0.01, 0.22)
    ax1.set_ylim(-0.05, 0.50)
    ax1.set_xlabel(r"Inverse ensemble count $1/N$", fontsize=10)
    ax1.set_ylabel(r"Squared bicoherence $b^2$", fontsize=10)
    ax1.set_title(r"(a) Squared Bicoherence versus $1/N$", fontsize=10.5, pad=6)
    ax1.grid(True, linestyle=':', alpha=0.35, color='gray')
    ax1.legend(loc='upper left', fontsize=8.2, framealpha=0.9)

    ax2.semilogy(x_line, m_p41 * x_line + b_p41, color='#2ca02c', linestyle='-', linewidth=1.5)
    ax2.semilogy(inv_N, psd_41, color='#2ca02c', marker='s', markersize=5.5, linestyle='None',
                 label=rf"Secondary mode ($41.0$ kHz): $P = {b_p41:.2e}$ V$^2$/Hz")
    ax2.semilogy(x_line, m_p89 * x_line + b_p89, color='#1f77b4', linestyle='-', linewidth=1.5)
    ax2.semilogy(inv_N, psd_89, color='#1f77b4', marker='o', markersize=5.5, linestyle='None',
                 label=rf"Primary mode ($89.8$ kHz): $P = {b_p89:.2e}$ V$^2$/Hz")
    ax2.semilogy(x_line, m_pn * x_line + b_pn, color='#7f7f7f', linestyle='--', linewidth=1.4)
    ax2.semilogy(inv_N, psd_noi, color='#7f7f7f', marker='^', markersize=5.5, linestyle='None',
                 label=rf"Noise baseline ($135.0$ kHz): $P = {b_pn:.2e}$ V$^2$/Hz")

    ax2.axvline(0.0, color='black', linewidth=0.8, linestyle='-')
    ax2.set_xlim(-0.01, 0.22)
    ax2.set_ylim(1e-8, 1.5e-5)
    ax2.set_xlabel(r"Inverse ensemble count $1/N$", fontsize=10)
    ax2.set_ylabel(r"Power spectral density (V$^2$/Hz)", fontsize=10)
    ax2.set_title(r"(b) Power Spectral Density versus $1/N$", fontsize=10.5, pad=6)
    ax2.grid(True, which='both', linestyle=':', alpha=0.35, color='gray')
    ax2.legend(loc='center right', fontsize=8.2, framealpha=0.9)

    plt.tight_layout()
    out_png = out_dir / f"mhd_asymptotic_bias_scaling_shot_{shot}.png"
    plt.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Asymptotic scaling plot saved to: '{out_png}'")
    return str(out_png)


if __name__ == "__main__":
    generate_sober_asymptotic_plot(88653)

