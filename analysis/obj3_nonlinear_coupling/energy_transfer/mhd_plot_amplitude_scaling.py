import sys
from pathlib import Path
import numpy as np
import scipy.signal as dsp
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

current = Path(__file__).resolve().parent
root_dir = None
for p in [current] + list(current.parents):
    if (p / "jpack").exists():
        root_dir = p
        break
if root_dir is None:
    root_dir = Path("c:/TFG")

for p_add in [root_dir / "jpack", root_dir / "analysis", root_dir / "analysis" / "common", root_dir / "analysis" / "heating_phase_comparison"]:
    if str(p_add) not in sys.path:
        sys.path.append(str(p_add))

from comparison_common import load_channel
from mhd_plot_bicoherence_sliding_trace import custom_abicoh2_complex

def compute_scaling_data(shot=88653):
    t_ms, ys, dt, fs = load_channel(shot, "MP1")
    if ys is None or t_ms is None:
        raise RuntimeError("MP1 signal not found.")

    win_len = 8.0
    step = 0.25
    NPERSEG = 1024
    NOVERLAP = 512

    # Noise floors / quiet intervals
    def get_band_amps(t0, t1):
        idx = np.where((t_ms >= t0) & (t_ms <= t1))[0]
        y_q = ys[idx]
        fq, Pxx_q = dsp.welch(y_q, fs=fs, nperseg=NPERSEG, noverlap=NOVERLAP, window='hann', scaling='density')
        df_q = fq[1] - fq[0]
        m50 = (fq >= 40e3) & (fq <= 55e3)
        m100 = (fq >= 90e3) & (fq <= 110e3)
        a50 = np.sqrt(np.sum(Pxx_q[m50]) * df_q)
        a100 = np.sqrt(np.sum(Pxx_q[m100]) * df_q)
        return np.log10(a50), np.log10(a100)

    q1_log50, q1_log100 = get_band_amps(150.0, 160.0)
    q2_log50, q2_log100 = get_band_amps(400.0, 410.0)

    all_data = []
    # Scan 291.3 to 310.0 step 0.25 (75 windows)
    for t_start in np.arange(291.3, 310.0, step):
        t_end = t_start + win_len
        if t_end > t_ms[-1]:
            break
        idx = np.where((t_ms >= t_start) & (t_ms <= t_end))[0]
        y_sub = ys[idx]
        if len(y_sub) < 512:
            continue

        f1, f2, b2, b_comp, den = custom_abicoh2_complex(y_sub, dt=dt, nfft=512, nensemble=30)
        ff1_k, ff2_k = np.meshgrid(f1 / 1000.0, f2 / 1000.0, indexing='ij')
        mask_diff = ((ff1_k > 95) & (ff1_k < 105) & (ff2_k < -40) & (np.abs(ff2_k / ff1_k + 0.5) < 0.1))

        if not np.any(mask_diff):
            continue
        best_c = np.argwhere(b2 == np.max(b2[mask_diff]))[0]

        t_center = t_start + win_len / 2.0
        b2_val = b2[best_c[0], best_c[1]]
        im_val_norm = -np.imag(b_comp[best_c[0], best_c[1]])
        den_val = den[best_c[0], best_c[1]]

        f_psd, Pxx = dsp.welch(y_sub, fs=fs, nperseg=NPERSEG, noverlap=NOVERLAP, window='hann', scaling='density')
        df = f_psd[1] - f_psd[0]
        mask_50 = (f_psd >= 40e3) & (f_psd <= 55e3)
        mask_100 = (f_psd >= 90e3) & (f_psd <= 110e3)

        A50_band = np.sqrt(np.sum(Pxx[mask_50]) * df)
        A100_band = np.sqrt(np.sum(Pxx[mask_100]) * df)
        A50_peak = np.sqrt(np.max(Pxx[mask_50]))
        A100_peak = np.sqrt(np.max(Pxx[mask_100]))

        all_data.append({
            't_start': t_start, 't_center': t_center, 't_end': t_end,
            'b2': b2_val, 'im_norm': im_val_norm, 'den': den_val,
            'A50_band': A50_band, 'A100_band': A100_band,
            'A50_peak': A50_peak, 'A100_peak': A100_peak,
        })

    return all_data, (q1_log50, q1_log100), (q2_log50, q2_log100)

def plot_amplitude_scaling(all_data, q1, q2, out_dir=None):
    if out_dir is None:
        out_dir = root_dir / "Images"
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_A50 = np.array([np.log10(d['A50_band']) for d in all_data])
    log_A100 = np.array([np.log10(d['A100_band']) for d in all_data])
    den_vals = np.array([d['den'] for d in all_data])

    valid_mask = den_vals > 2.5e-23
    x_valid = log_A50[valid_mask]
    y_valid = log_A100[valid_mask]

    # Regressions
    # 1. Valid 16
    res_valid = stats.linregress(x_valid, y_valid)
    slope_valid = res_valid.slope
    intercept_valid = res_valid.intercept

    # 2. All 75
    res_all = stats.linregress(log_A50, log_A100)
    slope_all = res_all.slope
    intercept_all = res_all.intercept

    # Centroid for reference slopes
    x_mean = np.mean(x_valid)
    y_mean = np.mean(y_valid)

    # FIGURE 1: amplitude_scaling_valid16.png
    fig1, ax1 = plt.subplots(figsize=(7, 5), dpi=300)
    ax1.scatter(x_valid, y_valid, color='blue', alpha=0.8, s=35, label=f'Valid (N={len(x_valid)})')

    x_line = np.linspace(-1.72, -0.85, 200)
    ax1.plot(x_line, slope_valid * x_line + intercept_valid, color='blue', lw=1.8,
             label=f'Valid slope={slope_valid:.2f}')
    ax1.plot(x_line, 1.0 * (x_line - x_mean) + y_mean, color='green', ls=':', lw=1.8,
             label='Ref slope=1.0')
    ax1.plot(x_line, 2.0 * (x_line - x_mean) + y_mean, color='red', ls=':', lw=1.8,
             label='Ref slope=2.0')

    ax1.set_title('Amplitude Scaling: Valid High-SNR Windows', fontsize=12)
    ax1.set_xlabel('log10(A_50)', fontsize=11)
    ax1.set_ylabel('log10(A_100)', fontsize=11)
    ax1.grid(True)
    ax1.legend(loc='upper left', fontsize=10)
    fig1.tight_layout()
    fig1.savefig(out_dir / "amplitude_scaling_valid16.png", dpi=300)
    plt.close(fig1)

    # FIGURE 2: amplitude_scaling_all_vs_valid.png
    fig2, ax2 = plt.subplots(figsize=(7, 5), dpi=300)
    x_low = log_A50[~valid_mask]
    y_low = log_A100[~valid_mask]
    ax2.scatter(x_low, y_low, color='grey', alpha=0.4, s=35, label=f'Low-SNR (N={len(x_low)})')
    ax2.scatter(x_valid, y_valid, color='blue', alpha=0.8, s=35, label=f'Valid (N={len(x_valid)})')

    # Quiet intervals
    ax2.scatter(q1[0], q1[1], color='red', marker='*', s=140, label='Quiet 1 (150-160 ms)')
    ax2.scatter(q2[0], q2[1], color='orange', marker='*', s=140, label='Quiet 2 (400-410 ms)')

    x_line_all = np.linspace(-2.52, -0.95, 200)
    ax2.plot(x_line_all, slope_all * x_line_all + intercept_all, color='black', ls='--', lw=1.6,
             label=f'All slope={slope_all:.2f}')
    ax2.plot(x_line_all, slope_valid * x_line_all + intercept_valid, color='blue', lw=1.8,
             label=f'Valid slope={slope_valid:.2f}')
    ax2.plot(x_line_all, 1.0 * (x_line_all - x_mean) + y_mean, color='green', ls=':', lw=1.8,
             label='Ref slope=1.0')
    ax2.plot(x_line_all, 2.0 * (x_line_all - x_mean) + y_mean, color='red', ls=':', lw=1.8,
             label='Ref slope=2.0')

    ax2.set_title('Amplitude Scaling: All vs Valid Windows', fontsize=12)
    ax2.set_xlabel('log10(A_50)', fontsize=11)
    ax2.set_ylabel('log10(A_100)', fontsize=11)
    ax2.grid(True)
    ax2.legend(loc='lower right', fontsize=9.5)
    fig2.tight_layout()
    fig2.savefig(out_dir / "amplitude_scaling_all_vs_valid.png", dpi=300)
    plt.close(fig2)

def main(out_dir=None):
    data, q1, q2 = compute_scaling_data()
    plot_amplitude_scaling(data, q1, q2, out_dir=out_dir)

if __name__ == "__main__":
    out_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(out_dir=out_arg)
