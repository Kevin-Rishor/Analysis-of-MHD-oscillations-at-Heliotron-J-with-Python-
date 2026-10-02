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

for p_add in [root_dir / "jpack", root_dir / "analysis", root_dir / "analysis" / "common", root_dir / "analysis" / "obj3_nonlinear_coupling", root_dir / "analysis" / "heating_phase_comparison"]:
    if str(p_add) not in sys.path:
        sys.path.append(str(p_add))

from comparison_common import load_channel
from mhd_plot_bicoherence_sliding_trace import custom_abicoh2_complex
from mhd_analysis_obj3 import extract_instantaneous_frequency

# Standard analysis parameters
WIN_LEN_MS = 8.0
STEP_MS = 0.25
T_SCAN_START_MS = 291.3
T_SCAN_END_MS = 310.0
T_NBI_BURST_START_MS = 291.3
T_NBI_BURST_END_MS = 300.0
T_BEAM_OFF_MS = 300.6
SNR_THRESHOLD = 2.5e-23
FLATNESS_GROWTH_TOL = 0.5  # 1.5x reference core

def compute_reference_flatness_core(t_ms, ifreq_khz, dt, t0=T_NBI_BURST_START_MS, t1=T_NBI_BURST_END_MS, scan_win_ms=WIN_LEN_MS):
    """Computes the reference flattest 8 ms core within the NBI-only burst window."""
    idx_domain = np.where((t_ms >= t0) & (t_ms <= t1))[0]
    if len(idx_domain) < 3:
        raise ValueError(f"Domain [{t0}, {t1}] ms has too few samples.")
    
    t_domain = t_ms[idx_domain]
    ifreq_domain = ifreq_khz[idx_domain]
    
    slope_raw = np.gradient(ifreq_domain, t_domain)
    smooth_samples = max(1, int(round(2.0 / (dt * 1000.0))))
    if smooth_samples % 2 == 0:
        smooth_samples += 1
    kernel = np.ones(smooth_samples) / smooth_samples
    slope_smooth = np.convolve(slope_raw, kernel, mode='same')
    slope_abs = np.abs(slope_smooth)
    
    scan_samples = int(round(scan_win_ms / (dt * 1000.0)))
    scan_kernel = np.ones(scan_samples) / scan_samples
    window_means = np.convolve(slope_abs, scan_kernel, mode='valid')
    best_mean = float(np.min(window_means))
    best_start_idx = int(np.argmin(window_means))
    tol_mean = best_mean * (1.0 + FLATNESS_GROWTH_TOL)
    
    return {
        'best_mean': best_mean,
        'tol_mean': tol_mean,
        'core_start_ms': float(t_domain[best_start_idx]),
        'core_end_ms': float(t_domain[best_start_idx + scan_samples - 1])
    }

def evaluate_energy_transfer_windows(shot=88653):
    """Shared helper evaluating the 3 filtering rules across all sliding windows:
    Rule (i): [t, t+8.0] contained in [291.3, 300.0] ms (with sensitivity at 300.6 ms)
    Rule (ii): flatness <= 1.5x reference 8 ms core
    Rule (iii): den > 2.5e-23 SNR
    """
    t_ms, ys, dt, fs = load_channel(shot, "MP1")
    if ys is None or t_ms is None:
        raise RuntimeError("MP1 channel signal not found.")

    # 1. Instantaneous frequency for flatness
    fl_hz = 40e3
    fu_hz = 120e3
    bessel_order = 4
    sg_win = 51
    _, _, _, ifreq_hz = extract_instantaneous_frequency(ys, fs, fl_hz, fu_hz, bessel_order, sg_win)
    ifreq_khz = ifreq_hz / 1000.0
    
    # Smooth slope over full trace
    smooth_samples = max(1, int(round(2.0 / (dt * 1000.0))))
    if smooth_samples % 2 == 0:
        smooth_samples += 1
    kernel = np.ones(smooth_samples) / smooth_samples
    full_slope_raw = np.gradient(ifreq_khz, t_ms)
    full_slope_smooth = np.convolve(full_slope_raw, kernel, mode='same')
    full_slope_abs = np.abs(full_slope_smooth)

    ref_core = compute_reference_flatness_core(t_ms, ifreq_khz, dt)

    # Welch parameters for amplitude scaling
    NPERSEG = 1024
    NOVERLAP = 512

    windows = []
    for t_start in np.arange(T_SCAN_START_MS, T_SCAN_END_MS, STEP_MS):
        t_end = t_start + WIN_LEN_MS
        if t_end > t_ms[-1]:
            break
        idx = np.where((t_ms >= t_start) & (t_ms <= t_end))[0]
        y_sub = ys[idx]
        if len(y_sub) < 512:
            continue

        f1, f2, b2, b_comp, den = custom_abicoh2_complex(y_sub, dt=dt, nfft=512, nensemble=30)
        ff1_k, ff2_k = np.meshgrid(f1 / 1000.0, f2 / 1000.0, indexing='ij')
        mask_diff = ((ff1_k > 95) & (ff1_k < 105) & (ff2_k < -40) & (np.abs(ff2_k / ff1_k + 0.5) < 0.1))

        if np.any(mask_diff):
            best_c = np.argwhere(b2 == np.max(b2[mask_diff]))[0]
            b2_val = b2[best_c[0], best_c[1]]
            im_val = -np.imag(b_comp[best_c[0], best_c[1]])
            den_val = den[best_c[0], best_c[1]]
        else:
            b2_val, im_val, den_val = np.nan, np.nan, 0.0

        win_mean_slope = float(np.mean(full_slope_abs[idx]))

        f_psd, Pxx = dsp.welch(y_sub, fs=fs, nperseg=NPERSEG, noverlap=NOVERLAP, window='hann', scaling='density')
        df = f_psd[1] - f_psd[0]
        mask_50 = (f_psd >= 40e3) & (f_psd <= 55e3)
        mask_100 = (f_psd >= 90e3) & (f_psd <= 110e3)
        A50_band = np.sqrt(np.sum(Pxx[mask_50]) * df)
        A100_band = np.sqrt(np.sum(Pxx[mask_100]) * df)
        f50_peak = f_psd[mask_50][np.argmax(Pxx[mask_50])]
        f100_peak = f_psd[mask_100][np.argmax(Pxx[mask_100])]

        # The 3 Rules
        pass_containment_strict = (t_end <= T_NBI_BURST_END_MS + 1e-6)
        pass_containment_beam = (t_end <= T_BEAM_OFF_MS + 1e-6)
        pass_flatness = (win_mean_slope <= ref_core['tol_mean'])
        pass_snr = (den_val > SNR_THRESHOLD)

        windows.append({
            't_start': t_start,
            't_center': t_start + WIN_LEN_MS / 2.0,
            't_end': t_end,
            'b2': b2_val,
            'im_val': im_val,
            'den': den_val,
            'slope': win_mean_slope,
            'A50_band': A50_band,
            'A100_band': A100_band,
            'f50_peak': f50_peak,
            'f100_peak': f100_peak,
            'pass_containment_strict': pass_containment_strict,
            'pass_containment_beam': pass_containment_beam,
            'pass_flatness': pass_flatness,
            'pass_snr': pass_snr,
            'pass_all_strict': pass_containment_strict and pass_flatness and pass_snr,
            'pass_all_beam': pass_containment_beam and pass_flatness and pass_snr
        })

    return windows, ref_core
