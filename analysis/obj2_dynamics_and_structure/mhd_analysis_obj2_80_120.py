"""
MHD Analysis - Specific Objective 2 (Primary Mode: 80 - 120 kHz Window)
========================================================================
This script performs the full energetic particle-driven mode (EPM) heating
correlation, Alfven scaling, inter-probe cross-spectral coherence, poloidal
mode number (m) decomposition, and energetic-particle distribution-function
response validation (Zhong et al. approach) specifically for the PRIMARY MODE
frequency band (80 kHz to 120 kHz).

Outputs:
  - mhd_analysis_objective2_{shot}_80_120kHz.png: 6-panel summary plot
  - mhd_analysis_objective2_zhong_{shot}_80_120kHz.png: 4-panel ECE/pressure validation plot
"""

# -------------------------------------------------------------
# SHOT CONFIGURATION 
# -------------------------------------------------------------
SHOTS_DEFAULT = [88653]

# Default frequency window for Primary Mode analysis
DEFAULT_LOWER_KHZ = 80.0
DEFAULT_UPPER_KHZ = 120.0

# -------------------------------------------------------------
#  Mirnov probe pairs used for inter-probe coherence panels
# -------------------------------------------------------------
PROBE_PAIRS = [("MP1", "MP3"), ("MP1", "MP4"), ("MP3", "MP4")]
PROBE_PAIR_STYLES = {
    ("MP1", "MP3"): dict(color="tab:blue", linestyle="-"),
    ("MP1", "MP4"): dict(color="tab:green", linestyle="--"),
    ("MP3", "MP4"): dict(color="tab:purple", linestyle="-."),
}

# -------------------------------------------------------------
# Poloidal probe array constants
# -------------------------------------------------------------
PMP_COIL_CONST = 4.5e-3   # coil parameter (height x width x turns)
PMP_GAIN = 2.0            # amplifier gain
PMP_CHANNELS = [f"PMP{i}" for i in range(1, 15)]  # PMP1..PMP14
PMP_RAW_ANGLE_LABELS_DEG = [0., 10., 20., 30., 40., 50., 60., 80., 90., 100., 110., 120., 150., 180.]
PMP_ANGLES_DEG = [(360.0 - a) % 360.0 for a in PMP_RAW_ANGLE_LABELS_DEG]
PMP_INVERT_CHANNELS_DEFAULT = ("PMP1", "PMP2", "PMP3", "PMP4")

import sys
import os
import argparse
import json
from pathlib import Path

# Ensure UTF-8 stdout/stderr encoding on Windows to prevent UnicodeEncodeError
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import scipy.signal as dsp
import scipy.stats as stats
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

# Add the jpack library path to Python's path
jpack_path = str(Path(__file__).parent.parent.resolve() / "jpack")
if jpack_path not in sys.path:
    sys.path.append(jpack_path)

import turnelib as TE
import libana_signal as LAS
from mhd_common import OPTIMAL_SG_WIN, extract_instantaneous_frequency, anti_alias_decimate


def estimate_acf(x, nlags=50):
    """Natively estimates the autocorrelation function (ACF) of a signal."""
    n = len(x)
    var = np.var(x)
    if var == 0:
        return np.ones(nlags + 1)
    xp = x - np.mean(x)
    corr = np.correlate(xp, xp, mode='full')
    center = len(xp) - 1
    return corr[center:center + nlags + 1] / (n * var)



def conservative_p_value(r, N_eff, n_control=0):
    """Computes a conservative p-value using the effective sample size N_eff."""
    if abs(r) >= 1.0:
        return 0.0
    df = N_eff - 2 - n_control
    if df <= 0:
        return 1.0
    t_stat = r * np.sqrt(df / (1.0 - r ** 2))
    return 2.0 * stats.t.sf(abs(t_stat), df)


def benjamini_hochberg(p_values, alpha=0.05):
    """Benjamini-Hochberg procedure for False Discovery Rate (FDR) control."""
    p_arr = np.asarray(p_values, dtype=float)
    n = len(p_arr)
    order = np.argsort(p_arr)
    ranked = p_arr[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha

    below = ranked <= thresholds
    significant_sorted = np.zeros(n, dtype=bool)
    if np.any(below):
        k_max = np.max(np.where(below)[0])
        significant_sorted[:k_max + 1] = True

    p_adj_sorted = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    p_adj_sorted = np.clip(p_adj_sorted, 0.0, 1.0)

    significant = np.empty(n, dtype=bool)
    p_adjusted = np.empty(n, dtype=float)
    significant[order] = significant_sorted
    p_adjusted[order] = p_adj_sorted
    return significant, p_adjusted


def format_p_value(p):
    """Formats a p-value for printing, avoiding numerical underflow (p=0.0)."""
    if p == 0.0 or p < 1e-300:
        return "p < 1e-300"
    return f"p = {p:.2e}"


def lagged_cross_correlation(x, y, dt_corr, max_lag_ms=20.0):
    """Normalized lagged cross-correlation between x and y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 4 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0, 0.0, np.array([0.0]), np.array([0.0])

    xn = (x - np.mean(x)) / (np.std(x) + 1e-30)
    yn = (y - np.mean(y)) / (np.std(y) + 1e-30)

    full_corr = dsp.correlate(xn, yn, mode='full') / n
    lags_samples = dsp.correlation_lags(n, n, mode='full')
    lags_ms = lags_samples * dt_corr * 1000.0

    max_lag_samples = max(1, int(round(max_lag_ms / (dt_corr * 1000.0))))
    center = len(lags_samples) // 2
    lo = max(0, center - max_lag_samples)
    hi = min(len(lags_samples), center + max_lag_samples + 1)

    window_lags = lags_ms[lo:hi]
    window_corr = full_corr[lo:hi]
    if len(window_corr) == 0:
        return 0.0, 0.0, np.array([0.0]), np.array([0.0])

    best_idx = int(np.argmax(np.abs(window_corr)))
    return float(window_lags[best_idx]), float(window_corr[best_idx]), window_lags, window_corr


def lagged_pearson_significance(x, y, dt_corr, lag_ms):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    lag_samples = int(round(lag_ms / (dt_corr * 1000.0)))
    if lag_samples >= 0:
        x_ov, y_ov = x[lag_samples:], y[: n - lag_samples] if lag_samples > 0 else y
    else:
        x_ov, y_ov = x[: n + lag_samples], y[-lag_samples:]
    n_ov = min(len(x_ov), len(y_ov))
    x_ov, y_ov = x_ov[:n_ov], y_ov[:n_ov]
    if n_ov < 10 or np.std(x_ov) == 0 or np.std(y_ov) == 0:
        return None, None, None, n_ov, None
    r, p_std = stats.pearsonr(x_ov, y_ov)
    acf = estimate_acf(x_ov, nlags=min(50, n_ov - 2))
    N_eff = n_ov / (1.0 + 2.0 * np.sum(acf[1:])) if n_ov > 2 else 3.0
    N_eff = max(3.0, min(float(n_ov), N_eff))
    p_adj = conservative_p_value(r, N_eff)
    return float(r), float(p_std), float(p_adj), n_ov, float(N_eff)


def compute_probe_pair_coherence(signal_dict, key, pairs, t_sec, dt, args, nfft=None):
    nfft_use = nfft if nfft is not None else args.nfft
    noverlap = nfft_use // 2
    results = {}
    for (pa, pb) in pairs:
        if pa in signal_dict and pb in signal_dict:
            xa = signal_dict[pa][key]
            xb = signal_dict[pb][key]
            n_ens = max(1, (len(xa) - nfft_use) // (nfft_use - noverlap) + 1)
            f_pair, Pxy_pair, Pyy_pair, Pxx_pair = LAS.csd(
                xa, t_sec, xb, dt=dt, nfft=nfft_use,
                noverlap=noverlap, nensemble=n_ens, window='hann', detrend='constant'
            )
            coh2_pair = LAS.xcoh2(Pxy_pair, Pyy_pair, Pxx_pair)
            mean_coh2_pair = np.mean(coh2_pair, axis=1) if coh2_pair.ndim > 1 else coh2_pair
            results[(pa, pb)] = (f_pair, mean_coh2_pair)
        else:
            results[(pa, pb)] = None
    return results


def load_poloidal_array(shot, data_dir, t_ms, invert_channels=PMP_INVERT_CHANNELS_DEFAULT):
    signals = {}
    plab_rad = {}
    missing = []
    for ch, angle_deg in zip(PMP_CHANNELS, PMP_ANGLES_DEG):
        pmp_file = data_dir / f"{ch}@{shot}.edf"
        if not pmp_file.exists():
            missing.append(ch)
            continue
        edf_pmp = TE.edf()
        dat_pmp = edf_pmp.load(str(pmp_file))
        t_pmp = dat_pmp[:, 0]
        t_pmp_ms = t_pmp if edf_pmp.DimUnit[0] == 'ms' else t_pmp * 1000.0
        ys_pmp = np.interp(t_ms, t_pmp_ms, dat_pmp[:, 1])
        ys_pmp = ys_pmp / (PMP_GAIN * PMP_COIL_CONST)
        if ch in invert_channels:
            ys_pmp = -ys_pmp
        signals[ch] = ys_pmp
        plab_rad[ch] = np.deg2rad(angle_deg)

    if missing:
        print(f"  ️ [M10] Warning: no file(s) found for poloidal probe(s) {missing}; using {len(signals)} channels.")
    inverted_found = [ch for ch in invert_channels if ch in signals]
    if inverted_found:
        print(f"  [M10] Loaded poloidal array: {len(signals)}/{len(PMP_CHANNELS)} channels found. "
              f"Inverted polarity (x -1) of {inverted_found}.")
    return signals, plab_rad


def _nudft_poloidal(angles_rad, complex_values, k_grid, sign=-1):
    n = len(angles_rad)
    return (1.0 / n) * np.dot(complex_values, np.exp(sign * 1j * k_grid * angles_rad[:, None]))


def poloidal_array_self_test(plab_rad, max_m_test=10, k_search_margin=6):
    channels = sorted(plab_rad.keys(), key=lambda c: int(c[3:]))
    if len(channels) < 3:
        print(" Fewer than 3 poloidal channels available; self-test skipped.")
        return None

    angles = np.array([plab_rad[ch] for ch in channels])
    k_search = max_m_test + k_search_margin
    k_grid = np.arange(-k_search, k_search + 1)
    test_modes = list(range(-max_m_test, max_m_test + 1))
    failures = []
    for m_true in test_modes:
        S_synth = np.exp(1j * m_true * angles)
        Sk = _nudft_poloidal(angles, S_synth, k_grid)
        power = (Sk * Sk.conj()).real
        m_recovered = int(k_grid[np.argmax(power)])
        if m_recovered != m_true:
            failures.append((m_true, m_recovered))

    print(f"  [M10-SELFTEST] Synthetic recovery test, m = -{max_m_test}..+{max_m_test} "
          f"(search range +/-{k_search}), using this array's {len(channels)}-probe geometry:")
    first_failure_abs_m = None
    if not failures:
        print(f"     All {len(test_modes)} synthetic mode numbers recovered correctly.")
    else:
        first_failure_abs_m = min(abs(m) for m, _ in failures)
        print(f"    ️ {len(failures)}/{len(test_modes)} synthetic mode numbers recovered INCORRECTLY "
              f"(first failure at |m_true| = {first_failure_abs_m}): {failures}")
    return {"test_modes": test_modes, "failures": failures, "max_m_test": max_m_test,
            "k_search": k_search, "first_failure_abs_m": first_failure_abs_m}


def _poloidal_power_map(sig_matrix, angles, fs, fl_hz, fu_hz, max_m, nfft_use):
    f, tave, S = dsp.spectrogram(
        sig_matrix, fs=fs, window='hann', nperseg=nfft_use, noverlap=nfft_use // 2,
        nfft=nfft_use, detrend='constant', return_onesided=False, scaling='density',
        axis=-1, mode='complex'
    )
    band_mask = (f >= fl_hz) & (f <= fu_hz)
    if not np.any(band_mask):
        return None

    k_grid = np.arange(-max_m, max_m + 1)
    f_idx_band = np.where(band_mask)[0]
    S_band = S[:, f_idx_band, :]
    E = np.exp(-1j * k_grid[:, None] * angles[None, :]) / len(angles)
    Sk = np.tensordot(E, S_band, axes=(1, 0))
    P2d = np.mean((Sk * Sk.conj()).real, axis=2)
    f_band_khz = f[f_idx_band] / 1000.0
    return k_grid, f_band_khz, P2d



def poloidal_mode_number_analysis(pmp_signals, plab_rad, dt, i0, i1, fl_hz, fu_hz, args):
    channels = sorted(pmp_signals.keys(), key=lambda c: int(c[3:]))
    if len(channels) < 3:
        return None

    sig_matrix = np.array([pmp_signals[ch][i0:i1] for ch in channels])
    fs = 1.0 / dt
    nfft_use = min(args.pmp_nfft, sig_matrix.shape[1])
    if nfft_use < 32:
        return None
    angles = np.array([plab_rad[ch] for ch in channels])

    result = _poloidal_power_map(sig_matrix, angles, fs, fl_hz, fu_hz, args.pmp_max_mode_number, nfft_use)
    if result is None:
        return None
    k_grid, f_band_khz, P2d = result

    m_per_f = k_grid[np.argmax(P2d, axis=0)]
    f_peak_local_idx = int(np.argmax(np.sum(P2d, axis=0)))
    m_dominant = int(m_per_f[f_peak_local_idx])
    f_peak_khz = float(f_band_khz[f_peak_local_idx])

    is_edge_pinned = abs(m_dominant) == args.pmp_max_mode_number
    verdict = "not_applicable"
    m_dominant_expanded = None
    k_grid_plot, f_band_khz_plot, P2d_plot = k_grid, f_band_khz, P2d
    was_expanded_for_plot = False
    if is_edge_pinned:
        expanded_max_m = min(2 * args.pmp_max_mode_number, args.pmp_max_mode_expanded)
        if expanded_max_m > args.pmp_max_mode_number:
            result_exp = _poloidal_power_map(sig_matrix, angles, fs, fl_hz, fu_hz, expanded_max_m, nfft_use)
            if result_exp is not None:
                k_grid_exp, f_band_khz_exp, P2d_exp = result_exp
                k_grid_plot, f_band_khz_plot, P2d_plot = k_grid_exp, f_band_khz_exp, P2d_exp
                was_expanded_for_plot = True
                m_per_f_exp = k_grid_exp[np.argmax(P2d_exp, axis=0)]
                f_peak_idx_exp = int(np.argmax(np.sum(P2d_exp, axis=0)))
                m_dominant_expanded = int(m_per_f_exp[f_peak_idx_exp])
                if abs(m_dominant_expanded) == expanded_max_m:
                    verdict = "boundary_artifact"
                elif m_dominant_expanded == m_dominant:
                    verdict = "confirmed_stable"
                else:
                    verdict = "moved_off_edge"
        else:
            verdict = "expand_capped"

    return {
        "k_grid": k_grid, "f_band_khz": f_band_khz, "P2d": P2d,
        "k_grid_plot": k_grid_plot, "f_band_khz_plot": f_band_khz_plot, "P2d_plot": P2d_plot,
        "was_expanded_for_plot": was_expanded_for_plot,
        "m_dominant": m_dominant, "f_peak_khz": f_peak_khz, "channels": channels,
        "is_edge_pinned": is_edge_pinned, "verdict": verdict, "m_dominant_expanded": m_dominant_expanded,
    }


def poloidal_phase_structure_analysis(pmp_signals, plab_rad, dt, i0, i1, f_peak_hz, args):
    """
    Computes the cross-spectral phase of each PMP probe relative to a reference
    probe (first channel, typically PMP1) at the dominant mode frequency f_peak_hz.
    
    This provides a direct, independent verification of the poloidal mode number m:
    for a mode with mode number m, the phase should increase linearly as
    phi(theta) = m * theta (wrapped to [-pi, pi]).
    """
    channels = sorted(pmp_signals.keys(), key=lambda c: int(c[3:]))
    if len(channels) < 3:
        return None

    sig_matrix = np.array([pmp_signals[ch][i0:i1] for ch in channels])
    angles_rad = np.array([plab_rad[ch] for ch in channels])
    angles_deg = np.rad2deg(angles_rad)
    fs = 1.0 / dt
    n_samples = sig_matrix.shape[1]

    nfft_use = min(getattr(args, 'pmp_nfft', 256), n_samples)
    if nfft_use < 32:
        return None

    f_spec, t_spec, S = dsp.spectrogram(
        sig_matrix, fs=fs, window='hann', nperseg=nfft_use, noverlap=nfft_use // 2,
        nfft=nfft_use, detrend='constant', return_onesided=True, scaling='density',
        axis=-1, mode='complex'
    )

    f_idx = int(np.argmin(np.abs(f_spec - f_peak_hz)))
    actual_f = float(f_spec[f_idx])

    S_peak = S[:, f_idx, :]
    ref_spectrum = S_peak[0, :]
    cross_spectra = S_peak * np.conj(ref_spectrum)[None, :]
    avg_cross = np.mean(cross_spectra, axis=1)
    n_seg = S_peak.shape[1]

    # Magnitude-squared coherence vs reference probe: gamma^2 = |avg_cross|^2 / (P_ref * P_ch)
    p_ref = np.mean(np.abs(ref_spectrum)**2)
    p_ch = np.mean(np.abs(S_peak)**2, axis=1)
    coherence_sq = np.abs(avg_cross)**2 / (p_ref * p_ch + 1e-30)
    coherence_sq = np.clip(coherence_sq, 0.0, 1.0)

    # Metrological phase uncertainty (Bendat & Piersol 1986):
    # sigma_phi = sqrt((1 - gamma^2) / (2 * gamma^2 * n_seg))
    # Cap maximum error at pi rad for physical plot bounds
    sigma_phase = np.sqrt(np.maximum(0.0, 1.0 - coherence_sq) / (2.0 * np.maximum(coherence_sq, 1e-4) * n_seg))
    sigma_phase = np.clip(sigma_phase, 0.0, np.pi)

    measured_phase = np.angle(avg_cross)
    measured_phase = measured_phase - measured_phase[0]
    measured_phase = np.arctan2(np.sin(measured_phase), np.cos(measured_phase))

    m_candidates = list(range(-6, 7))

    # Calculate circular phase alignment r_circ and mean absolute error for each candidate m
    theta_in_pi = np.arctan2(np.sin(angles_rad), np.cos(angles_rad))
    m_alignment = {}
    for m_val in m_candidates:
        th_theory = np.arctan2(np.sin(m_val * theta_in_pi), np.cos(m_val * theta_in_pi))
        diff = np.arctan2(np.sin(measured_phase - th_theory), np.cos(measured_phase - th_theory))
        r_c = float(np.sqrt(np.sum(np.cos(diff))**2 + np.sum(np.sin(diff))**2) / len(theta_in_pi))
        mean_err_deg = float(np.rad2deg(np.mean(np.abs(diff))))
        m_alignment[m_val] = {"r_circ": r_c, "mean_error_deg": mean_err_deg}

    return {
        "theta_deg": angles_deg,
        "theta_rad": angles_rad,
        "measured_phase": measured_phase,
        "coherence_sq": coherence_sq,
        "sigma_phase": sigma_phase,
        "mean_coherence": float(np.mean(coherence_sq)),
        "m_alignment": m_alignment,
        "channels": channels,
        "ref_channel": channels[0],
        "f_peak_hz": actual_f,
        "m_candidates": m_candidates,
        "n_seg": n_seg,
    }


def load_ece_channels(shot, args, t_ms, channels=None):
    data_dir = Path(args.data_dir_pattern.format(shot=shot))
    channels = args.ece_channels if channels is None else channels
    ece_signals = {}
    missing = []
    for ch in channels:
        ece_file = data_dir / args.ece_file_pattern.format(ch=ch, shot=shot)
        if ece_file.exists():
            edf_ece = TE.edf()
            dat_ece = edf_ece.load(str(ece_file))
            t_ece = dat_ece[:, 0]
            if edf_ece.DimUnit[0] == 's':
                t_ece = t_ece * 1000.0
            ece_signals[ch] = np.interp(t_ms, t_ece, dat_ece[:, 1])
        else:
            missing.append(ch)
    return ece_signals, missing


def detect_saturated_channel(sig, rail_frac_threshold=0.02, plateau_run_threshold=20):
    sig = np.asarray(sig, dtype=float)
    finite = sig[np.isfinite(sig)]
    if finite.size == 0:
        return True, {"reason": "empty_or_nonfinite"}

    sig_max, sig_min = np.max(finite), np.min(finite)
    span = sig_max - sig_min
    if span == 0:
        return True, {"reason": "flat_channel", "rail_frac_hi": 1.0, "rail_frac_lo": 1.0, "max_flat_run": finite.size}

    rail_tol = 0.005 * span 
    rail_frac_hi = float(np.mean(finite >= (sig_max - rail_tol)))
    rail_frac_lo = float(np.mean(finite <= (sig_min + rail_tol)))

    flat_tol = 0.002 * span
    is_flat_step = np.abs(np.diff(finite)) <= flat_tol
    if is_flat_step.any():
        padded = np.concatenate(([0], is_flat_step.astype(np.int8), [0]))
        change_points = np.flatnonzero(np.diff(padded))
        run_lengths = change_points[1::2] - change_points[0::2]
        max_flat_run = int(run_lengths.max()) + 1 if run_lengths.size else 1
    else:
        max_flat_run = 1

    diagnostics = {"rail_frac_hi": rail_frac_hi, "rail_frac_lo": rail_frac_lo, "max_flat_run": max_flat_run}
    is_saturated = (
        rail_frac_hi > rail_frac_threshold
        or rail_frac_lo > rail_frac_threshold
        or max_flat_run > plateau_run_threshold
    )
    return is_saturated, diagnostics


def filter_saturated_channels(ece_signals, rail_frac_threshold=0.02, plateau_run_threshold=20):
    clean_signals = {}
    saturated_report = {}
    for ch, sig in ece_signals.items():
        is_sat, diag = detect_saturated_channel(sig, rail_frac_threshold, plateau_run_threshold)
        if is_sat:
            saturated_report[ch] = diag
        else:
            clean_signals[ch] = sig
    return clean_signals, saturated_report


def select_core_ece_channel(ece_signals, ech_power, t_ms, decimate_factor):
    ech_corr = anti_alias_decimate(ech_power, decimate_factor)
    per_channel_r = {}
    for ch, sig in ece_signals.items():
        sig_corr = anti_alias_decimate(sig, decimate_factor)
        if np.std(sig_corr) == 0 or np.std(ech_corr) == 0:
            per_channel_r[ch] = 0.0
            continue
        r_val, _ = stats.pearsonr(sig_corr, ech_corr)
        per_channel_r[ch] = r_val
    if not per_channel_r:
        return None, 0.0, {}
    best_channel = max(per_channel_r, key=lambda k: per_channel_r[k])
    return best_channel, per_channel_r[best_channel], per_channel_r


def zhong_distribution_function_analysis(shot, args, t_ms, envelope, ech_power, density_val,
                                          density_detected, mask_active_win, decimate_factor, dt_corr,
                                          chirp_rate_khz_per_ms=None):
    print("\n--- Energetic-Particle Distribution-Function Response Validation (Zhong et al. approach) [M6] ---")

    if args.ece_core_channel is not None:
        ece_signals, missing_ece = load_ece_channels(shot, args, t_ms, channels=[args.ece_core_channel])
        if not ece_signals:
            print(f"  ️ Requested core channel {args.ece_core_channel} was specified but its file is missing; skipping.")
            return None
        core_ch = args.ece_core_channel
        core_r = None
        print(f"  Using explicitly requested ECE channel {core_ch}.")
    else:
        ece_signals_raw, missing_ece = load_ece_channels(shot, args, t_ms)
        if missing_ece:
            print(f"  ️ Warning: {len(missing_ece)} of {len(args.ece_channels)} requested ECE channels not found.")
        if not ece_signals_raw:
            print("  ️ No ECE channels found for this shot; [M6] validation SKIPPED.")
            return None

        ece_signals, saturated_report = filter_saturated_channels(
            ece_signals_raw,
            rail_frac_threshold=args.sat_rail_frac_threshold,
            plateau_run_threshold=args.sat_plateau_run_threshold,
        )
        if saturated_report:
            print(f"  [SAT-DETECT] Excluded {len(saturated_report)} saturated channel(s): {list(saturated_report.keys())}")
        if not ece_signals:
            print("  ️ Every candidate ECE channel was flagged saturated; [M6] validation SKIPPED.")
            return None

        core_ch, core_r, per_channel_r = select_core_ece_channel(ece_signals, ech_power, t_ms, decimate_factor)
        print(f"  [M6-HEURISTIC] Auto-selected ECE channel {core_ch} as core-proxy (r vs. ECH power = {core_r:+.3f}).")

    ece_core = ece_signals[core_ch]

    te_core_ev = None
    if args.te_calib_scale_ev_per_v is not None:
        te_core_ev = args.te_calib_scale_ev_per_v * ece_core + args.te_calib_offset_ev

    envelope_corr = anti_alias_decimate(envelope, decimate_factor)
    ece_core_corr = anti_alias_decimate(ece_core, decimate_factor)
    lag_ece_ms, r_ece_peak, lags_ece_curve, corr_ece_curve = lagged_cross_correlation(
        envelope_corr, ece_core_corr, dt_corr, max_lag_ms=args.m6_max_lag_ms
    )
    print(f"  - Envelope vs. ECE-core-proxy: peak |correlation| = {r_ece_peak:+.4f} at lag = {lag_ece_ms:+.2f} ms")

    r_ece_sig, p_ece_std, p_ece_adj, n_ece_sig, N_eff_ece_sig = lagged_pearson_significance(
        envelope_corr, ece_core_corr, dt_corr, lag_ece_ms
    )
    if r_ece_sig is not None:
        meets_ece = abs(r_ece_sig) > 0.7 and p_ece_adj < 0.05
        print(f"    -> At that lag: proper Pearson r = {r_ece_sig:.4f}, p_std = {format_p_value(p_ece_std)}, p_adj = {format_p_value(p_ece_adj)} "
              f"-- {'MEETS' if meets_ece else 'does NOT meet'} |r|>0.7 & p<0.05.")
    else:
        p_ece_adj = 1.0

    pressure_proxy = None
    lag_pressure_ms, r_pressure_peak = None, None
    lags_pressure_curve, corr_pressure_curve = None, None
    if density_detected:
        pressure_proxy = density_val * ece_core
        pressure_proxy_corr = anti_alias_decimate(pressure_proxy, decimate_factor)
        lag_pressure_ms, r_pressure_peak, lags_pressure_curve, corr_pressure_curve = lagged_cross_correlation(
            envelope_corr, pressure_proxy_corr, dt_corr, max_lag_ms=args.m6_max_lag_ms
        )
        print(f"  - Envelope vs. (density x ECE-core-proxy) [pressure proxy]: peak |correlation| = {r_pressure_peak:+.4f} at lag = {lag_pressure_ms:+.2f} ms")
        r_pressure_sig, p_pressure_std, p_pressure_adj, n_pressure_sig, N_eff_pressure_sig = lagged_pearson_significance(
            envelope_corr, pressure_proxy_corr, dt_corr, lag_pressure_ms
        )
        if r_pressure_sig is not None:
            meets_pressure = abs(r_pressure_sig) > 0.7 and p_pressure_adj < 0.05
            print(f"    -> At that lag: proper Pearson r = {r_pressure_sig:.4f}, p_std = {format_p_value(p_pressure_std)}, p_adj = {format_p_value(p_pressure_adj)} "
                  f"-- {'MEETS' if meets_pressure else 'does NOT meet'} |r|>0.7 & p<0.05.")
        else:
            p_pressure_adj = 1.0
    else:
        r_pressure_sig, p_pressure_adj, N_eff_pressure_sig = None, 1.0, None

    lag_chirp_ece_ms, r_chirp_ece_peak = None, None
    lag_chirp_pressure_ms, r_chirp_pressure_peak = None, None
    lags_chirp_ece_curve, corr_chirp_ece_curve = None, None
    lags_chirp_pressure_curve, corr_chirp_pressure_curve = None, None
    if chirp_rate_khz_per_ms is not None and np.any(mask_active_win):
        chirp_active = chirp_rate_khz_per_ms[mask_active_win]
        chirp_corr = anti_alias_decimate(chirp_active, decimate_factor)
        ece_core_active_corr = anti_alias_decimate(ece_core[mask_active_win], decimate_factor)
        if len(chirp_corr) > 4 and np.std(chirp_corr) > 0:
            lag_chirp_ece_ms, r_chirp_ece_peak, lags_chirp_ece_curve, corr_chirp_ece_curve = lagged_cross_correlation(
                chirp_corr, ece_core_active_corr, dt_corr, max_lag_ms=args.m6_max_lag_ms
            )
            r_chirp_ece_sig, p_chirp_ece_std, p_chirp_ece_adj, n_chirp_ece_sig, N_eff_chirp_ece_sig = lagged_pearson_significance(
                chirp_corr, ece_core_active_corr, dt_corr, lag_chirp_ece_ms
            )
            if density_detected and pressure_proxy is not None:
                pressure_proxy_active_corr = anti_alias_decimate(pressure_proxy[mask_active_win], decimate_factor)
                lag_chirp_pressure_ms, r_chirp_pressure_peak, lags_chirp_pressure_curve, corr_chirp_pressure_curve = lagged_cross_correlation(
                    chirp_corr, pressure_proxy_active_corr, dt_corr, max_lag_ms=args.m6_max_lag_ms
                )
                r_chirp_pressure_sig, p_chirp_pressure_std, p_chirp_pressure_adj, n_chirp_pressure_sig, N_eff_chirp_pressure_sig = lagged_pearson_significance(
                    chirp_corr, pressure_proxy_active_corr, dt_corr, lag_chirp_pressure_ms
                )
            else:
                r_chirp_pressure_sig, p_chirp_pressure_adj, N_eff_chirp_pressure_sig = None, 1.0, None
        else:
            r_chirp_ece_sig, p_chirp_ece_adj, N_eff_chirp_ece_sig = None, 1.0, None
            r_chirp_pressure_sig, p_chirp_pressure_adj, N_eff_chirp_pressure_sig = None, 1.0, None
    else:
        r_chirp_ece_sig, p_chirp_ece_adj, N_eff_chirp_ece_sig = None, 1.0, None
        r_chirp_pressure_sig, p_chirp_pressure_adj, N_eff_chirp_pressure_sig = None, 1.0, None

    tau_s_ms = None
    if te_core_ev is not None and density_detected:
        mask_scaling_win = mask_active_win
        te_active_ev = np.clip(te_core_ev[mask_scaling_win], 1.0, None)
        ne_active_cm3 = np.clip(density_val[mask_scaling_win], 0.01, None) * 1e19 * 1e-6
        A_b = 1.0 if args.beam_species == "H" else 2.0
        Z_b = 1.0
        ln_lambda = np.clip(24.0 - np.log(np.sqrt(ne_active_cm3) / te_active_ev), 5.0, 25.0)
        tau_s_s = 6.27e8 * A_b * te_active_ev**1.5 / (Z_b**2 * ne_active_cm3 * ln_lambda)
        tau_s_ms = float(np.mean(tau_s_s)) * 1000.0

    mask = mask_active_win
    t_plot = t_ms[mask]
    env_plot = envelope[mask]
    ech_plot = ech_power[mask]

    fig, axs = plt.subplots(1, 4 if density_detected else 3, figsize=(24 if density_detected else 18, 5))

    axs[0].plot(t_ms, ece_core, color='teal', alpha=0.8, label=f'ECE ch.{core_ch} (core-proxy, raw V)')
    ax0_twin = axs[0].twinx()
    ax0_twin.plot(t_ms, envelope, color='red', alpha=0.7, label='Mode Envelope')
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel("ECE-core-proxy (raw V)", color='teal')
    ax0_twin.set_ylabel("Envelope (V)", color='red')
    axs[0].set_title(f"Shot {shot}: ECE-core-proxy (ch.{core_ch}) & Mode Envelope (80-120 kHz)")
    axs[0].grid(True, alpha=0.3)

    sc = axs[1].scatter(ech_plot, env_plot, c=t_plot, cmap='viridis', s=6)
    axs[1].plot(ech_plot, env_plot, color='gray', alpha=0.15, linewidth=0.5)
    plt.colorbar(sc, ax=axs[1], label='Time (ms)')
    axs[1].set_xlabel("ECH Power (raw)")
    axs[1].set_ylabel("Mode Envelope (V)")
    axs[1].set_title("Envelope vs. ECH Power (80-120 kHz)\n(time-colored; loop = delayed/hysteretic response)")
    axs[1].grid(True, alpha=0.3)

    ax_lag = axs[3] if density_detected else axs[2]
    ax_lag.plot(lags_ece_curve, corr_ece_curve, color='teal', label='vs. ECE-core-proxy')
    ax_lag.axvline(lag_ece_ms, color='teal', linestyle=':', alpha=0.7)
    if lags_pressure_curve is not None:
        ax_lag.plot(lags_pressure_curve, corr_pressure_curve, color='darkorange', label='vs. pressure proxy')
        ax_lag.axvline(lag_pressure_ms, color='darkorange', linestyle=':', alpha=0.7)
    if lags_chirp_ece_curve is not None:
        ax_lag.plot(lags_chirp_ece_curve, corr_chirp_ece_curve, color='slateblue', linestyle='--',
                    label='chirp rate vs. ECE-core-proxy')
        ax_lag.axvline(lag_chirp_ece_ms, color='slateblue', linestyle=':', alpha=0.7)
    if lags_chirp_pressure_curve is not None:
        ax_lag.plot(lags_chirp_pressure_curve, corr_chirp_pressure_curve, color='darkgreen', linestyle='--',
                    label='chirp rate vs. pressure proxy')
        ax_lag.axvline(lag_chirp_pressure_ms, color='darkgreen', linestyle=':', alpha=0.7)
    ax_lag.axvspan(args.m6_max_lag_ms * 0.9, args.m6_max_lag_ms, color='red', alpha=0.08)
    ax_lag.axvspan(-args.m6_max_lag_ms, -args.m6_max_lag_ms * 0.9, color='red', alpha=0.08,
                   label='boundary zone')
    ax_lag.set_xlabel("Lag (ms)")
    ax_lag.set_ylabel("Normalized cross-correlation")
    ax_lag.set_title("Lag-correlation curves (80-120 kHz)")
    ax_lag.legend(loc='best', fontsize=8)
    ax_lag.grid(True, alpha=0.3)

    if density_detected:
        pressure_plot = pressure_proxy[mask]
        sc2 = axs[2].scatter(pressure_plot, env_plot, c=t_plot, cmap='viridis', s=6)
        axs[2].plot(pressure_plot, env_plot, color='gray', alpha=0.15, linewidth=0.5)
        plt.colorbar(sc2, ax=axs[2], label='Time (ms)')
        axs[2].set_xlabel("Density x ECE-core-proxy (pressure proxy, raw units)")
        axs[2].set_ylabel("Mode Envelope (V)")
        axs[2].set_title("Envelope vs. Pressure Proxy (80-120 kHz)\n(time-colored)")
        axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_png = f"mhd_analysis_objective2_zhong_{shot}_80_120kHz.png"
    plt.savefig(output_png, dpi=150)
    plt.close(fig)
    print(f"  Zhong-et-al. 80-120 kHz figure saved to: '{output_png}'")

    return {
        "core_ece_channel": core_ch,
        "core_ece_channel_r_vs_ech": core_r,
        "lag_ece_ms": lag_ece_ms,
        "r_ece_peak": r_ece_peak,
        "r_ece_sig": r_ece_sig, "p_ece_adj": p_ece_adj, "N_eff_ece_sig": N_eff_ece_sig,
        "lag_pressure_ms": lag_pressure_ms,
        "r_pressure_peak": r_pressure_peak,
        "r_pressure_sig": r_pressure_sig, "p_pressure_adj": p_pressure_adj, "N_eff_pressure_sig": N_eff_pressure_sig,
        "tau_s_ms": tau_s_ms,
        "lag_chirp_ece_ms": lag_chirp_ece_ms,
        "r_chirp_ece_peak": r_chirp_ece_peak,
        "r_chirp_ece_sig": r_chirp_ece_sig, "p_chirp_ece_adj": p_chirp_ece_adj, "N_eff_chirp_ece_sig": N_eff_chirp_ece_sig,
        "lag_chirp_pressure_ms": lag_chirp_pressure_ms,
        "r_chirp_pressure_peak": r_chirp_pressure_peak,
        "r_chirp_pressure_sig": r_chirp_pressure_sig, "p_chirp_pressure_adj": p_chirp_pressure_adj,
        "N_eff_chirp_pressure_sig": N_eff_chirp_pressure_sig,
    }


def load_obj1_reference_window(shot, args, fl_hz, fu_hz):
    if args.disable_obj1_reference:
        return None

    search_dirs = []
    if args.obj1_results_dir:
        search_dirs.append(Path(args.obj1_results_dir))
    search_dirs.append(Path("."))

    json_path = None
    for d in search_dirs:
        candidate = d / args.obj1_json_pattern.format(shot=shot)
        if candidate.exists():
            json_path = candidate
            break

    if json_path is None:
        return None

    try:
        with open(json_path) as fjson:
            obj1_data = json.load(fjson)
    except Exception:
        return None

    modes = obj1_data.get("discrete_modes", [])
    tol_hz = args.obj1_mode_freq_tol_khz * 1000.0
    band_center_hz = 0.5 * (fl_hz + fu_hz)
    in_band = [m for m in modes if (fl_hz - tol_hz) <= m.get("frequency_hz", -1e30) <= (fu_hz + tol_hz)]
    if not in_band:
        return None

    confirmed = [m for m in in_band if m.get("dual_criterion_pass")]
    pool = confirmed if confirmed else in_band

    chosen = min(pool, key=lambda m: abs(m.get("frequency_hz", 0.0) - band_center_hz))
    intervals = chosen.get("active_intervals_ms", [])
    if not intervals:
        return None

    start_ms, end_ms = max(intervals, key=lambda iv: iv[1] - iv[0])
    return start_ms, end_ms


def detect_flat_frequency_subwindow(ifreq_khz, t_ms, dt, i0_domain, i1_domain, args):
    n_domain = i1_domain - i0_domain
    fallback = (i0_domain, i1_domain, {"used_fallback": True, "reason": "domain too short"})

    if args.flat_window_start is not None and args.flat_window_end is not None:
        lo = max(args.flat_window_start, t_ms[i0_domain])
        hi = min(args.flat_window_end, t_ms[i1_domain - 1])
        if hi <= lo:
            return fallback
        idx = np.where((t_ms >= lo) & (t_ms <= hi))[0]
        i0f, i1f = int(idx[0]), int(idx[-1]) + 1
        return i0f, i1f, {"used_fallback": False, "manual": True}

    scan_samples = max(3, int(round(args.flat_scan_window_ms / (dt * 1000.0))))
    if scan_samples >= n_domain:
        return fallback

    ifreq_domain = ifreq_khz[i0_domain:i1_domain]
    t_domain = t_ms[i0_domain:i1_domain]

    slope_raw = np.gradient(ifreq_domain, t_domain)
    smooth_samples = max(1, int(round(args.flat_slope_smooth_ms / (dt * 1000.0))))
    if smooth_samples % 2 == 0:
        smooth_samples += 1
    if 3 <= smooth_samples < n_domain:
        kernel = np.ones(smooth_samples) / smooth_samples
        slope_smooth = np.convolve(slope_raw, kernel, mode='same')
    else:
        slope_smooth = slope_raw
    slope_abs = np.abs(slope_smooth)

    scan_kernel = np.ones(scan_samples) / scan_samples
    window_means = np.convolve(slope_abs, scan_kernel, mode='valid')
    best_start = int(np.argmin(window_means))
    best_end = best_start + scan_samples
    best_mean = float(window_means[best_start])

    tol_mean = best_mean * (1.0 + args.flat_growth_tolerance)
    cumsum = np.concatenate(([0.0], np.cumsum(slope_abs)))

    def window_mean(a, b):
        return (cumsum[b] - cumsum[a]) / (b - a)

    start, end = best_start, best_end
    while start > 0 and window_mean(start - 1, end) <= tol_mean:
        start -= 1
    while end < n_domain and window_mean(start, end + 1) <= tol_mean:
        end += 1

    duration_ms = (end - start) * dt * 1000.0
    touches_edge = (start == 0) or (end == n_domain)
    grown_mean = float(window_mean(start, end))

    i0_flat, i1_flat = i0_domain + start, i0_domain + end

    if duration_ms < args.flat_min_duration_ms:
        return fallback

    return i0_flat, i1_flat, {
        "used_fallback": False, "manual": False, "scan_mean_khz_per_ms": best_mean,
        "grown_mean_khz_per_ms": grown_mean, "duration_ms": duration_ms, "touches_edge": touches_edge,
    }


def process_shot(shot, args):
    data_dir = Path(args.data_dir_pattern.format(shot=shot))
    mag_file = data_dir / f"MP1@{shot}.edf"

    print(f"\n{'=' * 93}")
    print(f"--- Loading Mirnov Coil magnetic signal: {mag_file} (Shot {shot}, 80-120 kHz Primary Mode) ---")
    if not mag_file.exists():
        print(f"  ️ Warning: file {mag_file} does not exist; skipping shot {shot}.")
        return None

    edf_mag = TE.edf()
    dat_mag = edf_mag.load(str(mag_file))
    t = dat_mag[:, 0]
    ys = dat_mag[:, 1]

    if edf_mag.DimUnit[0] == 'ms':
        t_sec = t / 1000.0
    else:
        t_sec = t

    dt = (t_sec[100] - t_sec[0]) / 100.0
    fs = 1.0 / dt
    t_ms = t_sec * 1000.0
    print(f"Mirnov signal: {len(ys)} points, fs = {fs/1e6:.2f} MHz (Time range: {t_ms[0]:.1f} - {t_ms[-1]:.1f} ms)")

    print(f"\nApplying Bessel bandpass filter ({args.lower:.1f} - {args.upper:.1f} kHz) and Hilbert Transform...")
    fl_hz = args.lower * 1000.0
    fu_hz = args.upper * 1000.0

    envelope, phase, filtered, ifreq_hz = extract_instantaneous_frequency(
        ys, fs, fl_hz, fu_hz, args.order, args.smoothing
    )
    ifreq_khz = ifreq_hz / 1000.0

    probe_signals = {"MP1": {"filtered": filtered, "envelope": envelope}}
    probe_missing = []
    for probe in ("MP3", "MP4"):
        probe_file = data_dir / f"{probe}@{shot}.edf"
        if probe_file.exists():
            edf_probe = TE.edf()
            dat_probe = edf_probe.load(str(probe_file))
            t_probe = dat_probe[:, 0]
            t_probe_ms = t_probe if edf_probe.DimUnit[0] == 'ms' else t_probe * 1000.0
            ys_probe = np.interp(t_ms, t_probe_ms, dat_probe[:, 1])
            envelope_probe, _, filtered_probe, _ = extract_instantaneous_frequency(
                ys_probe, fs, fl_hz, fu_hz, args.order, args.smoothing
            )
            probe_signals[probe] = {"filtered": filtered_probe, "envelope": envelope_probe}
            print(f"  [M8] Loaded {probe}@{shot}.edf ({len(ys_probe)} points) and extracted carrier + Hilbert envelope.")
        else:
            probe_missing.append(probe)

    if not args.disable_poloidal_array:
        pmp_signals, pmp_plab_rad = load_poloidal_array(
            shot, data_dir, t_ms, invert_channels=tuple(args.pmp_invert_channels)
        )
        if not args.pmp_skip_self_test and pmp_plab_rad:
            poloidal_array_self_test(pmp_plab_rad, max_m_test=args.pmp_self_test_max_m)
    else:
        pmp_signals, pmp_plab_rad = {}, {}

    # Mode-active mask
    if args.mode_active_start is not None and args.mode_active_end is not None:
        mask_mode_active = (t_ms >= args.mode_active_start) & (t_ms <= args.mode_active_end)
    else:
        med_env = float(np.median(envelope))
        mad_env = float(np.median(np.abs(envelope - med_env))) * 1.4826
        mode_active_threshold = med_env + args.mode_active_k * mad_env
        mask_raw_threshold = envelope > mode_active_threshold

        gap_samples = max(1, int(round(args.mode_active_max_gap_ms * 1e-3 / dt)))
        structure = np.ones(2 * gap_samples + 1, dtype=bool)
        mask_closed = ndi.binary_closing(mask_raw_threshold, structure=structure)
        labeled, n_components = ndi.label(mask_closed)
        if n_components == 0:
            mask_mode_active = np.zeros_like(mask_raw_threshold)
        else:
            sizes = ndi.sum(mask_closed, labeled, index=np.arange(1, n_components + 1))
            best_label = int(np.argmax(sizes)) + 1
            mask_burst = labeled == best_label
            burst_duration_ms = float(np.sum(mask_burst)) * dt * 1000.0
            if burst_duration_ms < args.mode_active_min_duration_ms:
                mask_mode_active = np.zeros_like(mask_raw_threshold)
            else:
                mask_mode_active = mask_burst

    n_mode_active = int(np.sum(mask_mode_active))
    if n_mode_active >= 200:
        active_idx_mode = np.where(mask_mode_active)[0]
        i0_mode, i1_mode = int(active_idx_mode[0]), int(active_idx_mode[-1]) + 1
        print(f"  Mode-active span: {t_ms[i0_mode]:.1f}-{t_ms[i1_mode-1]:.1f} ms ({n_mode_active} samples).")
    else:
        active_idx_mode = np.array([], dtype=int)
        i0_mode, i1_mode = 0, 0
        print(f"  ️ Only {n_mode_active} mode-active samples found.")

    # Flat-frequency window
    if n_mode_active >= 200:
        obj1_ref = load_obj1_reference_window(shot, args, fl_hz, fu_hz)
        if obj1_ref is not None:
            ref_start_ms, ref_end_ms = obj1_ref
            lo = max(ref_start_ms, t_ms[i0_mode])
            hi = min(ref_end_ms, t_ms[i1_mode - 1])
            if hi > lo:
                idx_ref = np.where((t_ms >= lo) & (t_ms <= hi))[0]
                i0_domain, i1_domain = int(idx_ref[0]), int(idx_ref[-1]) + 1
            else:
                i0_domain, i1_domain = i0_mode, i1_mode
        else:
            i0_domain, i1_domain = i0_mode, i1_mode

        i0_flat, i1_flat, flat_info = detect_flat_frequency_subwindow(
            ifreq_khz, t_ms, dt, i0_domain, i1_domain, args
        )
        n_flat_active = i1_flat - i0_flat
    else:
        i0_flat, i1_flat, flat_info = i0_mode, i1_mode, {"used_fallback": True}
        n_flat_active = 0

    # Heating channels
    ech_file = data_dir / f"ECHRG500@{shot}.edf"
    ech_power = np.zeros_like(t_ms)
    if ech_file.exists():
        edf_ech = TE.edf()
        dat_ech = edf_ech.load(str(ech_file))
        t_ech = dat_ech[:, 0] * (1000.0 if edf_ech.DimUnit[0] == 's' else 1.0)
        ech_power = np.interp(t_ms, t_ech, dat_ech[:, 1])

    nbi_channels = ["NBIS3I", "NBIS4I", "NBIS9I", "NBIS10I"]
    nbi_signals = {}
    total_nbi_power = np.zeros_like(t_ms)
    nbi_mismatch_detected = False
    nbi_turn_on_times = {}

    for nbi in nbi_channels:
        nbi_file = data_dir / f"{nbi}@{shot}.edf"
        if nbi_file.exists():
            edf_nbi = TE.edf()
            dat_nbi = edf_nbi.load(str(nbi_file))
            t_nbi = dat_nbi[:, 0] * (1000.0 if edf_nbi.DimUnit[0] == 's' else 1.0)
            nbi_val = np.interp(t_ms, t_nbi, dat_nbi[:, 1])
            nbi_signals[nbi] = nbi_val
            total_nbi_power += nbi_val

            active_idx = np.where(dat_nbi[:, 1] > 0.5)[0]
            if len(active_idx) > 0:
                t_on = dat_nbi[active_idx[0], 0] * (1000.0 if edf_nbi.DimUnit[0] == 's' else 1.0)
                nbi_turn_on_times[nbi] = t_on
                if t_on > t_ms[-1]:
                    nbi_mismatch_detected = True

    # Density
    nave_file = data_dir / f"nave@{shot}.edf"
    density_val = np.zeros_like(t_ms)
    density_detected = False
    if nave_file.exists():
        edf_den = TE.edf()
        dat_den = edf_den.load(str(nave_file))
        t_den = dat_den[:, 0] * (1000.0 if edf_den.DimUnit[0] == 's' else 1.0)
        raw_density = dat_den[:, 1]
        invalid_mask = raw_density <= 0.0
        if np.any(invalid_mask):
            valid_idx = np.where(~invalid_mask)[0]
            if len(valid_idx) > 0:
                raw_density = np.interp(np.arange(len(raw_density)), valid_idx, raw_density[valid_idx])
            else:
                raw_density = np.clip(raw_density, 1e-5, None)

        med_kernel = args.nave_medfilt if args.nave_medfilt % 2 == 1 else args.nave_medfilt + 1
        if med_kernel >= 3 and len(raw_density) > med_kernel:
            raw_density_clean = dsp.medfilt(raw_density, kernel_size=med_kernel)
        else:
            raw_density_clean = raw_density
        density_val = np.interp(t_ms, t_den, raw_density_clean)

        ech_glitch_mask = (t_ms >= args.ech_glitch_start) & (t_ms <= args.ech_glitch_end)
        if np.any(ech_glitch_mask):
            mask_before = t_ms < args.ech_glitch_start
            mask_after = t_ms > args.ech_glitch_end
            pre_val = density_val[mask_before][-1] if np.any(mask_before) else density_val[0]
            post_val = density_val[mask_after][0] if np.any(mask_after) else density_val[-1]
            density_val[ech_glitch_mask] = np.linspace(pre_val, post_val, np.sum(ech_glitch_mask))
        density_detected = True

    # Correlations
    decimate_factor = 100
    dt_corr = dt * decimate_factor
    t_corr = t_ms[::decimate_factor]
    envelope_corr = anti_alias_decimate(envelope, decimate_factor)
    total_nbi_corr = anti_alias_decimate(total_nbi_power, decimate_factor)
    ech_power_corr = anti_alias_decimate(ech_power, decimate_factor)

    acf_full = estimate_acf(envelope_corr, nlags=50)
    sum_acf_full = np.sum(acf_full[1:])
    N_full = len(envelope_corr)
    N_eff_full = max(3.0, min(float(N_full), N_full / (1.0 + 2.0 * sum_acf_full)))

    r_nbi_tot, p_nbi_tot_std = stats.pearsonr(envelope_corr, total_nbi_corr) if np.max(total_nbi_corr) > 0 else (0.0, 1.0)
    p_nbi_tot = conservative_p_value(r_nbi_tot, N_eff_full) if np.max(total_nbi_corr) > 0 else 1.0

    r_ech, p_ech_std = stats.pearsonr(envelope_corr, ech_power_corr)
    p_ech = conservative_p_value(r_ech, N_eff_full)

    mask_active_win = (t_ms >= args.ech_active_start) & (t_ms <= args.ech_active_end)
    envelope_corr_active = anti_alias_decimate(envelope[mask_active_win], decimate_factor)
    ech_power_corr_active = anti_alias_decimate(ech_power[mask_active_win], decimate_factor)

    acf_active = estimate_acf(envelope_corr_active, nlags=50)
    nlags_act = min(50, len(envelope_corr_active) - 2)
    sum_acf_active = np.sum(acf_active[1:nlags_act + 1]) if nlags_act > 0 else 0.0
    N_active = len(envelope_corr_active)
    N_eff_active = max(3.0, min(float(N_active), N_active / (1.0 + 2.0 * sum_acf_active)))

    r_ech_active, p_ech_active_std = stats.pearsonr(envelope_corr_active, ech_power_corr_active) if len(envelope_corr_active) > 2 else (0.0, 1.0)
    p_ech_active = conservative_p_value(r_ech_active, N_eff_active) if len(envelope_corr_active) > 2 else 1.0

    step_regressor = (ech_power_corr > 0.5 * np.max(ech_power_corr)).astype(float) if np.max(ech_power_corr) > 0 else np.zeros_like(ech_power_corr)
    r_xy = r_ech
    r_xz, _ = stats.pearsonr(envelope_corr, step_regressor) if np.max(step_regressor) > 0 else (0.0, 1.0)
    r_yz, _ = stats.pearsonr(ech_power_corr, step_regressor) if np.max(step_regressor) > 0 else (0.0, 1.0)

    denom_partial = np.sqrt((1.0 - r_xz**2) * (1.0 - r_yz**2))
    r_partial = (r_xy - r_xz * r_yz) / denom_partial if denom_partial > 0 else 0.0
    p_partial_adj = conservative_p_value(r_partial, N_eff_full, n_control=1)

    nbi_corrs = {}
    for nbi, nbi_val in nbi_signals.items():
        nbi_corr_val = anti_alias_decimate(nbi_val, decimate_factor)
        r_val, p_val_std = stats.pearsonr(envelope_corr, nbi_corr_val) if np.max(nbi_corr_val) > 0 else (0.0, 1.0)
        p_val_adj = conservative_p_value(r_val, N_eff_full) if np.max(nbi_corr_val) > 0 else 1.0
        nbi_corrs[nbi] = (r_val, p_val_std, p_val_adj)

    lag_nbi_ms, r_nbi_peak, _, _ = lagged_cross_correlation(
        envelope_corr, total_nbi_corr, dt_corr, max_lag_ms=20.0
    ) if np.max(total_nbi_corr) > 0 else (0.0, 0.0, np.array([0.0]), np.array([0.0]))
    lag_ech_ms, r_ech_peak, _, _ = lagged_cross_correlation(
        envelope_corr, ech_power_corr, dt_corr, max_lag_ms=20.0
    )

    # Frequency vs heating
    freq_heating_results = None
    if n_mode_active >= 200:
        ifreq_active_mode = ifreq_khz[mask_mode_active]
        nbi_active_mode = total_nbi_power[mask_mode_active]
        ech_active_mode = ech_power[mask_mode_active]

        ifreq_freq_corr = anti_alias_decimate(ifreq_active_mode, decimate_factor)
        nbi_freq_corr = anti_alias_decimate(nbi_active_mode, decimate_factor)
        ech_freq_corr = anti_alias_decimate(ech_active_mode, decimate_factor)

        acf_freq = estimate_acf(ifreq_freq_corr, nlags=min(50, len(ifreq_freq_corr) - 2))
        N_freq = len(ifreq_freq_corr)
        N_eff_freq = max(3.0, min(float(N_freq), N_freq / (1.0 + 2.0 * np.sum(acf_freq[1:])))) if N_freq > 2 else 3.0

        if not nbi_mismatch_detected and np.max(nbi_freq_corr) > 0:
            r_ifreq_nbi, _ = stats.pearsonr(ifreq_freq_corr, nbi_freq_corr)
            p_ifreq_nbi = conservative_p_value(r_ifreq_nbi, N_eff_freq)
        else:
            r_ifreq_nbi, p_ifreq_nbi = 0.0, 1.0

        r_ifreq_ech, _ = stats.pearsonr(ifreq_freq_corr, ech_freq_corr) if np.std(ech_freq_corr) > 0 else (0.0, 1.0)
        p_ifreq_ech = conservative_p_value(r_ifreq_ech, N_eff_freq)

        max_ech_freq_corr = np.max(ech_freq_corr) if len(ech_freq_corr) else 0.0
        step_regressor_freq = (ech_freq_corr > 0.5 * max_ech_freq_corr).astype(float) if max_ech_freq_corr > 0 else np.zeros_like(ech_freq_corr)
        if np.std(step_regressor_freq) > 0 and np.std(ifreq_freq_corr) > 0:
            r_xz_freq, _ = stats.pearsonr(ifreq_freq_corr, step_regressor_freq)
            r_yz_freq, _ = stats.pearsonr(ech_freq_corr, step_regressor_freq)
            denom_partial_freq = np.sqrt(max(0.0, (1.0 - r_xz_freq ** 2) * (1.0 - r_yz_freq ** 2)))
            r_partial_freq = (r_ifreq_ech - r_xz_freq * r_yz_freq) / denom_partial_freq if denom_partial_freq > 0 else 0.0
            p_partial_freq_adj = conservative_p_value(r_partial_freq, N_eff_freq, n_control=1)
        else:
            r_partial_freq, p_partial_freq_adj = r_ifreq_ech, p_ifreq_ech

        lag_ifreq_nbi_ms, r_ifreq_nbi_peak, _, _ = lagged_cross_correlation(
            ifreq_freq_corr, nbi_freq_corr, dt_corr, max_lag_ms=args.m7_max_lag_ms
        ) if (not nbi_mismatch_detected and np.max(nbi_freq_corr) > 0) else (0.0, 0.0, np.array([0.0]), np.array([0.0]))
        lag_ifreq_ech_ms, r_ifreq_ech_peak, _, _ = lagged_cross_correlation(
            ifreq_freq_corr, ech_freq_corr, dt_corr, max_lag_ms=args.m7_max_lag_ms
        )

        freq_heating_results = {
            "r_ifreq_nbi": r_ifreq_nbi, "p_ifreq_nbi": p_ifreq_nbi,
            "r_ifreq_ech": r_ifreq_ech, "p_ifreq_ech": p_ifreq_ech,
            "r_partial_freq": r_partial_freq, "p_partial_freq_adj": p_partial_freq_adj,
            "lag_ifreq_nbi_ms": lag_ifreq_nbi_ms, "r_ifreq_nbi_peak": r_ifreq_nbi_peak,
            "lag_ifreq_ech_ms": lag_ifreq_ech_ms, "r_ifreq_ech_peak": r_ifreq_ech_peak,
        }

    # Alfven scaling
    f_theoretical_alfven = np.zeros_like(t_ms)
    r_val_scaling, p_val_scaling = 0.0, 1.0
    bfield_available = False
    b_is_constant = False
    if density_detected:
        density_val_clean = np.clip(density_val, 0.01, None)
        if args.bfield_constant_tesla is not None:
            b_val = np.full_like(t_ms, args.bfield_constant_tesla)
            bfield_available = True
            b_is_constant = True
            mu0, m_ion = 4 * np.pi * 1e-7, args.ion_mass_amu * 1.6726e-27
            n_i_m3 = density_val_clean * 1e19
            f_alfven_scaling = np.abs(b_val) / np.sqrt(mu0 * n_i_m3 * m_ion)
        else:
            f_alfven_scaling = 1.0 / np.sqrt(density_val_clean)

        mask_scaling_win = (t_ms >= args.alfven_cal_start) & (t_ms <= args.alfven_cal_end)
        mean_measured_f = np.mean(ifreq_khz[mask_scaling_win]) if np.sum(mask_scaling_win) >= 5 else np.mean(ifreq_khz)
        mean_scaling_val = np.mean(f_alfven_scaling[mask_scaling_win]) if np.sum(mask_scaling_win) >= 5 else np.mean(f_alfven_scaling)
        norm_constant = mean_measured_f / mean_scaling_val if mean_scaling_val > 0 else 1.0
        f_theoretical_alfven = f_alfven_scaling * norm_constant

        ifreq_active = ifreq_khz[mask_active_win]
        f_theoretical_alfven_active = f_theoretical_alfven[mask_active_win]
        ifreq_active_corr = anti_alias_decimate(ifreq_active, decimate_factor)
        f_theo_active_corr = anti_alias_decimate(f_theoretical_alfven_active, decimate_factor)

        r_val_scaling, p_val_scaling_std = stats.pearsonr(ifreq_active_corr, f_theo_active_corr) if len(ifreq_active_corr) > 2 else (0.0, 1.0)
        acf_alfven = estimate_acf(ifreq_active_corr, nlags=50)
        nlags_alf = min(50, len(ifreq_active_corr) - 2)
        sum_acf_alfven = np.sum(acf_alfven[1:nlags_alf + 1]) if nlags_alf > 0 else 0.0
        N_alfven = len(ifreq_active_corr)
        N_eff_alfven = max(3.0, min(float(N_alfven), N_alfven / (1.0 + 2.0 * sum_acf_alfven)))
        p_val_scaling = conservative_p_value(r_val_scaling, N_eff_alfven) if len(ifreq_active_corr) > 2 else 1.0

    # Inter-probe coherence
    if n_mode_active >= 200 and n_flat_active >= 200:
        t_sec_probe_span = t_sec[i0_flat:i1_flat]
        probe_signals_active = {
            probe: {k: v[i0_flat:i1_flat] for k, v in sig.items()}
            for probe, sig in probe_signals.items()
        }
        nfft_probe = min(args.nfft, max(64, 2 ** int(np.floor(np.log2(
            max(8, (i1_flat - i0_flat) // max(args.ensemble, 1)))))))
        carrier_coh_results = compute_probe_pair_coherence(
            probe_signals_active, "filtered", PROBE_PAIRS, t_sec_probe_span, dt, args, nfft=nfft_probe
        )
        envelope_coh_results = compute_probe_pair_coherence(
            probe_signals_active, "envelope", PROBE_PAIRS, t_sec_probe_span, dt, args, nfft=nfft_probe
        )
    else:
        carrier_coh_results = {pair: None for pair in PROBE_PAIRS}
        envelope_coh_results = {pair: None for pair in PROBE_PAIRS}

    # Poloidal mode-number (m) decomposition
    if n_mode_active >= 200 and n_flat_active >= 200 and len(pmp_signals) >= 3:
        poloidal_result = poloidal_mode_number_analysis(
            pmp_signals, pmp_plab_rad, dt, i0_flat, i1_flat, fl_hz, fu_hz, args
        )
        if poloidal_result is not None:
            print(f"  [M10] Dominant poloidal mode number (80-120 kHz): m = {poloidal_result['m_dominant']:+d} at {poloidal_result['f_peak_khz']:.2f} kHz")
    else:
        poloidal_result = None

    # Phase-structure verification: cross-spectral phase at the dominant frequency
    phase_structure_result = None
    if poloidal_result is not None and poloidal_result["f_peak_khz"] is not None:
        f_peak_for_phase = poloidal_result["f_peak_khz"] * 1000.0
        phase_structure_result = poloidal_phase_structure_analysis(
            pmp_signals, pmp_plab_rad, dt, i0_flat, i1_flat, f_peak_for_phase, args
        )
        if phase_structure_result is not None:
            align_dom = phase_structure_result.get("m_alignment", {}).get(poloidal_result["m_dominant"], {"r_circ": 0.0, "mean_error_deg": 90.0})
            is_conf = (align_dom["r_circ"] >= 0.70) and (align_dom["mean_error_deg"] <= 45.0) and (phase_structure_result["mean_coherence"] >= 0.40)
            status_str = f"CONFIRMED (r_circ = {align_dom['r_circ']:.2f}, error = {align_dom['mean_error_deg']:.1f} deg)" if is_conf else f"UNCONFIRMED / NOISY FIT (r_circ = {align_dom['r_circ']:.2f} < 0.70, error = {align_dom['mean_error_deg']:.1f} deg > 45 deg)"
            print(f"  [M10-PHASE] Phase-structure double-verification at {phase_structure_result['f_peak_hz']/1000:.2f} kHz "
                  f"(m = {poloidal_result['m_dominant']:+d}, ref: {phase_structure_result['ref_channel']}, mean gamma^2: {phase_structure_result['mean_coherence']:.2f}):\n"
                  f"    -> Verdict: {status_str}")

    chirp_rate_khz_per_ms = dsp.savgol_filter(ifreq_khz, args.smoothing, 2, deriv=1) / (dt * 1000.0)

    # Zhong et al. analysis
    zhong_results = zhong_distribution_function_analysis(
        shot, args, t_ms, envelope, ech_power, density_val, density_detected,
        mask_active_win, decimate_factor, dt_corr, chirp_rate_khz_per_ms
    )

    # Multiple comparisons correction
    test_labels = ["Full ECH", "Partial ECH (Step)", f"Active ECH ({args.ech_active_start:.0f}-{args.ech_active_end:.0f}ms)"]
    test_pvalues = [p_ech, p_partial_adj, p_ech_active]
    test_rvalues = [r_ech, r_partial, r_ech_active]

    if not nbi_mismatch_detected:
        test_labels.insert(0, "Total NBI")
        test_pvalues.insert(0, p_nbi_tot)
        test_rvalues.insert(0, r_nbi_tot)
        for nbi, corr_tuple in nbi_corrs.items():
            test_labels.append(f"Individual NBI ({nbi})")
            test_pvalues.append(corr_tuple[2])
            test_rvalues.append(corr_tuple[0])

    if density_detected:
        test_labels.append("Alfven (f_measured vs f_theoretical)")
        test_pvalues.append(p_val_scaling)
        test_rvalues.append(r_val_scaling)

    if freq_heating_results is not None:
        test_labels.append("Freq vs NBI (mode-active) [M7]")
        test_pvalues.append(freq_heating_results["p_ifreq_nbi"])
        test_rvalues.append(freq_heating_results["r_ifreq_nbi"])
        test_labels.append("Freq vs ECH (mode-active) [M7]")
        test_pvalues.append(freq_heating_results["p_ifreq_ech"])
        test_rvalues.append(freq_heating_results["r_ifreq_ech"])
        test_labels.append("Freq vs ECH, step-ctrl [M7]")
        test_pvalues.append(freq_heating_results["p_partial_freq_adj"])
        test_rvalues.append(freq_heating_results["r_partial_freq"])

    if zhong_results is not None:
        if zhong_results.get("r_ece_sig") is not None:
            test_labels.append("[M6] Envelope vs ECE-proxy")
            test_pvalues.append(zhong_results["p_ece_adj"])
            test_rvalues.append(zhong_results["r_ece_sig"])
        if zhong_results.get("r_pressure_sig") is not None:
            test_labels.append("[M6] Envelope vs pressure proxy")
            test_pvalues.append(zhong_results["p_pressure_adj"])
            test_rvalues.append(zhong_results["r_pressure_sig"])

    n_tests = len(test_pvalues)
    alpha_bonferroni = 0.05 / n_tests
    bh_significant, bh_p_adjusted = benjamini_hochberg(test_pvalues, alpha=0.05)

    print(f"\n--- Multiple Comparisons Correction (n = {n_tests} tests, 80-120 kHz Primary Mode) ---")
    for label, r_v, p_v, p_bh, sig_bh in zip(test_labels, test_rvalues, test_pvalues, bh_p_adjusted, bh_significant):
        sig_bonf = p_v < alpha_bonferroni
        print(f"    {label:<32} r = {r_v:+.3f}, {format_p_value(p_v):<16} "
              f"Bonferroni: {'Sig.' if sig_bonf else 'Not sig.':<8} | "
              f"FDR-BH: p_adj = {p_bh:.3e} ({'Sig.' if sig_bh else 'Not sig.'})")

    # Generating 6-panel per-shot plot
    print(f"\n--- Generating 80-120 kHz Primary Mode Results Plot ---")
    n_panels = 7
    fig, axs = plt.subplots(n_panels, 1, figsize=(12.5, 15 * n_panels / 4.0), sharex=False)

    axs[0].plot(t_ms, filtered, color='gray', alpha=0.5, label='Filtered Signal (Bessel Bandpass 80-120 kHz)')
    axs[0].plot(t_ms, envelope, color='red', linewidth=1.8, label='Hilbert Envelope')
    axs[0].set_ylabel("Fluctuation Voltage (V)")
    axs[0].set_title(f"Primary EPM Mode Isolation via Bessel Filter (80-120 kHz) and Hilbert Envelope - Shot {shot}")
    axs[0].legend(loc='upper right')
    axs[0].grid(True, alpha=0.3)

    color = 'tab:blue'
    axs[1].plot(t_ms, ifreq_khz, color=color, alpha=0.6, label='Measured Instantaneous Freq.')
    if density_detected:
        scaling_label = (r"$\propto B/\sqrt{n_e m_i}$ (full physical)" if bfield_available
                          else r"$\propto 1/\sqrt{n_e}$ (SIMPLIFIED, no B)")
        axs[1].plot(t_ms, f_theoretical_alfven, color='tab:red', linestyle='--', linewidth=2.0,
                    label=fr'Theoretical Alfven Scaling ({scaling_label}, r={r_val_scaling:.2f})')
        axs[1].axvspan(args.alfven_cal_start, args.alfven_cal_end, color='gray', alpha=0.15,
                        label=f'Calibration window ({args.alfven_cal_start:.0f}-{args.alfven_cal_end:.0f} ms)')
    if n_mode_active >= 200:
        axs[1].axvspan(t_ms[i0_mode], t_ms[i1_mode - 1], color='blue', alpha=0.06,
                        label='Mode-active window [M7]')
        if n_flat_active >= 200 and not flat_info.get("used_fallback"):
            axs[1].axvspan(t_ms[i0_flat], t_ms[i1_flat - 1], color='darkgreen', alpha=0.18,
                            label='Flat-frequency sub-window [M9]')
    axs[1].set_ylabel("Frequency (kHz)", color=color)
    axs[1].tick_params(axis='y', labelcolor=color)
    axs[1].set_ylim(args.lower - 5, args.upper + 5)
    axs[1].set_title("Physical Validation: Measured Instantaneous Frequency vs. Theoretical Alfven Scaling (80-120 kHz)")
    axs[1].grid(True, alpha=0.3)

    ax1_twin = axs[1].twinx()
    colors_nbi = {"NBIS3I": "darkorange", "NBIS4I": "green", "NBIS9I": "purple", "NBIS10I": "brown"}
    for nbi, val in nbi_signals.items():
        max_val = np.max(val)
        if max_val > 0.5:
            norm_val = val / max_val
            ax1_twin.plot(t_ms, norm_val, color=colors_nbi[nbi], alpha=0.8, linestyle='-',
                          label=f"{nbi} (Norm. Max: {max_val:.2f}A)")
        else:
            ax1_twin.plot(t_ms, val, color=colors_nbi[nbi], alpha=0.25, linestyle=':',
                          label=f"{nbi} (Inactive)")

    max_ech = np.max(ech_power)
    norm_ech = ech_power / max_ech if max_ech > 0 else ech_power
    ax1_twin.plot(t_ms, norm_ech, color='magenta', linewidth=2, linestyle='--',
                  label=f"ECH (Norm. Max: {max_ech:.2f}V)")
    ax1_twin.set_ylabel("Normalized Heating (a.u.)", color='magenta')
    ax1_twin.tick_params(axis='y', labelcolor='magenta')
    ax1_twin.set_ylim(-0.1, 1.1)

    lines, labels = axs[1].get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    axs[1].legend(lines + lines2, labels + labels2, loc='upper right', fontsize=9)

    axs[2].plot(t_corr, envelope_corr / np.max(envelope_corr) if np.max(envelope_corr) > 0 else envelope_corr,
                color='red', label='Normalized Envelope (80-120 kHz)', linewidth=2)
    axs[2].plot(t_corr, total_nbi_corr / np.max(total_nbi_corr) if np.max(total_nbi_corr) > 0 else total_nbi_corr,
                color='darkorange', label=f"Normalized Aggregate NBI{' (Inactive)' if nbi_mismatch_detected else ''}", linestyle='-.', linewidth=2)
    axs[2].plot(t_corr, ech_power_corr / np.max(ech_power_corr) if np.max(ech_power_corr) > 0 else ech_power_corr,
                color='magenta', label='Normalized ECH', linestyle=':', linewidth=2)
    axs[2].set_ylabel("Normalized Amplitude (a.u.)")
    axs[2].set_xlabel("Time (ms)")
    axs[2].set_title(f"Time Synchronization and Pearson Correlations (80-120 kHz)\n"
                      f"(Lagged peak: NBI @ {lag_nbi_ms:+.1f} ms, ECH @ {lag_ech_ms:+.1f} ms)")
    axs[2].legend(loc='upper right')
    axs[2].grid(True, alpha=0.3)

    # Panel 4: Carrier coherence
    for (pa, pb) in PROBE_PAIRS:
        res = carrier_coh_results[(pa, pb)]
        if res is None:
            continue
        f_pair, mean_coh2_pair = res
        mask_f_pair = (f_pair >= fl_hz) & (f_pair <= fu_hz)
        style = PROBE_PAIR_STYLES[(pa, pb)]
        axs[3].plot(f_pair[mask_f_pair] / 1000.0, mean_coh2_pair[mask_f_pair], linewidth=2,
                    label=f'{pa}-{pb} Carrier Coherence', **style)
    axs[3].axhline(y=0.5, color='black', linestyle=':', label='Significance Threshold (0.5)')
    axs[3].set_title("Cross-Spectral Coherence Between Mirnov Probes: Carrier-Mode Oscillations [M8] (80-120 kHz)")
    axs[3].set_xlabel("Frequency (kHz)")
    axs[3].set_ylabel(r"Coherence $\gamma^2$")
    axs[3].set_xlim(args.lower, args.upper)
    axs[3].set_ylim(0, 1.05)
    axs[3].grid(True, alpha=0.3, which='both', linestyle=':')
    axs[3].legend(loc='upper right')

    # Panel 5: Envelope coherence
    for (pa, pb) in PROBE_PAIRS:
        res = envelope_coh_results[(pa, pb)]
        if res is None:
            continue
        f_pair, mean_coh2_pair = res
        mask_f_pair = (f_pair >= 0) & (f_pair <= 10000.0)
        style = PROBE_PAIR_STYLES[(pa, pb)]
        axs[4].plot(f_pair[mask_f_pair] / 1000.0, mean_coh2_pair[mask_f_pair], linewidth=2,
                    label=f'{pa}-{pb} Envelope Coherence', **style)
    axs[4].axhline(y=0.5, color='black', linestyle=':', label='Significance Threshold (0.5)')
    axs[4].set_title("Cross-Spectral Coherence Between Probe Envelopes: Modal-Structure Estimation [M8] (80-120 kHz)")
    axs[4].set_xlabel("Modulation Frequency (kHz)")
    axs[4].set_ylabel(r"Coherence $\gamma^2$")
    axs[4].set_ylim(0, 1.05)
    axs[4].grid(True, alpha=0.3, which='both', linestyle=':')
    axs[4].legend(loc='upper right')

    # Panel 6: Poloidal mode number
    if poloidal_result is not None:
        pcm = axs[5].pcolormesh(
            poloidal_result["f_band_khz_plot"], poloidal_result["k_grid_plot"],
            np.log10(poloidal_result["P2d_plot"] + 1e-30), cmap='jet', shading='auto'
        )
        cb = plt.colorbar(pcm, ax=axs[5])
        cb.set_label(r"log$_{10}$ Poloidal Power (a.u.)")
        axs[5].axhline(y=poloidal_result["m_dominant"], color='white', linestyle='--', linewidth=1.4,
                        label=f"Dominant m = {poloidal_result['m_dominant']:+d}")
        axs[5].set_xlim(args.lower, args.upper)
        axs[5].legend(loc='upper right')
    axs[5].set_title(f"Poloidal Mode-Number Decomposition (PMP1-PMP14 Array) [M10] (80-120 kHz Primary Mode)")
    axs[5].set_xlabel("Frequency (kHz)")
    axs[5].set_ylabel("Poloidal Mode Number m")

    # -----------------------------------------------------------------------------------
    # Panel 7: Poloidal phase structure verification (phase vs. theta in radians)
    # -----------------------------------------------------------------------------------
    window_str = f"{t_ms[i0_flat]:.1f}-{t_ms[i1_flat - 1]:.1f} ms" if n_flat_active >= 200 else "N/A"
    if phase_structure_result is not None:
        ps = phase_structure_result
        # Map physical poloidal angles into [-pi, pi] radians for contiguous spatial ordering
        theta_plot_rad = np.arctan2(np.sin(ps["theta_rad"]), np.cos(ps["theta_rad"]))
        phase_meas = ps["measured_phase"]
        m_dom = poloidal_result["m_dominant"] if poloidal_result is not None else 0

        # Plot theoretical lines for candidate m values over [-pi, pi]
        th_theory_grid = np.linspace(-np.pi, np.pi, 1200)
        # Display selected candidate m numbers
        candidate_ms_to_show = sorted(set([-4, -3, -2, -1, 1, 2, 3, 4, m_dom]))
        for m_val in candidate_ms_to_show:
            ph_theory = np.arctan2(np.sin(m_val * th_theory_grid), np.cos(m_val * th_theory_grid))
            # Insert NaNs at wrap discontinuities (|diff| > pi) to avoid vertical/diagonal connecting lines
            diff_ph = np.abs(np.diff(ph_theory))
            jump_indices = np.where(diff_ph > np.pi)[0]
            th_plot_line = th_theory_grid.copy()
            ph_plot_line = ph_theory.copy()
            if len(jump_indices) > 0:
                th_plot_line = np.insert(th_plot_line, jump_indices + 1, np.nan)
                ph_plot_line = np.insert(ph_plot_line, jump_indices + 1, np.nan)

            is_dom = (m_val == m_dom)
            alpha_line = 0.95 if is_dom else 0.25
            lw = 2.2 if is_dom else 0.8
            color = 'red' if is_dom else 'gray'
            label = f"m = {m_val:+d} (Dominant)" if is_dom else (f"m = {m_val:+d}" if abs(m_val) <= 3 else None)
            axs[6].plot(th_plot_line, ph_plot_line, color=color, alpha=alpha_line,
                        linewidth=lw, label=label, zorder=3 if is_dom else 1)

        # Plot error bars (Bendat & Piersol standard error of phase)
        axs[6].errorbar(theta_plot_rad, phase_meas, yerr=ps["sigma_phase"],
                        fmt='none', ecolor='tab:blue', elinewidth=1.4, capsize=3.5, capthick=1.0,
                        alpha=0.75, zorder=4, label=r'Phase uncertainty $\pm 1\sigma_\phi$ (Bendat & Piersol)')

        # Plot measured data points
        axs[6].scatter(theta_plot_rad, phase_meas, s=80, c='blue', marker='o', edgecolors='black',
                        linewidths=1.2, zorder=5, label=f"Measured phase (ref: {ps['ref_channel']})")

        # Label each probe
        for i, ch in enumerate(ps["channels"]):
            axs[6].annotate(ch, (theta_plot_rad[i], phase_meas[i]),
                            textcoords="offset points", xytext=(5, 6),
                            fontsize=7, color='darkblue', fontweight='bold')

        # Statistical phase alignment verdict
        align_info = ps.get("m_alignment", {}).get(m_dom, {"r_circ": 0.0, "mean_error_deg": 90.0})
        r_circ_dom = align_info["r_circ"]
        mean_err_dom = align_info["mean_error_deg"]
        is_phase_confirmed = (r_circ_dom >= 0.70) and (mean_err_dom <= 45.0) and (ps["mean_coherence"] >= 0.40)

        # Add diagnostic text badge
        if is_phase_confirmed:
            axs[6].text(0.02, 0.05, f"[CONFIRMED] Statistically Validated $m = {m_dom:+d}$ Fit\n"
                        f"Alignment $r_{{circ}}$ = {r_circ_dom:.2f} (>= 0.70), Mean Error = {mean_err_dom:.1f} deg (<= 45 deg)",
                        transform=axs[6].transAxes, fontsize=8.5, color='darkgreen', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='honeydew', edgecolor='darkgreen', alpha=0.9))
        else:
            axs[6].text(0.02, 0.05, f"[UNCONFIRMED / NOISY FIT] $m = {m_dom:+d}$ Failed Verification\n"
                        f"Alignment $r_{{circ}}$ = {r_circ_dom:.2f} (< 0.70), Mean Error = {mean_err_dom:.1f} deg (> 45 deg)",
                        transform=axs[6].transAxes, fontsize=8.5, color='darkred', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='linen', edgecolor='darkred', alpha=0.9))

        axs[6].set_xlim(-np.pi * 1.05, np.pi * 1.05)
        axs[6].set_ylim(-np.pi * 1.15, np.pi * 1.15)
        axs[6].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        axs[6].set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
        axs[6].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        axs[6].set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
        axs[6].axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
        axs[6].axvline(x=0, color='black', linewidth=0.5, alpha=0.3)
        axs[6].set_title(f"Poloidal Phase Structure Verification [M10-PHASE] (80-120 kHz Primary Mode)\n"
                          f"(cross-spectral phase at {ps['f_peak_hz']/1000:.2f} kHz, "
                          f"ref: {ps['ref_channel']}, mean $\\gamma^2$ = {ps['mean_coherence']:.2f}, window: {window_str})")
        axs[6].set_xlabel(r"Poloidal Angle $\theta$ (rad)")
        axs[6].set_ylabel(r"Cross-Spectral Phase $\phi$ (rad)")
        axs[6].grid(True, alpha=0.3)
        axs[6].legend(loc='upper right', fontsize=8, ncol=2)
    else:
        placeholder_msg = ("Phase structure unavailable\n(no poloidal NUDFT result)" if poloidal_result is None
                            else "Phase structure unavailable\n(insufficient data)")
        axs[6].text(0.5, 0.5, placeholder_msg, ha='center', va='center',
                    transform=axs[6].transAxes, fontsize=11, color='gray')
        axs[6].set_title("Poloidal Phase Structure Verification [M10-PHASE] (80-120 kHz)")
        axs[6].set_xlabel(r"Poloidal Angle $\theta$ (rad)")
        axs[6].set_ylabel(r"Cross-Spectral Phase $\phi$ (rad)")

    plt.tight_layout()
    output_png = f"mhd_analysis_objective2_{shot}_80_120kHz.png"
    plt.savefig(output_png, dpi=150)
    plt.close(fig)
    print(f"Results successfully saved to: '{output_png}'")

    return {
        "shot": shot,
        "nbi_mismatch_detected": nbi_mismatch_detected,
        "density_detected": density_detected,
        "r_nbi_tot": r_nbi_tot,
        "p_nbi_tot": p_nbi_tot,
        "r_ech": r_ech,
        "p_ech": p_ech,
        "r_partial": r_partial,
        "r_val_scaling": r_val_scaling,
        "p_val_scaling": p_val_scaling,
        "lag_nbi_ms": lag_nbi_ms,
        "r_nbi_peak": r_nbi_peak,
        "lag_ech_ms": lag_ech_ms,
        "r_ech_peak": r_ech_peak,
        "envelope_corr": envelope_corr,
        "total_nbi_corr": total_nbi_corr,
        "ech_power_corr": ech_power_corr,
        "dt_corr": dt_corr,
        "zhong_results": zhong_results,
        "freq_heating_results": freq_heating_results,
        "n_mode_active": n_mode_active,
        "poloidal_m_dominant": poloidal_result["m_dominant"] if poloidal_result is not None else None,
        "poloidal_f_peak_khz": poloidal_result["f_peak_khz"] if poloidal_result is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description="EPM Primary Mode (80-120 kHz) vs. Real Multichannel Heating Correlation Analysis")
    parser.add_argument("--shots", type=int, nargs="+", default=SHOTS_DEFAULT,
                         help=f"List of shot numbers to analyze (def: {SHOTS_DEFAULT})")
    parser.add_argument("--data-dir-pattern", type=str, default="data/hj{shot}",
                         help="Data directory pattern (def: data/hj{shot})")
    parser.add_argument("-l", "--lower", type=float, default=DEFAULT_LOWER_KHZ, help=f"Bandpass filter lower frequency in kHz (def: {DEFAULT_LOWER_KHZ})")
    parser.add_argument("-u", "--upper", type=float, default=DEFAULT_UPPER_KHZ, help=f"Bandpass filter upper frequency in kHz (def: {DEFAULT_UPPER_KHZ})")
    parser.add_argument("-o", "--order", type=int, default=4, help="Bessel filter order (def: 4)")
    parser.add_argument("-s", "--smoothing", type=int, default=OPTIMAL_SG_WIN, help=f"Savitzky-Golay smoothing window (def: {OPTIMAL_SG_WIN})")
    parser.add_argument("-n", "--nfft", type=int, default=1024, help="FFT size for cross-coherence (def: 1024)")
    parser.add_argument("-e", "--ensemble", type=int, default=10, help="Number of ensembles for cross-coherence (def: 10)")
    parser.add_argument("--pmp-nfft", type=int, default=256, help="FFT size for poloidal spectrogram (def: 256)")
    parser.add_argument("--pmp-max-mode-number", type=int, default=6, help="Max mode number search (def: 6)")
    parser.add_argument("--pmp-max-mode-expanded", type=int, default=20, help="Cap for auto-expanded mode search (def: 20)")
    parser.add_argument("--pmp-skip-self-test", action="store_true", help="Skip poloidal self test")
    parser.add_argument("--pmp-self-test-max-m", type=int, default=10, help="Poloidal self test max m (def: 10)")
    parser.add_argument("--pmp-invert-channels", type=str, nargs="*", default=list(PMP_INVERT_CHANNELS_DEFAULT))
    parser.add_argument("--disable-poloidal-array", action="store_true")
    parser.add_argument("--nave-medfilt", type=int, default=5)
    parser.add_argument("--alfven-cal-start", type=float, default=250.0)
    parser.add_argument("--alfven-cal-end", type=float, default=270.0)
    parser.add_argument("--mode-active-k", type=float, default=6.0)
    parser.add_argument("--mode-active-max-gap-ms", type=float, default=3.0)
    parser.add_argument("--mode-active-min-duration-ms", type=float, default=10.0)
    parser.add_argument("--flat-slope-smooth-ms", type=float, default=2.0)
    parser.add_argument("--flat-scan-window-ms", type=float, default=8.0)
    parser.add_argument("--flat-growth-tolerance", type=float, default=0.5)
    parser.add_argument("--flat-min-duration-ms", type=float, default=5.0)
    parser.add_argument("--flat-window-start", type=float, default=None)
    parser.add_argument("--flat-window-end", type=float, default=None)
    parser.add_argument("--obj1-results-dir", type=str, default=".")
    parser.add_argument("--obj1-json-pattern", type=str, default="discrete_modes_shot_{shot}.json")
    parser.add_argument("--obj1-mode-freq-tol-khz", type=float, default=3.0)
    parser.add_argument("--disable-obj1-reference", action="store_true")
    parser.add_argument("--mode-active-start", type=float, default=None)
    parser.add_argument("--mode-active-end", type=float, default=None)
    parser.add_argument("--m7-max-lag-ms", type=float, default=40.0)
    parser.add_argument("--bfield-constant-tesla", type=float, default=1.25)
    parser.add_argument("--ion-mass-amu", type=float, default=1.0)
    parser.add_argument("--ech-active-start", type=float, default=170.0)
    parser.add_argument("--ech-active-end", type=float, default=290.0)
    parser.add_argument("--ech-glitch-start", type=float, default=170.0)
    parser.add_argument("--ech-glitch-end", type=float, default=190.0)
    parser.add_argument("--ece-channels", type=int, nargs="+", default=list(range(1, 17)))
    parser.add_argument("--ece-core-channel", type=int, default=None)
    parser.add_argument("--sat-rail-frac-threshold", type=float, default=0.02)
    parser.add_argument("--sat-plateau-run-threshold", type=int, default=20)
    parser.add_argument("--ece-file-pattern", type=str, default="ECE{ch}FAST@{shot}.edf")
    parser.add_argument("--beam-species", type=str, choices=["H", "D"], default="H")
    parser.add_argument("--beam-energy-kev", type=float, default=30.0)
    parser.add_argument("--m6-max-lag-ms", type=float, default=60.0)
    parser.add_argument("--te-calib-scale-ev-per-v", type=float, default=None)
    parser.add_argument("--te-calib-offset-ev", type=float, default=0.0)
    args = parser.parse_args()

    print(f"=== Objective 2 Primary Mode (80-120 kHz) Analysis: shots {args.shots} ===")
    results_list = []
    for shot in args.shots:
        result = process_shot(shot, args)
        results_list.append(result)

    n_ok = sum(1 for r in results_list if r is not None)
    n_fail = len(results_list) - n_ok
    print(f"\n{'=' * 93}")
    print(f"Per-shot processing complete: {n_ok} shot(s) succeeded, {n_fail} shot(s) skipped.")


if __name__ == "__main__":
    main()

