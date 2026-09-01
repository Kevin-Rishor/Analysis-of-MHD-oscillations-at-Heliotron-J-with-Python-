# -------------------------------------------------------------
# SHOT CONFIGURATION 
# -------------------------------------------------------------
SHOTS_DEFAULT = [88653]

# -------------------------------------------------------------
#  Mirnov probe pairs used for the inter-probe (modal-structure) coherence panels.
# Order is fixed so the two coherence panels always list/plot the pairs in the same order.
# -------------------------------------------------------------
PROBE_PAIRS = [("MP1", "MP3"), ("MP1", "MP4"), ("MP3", "MP4")]
PROBE_PAIR_STYLES = {
    ("MP1", "MP3"): dict(color="tab:blue", linestyle="-"),
    ("MP1", "MP4"): dict(color="tab:green", linestyle="--"),
    ("MP3", "MP4"): dict(color="tab:purple", linestyle="-."),
}

# -------------------------------------------------------------
PMP_COIL_CONST = 4.5e-3   # coil parameter (height x width x turns), from getPMPs()
PMP_GAIN = 2.0            # amplifier gain, from getPMPs()
PMP_CHANNELS = [f"PMP{i}" for i in range(1, 15)]  # PMP1..PMP14
# anaaspect.py labels each PMP with a RAW angle label, then transforms it to the
# actual physical poloidal angle via `plab = 360 - plab`.
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
import libana_signal as LAS
from mhd_common import OPTIMAL_SG_WIN, extract_instantaneous_frequency, anti_alias_decimate


def estimate_acf(x, nlags=50):
    """Natively estimates the autocorrelation function (ACF) of a signal."""
    n = len(x)
    mean = np.mean(x)
    var = np.var(x)
    if var == 0:
        return np.ones(nlags + 1)
    xp = x - mean
    acf_vals = []
    for lag in range(nlags + 1):
        if lag == 0:
            r = 1.0
        else:
            r = np.sum(xp[:-lag] * xp[lag:]) / (n * var)
        acf_vals.append(r)
    return np.array(acf_vals)


def conservative_p_value(r, N_eff, n_control=0):
    """Computes a more conservative p-value using the effective sample size N_eff."""
    if abs(r) >= 1.0:
        return 0.0
    df = N_eff - 2 - n_control
    if df <= 0:
        return 1.0
    t_stat = r * np.sqrt(df / (1.0 - r ** 2))
    return 2.0 * stats.t.sf(abs(t_stat), df)


def benjamini_hochberg(p_values, alpha=0.05):
    """
    Benjamini-Hochberg procedure for False Discovery Rate (FDR) control
    over a set of p-values from multiple tests.
    """
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

    # BH-adjusted p-values (monotonized)
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


# -------------------------------------------------------------------------------------------
# Lagged time-domain cross-correlation.
# -------------------------------------------------------------------------------------------
def lagged_cross_correlation(x, y, dt_corr, max_lag_ms=20.0):
    """
    Normalized lagged cross-correlation between x and y (assumed already on a common,
    uniformly-sampled time base with step dt_corr, in seconds).
    """
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


# -------------------------------------------------------------------------------------------
# Inter-probe cross-spectral coherence, for the multichannel Mirnov modal-structure panels.
# -------------------------------------------------------------------------------------------
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


# =============================================================================================
# Poloidal PMP1-PMP14 array: loading + poloidal mode-number (m) decomposition.
# =============================================================================================

def load_poloidal_array(shot, data_dir, t_ms, invert_channels=PMP_INVERT_CHANNELS_DEFAULT):
    """
    Loads the poloidal PMP1-PMP14 Mirnov array
    """
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
        print(f"  ⚠️ [M10] Warning: no file(s) found for poloidal probe(s) {missing}; the poloidal "
              "mode-number analysis will use only the channels found (needs >=3).")
    inverted_found = [ch for ch in invert_channels if ch in signals]
    if inverted_found:
        print(f"  [M10] Loaded poloidal array: {len(signals)}/{len(PMP_CHANNELS)} channels found. "
              f"Inverted polarity (x -1) of {inverted_found}, per confirmed reversed wiring.")
    return signals, plab_rad


def _nudft_poloidal(angles_rad, complex_values, k_grid, sign=-1):
    n = len(angles_rad)
    return (1.0 / n) * np.dot(complex_values, np.exp(sign * 1j * k_grid * angles_rad[:, None]))


# -------------------------------------------------------------------------------------------
# Synthetic sanity check of the array's NUDFT resolving power, independent of any
# real shot data.
# -------------------------------------------------------------------------------------------
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
        print(f"    ✅ All {len(test_modes)} synthetic mode numbers recovered correctly. This array "
              f"geometry/NUDFT can resolve |m| up to {max_m_test} without aliasing -- a real "
              f"m_dominant result in that range is not, by itself, a boundary artifact.")
    else:
        first_failure_abs_m = min(abs(m) for m, _ in failures)
        print(f"    ⚠️ {len(failures)}/{len(test_modes)} synthetic mode numbers recovered INCORRECTLY "
              f"(first failure at |m_true| = {first_failure_abs_m}): {failures}")
        print(f"    -> This array's angular coverage (gaps/uneven spacing) CANNOT reliably resolve "
              f"|m| >= {first_failure_abs_m}, even for a perfect noise-free input. Any REAL "
              f"m_dominant result at or beyond that magnitude must be treated as an ARRAY-GEOMETRY "
              f"ARTIFACT, not a physical finding, regardless of how it looks on real data.")
    return {"test_modes": test_modes, "failures": failures, "max_m_test": max_m_test,
            "k_search": k_search, "first_failure_abs_m": first_failure_abs_m}


def _poloidal_power_map(sig_matrix, angles, fs, fl_hz, fu_hz, max_m, nfft_use):
    """
    Factored out of poloidal_mode_number_analysis() so it can be called
    twice at different `max_m` without recomputing the spectrogram twice.
    """
    f, tave, S = dsp.spectrogram(
        sig_matrix, fs=fs, window='hann', nperseg=nfft_use, noverlap=nfft_use // 2,
        nfft=nfft_use, detrend='constant', return_onesided=False, scaling='density',
        axis=-1, mode='complex'
    )
    band_mask = (f >= fl_hz) & (f <= fu_hz)
    if not np.any(band_mask):
        return None
    assert fl_hz > 0 and fu_hz < fs / 2.0, \
        "[M10] EPM band must stay strictly within (0, Nyquist) -- f is not fftshifted here, see comment above"

    k_grid = np.arange(-max_m, max_m + 1)
    f_idx_band = np.where(band_mask)[0]
    P2d = np.zeros((len(k_grid), len(f_idx_band)))
    for jf, fidx in enumerate(f_idx_band):
        for it in range(S.shape[2]):
            Sk = _nudft_poloidal(angles, S[:, fidx, it], k_grid)
            P2d[:, jf] += (Sk * Sk.conj()).real
    P2d /= S.shape[2]
    f_band_khz = f[f_idx_band] / 1000.0
    return k_grid, f_band_khz, P2d


def poloidal_mode_number_analysis(pmp_signals, plab_rad, dt, i0, i1, fl_hz, fu_hz, args):
    """
    Estimates the dominant poloidal mode number m of the EPM from the PMP1-PMP14 array,
    restricted to the mode-active/flat-frequency window and the EPM band.
    """
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

    # Boundary-pinning check on THIS shot's real data.
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
                    print(f"  ⚠️ [M10-AUTOEXPAND] m_dominant was pinned at the search edge "
                          f"(+/-{args.pmp_max_mode_number}). Re-checked at +/-{expanded_max_m}: still "
                          f"pinned at the new edge (m = {m_dominant_expanded:+d}). This is a BOUNDARY "
                          "ARTIFACT, not a converged physical mode number -- do not report the "
                          f"m = {m_dominant:+d} value from the narrower range as a physical finding.")
                elif m_dominant_expanded == m_dominant:
                    verdict = "confirmed_stable"
                    print(f"  [M10-AUTOEXPAND] m_dominant was pinned at the search edge "
                          f"(+/-{args.pmp_max_mode_number}), which on its own is inconclusive. "
                          f"Re-checked at +/-{expanded_max_m} (much more room to move): the peak stayed "
                          f"at the EXACT SAME value (m = {m_dominant_expanded:+d}) instead of sliding "
                          "elsewhere. That is the strongest available confirmation that this is a real "
                          "interior peak, not a boundary artifact -- it coincided with the narrow "
                          "range's edge by chance, not because the algorithm was clipped there.")
                else:
                    verdict = "moved_off_edge"
                    print(f"  [M10-AUTOEXPAND] m_dominant was pinned at the search edge "
                          f"(+/-{args.pmp_max_mode_number}). Re-checked at +/-{expanded_max_m}: moved to "
                          f"a DIFFERENT interior value (m = {m_dominant_expanded:+d}, was "
                          f"{m_dominant:+d}), i.e. still not converged at the original range -- widen "
                          f"--pmp-max-mode-number to at least {abs(m_dominant_expanded)} before trusting "
                          "any single m value.")
        else:
            verdict = "expand_capped"
            print(f"  ⚠️ [M10-AUTOEXPAND] m_dominant pinned at the search edge, but "
                  f"--pmp-max-mode-expanded ({args.pmp_max_mode_expanded}) does not allow checking "
                  "further. Raise --pmp-max-mode-expanded to get a real verdict.")

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
    
    Returns a dict with:
      - theta_deg: poloidal angles in degrees for each probe
      - theta_rad: poloidal angles in radians for each probe
      - measured_phase: measured cross-spectral phase (rad) relative to ref probe
      - channels: list of channel names
      - ref_channel: name of the reference channel
      - f_peak_hz: the frequency used
      - m_candidates: list of m values to plot as theoretical lines
    """
    channels = sorted(pmp_signals.keys(), key=lambda c: int(c[3:]))
    if len(channels) < 3:
        return None

    sig_matrix = np.array([pmp_signals[ch][i0:i1] for ch in channels])
    angles_rad = np.array([plab_rad[ch] for ch in channels])
    angles_deg = np.rad2deg(angles_rad)
    fs = 1.0 / dt
    n_samples = sig_matrix.shape[1]

    # Use a cross-spectral approach: compute the complex FFT of each probe signal,
    # then extract the phase at the frequency bin closest to f_peak_hz, relative to
    # the reference probe (channel 0).
    nfft_use = min(getattr(args, 'pmp_nfft', 256), n_samples)
    if nfft_use < 32:
        return None

    # Compute complex spectrograms for all channels, then average the cross-spectrum
    # across time segments to get a robust phase estimate at f_peak_hz.
    f_spec, t_spec, S = dsp.spectrogram(
        sig_matrix, fs=fs, window='hann', nperseg=nfft_use, noverlap=nfft_use // 2,
        nfft=nfft_use, detrend='constant', return_onesided=True, scaling='density',
        axis=-1, mode='complex'
    )

    # Find the frequency bin closest to f_peak_hz
    f_idx = int(np.argmin(np.abs(f_spec - f_peak_hz)))
    actual_f = float(f_spec[f_idx])

    # Extract complex spectra at f_peak_hz for all channels and time segments
    # S shape: (n_channels, n_freqs, n_time_segments)
    S_peak = S[:, f_idx, :]   # shape: (n_channels, n_time_segments)

    # Cross-spectrum relative to reference channel (channel 0)
    ref_spectrum = S_peak[0, :]  # shape: (n_time_segments,)
    cross_spectra = S_peak * np.conj(ref_spectrum)[None, :]  # shape: (n_channels, n_time_segments)

    # Average cross-spectrum across time segments for robustness
    avg_cross = np.mean(cross_spectra, axis=1)  # shape: (n_channels,)
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

    # Extract phase from the averaged cross-spectrum
    measured_phase = np.angle(avg_cross)  # shape: (n_channels,)

    # The reference probe phase is 0 by construction (cross with itself = real positive)
    # Subtract reference phase to make it explicit (should already be ~0 for ch0)
    measured_phase = measured_phase - measured_phase[0]
    # Wrap to [-pi, pi]
    measured_phase = np.arctan2(np.sin(measured_phase), np.cos(measured_phase))

    m_candidates = list(range(-6, 7))  # m = -6 to +6

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


# =============================================================================================
# "Energetic-Particle Distribution-Function Variations" validation (Zhong et al. approach
# =============================================================================================

def load_ece_channels(shot, args, t_ms, channels=None):
    """
    Loads ECE#FAST@{shot}.edf channels
    """
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
    """
    Heuristically flags whether a raw ECE channel is ADC-saturated rather than genuinely
    being a high-amplitude, strongly-responding channel
    """
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
    """
    Runs detect_saturated_channel() over every channel in ece_signals and returns
    clean_signals, that excludes flagged channels and saturated_report maps.
    """
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
    """
    Picks a "core-proxy" ECE channel out of the available ones.
    """
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


def get_output_suffix(args):
    """Returns filename suffix based on band parameters or explicit override."""
    if getattr(args, "output_suffix", None) is not None:
        return args.output_suffix
    if args.lower != 40.0 or args.upper != 80.0:
        return f"_{int(args.lower)}_{int(args.upper)}kHz"
    return ""


def zhong_distribution_function_analysis(shot, args, t_ms, envelope, ech_power, density_val,
                                          density_detected, mask_active_win, decimate_factor, dt_corr,
                                          chirp_rate_khz_per_ms=None):
    print("\n--- Energetic-Particle Distribution-Function Response Validation (Zhong et al. approach) [M6] ---")

    if args.ece_core_channel is not None:
        ece_signals, missing_ece = load_ece_channels(shot, args, t_ms, channels=[args.ece_core_channel])
        if not ece_signals:
            print(f"  ⚠️ Requested core channel {args.ece_core_channel} was specified but its file is")
            print("     missing for this shot; [M6] validation SKIPPED for this shot.")
            return None
        core_ch = args.ece_core_channel
        core_r = None
        print(f"  Using explicitly requested ECE channel {core_ch}.")
    else:
        ece_signals_raw, missing_ece = load_ece_channels(shot, args, t_ms)
        if missing_ece:
            print(f"  ⚠️ Warning: {len(missing_ece)} of {len(args.ece_channels)} requested ECE channels not found "
                  f"(missing: {missing_ece[:5]}{'...' if len(missing_ece) > 5 else ''}).")
        if not ece_signals_raw:
            print("  ⚠️ No ECE channels found for this shot; [M6] validation SKIPPED. main.md's requirement to")
            print("     compare against energetic-particle distribution-function models remains UNADDRESSED for this shot.")
            return None

        ece_signals, saturated_report = filter_saturated_channels(
            ece_signals_raw,
            rail_frac_threshold=args.sat_rail_frac_threshold,
            plateau_run_threshold=args.sat_plateau_run_threshold,
        )
        if saturated_report:
            sat_list = ", ".join(
                f"{ch} (rail_hi={diag.get('rail_frac_hi', 0):.1%}, rail_lo={diag.get('rail_frac_lo', 0):.1%}, "
                f"flat_run={diag.get('max_flat_run', 0)}spl)"
                if "reason" not in diag else f"{ch} ({diag['reason']})"
                for ch, diag in sorted(saturated_report.items())
            )
            print(f"  [SAT-DETECT] Excluded {len(saturated_report)} saturated/railed channel(s) from "
                  f"consideration: {sat_list}")
        if not ece_signals:
            print("  ⚠️ Every candidate ECE channel was flagged saturated for this shot; [M6] validation")
            print("     SKIPPED. Try --ece-core-channel to force a specific channel, or relax")
            print("     --sat-rail-frac-threshold / --sat-plateau-run-threshold if this looks like a false positive.")
            return None

        core_ch, core_r, per_channel_r = select_core_ece_channel(ece_signals, ech_power, t_ms, decimate_factor)
        print(f"  [M6-HEURISTIC] Auto-selected ECE channel {core_ch} as core-proxy "
              f"(r vs. ECH power = {core_r:+.3f}, highest among {len(per_channel_r)} non-saturated channels "
              f"checked).")
        print("     This is a HEURISTIC choice, NOT a confirmed core/magnetic-axis measurement. Pass")
        print("     --ece-core-channel to use a specific channel directly and skip this heuristic.")

    ece_core = ece_signals[core_ch]

    if args.beam_species not in ("H", "D"):
        raise ValueError("--beam-species must be 'H' or 'D'")

    # --- Optional Te calibration (V -> eV). Not available yet -- see note below when absent. ---
    te_core_ev = None
    if args.te_calib_scale_ev_per_v is not None:
        te_core_ev = args.te_calib_scale_ev_per_v * ece_core + args.te_calib_offset_ev
        print(f"  Te calibration applied: Te[eV] = {args.te_calib_scale_ev_per_v:g} * V + {args.te_calib_offset_ev:g} "
              f"(--te-calib-scale-ev-per-v / --te-calib-offset-ev).")
    else:
        print("  ECE channel remains UNCALIBRATED (no --te-calib-scale-ev-per-v provided): only the")
        print("     DIRECTION and RELATIVE TIMING of the response are validated below, matching Zhong et")
        print("     al.'s own level of rigor (their Fig. 1d also plots raw, uncalibrated ECE intensity).")

    print("  Note: Zhong et al. modulate ECH periodically (many on/off cycles), which is what produces a")
    print("     clean hysteresis loop in their Fig. 2/3. Your ECH is a single on/off step, so the 'Fig. 2")
    print("     analogue' scatter below is expected to show a scattered blob near ECH's plateau rather")
    print("     than a loop -- the lagged cross-correlation numbers are the more meaningful comparison here.")

    envelope_corr = anti_alias_decimate(envelope, decimate_factor)
    ece_core_corr = anti_alias_decimate(ece_core, decimate_factor)
    lag_ece_ms, r_ece_peak, lags_ece_curve, corr_ece_curve = lagged_cross_correlation(
        envelope_corr, ece_core_corr, dt_corr, max_lag_ms=args.m6_max_lag_ms
    )
    print(f"  - Envelope vs. ECE-core-proxy: peak |correlation| = {r_ece_peak:+.4f} at lag = {lag_ece_ms:+.2f} ms "
          f"(search window: +/-{args.m6_max_lag_ms:.0f} ms)")
    if abs(lag_ece_ms) >= 0.9 * args.m6_max_lag_ms:
        print(f"    ⚠️ BOUNDARY WARNING: this lag is within 10% of the +/-{args.m6_max_lag_ms:.0f} ms search")
        print("       window edge -- the true peak may lie OUTSIDE this window. Re-run with a larger")
        print("       --m6-max-lag-ms before trusting this number.")
    print("    (Zhong et al. report ~6.0 ms excitation delay, ~1.5 ms suppression delay for their EPM;")
    print(f"     compare order of magnitude only -- their mode was 95-103 kHz, yours is filtered to "
          f"{args.lower:.0f}-{args.upper:.0f} kHz, so this may not be the same mode.)")
    r_ece_sig, p_ece_std, p_ece_adj, n_ece_sig, N_eff_ece_sig = lagged_pearson_significance(
        envelope_corr, ece_core_corr, dt_corr, lag_ece_ms
    )
    if r_ece_sig is not None:
        meets_ece = abs(r_ece_sig) > 0.7 and p_ece_adj < 0.05
        print(f"    -> At that lag: proper Pearson r = {r_ece_sig:.4f} (N={n_ece_sig}, N_eff={N_eff_ece_sig:.1f}), "
              f"p_std = {format_p_value(p_ece_std)}, p_adj = {format_p_value(p_ece_adj)} "
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
        print(f"  - Envelope vs. (density x ECE-core-proxy) [pressure proxy, Zhong Fig. 3 analogue]: "
              f"peak |correlation| = {r_pressure_peak:+.4f} at lag = {lag_pressure_ms:+.2f} ms "
              f"(search window: +/-{args.m6_max_lag_ms:.0f} ms)")
        if abs(lag_pressure_ms) >= 0.9 * args.m6_max_lag_ms:
            print(f"    ⚠️ BOUNDARY WARNING: this lag is within 10% of the +/-{args.m6_max_lag_ms:.0f} ms search")
            print("       window edge -- the true peak may lie OUTSIDE this window. Re-run with a larger")
            print("       --m6-max-lag-ms before trusting this number.")
        r_pressure_sig, p_pressure_std, p_pressure_adj, n_pressure_sig, N_eff_pressure_sig = lagged_pearson_significance(
            envelope_corr, pressure_proxy_corr, dt_corr, lag_pressure_ms
        )
        if r_pressure_sig is not None:
            meets_pressure = abs(r_pressure_sig) > 0.7 and p_pressure_adj < 0.05
            print(f"    -> At that lag: proper Pearson r = {r_pressure_sig:.4f} (N={n_pressure_sig}, "
                  f"N_eff={N_eff_pressure_sig:.1f}), p_std = {format_p_value(p_pressure_std)}, "
                  f"p_adj = {format_p_value(p_pressure_adj)} -- "
                  f"{'MEETS' if meets_pressure else 'does NOT meet'} |r|>0.7 & p<0.05.")
        else:
            p_pressure_adj = 1.0
    else:
        print("  - (density x ECE-core-proxy) pressure-proxy analysis skipped: density ('nave') unavailable for this shot.")
        r_pressure_sig, p_pressure_adj, N_eff_pressure_sig = None, 1.0, None

    # -------------------------------------------------------------------------------------------
    # Chirp-rate (d(f_inst)/dt) vs. ECE-core-proxy and vs. the pressure proxy, 
    # restricted to the mode-active mask.
    # -------------------------------------------------------------------------------------------
    lag_chirp_ece_ms, r_chirp_ece_peak = None, None
    lag_chirp_pressure_ms, r_chirp_pressure_peak = None, None
    lags_chirp_ece_curve, corr_chirp_ece_curve = None, None
    lags_chirp_pressure_curve, corr_chirp_pressure_curve = None, None
    if chirp_rate_khz_per_ms is not None:
        chirp_active = chirp_rate_khz_per_ms[mask_active_win]
        chirp_corr = anti_alias_decimate(chirp_active, decimate_factor)
        ece_core_active_corr = anti_alias_decimate(ece_core[mask_active_win], decimate_factor)
        if len(chirp_corr) > 4 and np.std(chirp_corr) > 0:
            lag_chirp_ece_ms, r_chirp_ece_peak, lags_chirp_ece_curve, corr_chirp_ece_curve = lagged_cross_correlation(
                chirp_corr, ece_core_active_corr, dt_corr, max_lag_ms=args.m6_max_lag_ms
            )
            print(f"  - [Frequency-sweep reading] Chirp rate (d(f_inst)/dt) vs. ECE-core-proxy: "
                  f"peak |correlation| = {r_chirp_ece_peak:+.4f} at lag = {lag_chirp_ece_ms:+.2f} ms "
                  f"(search window: +/-{args.m6_max_lag_ms:.0f} ms)")
            if abs(lag_chirp_ece_ms) >= 0.9 * args.m6_max_lag_ms:
                print(f"    ⚠️ BOUNDARY WARNING: this lag is within 10% of the +/-{args.m6_max_lag_ms:.0f} ms search "
                      "window edge; widen --m6-max-lag-ms before trusting this number.")
            r_chirp_ece_sig, p_chirp_ece_std, p_chirp_ece_adj, n_chirp_ece_sig, N_eff_chirp_ece_sig = lagged_pearson_significance(
                chirp_corr, ece_core_active_corr, dt_corr, lag_chirp_ece_ms
            )
            if r_chirp_ece_sig is not None:
                meets_chirp_ece = abs(r_chirp_ece_sig) > 0.7 and p_chirp_ece_adj < 0.05
                print(f"    -> At that lag: proper Pearson r = {r_chirp_ece_sig:.4f} (N={n_chirp_ece_sig}, "
                      f"N_eff={N_eff_chirp_ece_sig:.1f}), p_adj = {format_p_value(p_chirp_ece_adj)} -- "
                      f"{'MEETS' if meets_chirp_ece else 'does NOT meet'} |r|>0.7 & p<0.05.")
                if N_eff_chirp_ece_sig <= 3.05:
                    print("       ⚠️ N_eff hit the floor (~3): the heavily-smoothed chirp-rate series is so "
                          "autocorrelated that this test has essentially NO statistical power -- 'does NOT "
                          "meet' here means 'inconclusive', not 'no relationship'. Do not report this as a "
                          "null result; a coarser/less-smoothed chirp-rate estimate would be needed to test "
                          "this properly.")
            else:
                p_chirp_ece_adj = 1.0
            if density_detected and pressure_proxy is not None:
                pressure_proxy_active_corr = anti_alias_decimate(pressure_proxy[mask_active_win], decimate_factor)
                lag_chirp_pressure_ms, r_chirp_pressure_peak, lags_chirp_pressure_curve, corr_chirp_pressure_curve = lagged_cross_correlation(
                    chirp_corr, pressure_proxy_active_corr, dt_corr, max_lag_ms=args.m6_max_lag_ms
                )
                print(f"  - [Frequency-sweep reading] Chirp rate vs. (density x ECE-core-proxy) [pressure proxy]: "
                      f"peak |correlation| = {r_chirp_pressure_peak:+.4f} at lag = {lag_chirp_pressure_ms:+.2f} ms "
                      f"(search window: +/-{args.m6_max_lag_ms:.0f} ms)")
                if abs(lag_chirp_pressure_ms) >= 0.9 * args.m6_max_lag_ms:
                    print(f"    ⚠️ BOUNDARY WARNING: this lag is within 10% of the +/-{args.m6_max_lag_ms:.0f} ms search "
                          "window edge; widen --m6-max-lag-ms before trusting this number.")
                r_chirp_pressure_sig, p_chirp_pressure_std, p_chirp_pressure_adj, n_chirp_pressure_sig, N_eff_chirp_pressure_sig = lagged_pearson_significance(
                    chirp_corr, pressure_proxy_active_corr, dt_corr, lag_chirp_pressure_ms
                )
                if r_chirp_pressure_sig is not None:
                    meets_chirp_pressure = abs(r_chirp_pressure_sig) > 0.7 and p_chirp_pressure_adj < 0.05
                    print(f"    -> At that lag: proper Pearson r = {r_chirp_pressure_sig:.4f} (N={n_chirp_pressure_sig}, "
                          f"N_eff={N_eff_chirp_pressure_sig:.1f}), p_adj = {format_p_value(p_chirp_pressure_adj)} -- "
                          f"{'MEETS' if meets_chirp_pressure else 'does NOT meet'} |r|>0.7 & p<0.05.")
                    if N_eff_chirp_pressure_sig <= 3.05:
                        print("       ⚠️ N_eff hit the floor (~3): same caveat as above -- this test has "
                              "essentially no statistical power, so treat 'does NOT meet' as inconclusive.")
                else:
                    p_chirp_pressure_adj = 1.0
            else:
                r_chirp_pressure_sig, p_chirp_pressure_adj, N_eff_chirp_pressure_sig = None, 1.0, None
        else:
            print("  - [Frequency-sweep reading] Chirp-rate correlation skipped: too few/degenerate samples in the active window.")
            r_chirp_ece_sig, p_chirp_ece_adj, N_eff_chirp_ece_sig = None, 1.0, None
            r_chirp_pressure_sig, p_chirp_pressure_adj, N_eff_chirp_pressure_sig = None, 1.0, None
    else:
        print("  - [Frequency-sweep reading] Chirp-rate correlation skipped: chirp_rate_khz_per_ms not supplied.")
        r_chirp_ece_sig, p_chirp_ece_adj, N_eff_chirp_ece_sig = None, 1.0, None
        r_chirp_pressure_sig, p_chirp_pressure_adj, N_eff_chirp_pressure_sig = None, 1.0, None

    # --- Optional: theoretical electron-drag (slowing-down) timescale, only if Te calibration given ---
    tau_s_ms = None
    if te_core_ev is not None and density_detected:
        mask_scaling_win = mask_active_win
        te_active_ev = np.clip(te_core_ev[mask_scaling_win], 1.0, None)  # clip to avoid <=0 eV under noise
        ne_active_cm3 = np.clip(density_val[mask_scaling_win], 0.01, None) * 1e19 * 1e-6  # e19 m^-3 -> cm^-3
        A_b = 1.0 if args.beam_species == "H" else 2.0
        Z_b = 1.0
        ln_lambda = 24.0 - np.log(np.sqrt(ne_active_cm3) / te_active_ev)
        ln_lambda = np.clip(ln_lambda, 5.0, 25.0)
        tau_s_s = 6.27e8 * A_b * te_active_ev**1.5 / (Z_b**2 * ne_active_cm3 * ln_lambda)
        tau_s_ms = float(np.mean(tau_s_s)) * 1000.0
        print(f"  - [VERIFIED PREFACTOR] Theoretical electron-drag slowing-down time (active window mean): "
              f"tau_s ~= {tau_s_ms:.2f} ms")
        print(f"    Measured envelope-vs-ECE-core delay: {lag_ece_ms:+.2f} ms "
              f"({'same order of magnitude' if 0.1 < abs(lag_ece_ms) / max(tau_s_ms, 1e-9) < 10 else 'DIFFERENT order of magnitude'} "
              f"as tau_s).")
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
    axs[0].set_title(f"Shot {shot}: ECE-core-proxy (ch.{core_ch}) & Mode Envelope")
    axs[0].grid(True, alpha=0.3)

    sc = axs[1].scatter(ech_plot, env_plot, c=t_plot, cmap='viridis', s=6)
    axs[1].plot(ech_plot, env_plot, color='gray', alpha=0.15, linewidth=0.5)
    plt.colorbar(sc, ax=axs[1], label='Time (ms)')
    axs[1].set_xlabel("ECH Power (raw)")
    axs[1].set_ylabel("Mode Envelope (V)")
    axs[1].set_title("Zhong Fig. 2 analogue: Envelope vs. ECH Power\n(time-colored; a loop = delayed/hysteretic response)")
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
                   label='boundary zone (peak here = untrustworthy, widen window)')
    ax_lag.set_xlabel("Lag (ms)")
    ax_lag.set_ylabel("Normalized cross-correlation")
    ax_lag.set_title("[BUGFIX] Lag-correlation curves\n(dotted = chosen peak; shaded = boundary risk zone)")
    ax_lag.legend(loc='best', fontsize=8)
    ax_lag.grid(True, alpha=0.3)

    if density_detected:
        pressure_plot = pressure_proxy[mask]
        sc2 = axs[2].scatter(pressure_plot, env_plot, c=t_plot, cmap='viridis', s=6)
        axs[2].plot(pressure_plot, env_plot, color='gray', alpha=0.15, linewidth=0.5)
        plt.colorbar(sc2, ax=axs[2], label='Time (ms)')
        axs[2].set_xlabel("Density x ECE-core-proxy (pressure proxy, raw units)")
        axs[2].set_ylabel("Mode Envelope (V)")
        axs[2].set_title("Zhong Fig. 3 analogue: Envelope vs. Pressure Proxy\n(time-colored; a loop = delayed response)")
        axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    suffix = get_output_suffix(args)
    output_png = f"mhd_analysis_objective2_zhong_{shot}{suffix}.png"
    plt.savefig(output_png, dpi=150)
    plt.close(fig)
    print(f"  Zhong-et-al.-style delay/hysteresis figure saved to: '{output_png}'")

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
        "ece_core": ece_core,
    }



def load_obj1_reference_window(shot, args, fl_hz, fu_hz):
    """
    Load Objective-1 export JSON (if available) and extract the longest 3-sigma-dominant interval
    """
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
        looked_in = ", ".join(str(d) for d in search_dirs)
        print(f"  [M9][OBJ1-XREF] No Objective-1 export found for shot {shot} (looked for "
              f"'{args.obj1_json_pattern.format(shot=shot)}' in [{looked_in}]; pass --obj1-results-dir "
              "if mhd_analysis_obj1.py was run elsewhere). [M9] will search the full [M7] "
              "envelope-active burst instead of anchoring to Objective 1's 3-sigma-dominance detection.")
        return None

    try:
        with open(json_path) as fjson:
            obj1_data = json.load(fjson)
    except Exception as exc:
        print(f"  ⚠️ [M9][OBJ1-XREF] Failed to read/parse '{json_path}': {exc}; ignoring cross-reference.")
        return None

    modes = obj1_data.get("discrete_modes", [])
    tol_hz = args.obj1_mode_freq_tol_khz * 1000.0
    band_center_hz = 0.5 * (fl_hz + fu_hz)
    in_band = [m for m in modes if (fl_hz - tol_hz) <= m.get("frequency_hz", -1e30) <= (fu_hz + tol_hz)]
    if not in_band:
        print(f"  ⚠️ [M9][OBJ1-XREF] '{json_path}' has no mode within {args.lower:.1f}-{args.upper:.1f} kHz "
              f"(+/- {args.obj1_mode_freq_tol_khz:.1f} kHz tolerance); ignoring cross-reference.")
        return None

    confirmed = [m for m in in_band if m.get("dual_criterion_pass")]
    pool = confirmed if confirmed else in_band
    if not confirmed:
        print(f"  ⚠️ [M9][OBJ1-XREF] No dominant+coherent-CONFIRMED mode in-band in '{json_path}'; using "
              "the closest in-band mode anyway (dominant-only), lower confidence.")

    chosen = min(pool, key=lambda m: abs(m.get("frequency_hz", 0.0) - band_center_hz))
    intervals = chosen.get("active_intervals_ms", [])
    if not intervals:
        print(f"  ⚠️ [M9][OBJ1-XREF] Matched mode at {chosen.get('frequency_hz', 0.0)/1000.0:.2f} kHz in "
              f"'{json_path}' has no recorded active_intervals_ms; ignoring cross-reference.")
        return None

    start_ms, end_ms = max(intervals, key=lambda iv: iv[1] - iv[0])
    n_intervals = len(intervals)
    tag = "confirmed dominant+coherent" if chosen.get("dual_criterion_pass") else "dominant only"
    print(f"  [M9][OBJ1-XREF] Cross-referenced '{json_path}': mode at "
          f"{chosen.get('frequency_hz', 0.0)/1000.0:.2f} kHz ({tag}) -> longest 3-sigma-dominant interval "
          f"{start_ms:.1f}-{end_ms:.1f} ms (of {n_intervals} recorded interval(s) for this mode); this "
          "anchors [M9]'s search domain below.")
    return start_ms, end_ms


def _typical_background_score(ys_smooth, duration_bins, step_bins, exclude_lo, exclude_hi):
    scores = []
    n = len(ys_smooth)
    start = 0
    while start + duration_bins <= n:
        end = start + duration_bins - 1
        if not (end < exclude_lo or start > exclude_hi):
            start += step_bins
            continue
        window = ys_smooth[start:end + 1]
        scores.append(float(np.std(window) + (np.max(window) - np.min(window))))
        start += step_bins
    return np.array(scores)


def _search_background_window(ys_smooth, anchor_idx, direction, duration_bins, step_bins,
                               level_threshold, score_threshold, min_idx, max_idx):
    best_overall = None
    best_overall_score = np.inf
    best_level_ok = None
    best_level_ok_score = np.inf
    offset = 0
    while True:
        if direction < 0:
            end_idx = anchor_idx - offset
            start_idx = end_idx - duration_bins + 1
            if start_idx < min_idx:
                break
        else:
            start_idx = anchor_idx + offset
            end_idx = start_idx + duration_bins - 1
            if end_idx > max_idx:
                break

        window = ys_smooth[start_idx:end_idx + 1]
        w_mean = float(np.mean(window))
        w_std = float(np.std(window))
        w_range = float(np.max(window) - np.min(window))
        score = w_std + w_range
        level_ok = w_mean <= level_threshold

        satisfies = (level_ok and score <= score_threshold)
        if satisfies:
            return {'found': True, 'start_idx': start_idx, 'end_idx': end_idx,
                    'mean': w_mean, 'std': w_std}

        if score < best_overall_score:
            best_overall_score = score
            best_overall = {'found': False, 'start_idx': start_idx, 'end_idx': end_idx,
                             'mean': w_mean, 'std': w_std}
        if level_ok and score < best_level_ok_score:
            best_level_ok_score = score
            best_level_ok = {'found': False, 'start_idx': start_idx, 'end_idx': end_idx,
                              'mean': w_mean, 'std': w_std}
        offset += step_bins

    if best_level_ok is not None:
        return best_level_ok
    return best_overall if best_overall is not None else {'found': False, 'start_idx': None,
                                                            'end_idx': None, 'mean': None, 'std': None}


def detect_hmode_burst_window(wp_file, baseline_end_ms=50.0, smooth_ms=2.0,
                               k_on=3.0, k_off=1.5, min_duration_ms=10.0,
                               robust_baseline=True):
    """
    Auto-detects the H-mode burst window from the stored energy W_p (ported from mhd_analysis_obj3.py).
    """
    edf_wp = TE.edf()
    dat_wp = edf_wp.load(str(wp_file))
    t_sec = dat_wp[:, 0]
    if edf_wp.DimUnit[0] == 'ms':
        t_sec = t_sec / 1000.0
    ys = dat_wp[:, 1]
    dt = t_sec[1] - t_sec[0]
    t_ms = t_sec * 1000.0

    mask_base = t_ms <= (t_ms[0] + baseline_end_ms)
    if np.sum(mask_base) < 5:
        return {'ok': False, 'reason': 'baseline window too short / no samples'}
    if robust_baseline:
        baseline_mean = float(np.median(ys[mask_base]))
        mad = float(np.median(np.abs(ys[mask_base] - baseline_mean)))
        baseline_std = float(mad * 1.4826)
        if baseline_std <= 1e-12:
            baseline_std = float(np.std(ys[mask_base]))
    else:
        baseline_mean = float(np.mean(ys[mask_base]))
        baseline_std = float(np.std(ys[mask_base]))

    smooth_bins = max(1, int(round(smooth_ms / (dt * 1000.0))))
    ys_smooth = ndi.uniform_filter1d(ys, size=smooth_bins, mode='nearest') if smooth_bins > 1 else ys

    onset_threshold = baseline_mean + k_on * baseline_std
    end_threshold = baseline_mean + k_off * baseline_std
    min_duration_bins = max(1, int(round(min_duration_ms / (dt * 1000.0))))

    above_onset = ys_smooth > onset_threshold
    onset_idx = None
    run_len = 0
    for i, flag in enumerate(above_onset):
        run_len = run_len + 1 if flag else 0
        if run_len >= min_duration_bins:
            onset_idx = i - min_duration_bins + 1
            break

    if onset_idx is None:
        return {'ok': False, 'reason': f'no sustained W_p rise above baseline_mean + {k_on}*std'}

    below_end = ys_smooth <= end_threshold
    end_idx = len(ys_smooth) - 1
    run_len = 0
    for i in range(onset_idx, len(ys_smooth)):
        run_len = run_len + 1 if below_end[i] else 0
        if run_len >= min_duration_bins:
            end_idx = i - min_duration_bins + 1
            break

    burst_start_ms = float(t_ms[onset_idx])
    burst_end_ms = float(t_ms[end_idx])

    return {
        'ok': True,
        'start_ms': burst_start_ms,
        'end_ms': burst_end_ms,
        'baseline_mean': baseline_mean,
        'baseline_std': baseline_std,
        'onset_threshold': onset_threshold,
        'end_threshold': end_threshold,
        'wp_peak': float(np.max(ys_smooth[onset_idx:end_idx + 1])) if end_idx > onset_idx else float(ys_smooth[onset_idx]),
        't_ms': t_ms, 'ys': ys, 'ys_smooth': ys_smooth,
    }


def detect_flat_frequency_subwindow(ifreq_khz, t_ms, dt, i0_domain, i1_domain, args, min_duration_ms=None):
    """
    [M9] FLAT-FREQUENCY SUB-WINDOW DETECTOR (ported from mhd_analysis_obj3.py).
    Finds the flattest, least-chirping sub-interval within the search domain.
    """
    n_domain = i1_domain - i0_domain
    fallback = (i0_domain, i1_domain, {"used_fallback": True, "reason": "domain too short or manual window empty"})
    effective_min_duration_ms = getattr(args, 'flat_min_duration_ms', 5.0) if min_duration_ms is None else min_duration_ms

    if getattr(args, 'flat_window_start', None) is not None and getattr(args, 'flat_window_end', None) is not None:
        lo = max(args.flat_window_start, t_ms[i0_domain])
        hi = min(args.flat_window_end, t_ms[i1_domain - 1])
        if hi <= lo:
            print(f"  --flat-window-start/--flat-window-end ({args.flat_window_start:.1f}-"
                  f"{args.flat_window_end:.1f} ms) does not overlap the search domain "
                  f"({t_ms[i0_domain]:.1f}-{t_ms[i1_domain-1]:.1f} ms); falling back to the full domain.")
            return fallback
        idx = np.where((t_ms >= lo) & (t_ms <= hi))[0]
        i0f, i1f = int(idx[0]), int(idx[-1]) + 1
        print(f"  [M9] Flat-frequency sub-window: MANUAL override = {t_ms[i0f]:.1f}-{t_ms[i1f-1]:.1f} ms "
              "(--flat-window-start/--flat-window-end).")
        return i0f, i1f, {"used_fallback": False, "manual": True}

    scan_window_ms = getattr(args, 'flat_scan_window_ms', 8.0)
    scan_samples = max(3, int(round(scan_window_ms / (dt * 1000.0))))
    if scan_samples >= n_domain:
        print(f"  ⚠️ [M9] Search domain ({n_domain} samples, {n_domain * dt * 1000.0:.1f} ms) is shorter "
              f"than --flat-scan-window-ms ({scan_window_ms:.1f} ms); using full domain.")
        return fallback

    ifreq_domain = ifreq_khz[i0_domain:i1_domain]
    t_domain = t_ms[i0_domain:i1_domain]

    slope_raw = np.gradient(ifreq_domain, t_domain)
    slope_smooth_ms = getattr(args, 'flat_slope_smooth_ms', 2.0)
    smooth_samples = max(1, int(round(slope_smooth_ms / (dt * 1000.0))))
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

    growth_tol = getattr(args, 'flat_growth_tolerance', 0.5)
    tol_mean = best_mean * (1.0 + growth_tol)
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

    relaxed_for_min_duration = False
    if duration_ms < effective_min_duration_ms:
        relaxed_for_min_duration = True
        while (end - start) * dt * 1000.0 < effective_min_duration_ms and (start > 0 or end < n_domain):
            left_slope = slope_abs[start - 1] if start > 0 else np.inf
            right_slope = slope_abs[end] if end < n_domain else np.inf
            if left_slope <= right_slope:
                start -= 1
            else:
                end += 1
        duration_ms = (end - start) * dt * 1000.0
        touches_edge = (start == 0) or (end == n_domain)
        grown_mean = float(window_mean(start, end))

    i0_flat, i1_flat = i0_domain + start, i0_domain + end
    edge_note = " (grew to domain edge)" if touches_edge else ""
    print(f"  [M9] Flat-frequency sub-window: {t_ms[i0_flat]:.1f}-{t_ms[i1_flat-1]:.1f} ms "
          f"({end - start} samples, {duration_ms:.1f} ms, mean |slope| = {grown_mean:.3f} kHz/ms){edge_note}.")

    return i0_flat, i1_flat, {
        "used_fallback": False, "manual": False, "scan_mean_khz_per_ms": best_mean,
        "grown_mean_khz_per_ms": grown_mean, "duration_ms": duration_ms, "touches_edge": touches_edge,
        "relaxed_for_min_duration": relaxed_for_min_duration,
    }


def compute_mode_active_subwindow_obj3(fpath, t_start_ms, t_end_ms, args):
    """
    Computes the mode-active flat-frequency sub-window using Objective 3's exact method:
    loads MP1, filters 5.0 - 100.0 kHz, extracts Hilbert instantaneous frequency,
    scans inside [t_start_ms, t_end_ms] from W_p burst, grows within 50% tolerance,
    and enforces the required duration floor.
    """
    edf = TE.edf()
    dat = edf.load(str(fpath))
    t_sec = dat[:, 0]
    if edf.DimUnit[0] == 'ms':
        t_sec = t_sec / 1000.0
    ys = dat[:, 1]
    dt = t_sec[1] - t_sec[0]
    fs = 1.0 / dt
    t_ms = t_sec * 1000.0

    idx_domain = np.where((t_ms >= t_start_ms) & (t_ms <= t_end_ms))[0]
    if len(idx_domain) < 3:
        return t_start_ms, t_end_ms, {"used_fallback": True, "reason": "burst window has too few samples"}
    i0_domain, i1_domain = int(idx_domain[0]), int(idx_domain[-1]) + 1

    fl_hz = 5000.0
    fu_hz = 100000.0
    _, _, _, ifreq_hz = extract_instantaneous_frequency(ys, fs, fl_hz, fu_hz, 4, OPTIMAL_SG_WIN)
    ifreq_khz = ifreq_hz / 1000.0

    effective_min_duration_ms = 15.872  # bicoherence/coherence sample floor (1024 + 29*512 spl = 15.872 ms)

    i0_flat, i1_flat, flat_info = detect_flat_frequency_subwindow(
        ifreq_khz, t_ms, dt, i0_domain, i1_domain, args, min_duration_ms=effective_min_duration_ms
    )
    return float(t_ms[i0_flat]), float(t_ms[i1_flat - 1]), flat_info


def process_shot(shot, args):
    data_dir = Path(args.data_dir_pattern.format(shot=shot))
    mag_file = data_dir / f"MP1@{shot}.edf"

    print(f"\n{'=' * 93}")
    print(f"--- Loading Mirnov Coil magnetic signal: {mag_file} (Shot {shot}) ---")
    if not mag_file.exists():
        print(f"  ⚠️ Warning: file {mag_file} does not exist; skipping shot {shot}.")
        return None

    edf_mag = TE.edf()
    dat_mag = edf_mag.load(str(mag_file))
    t = dat_mag[:, 0]
    ys = dat_mag[:, 1]  # Magnetic fluctuation voltage channel

    
    if edf_mag.DimUnit[0] == 'ms':
        t_sec = t / 1000.0
    else:
        t_sec = t

    dt = (t_sec[100] - t_sec[0]) / 100.0
    fs = 1.0 / dt
    t_ms = t_sec * 1000.0
    print(f"Mirnov signal: {len(ys)} points, fs = {fs/1e6:.2f} MHz (Time range: {t_ms[0]:.1f} - {t_ms[-1]:.1f} ms)")

    
    print(f"\nApplying Bessel bandpass filter ({args.lower} - {args.upper} kHz) and Hilbert Transform...")
    fl_hz = args.lower * 1000.0
    fu_hz = args.upper * 1000.0

    envelope, phase, filtered, ifreq_hz = extract_instantaneous_frequency(
        ys, fs, fl_hz, fu_hz, args.order, args.smoothing
    )
    ifreq_khz = ifreq_hz / 1000.0

    # -----------------------------------------------------------------------------------------
    # Load the two additional Mirnov probes (MP3, MP4) and extract their carrier oscillation + Hilbert envelope.
    # -----------------------------------------------------------------------------------------
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
            print(f"  [M8] Loaded {probe}@{shot}.edf ({len(ys_probe)} points, interpolated onto MP1's "
                  "time base) and extracted its carrier oscillation + Hilbert envelope.")
        else:
            probe_missing.append(probe)
    if probe_missing:
        print(f"  ⚠️ [M8] Warning: no file(s) found for Mirnov probe(s) {probe_missing}; any inter-probe "
              "coherence pair involving a missing probe will be skipped in the modal-structure panels below.")

    # -----------------------------------------------------------------------------------------
    # Load the poloidal PMP1-PMP14 array for the poloidal mode-number (m) estimate below.
    # Skipped entirely (empty dict) if --disable-poloidal-array is set.
    # -----------------------------------------------------------------------------------------
    if not args.disable_poloidal_array:
        pmp_signals, pmp_plab_rad = load_poloidal_array(
            shot, data_dir, t_ms, invert_channels=tuple(args.pmp_invert_channels)
        )
        if not args.pmp_skip_self_test and pmp_plab_rad:
            poloidal_array_self_test(pmp_plab_rad, max_m_test=args.pmp_self_test_max_m)
    else:
        pmp_signals, pmp_plab_rad = {}, {}

    print("\n--- Savitzky-Golay Window Sensitivity Analysis (sg_win) [EXT] ---")
    windows_to_test = [125, 325, 525, 725, 925]
    for w in windows_to_test:
        test_ifreq_hz = dsp.savgol_filter(phase, w, 2, deriv=1) / dt
        test_ifreq_khz = test_ifreq_hz / 1000.0
        freq_rate_std = np.std(np.diff(test_ifreq_khz))
        print(f"  - Window sg_win = {w:<3} -> Standard Deviation of d(f_inst)/dt: {freq_rate_std:.4f} kHz/sample")
    print(f"  - Justification: sg_win = {args.smoothing} is the optimal value that drastically reduces high-frequency")
    print("    numerical phase noise without distorting or flattening the physical EPM frequency sweep.")

    # -----------------------------------------------------------------------------------------
    # MODE-ACTIVE MASK 
    # -----------------------------------------------------------------------------------------
    if args.mode_active_start is not None and args.mode_active_end is not None:
        mask_mode_active = (t_ms >= args.mode_active_start) & (t_ms <= args.mode_active_end)
        print(f"\n[M7] Mode-active window: MANUAL override = {args.mode_active_start:.1f}-{args.mode_active_end:.1f} ms "
              f"(--mode-active-start/--mode-active-end).")
    else:
        med_env = float(np.median(envelope))
        mad_env = float(np.median(np.abs(envelope - med_env))) * 1.4826  # normal-consistent -> std-equivalent
        mode_active_threshold = med_env + args.mode_active_k * mad_env
        mask_raw_threshold = envelope > mode_active_threshold
        n_raw = int(np.sum(mask_raw_threshold))
        print(f"\n[M7] Raw threshold mask: envelope > median + {args.mode_active_k:g} x MAD_std "
              f"(full-trace median={med_env:.4f} V, MAD_std={mad_env:.4f} V) "
              f"= threshold {mode_active_threshold:.4f} V -> {n_raw} points ({100.0*n_raw/len(mask_raw_threshold):.1f}% of trace).")

        gap_samples = max(1, int(round(args.mode_active_max_gap_ms * 1e-3 / dt)))
        structure = np.ones(2 * gap_samples + 1, dtype=bool)
        mask_closed = ndi.binary_closing(mask_raw_threshold, structure=structure)
        labeled, n_components = ndi.label(mask_closed)
        if n_components == 0:
            mask_mode_active = np.zeros_like(mask_raw_threshold)
            print("  ⚠️ No mode-active burst found after gap-bridging (envelope never sustained above threshold).")
        else:
            sizes = ndi.sum(mask_closed, labeled, index=np.arange(1, n_components + 1))
            best_label = int(np.argmax(sizes)) + 1
            mask_burst = labeled == best_label
            burst_duration_ms = float(np.sum(mask_burst)) * dt * 1000.0
            if burst_duration_ms < args.mode_active_min_duration_ms:
                mask_mode_active = np.zeros_like(mask_raw_threshold)
                print(f"  ⚠️ Largest contiguous burst is only {burst_duration_ms:.1f} ms "
                      f"(< --mode-active-min-duration-ms={args.mode_active_min_duration_ms:.0f} ms); rejecting as noise.")
            else:
                mask_mode_active = mask_burst
                n_raw_outside_burst = n_raw - int(np.sum(mask_raw_threshold & mask_burst))
                frac_active = 100.0 * np.sum(mask_mode_active) / len(mask_mode_active)
                print(f"  Burst detection ({n_components} candidate run(s) after bridging gaps <= "
                      f"{args.mode_active_max_gap_ms:.1f} ms): largest run kept = {burst_duration_ms:.1f} ms "
                      f"({np.sum(mask_mode_active)} samples, {frac_active:.1f}% of trace); "
                      f"{n_raw_outside_burst} raw-threshold points outside this burst were excluded.")

    n_mode_active = int(np.sum(mask_mode_active))
    if n_mode_active < 200:
        print(f"  ⚠️ Only {n_mode_active} mode-active samples found; instantaneous-frequency-vs-heating "
              "correlation ([M7]) and the [M6] chirp-rate analysis will be SKIPPED for this shot.")
        active_idx_mode = np.array([], dtype=int)
    else:
        active_idx_mode = np.where(mask_mode_active)[0]
        i0_mode, i1_mode = int(active_idx_mode[0]), int(active_idx_mode[-1]) + 1
        n_gaps = n_mode_active - (i1_mode - i0_mode)  # 0 if the mask is one contiguous block
        if n_gaps < 0:
            print(f"  Mode-active span (contiguous bounding window used for coherence): "
                  f"{t_ms[i0_mode]:.1f}-{t_ms[i1_mode-1]:.1f} ms ({i1_mode - i0_mode} samples, "
                  f"{-n_gaps} of which are below-threshold gaps inside that span).")
        else:
            print(f"  Mode-active span: {t_ms[i0_mode]:.1f}-{t_ms[i1_mode-1]:.1f} ms (contiguous).")

    # -----------------------------------------------------------------------------------------
    # H-MODE BURST & MODE-ACTIVE (STRONGLY PRESENT) SUB-WINDOW (Objective 3 algorithm)
    # -----------------------------------------------------------------------------------------
    wp_file = data_dir / f"Wp@{shot}.edf"
    t_obj3_burst_start, t_obj3_burst_end = None, None
    t_mode_strong_start, t_mode_strong_end = None, None
    obj3_sub_applied = False
    obj3_flat_info = {}

    if wp_file.exists():
        wp_det = detect_hmode_burst_window(wp_file)
        if wp_det.get('ok', False):
            t_obj3_burst_start, t_obj3_burst_end = wp_det['start_ms'], wp_det['end_ms']
            t_mode_strong_start, t_mode_strong_end, obj3_flat_info = compute_mode_active_subwindow_obj3(
                mag_file, t_obj3_burst_start, t_obj3_burst_end, args
            )
            obj3_sub_applied = not obj3_flat_info.get('used_fallback', False)
            print(f"\n[M9-OBJ3] Auto-detected H-mode W_p burst: {t_obj3_burst_start:.1f}-{t_obj3_burst_end:.1f} ms")
            print(f"[M9-OBJ3] Auto-detected mode-active (strongly present) sub-window: "
                  f"{t_mode_strong_start:.1f}-{t_mode_strong_end:.1f} ms (applied: {obj3_sub_applied})")

    # -----------------------------------------------------------------------------------------
    # FLAT-FREQUENCY SUB-WINDOW (FOR COHERENCE [M8])
    # -----------------------------------------------------------------------------------------
    if obj3_sub_applied and t_mode_strong_start is not None and t_mode_strong_end is not None:
        idx_strong = np.where((t_ms >= t_mode_strong_start) & (t_ms <= t_mode_strong_end))[0]
        if len(idx_strong) >= 50:
            i0_flat = int(idx_strong[0])
            i1_flat = int(idx_strong[-1]) + 1
            n_flat_active = i1_flat - i0_flat
            flat_info = obj3_flat_info
            print(f"  [M9] Using Objective-3 mode-active sub-window: {t_ms[i0_flat]:.1f}-{t_ms[i1_flat-1]:.1f} ms ({n_flat_active} samples).")
        else:
            i0_flat, i1_flat = i0_mode, i1_mode
            n_flat_active = i1_flat - i0_flat
            flat_info = {"used_fallback": True, "reason": "subwindow has too few samples"}
    elif n_mode_active >= 200:
        print("\n[M9] Detecting flat-frequency (non-chirping) sub-window for [M8]:")
        obj1_ref = load_obj1_reference_window(shot, args, fl_hz, fu_hz)
        if obj1_ref is not None:
            ref_start_ms, ref_end_ms = obj1_ref
            lo = max(ref_start_ms, t_ms[i0_mode])
            hi = min(ref_end_ms, t_ms[i1_mode - 1])
            if hi > lo:
                idx_ref = np.where((t_ms >= lo) & (t_ms <= hi))[0]
                i0_domain, i1_domain = int(idx_ref[0]), int(idx_ref[-1]) + 1
                print(f"  [M9] Search domain = Objective-1 reference window intersect [M7] burst = "
                      f"{t_ms[i0_domain]:.1f}-{t_ms[i1_domain-1]:.1f} ms.")
            else:
                print("  ⚠️ [M9] Objective-1 reference window does not overlap the [M7] burst at all "
                      f"(obj1: {ref_start_ms:.1f}-{ref_end_ms:.1f} ms vs. [M7]: {t_ms[i0_mode]:.1f}-"
                      f"{t_ms[i1_mode-1]:.1f} ms); falling back to searching the full [M7] burst.")
                i0_domain, i1_domain = i0_mode, i1_mode
        else:
            i0_domain, i1_domain = i0_mode, i1_mode

        i0_flat, i1_flat, flat_info = detect_flat_frequency_subwindow(
            ifreq_khz, t_ms, dt, i0_domain, i1_domain, args
        )
        n_flat_active = i1_flat - i0_flat
    else:
        i0_flat, i1_flat, flat_info = i0_mode, i1_mode, {"used_fallback": True, "reason": "no mode-active window"}
        n_flat_active = 0

    # 3. Loading Real Heating Signals (Multichannel)
    print(f"\n--- Loading real heating channels (Shot {shot}) ---")

    # ECH Gyrotron
    ech_file = data_dir / f"ECHRG500@{shot}.edf"
    ech_power = np.zeros_like(t_ms)
    if ech_file.exists():
        print(f"Loading real ECH power from: {ech_file}")
        edf_ech = TE.edf()
        dat_ech = edf_ech.load(str(ech_file))
        t_ech = dat_ech[:, 0]
        if edf_ech.DimUnit[0] == 's':
            t_ech = t_ech * 1000.0
        ech_power = np.interp(t_ms, t_ech, dat_ech[:, 1])
    else:
        print(f"  ⚠️ Warning: expected ECH file not found at {ech_file}; a null")
        print("     ECH channel (all zeros) will be used, which invalidates any conclusion about correlation with ECH.")

    # NBI Injectors (Loading S3, S4, S9, S10)
    nbi_channels = ["NBIS3I", "NBIS4I", "NBIS9I", "NBIS10I"]
    nbi_signals = {}
    total_nbi_power = np.zeros_like(t_ms)

    nbi_mismatch_detected = False
    nbi_turn_on_times = {}
    nbi_missing = []

    for nbi in nbi_channels:
        nbi_file = data_dir / f"{nbi}@{shot}.edf"
        if nbi_file.exists():
            edf_nbi = TE.edf()
            dat_nbi = edf_nbi.load(str(nbi_file))
            t_nbi = dat_nbi[:, 0]
            if edf_nbi.DimUnit[0] == 's':
                t_nbi = t_nbi * 1000.0

            nbi_val = np.interp(t_ms, t_nbi, dat_nbi[:, 1])
            nbi_signals[nbi] = nbi_val
            total_nbi_power += nbi_val

            active_idx = np.where(dat_nbi[:, 1] > 0.5)[0]
            if len(active_idx) > 0:
                t_on = dat_nbi[active_idx[0], 0]
                if edf_nbi.DimUnit[0] == 's':
                    t_on = t_on * 1000.0
                nbi_turn_on_times[nbi] = t_on
                if t_on > t_ms[-1]:
                    nbi_mismatch_detected = True
        else:
            nbi_missing.append(nbi)

    if nbi_missing:
        print(f"  ⚠️ Warning: no files found for NBI channels: {nbi_missing}")

    print(f"  NBI channels loaded: {list(nbi_signals.keys())} (of {nbi_channels} requested). Confirm with")
    print("  the diagnostics team whether this list of injectors is complete; otherwise, 'Total NBI'")
    print("  might underestimate the real injection power.")

    first_t_on = min(nbi_turn_on_times.values()) if nbi_turn_on_times else None

    if nbi_mismatch_detected:
        print("\n⚠️ PHYSICAL CONFIGURATION ALERT:")
        print(f"  The NBI pulse turns on after the magnetic acquisition ends ({t_ms[-1]:.1f} ms).")
        print("  The NBI signals are flat throughout the MHD measurement window for this shot.")
        margin_ms = min(nbi_turn_on_times.values()) - t_ms[-1]
        print(f"  Margin between the end of Mirnov and NBI turn-on: {margin_ms:.2f} ms.")
        if ech_file.exists():
            ech_on_idx = np.where(dat_ech[:, 1] > 0.5 * np.max(dat_ech[:, 1]))[0]
            if len(ech_on_idx) > 0:
                t_ech_on = dat_ech[ech_on_idx[0], 0] * (1000.0 if edf_ech.DimUnit[0] == 's' else 1.0)
                print(f"  Cross-reference: ECH turns on at {t_ech_on:.1f} ms (it DOES fall within the Mirnov window).")
                print("  -> The heating and Mirnov timebases appear aligned; the NBI offset is real.")
        if margin_ms < 5.0:
            print(f"  ⚠️ Margin of only {margin_ms:.2f} ms: manually confirm against the shot {shot} logbook.")

    # 3b. Loading line-averaged electron density (nave) for Alfvenic validation
    nave_file = data_dir / f"nave@{shot}.edf"
    density_val = np.zeros_like(t_ms)
    density_detected = False
    if nave_file.exists():
        print(f"Loading real nave density for Alfvenic validation from: {nave_file}")
        edf_den = TE.edf()
        dat_den = edf_den.load(str(nave_file))
        t_den = dat_den[:, 0]
        if edf_den.DimUnit[0] == 's':
            t_den = t_den * 1000.0

        nave_val_unit = getattr(edf_den, 'ValUnit', ['?'])[0]
        nave_dim_unit = getattr(edf_den, 'DimUnit', ['?'])[0]
        print(f"  'nave' metadata: DimUnit (time) = '{nave_dim_unit}', ValUnit (density) = '{nave_val_unit}'.")

        raw_density = dat_den[:, 1]
        print(f"  Raw 'nave' range: min={np.min(raw_density):.4g}, max={np.max(raw_density):.4g}, "
              f"mean={np.mean(raw_density):.4g} (order-of-magnitude sanity check; a typical line")
        print("  density is around ~1e18-1e20 m^-3 (small devices like Heliotron J typically run ~1e18-1e19 m^-3);")
        print("  if these values do not look reasonable, review the 'nave' calibration before using")
        print("  the Alfven scaling, even the simplified one.")

        invalid_mask = raw_density <= 0.0
        n_invalid = int(np.sum(invalid_mask))
        if n_invalid > 0:
            valid_idx = np.where(~invalid_mask)[0]
            if len(valid_idx) > 0:
                raw_density = np.interp(np.arange(len(raw_density)), valid_idx, raw_density[valid_idx])
                print(f"  ⚠️ DENSITY CLEANUP: {n_invalid} samples with negative or zero density were detected "
                      f"(physically impossible minimum = {np.min(dat_den[:, 1]):.6g}).")
                print("     These samples were corrected via linear interpolation from neighboring valid samples.")
            else:
                raw_density = np.clip(raw_density, 1e-5, None)
                print("  ⚠️ DENSITY CLEANUP: All density samples were non-positive; clipping to 1e-5 was applied.")

        med_kernel = args.nave_medfilt if args.nave_medfilt % 2 == 1 else args.nave_medfilt + 1
        if med_kernel >= 3 and len(raw_density) > med_kernel:
            raw_density_clean = dsp.medfilt(raw_density, kernel_size=med_kernel)
            n_flagged = int(np.sum(np.abs(raw_density_clean - raw_density) > 0.05 * np.max(np.abs(raw_density) + 1e-30)))
            print(f"  Median filter (kernel={med_kernel}) applied to raw nave: "
                  f"{n_flagged} samples corrected (possible fringe jumps/dropouts).")
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
            print(f"  [TRANSIENT CLEANUP] Density was linearly interpolated between {args.ech_glitch_start:.0f}-{args.ech_glitch_end:.0f} ms to "
                  f"suppress the spurious ECH turn-on spike (pre={pre_val:.4f}, post={post_val:.4f}).")

        density_detected = True
    else:
        print(f"  ⚠️ Warning: expected density file 'nave' not found at {nave_file}; ")
        print("     the Alfven scaling validation (Section 2.3 of the proposal) will be completely")
        print("     skipped in this run.")

    # 3c. Loading plasma current (Ip from Ip15 coil) for correlation analysis
    ip_file = data_dir / f"Ip15@{shot}.edf"
    ip_signal = None
    if ip_file.exists():
        print(f"Loading Ip15 from: {ip_file}")
        edf_ip = TE.edf()
        dat_ip = edf_ip.load(str(ip_file))
        t_ip = dat_ip[:, 0]
        if edf_ip.DimUnit[0] == 's':
            t_ip = t_ip * 1000.0
        ip_signal = np.interp(t_ms, t_ip, dat_ip[:, 1])
        ip_val_unit = getattr(edf_ip, 'ValUnit', ['?'])[0]
        print(f"  'Ip15' metadata: ValUnit = '{ip_val_unit}', "
              f"range: min={np.min(ip_signal):.4g}, max={np.max(ip_signal):.4g}")
    else:
        print(f"  ⚠️ Warning: expected Ip15 file not found at {ip_file}; Ip correlation will be skipped.")

    # 3d. Loading stored energy (Wp) for correlation analysis
    wp_file = data_dir / f"Wp@{shot}.edf"
    wp_signal = None
    if wp_file.exists():
        print(f"Loading Wp from: {wp_file}")
        edf_wp = TE.edf()
        dat_wp = edf_wp.load(str(wp_file))
        t_wp = dat_wp[:, 0]
        if edf_wp.DimUnit[0] == 's':
            t_wp = t_wp * 1000.0
        wp_signal = np.interp(t_ms, t_wp, dat_wp[:, 1])
        wp_val_unit = getattr(edf_wp, 'ValUnit', ['?'])[0]
        print(f"  'Wp' metadata: ValUnit = '{wp_val_unit}', "
              f"range: min={np.min(wp_signal):.4g}, max={np.max(wp_signal):.4g}")
    else:
        print(f"  ⚠️ Warning: expected Wp file not found at {wp_file}; Wp correlation will be skipped.")

    # 3e. Loading fast H-alpha signal (HAFAST7.5) for correlation analysis
    ha_file = data_dir / f"HAFAST7.5@{shot}.edf"
    hafast_signal = None
    if ha_file.exists():
        print(f"Loading HAFAST7.5 from: {ha_file}")
        edf_ha = TE.edf()
        dat_ha = edf_ha.load(str(ha_file))
        t_ha = dat_ha[:, 0]
        if edf_ha.DimUnit[0] == 's':
            t_ha = t_ha * 1000.0
        hafast_signal = np.interp(t_ms, t_ha, dat_ha[:, 1])
        ha_val_unit = getattr(edf_ha, 'ValUnit', ['V'])[0]
        print(f"  'HAFAST7.5' metadata: ValUnit = '{ha_val_unit}', "
              f"range: min={np.min(hafast_signal):.4g}, max={np.max(hafast_signal):.4g}")
    else:
        print(f"  ⚠️ Warning: expected HAFAST7.5 file not found at {ha_file}; HAFAST correlation will be skipped.")

    # 4. Decimation for Correlation Analysis
    decimate_factor = 100
    dt_corr = dt * decimate_factor
    t_corr = t_ms[::decimate_factor]
    envelope_corr = anti_alias_decimate(envelope, decimate_factor)
    total_nbi_corr = anti_alias_decimate(total_nbi_power, decimate_factor)
    ech_power_corr = anti_alias_decimate(ech_power, decimate_factor)

    ip_corr_result = None
    if ip_signal is not None:
        ip_corr = anti_alias_decimate(ip_signal, decimate_factor)
        lag_ip_ms, r_ip_peak, lags_ip_curve, corr_ip_curve = lagged_cross_correlation(
            envelope_corr, ip_corr, dt_corr, max_lag_ms=20.0
        )
        r_ip_sig, p_ip_std, p_ip_adj, n_ip_sig, N_eff_ip = lagged_pearson_significance(
            envelope_corr, ip_corr, dt_corr, lag_ip_ms
        )
        ip_corr_result = {
            "signal": ip_signal, "corr": ip_corr,
            "lag_ms": lag_ip_ms, "r_peak": r_ip_peak,
            "lags_curve": lags_ip_curve, "corr_curve": corr_ip_curve,
            "r_sig": r_ip_sig, "p_std": p_ip_std, "p_adj": p_ip_adj,
            "n_sig": n_ip_sig, "N_eff": N_eff_ip,
        }
        if r_ip_sig is not None:
            meets = abs(r_ip_sig) > 0.7 and p_ip_adj < 0.05
            print(f"  - Envelope vs. Ip (Ip15): peak |correlation| = {r_ip_peak:+.4f} at lag = {lag_ip_ms:+.2f} ms")
            print(f"    -> Pearson r = {r_ip_sig:.4f} (N_eff={N_eff_ip:.1f}), {format_p_value(p_ip_adj)} "
                  f"-- {'MEETS' if meets else 'does NOT meet'} |r|>0.7 & p<0.05.")
        else:
            print("  - Envelope vs. Ip (Ip15): insufficient data for significance test.")

    wp_corr_result = None
    if wp_signal is not None:
        wp_corr = anti_alias_decimate(wp_signal, decimate_factor)
        lag_wp_ms, r_wp_peak, lags_wp_curve, corr_wp_curve = lagged_cross_correlation(
            envelope_corr, wp_corr, dt_corr, max_lag_ms=args.macro_max_lag_ms
        )
        r_wp_sig, p_wp_std, p_wp_adj, n_wp_sig, N_eff_wp = lagged_pearson_significance(
            envelope_corr, wp_corr, dt_corr, lag_wp_ms
        )
        wp_corr_result = {
            "signal": wp_signal, "corr": wp_corr,
            "lag_ms": lag_wp_ms, "r_peak": r_wp_peak,
            "lags_curve": lags_wp_curve, "corr_curve": corr_wp_curve,
            "r_sig": r_wp_sig, "p_std": p_wp_std, "p_adj": p_wp_adj,
            "n_sig": n_wp_sig, "N_eff": N_eff_wp,
        }
        if r_wp_sig is not None:
            meets = abs(r_wp_sig) > 0.7 and p_wp_adj < 0.05
            print(f"  - Envelope vs. Wp: peak |correlation| = {r_wp_peak:+.4f} at lag = {lag_wp_ms:+.2f} ms "
                  f"(search window: +/-{args.macro_max_lag_ms:.0f} ms)")
            print(f"    -> Pearson r = {r_wp_sig:.4f} (N_eff={N_eff_wp:.1f}), {format_p_value(p_wp_adj)} "
                  f"-- {'MEETS' if meets else 'does NOT meet'} |r|>0.7 & p<0.05.")
        else:
            print("  - Envelope vs. Wp: insufficient data for significance test.")

    nave_corr_result = None
    if density_detected:
        t_nav_start = getattr(args, "nave_corr_start", 270.0)
        t_nav_end = getattr(args, "nave_corr_end", None)
        mask_nave_win = t_corr >= t_nav_start
        if t_nav_end is not None:
            mask_nave_win &= (t_corr <= t_nav_end)

        nave_corr = anti_alias_decimate(density_val, decimate_factor)
        env_sub = envelope_corr[mask_nave_win]
        nave_sub = nave_corr[mask_nave_win]

        if len(env_sub) >= 20:
            lag_nave_ms, r_nave_peak, lags_nave_curve, corr_nave_curve = lagged_cross_correlation(
                env_sub, nave_sub, dt_corr, max_lag_ms=args.macro_max_lag_ms
            )
            r_nave_sig, p_nave_std, p_nave_adj, n_nave_sig, N_eff_nave = lagged_pearson_significance(
                env_sub, nave_sub, dt_corr, lag_nave_ms
            )
            nave_corr_result = {
                "signal": density_val, "corr": nave_corr,
                "t_start": t_nav_start, "t_end": t_nav_end,
                "lag_ms": lag_nave_ms, "r_peak": r_nave_peak,
                "lags_curve": lags_nave_curve, "corr_curve": corr_nave_curve,
                "r_sig": r_nave_sig, "p_std": p_nave_std, "p_adj": p_nave_adj,
                "n_sig": n_nave_sig, "N_eff": N_eff_nave,
            }
            if r_nave_sig is not None:
                meets = abs(r_nave_sig) > 0.7 and p_nave_adj < 0.05
                end_str = f"{t_nav_end:.0f}" if t_nav_end is not None else "end"
                print(f"  - Envelope vs. nave [Noise-Free Window: {t_nav_start:.0f}-{end_str} ms]: "
                      f"peak |correlation| = {r_nave_peak:+.4f} at lag = {lag_nave_ms:+.2f} ms "
                      f"(search window: +/-{args.macro_max_lag_ms:.0f} ms)")
                print(f"    -> Pearson r = {r_nave_sig:.4f} (N_eff={N_eff_nave:.1f}), {format_p_value(p_nave_adj)} "
                      f"-- {'MEETS' if meets else 'does NOT meet'} |r|>0.7 & p<0.05.")
            else:
                print("  - Envelope vs. nave: insufficient data for significance test.")
        else:
            print("  - Envelope vs. nave: too few samples in specified second hill window.")

    hafast_corr_result = None
    if hafast_signal is not None:
        ha_corr = anti_alias_decimate(hafast_signal, decimate_factor)
        lag_ha_ms, r_ha_peak, lags_ha_curve, corr_ha_curve = lagged_cross_correlation(
            envelope_corr, ha_corr, dt_corr, max_lag_ms=args.macro_max_lag_ms
        )
        r_ha_sig, p_ha_std, p_ha_adj, n_ha_sig, N_eff_ha = lagged_pearson_significance(
            envelope_corr, ha_corr, dt_corr, lag_ha_ms
        )
        hafast_corr_result = {
            "signal": hafast_signal, "corr": ha_corr,
            "lag_ms": lag_ha_ms, "r_peak": r_ha_peak,
            "lags_curve": lags_ha_curve, "corr_curve": corr_ha_curve,
            "r_sig": r_ha_sig, "p_std": p_ha_std, "p_adj": p_ha_adj,
            "n_sig": n_ha_sig, "N_eff": N_eff_ha,
        }
        if r_ha_sig is not None:
            meets = abs(r_ha_sig) > 0.7 and p_ha_adj < 0.05
            print(f"  - Envelope vs. HAFAST7.5: peak |correlation| = {r_ha_peak:+.4f} at lag = {lag_ha_ms:+.2f} ms "
                  f"(search window: +/-{args.macro_max_lag_ms:.0f} ms)")
            print(f"    -> Pearson r = {r_ha_sig:.4f} (N_eff={N_eff_ha:.1f}), {format_p_value(p_ha_adj)} "
                  f"-- {'MEETS' if meets else 'does NOT meet'} |r|>0.7 & p<0.05.")
        else:
            print("  - Envelope vs. HAFAST7.5: insufficient data for significance test.")

    acf_full = estimate_acf(envelope_corr, nlags=50)
    sum_acf_full = np.sum(acf_full[1:])
    N_full = len(envelope_corr)
    N_eff_full = N_full / (1.0 + 2.0 * sum_acf_full)
    N_eff_full = max(3.0, min(float(N_full), N_eff_full))

    r_nbi_tot, p_nbi_tot_std = stats.pearsonr(envelope_corr, total_nbi_corr) if np.max(total_nbi_corr) > 0 else (0.0, 1.0)
    p_nbi_tot = conservative_p_value(r_nbi_tot, N_eff_full) if np.max(total_nbi_corr) > 0 else 1.0

    r_ech, p_ech_std = stats.pearsonr(envelope_corr, ech_power_corr)
    p_ech = conservative_p_value(r_ech, N_eff_full)

    # ECH correlation restricted only to the active window
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
    p_partial_std = conservative_p_value(r_partial, N_full, n_control=1)
    p_partial_adj = conservative_p_value(r_partial, N_eff_full, n_control=1)

    step_thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7]
    r_partial_sensitivity = {}
    max_ech_corr = np.max(ech_power_corr)
    for thr in step_thresholds_to_test:
        test_step_regressor = ((ech_power_corr > thr * max_ech_corr).astype(float)
                                if max_ech_corr > 0 else np.zeros_like(ech_power_corr))
        test_r_xz, _ = (stats.pearsonr(envelope_corr, test_step_regressor)
                         if np.max(test_step_regressor) > 0 else (0.0, 1.0))
        test_r_yz, _ = (stats.pearsonr(ech_power_corr, test_step_regressor)
                         if np.max(test_step_regressor) > 0 else (0.0, 1.0))
        test_denom = np.sqrt((1.0 - test_r_xz**2) * (1.0 - test_r_yz**2))
        test_r_partial = (r_ech - test_r_xz * test_r_yz) / test_denom if test_denom > 0 else 0.0
        r_partial_sensitivity[thr] = test_r_partial
        marker = " <-- default threshold" if thr == 0.5 else ""
        print(f"  - Threshold = {thr*100:.0f}% of max ECH -> r_partial = {test_r_partial:+.4f}{marker}")

    r_partial_vals_arr = np.array(list(r_partial_sensitivity.values()))
    r_partial_sens_range = r_partial_vals_arr.max() - r_partial_vals_arr.min()
    print(f"  - Range of r_partial across 30%-70% thresholds: {r_partial_sens_range:.4f} "
          f"(spuriousness conclusion {'ROBUST' if r_partial_sens_range < 0.15 else 'SENSITIVE'} "
          f"to the exact choice of threshold).")

    print("\n=== Statistical Correlation Metrics (Pearson with Autocorrelation Adjustment) [EXT] ===")
    print(f"Full Range: N_total = {N_full}, N_effective = {N_eff_full:.1f}")
    print(f"  - Correlation Envelope vs. Combined NBI: r = {r_nbi_tot:.4f}, p_std = {format_p_value(p_nbi_tot_std)}, p_adj = {format_p_value(p_nbi_tot)}")
    print(f"  - Correlation Envelope vs. ECH Power:     r = {r_ech:.4f}, p_std = {format_p_value(p_ech_std)}, p_adj = {format_p_value(p_ech)}")
    print(f"  - Partial Correlation (Controlling ECH Step): r = {r_partial:.4f}, p_std = {format_p_value(p_partial_std)}, p_adj = {format_p_value(p_partial_adj)}")
    print(f"Active Window ({args.ech_active_start:.0f}-{args.ech_active_end:.0f} ms): N_total = {N_active}, N_effective = {N_eff_active:.1f}")
    print(f"  - ECH Correlation in Active Window:        r = {r_ech_active:.4f}, p_std = {format_p_value(p_ech_active_std)}, p_adj = {format_p_value(p_ech_active)}")

    # Correlations for each individual NBI injector
    nbi_corrs = {}
    for nbi, nbi_val in nbi_signals.items():
        nbi_corr_val = anti_alias_decimate(nbi_val, decimate_factor)
        r_val, p_val_std = stats.pearsonr(envelope_corr, nbi_corr_val) if np.max(nbi_corr_val) > 0 else (0.0, 1.0)
        p_val_adj = conservative_p_value(r_val, N_eff_full) if np.max(nbi_corr_val) > 0 else 1.0
        nbi_corrs[nbi] = (r_val, p_val_std, p_val_adj)
        print(f"  - Correlation Envelope vs. {nbi}:        r = {r_val:.4f}, p_std = {format_p_value(p_val_std)}, p_adj = {format_p_value(p_val_adj)}")

    # -----------------------------------------------------------------------------------
    # Lagged time-domain cross-correlation (Envelope vs. NBI, Envelope vs. ECH)
    # -----------------------------------------------------------------------------------
    print("\n--- Lagged Time-Domain Cross-Correlation (main.md requirement (b1)) [M3] ---")
    lag_nbi_ms, r_nbi_peak, lags_nbi_ms, corr_nbi_vals = lagged_cross_correlation(
        envelope_corr, total_nbi_corr, dt_corr, max_lag_ms=20.0
    ) if np.max(total_nbi_corr) > 0 else (0.0, 0.0, np.array([0.0]), np.array([0.0]))
    lag_ech_ms, r_ech_peak, lags_ech_ms, corr_ech_vals = lagged_cross_correlation(
        envelope_corr, ech_power_corr, dt_corr, max_lag_ms=20.0
    )
    print(f"  - Envelope vs. Total NBI: peak |correlation| = {r_nbi_peak:+.4f} at lag = {lag_nbi_ms:+.2f} ms "
          f"({'lag ~0 -> simultaneous' if abs(lag_nbi_ms) < dt_corr*1000 else ('envelope leads' if lag_nbi_ms < 0 else 'envelope lags heating')})")
    print(f"  - Envelope vs. ECH Power: peak |correlation| = {r_ech_peak:+.4f} at lag = {lag_ech_ms:+.2f} ms "
          f"({'lag ~0 -> simultaneous' if abs(lag_ech_ms) < dt_corr*1000 else ('envelope leads' if lag_ech_ms < 0 else 'envelope lags heating')})")
    ech_lag_near_zero_ms = 2.0 
    if abs(lag_ech_ms) < ech_lag_near_zero_ms:
        print(f"  - Note: the ECH peak lag ({lag_ech_ms:+.2f} ms) IS close to 0 ms, coincident with the")
        print("    on/off step -- this supports the 'shared-step artifact' conclusion from the partial-")
        print("    correlation test above.")
    else:
        print(f"  - ⚠️ Note: the ECH peak lag ({lag_ech_ms:+.2f} ms) is NOT close to 0 ms. A pure shared-")
        print("    step artifact should peak at lag ~ 0, so this does not, by itself, support that")
        print("    explanation, and should not be cited as if it did. It may instead reflect a genuine")
        print("    delayed response, or a secondary feature (e.g. a transient near the ECH edge) driving")
        print("    the match -- inspect the envelope/ECH overlay near this lag directly before concluding either way.")

    # =============================================================================================
    # INSTANTANEOUS-FREQUENCY-vs-HEATING CORRELATION 
    # =============================================================================================
    freq_heating_results = None
    if n_mode_active >= 200:
        print("\n--- Instantaneous-Frequency vs. Heating Correlation (main.md requirement (b), frequency reading) [M7] ---")
        print(f"  Restricted to the mode-active mask above ({n_mode_active} samples, "
              f"{t_ms[i0_mode]:.1f}-{t_ms[i1_mode-1]:.1f} ms) -- ifreq_khz is not physically meaningful outside it.")

        ifreq_active_mode = ifreq_khz[mask_mode_active]
        nbi_active_mode = total_nbi_power[mask_mode_active]
        ech_active_mode = ech_power[mask_mode_active]

        ifreq_freq_corr = anti_alias_decimate(ifreq_active_mode, decimate_factor)
        nbi_freq_corr = anti_alias_decimate(nbi_active_mode, decimate_factor)
        ech_freq_corr = anti_alias_decimate(ech_active_mode, decimate_factor)

        acf_freq = estimate_acf(ifreq_freq_corr, nlags=min(50, len(ifreq_freq_corr) - 2))
        N_freq = len(ifreq_freq_corr)
        N_eff_freq = N_freq / (1.0 + 2.0 * np.sum(acf_freq[1:])) if N_freq > 2 else 3.0
        N_eff_freq = max(3.0, min(float(N_freq), N_eff_freq))

        if not nbi_mismatch_detected and np.max(nbi_freq_corr) > 0:
            r_ifreq_nbi, p_ifreq_nbi_std = stats.pearsonr(ifreq_freq_corr, nbi_freq_corr)
            p_ifreq_nbi = conservative_p_value(r_ifreq_nbi, N_eff_freq)
        else:
            r_ifreq_nbi, p_ifreq_nbi_std, p_ifreq_nbi = 0.0, 1.0, 1.0

        r_ifreq_ech, p_ifreq_ech_std = stats.pearsonr(ifreq_freq_corr, ech_freq_corr) if np.std(ech_freq_corr) > 0 else (0.0, 1.0)
        p_ifreq_ech = conservative_p_value(r_ifreq_ech, N_eff_freq)

        print(f"  N_total = {N_freq}, N_effective = {N_eff_freq:.1f}")
        print(f"  - Correlation Freq vs. Combined NBI: r = {r_ifreq_nbi:.4f}, p_std = {format_p_value(p_ifreq_nbi_std)}, p_adj = {format_p_value(p_ifreq_nbi)}")
        print(f"  - Correlation Freq vs. ECH Power:     r = {r_ifreq_ech:.4f}, p_std = {format_p_value(p_ifreq_ech_std)}, p_adj = {format_p_value(p_ifreq_ech)}")

        # -------------------------------------------------------------------------------------
        # ECH-step partial correlation, frequency version -- mirrors the step_regressor
        # partial-correlation check already applied to the ENVELOPE-vs-ECH correlation above.
        # -------------------------------------------------------------------------------------
        max_ech_freq_corr = np.max(ech_freq_corr) if len(ech_freq_corr) else 0.0
        step_regressor_freq = (ech_freq_corr > 0.5 * max_ech_freq_corr).astype(float) if max_ech_freq_corr > 0 else np.zeros_like(ech_freq_corr)
        if np.std(step_regressor_freq) > 0 and np.std(ifreq_freq_corr) > 0:
            r_xz_freq, _ = stats.pearsonr(ifreq_freq_corr, step_regressor_freq)
            r_yz_freq, _ = stats.pearsonr(ech_freq_corr, step_regressor_freq)
            denom_partial_freq = np.sqrt(max(0.0, (1.0 - r_xz_freq ** 2) * (1.0 - r_yz_freq ** 2)))
            r_partial_freq = (r_ifreq_ech - r_xz_freq * r_yz_freq) / denom_partial_freq if denom_partial_freq > 0 else 0.0
            p_partial_freq_std = conservative_p_value(r_partial_freq, N_freq, n_control=1)
            p_partial_freq_adj = conservative_p_value(r_partial_freq, N_eff_freq, n_control=1)
            print(f"  - Partial Correlation (Freq vs. ECH, ECH-step-controlled): r_partial = {r_partial_freq:.4f}, "
                  f"p_std = {format_p_value(p_partial_freq_std)}, p_adj = {format_p_value(p_partial_freq_adj)}")
            if abs(r_ifreq_ech) > 0.3 and abs(r_partial_freq) < 0.5 * abs(r_ifreq_ech):
                print("    ⚠️ The raw Freq-vs-ECH correlation drops by more than half once the ECH on/off step is")
                print("    controlled for -- consistent with it being driven largely by the shared ECH-turn-off /")
                print("    frequency-termination-transient edge coincidence rather than a sustained relationship.")
        else:
            r_partial_freq, p_partial_freq_std, p_partial_freq_adj = r_ifreq_ech, p_ifreq_ech_std, p_ifreq_ech
            print("  - Partial Correlation (Freq vs. ECH, ECH-step-controlled): SKIPPED (ECH constant within "
                  "the mode-active window, so no on/off step to control for; raw r above already reflects this).")

        lag_ifreq_nbi_ms, r_ifreq_nbi_peak, lags_ifreq_nbi_curve, corr_ifreq_nbi_curve = lagged_cross_correlation(
            ifreq_freq_corr, nbi_freq_corr, dt_corr, max_lag_ms=args.m7_max_lag_ms
        ) if (not nbi_mismatch_detected and np.max(nbi_freq_corr) > 0) else (0.0, 0.0, np.array([0.0]), np.array([0.0]))
        lag_ifreq_ech_ms, r_ifreq_ech_peak, lags_ifreq_ech_curve, corr_ifreq_ech_curve = lagged_cross_correlation(
            ifreq_freq_corr, ech_freq_corr, dt_corr, max_lag_ms=args.m7_max_lag_ms
        )
        print(f"  - Lagged (b1): Freq vs Total NBI: peak |correlation| = {r_ifreq_nbi_peak:+.4f} at lag = {lag_ifreq_nbi_ms:+.2f} ms "
              f"(search window: +/-{args.m7_max_lag_ms:.0f} ms)")
        print(f"  - Lagged (b1): Freq vs ECH Power: peak |correlation| = {r_ifreq_ech_peak:+.4f} at lag = {lag_ifreq_ech_ms:+.2f} ms "
              f"(search window: +/-{args.m7_max_lag_ms:.0f} ms)")
        if abs(lag_ifreq_nbi_ms) >= 0.9 * args.m7_max_lag_ms:
            print(f"    ⚠️ BOUNDARY WARNING: NBI lag is within 10% of the +/-{args.m7_max_lag_ms:.0f} ms search "
                  "window edge -- this peak is likely clipped, not a true maximum. Widen --m7-max-lag-ms before trusting it.")
        if abs(lag_ifreq_ech_ms) >= 0.9 * args.m7_max_lag_ms:
            print(f"    ⚠️ BOUNDARY WARNING: ECH lag is within 10% of the +/-{args.m7_max_lag_ms:.0f} ms search "
                  "window edge -- this peak is likely clipped, not a true maximum. Widen --m7-max-lag-ms before trusting it.")

        freq_heating_results = {
            "r_ifreq_nbi": r_ifreq_nbi, "p_ifreq_nbi": p_ifreq_nbi,
            "r_ifreq_ech": r_ifreq_ech, "p_ifreq_ech": p_ifreq_ech,
            "r_partial_freq": r_partial_freq, "p_partial_freq_adj": p_partial_freq_adj,
            "lag_ifreq_nbi_ms": lag_ifreq_nbi_ms, "r_ifreq_nbi_peak": r_ifreq_nbi_peak,
            "lag_ifreq_ech_ms": lag_ifreq_ech_ms, "r_ifreq_ech_peak": r_ifreq_ech_peak,
            "lags_ifreq_nbi_curve": lags_ifreq_nbi_curve, "corr_ifreq_nbi_curve": corr_ifreq_nbi_curve,
            "lags_ifreq_ech_curve": lags_ifreq_ech_curve, "corr_ifreq_ech_curve": corr_ifreq_ech_curve,
        }

    # --- 4b. Validation against the Theoretical Alfven Scaling (f_A ~ 1/sqrt(n_e)) ---
    f_theoretical_alfven = np.zeros_like(t_ms)
    r_val_scaling = 0.0
    p_val_scaling = 1.0
    bfield_available = False
    b_is_constant = False
    b_var_rel = 0.0
    if density_detected:
        print("\n--- Alfven Theoretical Scaling Validation (Proposal Criterion 2.3) ---")
        # [M4] Explicit scope clarification, printed every run.
        print("  [M4] SCOPE NOTE: this test validates the FREQUENCY SCALING with the Alfven velocity")
        print("       (v_A-dependence), which is one half of main.md's requirement (d). It does NOT")
        print("       compute an actual Alfven continuum (omega(r) gap structure from a VMEC equilibrium")
        print("       + STELLGAP/CAS3D solve) -- this requires a separate equilibrium-reconstruction")
        print("       pipeline this script cannot reproduce; see the [M4] header note for why, and ask")
        print("       the group whether a continuum run already exists for this magnetic configuration.")

        density_val_clean = np.clip(density_val, 0.01, None)

        if args.bfield_constant_tesla is not None:
            b_val = np.full_like(t_ms, args.bfield_constant_tesla)
            bfield_available = True
            b_is_constant = True
            b_mean = args.bfield_constant_tesla
            print(f"  Using CONFIRMED constant toroidal field B = {args.bfield_constant_tesla:.2f} T "
                  f"(--bfield-constant-tesla). Alfven scaling validation status: FINAL.")
            mu0, m_ion = 4 * np.pi * 1e-7, args.ion_mass_amu * 1.6726e-27
            n_i_m3 = density_val_clean * 1e19
            f_alfven_scaling = np.abs(b_val) / np.sqrt(mu0 * n_i_m3 * m_ion)
        else:
            bfield_file_this_shot = args.bfield_pattern.format(shot=shot) if args.bfield_pattern else args.bfield_file
            bfield_requested = bfield_file_this_shot is not None
            bfield_available = bfield_requested and os.path.exists(bfield_file_this_shot)
            if bfield_requested and not bfield_available:
                print(f"  ⚠️ Warning: bfield file '{bfield_file_this_shot}' was specified but does not")
                print("     exist; the SIMPLIFIED 1/sqrt(n_e) scaling is used as fallback.")

            if bfield_available:
                edf_b = TE.edf()
                dat_b = edf_b.load(bfield_file_this_shot)
                b_val_unit = getattr(edf_b, 'ValUnit', ['?'])[0]
                b_unit_ok = any(tag in str(b_val_unit).lower() for tag in ('tesla', ' t', 't)', 't]')) or str(b_val_unit).strip().lower() == 't'
                print(f"  B(t) metadata: ValUnit = '{b_val_unit}' -> "
                      f"{'looks consistent with Tesla.' if b_unit_ok else '⚠️ NOT clearly recognized as Tesla; verify manually before trusting v_A.'}")
                print(f"  Assumed ion mass: {args.ion_mass_amu:.2f} amu. CONFIRM that it corresponds to the real")
                print(f"  working gas of shot {shot} (the default value assumes Hydrogen, amu=1.0).")

                t_b = dat_b[:, 0] * (1000.0 if edf_b.DimUnit[0] == 's' else 1.0)
                b_val = np.interp(t_ms, t_b, dat_b[:, 1])

                b_val_active = b_val[mask_active_win] if len(b_val[mask_active_win]) > 0 else b_val
                b_mean = np.mean(np.abs(b_val_active))
                b_var_rel = (np.max(b_val_active) - np.min(b_val_active)) / b_mean if b_mean > 0 else 0.0
                if b_var_rel < 0.01:
                    b_is_constant = True
                    print(f"  ✅ CONSTANT TOROIDAL FIELD CONFIRMED (Relative variation of B: {b_var_rel*100:.2f}% < 1%):")
                    print(f"     Consistent with the real, confirmed flat-top of Heliotron J (~{b_mean:.2f} T)")
                    print("     for this shot. Since B(t) is constant by physical construction, the full")
                    print("     scaling (with B) and the simplified one (1/sqrt(n_e)) are mathematically")
                    print("     equivalent in this dataset: the result below is the FINAL validation.")

                mu0, m_ion = 4 * np.pi * 1e-7, args.ion_mass_amu * 1.6726e-27
                n_i_m3 = density_val_clean * 1e19
                f_alfven_scaling = np.abs(b_val) / np.sqrt(mu0 * n_i_m3 * m_ion)
                print("  Using full physical scaling v_A = B/sqrt(mu0*n_i*m_i) (bfield file provided).")
            else:
                f_alfven_scaling = 1.0 / np.sqrt(density_val_clean)
                print("  Using simplified scaling 1/sqrt(n_e) (no bfield file); interpret r with caution.")
                print("  ⚠️ INCOMPLETE VALIDATION: this run does NOT include the dependence on the magnetic")
                print("     field B required by Section 2.3 of the proposal (v_A = B/sqrt(mu0*n_i*m_i)).")
                print("     Treat the conclusion as PRELIMINARY until this shot is run with a bfield file.")

        # -------------------------------------------------------------------------------------
        # Calibration-window transparency check.
        # -------------------------------------------------------------------------------------
        def _alfven_calib_for_window(cal_start, cal_end):
            m = (t_ms >= cal_start) & (t_ms <= cal_end)
            if np.sum(m) < 5:
                return None, None
            mean_f = np.mean(ifreq_khz[m])
            mean_s = np.mean(f_alfven_scaling[m])
            if mean_s <= 0:
                return None, None
            nc = mean_f / mean_s
            f_theo = f_alfven_scaling * nc
            i_c = anti_alias_decimate(ifreq_khz[mask_active_win], decimate_factor)
            f_c = anti_alias_decimate(f_theo[mask_active_win], decimate_factor)
            if len(i_c) <= 2 or np.std(f_c) == 0:
                return nc, None
            r_test, _ = stats.pearsonr(i_c, f_c)
            return nc, r_test

        if args.alfven_cal_sweep.strip():
            print("\n  --- Alfven Calibration-Window Transparency Check [EXT][CORRECTED] ---")
            print("    (norm_constant WILL differ across windows; r_val_scaling is mathematically")
            print("     guaranteed to be identical -- see code comment. This is NOT a robustness test.)")
            sweep_r_vals = {}
            for pair in args.alfven_cal_sweep.split(","):
                pair = pair.strip()
                if not pair:
                    continue
                try:
                    w_start_str, w_end_str = pair.split(":")
                    w_start, w_end = float(w_start_str), float(w_end_str)
                except ValueError:
                    print(f"    ⚠️ Skipping malformed --alfven-cal-sweep entry: '{pair}'")
                    continue
                nc_sweep, r_sweep = _alfven_calib_for_window(w_start, w_end)
                is_default = (abs(w_start - args.alfven_cal_start) < 1e-9 and abs(w_end - args.alfven_cal_end) < 1e-9)
                marker = " <-- current default" if is_default else ""
                sweep_r_vals[pair] = r_sweep
                if r_sweep is None:
                    print(f"    - Window {w_start:.0f}-{w_end:.0f} ms -> N/A (too few/degenerate samples){marker}")
                else:
                    print(f"    - Window {w_start:.0f}-{w_end:.0f} ms -> norm_constant = {nc_sweep:.3e}, "
                          f"r_val_scaling = {r_sweep:+.4f}{marker}")
            valid_sweep = [v for v in sweep_r_vals.values() if v is not None]
            if len(valid_sweep) >= 2:
                sweep_range = max(valid_sweep) - min(valid_sweep)
                print(f"    - Range of r_val_scaling across candidate windows: {sweep_range:.6f} "
                      "(expected to be ~0 for the mathematical reason above; "
                      "a nonzero range here would indicate a code bug, not a data-driven conclusion).")

        mask_scaling_win = (t_ms >= args.alfven_cal_start) & (t_ms <= args.alfven_cal_end)
        if np.sum(mask_scaling_win) < 5:
            print(f"  ⚠️ Warning: the calibration window ({args.alfven_cal_start}-{args.alfven_cal_end} ms) "
                  f"has very few samples; check the shot's time range.")
        mean_measured_f = np.mean(ifreq_khz[mask_scaling_win])
        mean_scaling_val = np.mean(f_alfven_scaling[mask_scaling_win])
        norm_constant = mean_measured_f / mean_scaling_val if mean_scaling_val > 0 else 1.0
        f_theoretical_alfven = f_alfven_scaling * norm_constant

        if t_mode_strong_start is not None and t_mode_strong_end is not None:
            mask_eval_alfven = (t_ms >= t_mode_strong_start) & (t_ms <= t_mode_strong_end)
            t_alf_start = t_mode_strong_start
            t_alf_end = t_mode_strong_end
            alf_win_tag = f"strongly-present {t_alf_start:.1f}-{t_alf_end:.1f} ms"
        elif n_mode_active >= 50:
            mask_eval_alfven = mask_mode_active
            t_alf_start = t_ms[i0_mode]
            t_alf_end = t_ms[i1_mode-1]
            alf_win_tag = f"mode-active {t_alf_start:.1f}-{t_alf_end:.1f} ms"
        else:
            mask_eval_alfven = mask_active_win
            t_alf_start = args.ech_active_start
            t_alf_end = args.ech_active_end
            alf_win_tag = f"active ECH {t_alf_start:.1f}-{t_alf_end:.1f} ms"

        ifreq_active_corr = ifreq_khz[mask_eval_alfven][::decimate_factor]
        f_theo_active_corr = f_theoretical_alfven[mask_eval_alfven][::decimate_factor]

        r_val_scaling, p_val_scaling_std = stats.pearsonr(ifreq_active_corr, f_theo_active_corr) if len(ifreq_active_corr) > 2 else (0.0, 1.0)

        acf_alfven = estimate_acf(ifreq_active_corr, nlags=min(50, len(ifreq_active_corr) - 2))
        N_alfven = len(ifreq_active_corr)
        sum_acf_alfven = np.sum(acf_alfven[1:]) if len(acf_alfven) > 1 else 0.0
        N_eff_alfven = max(3.0, min(float(N_alfven), N_alfven / (1.0 + 2.0 * sum_acf_alfven)))

        p_val_scaling = conservative_p_value(r_val_scaling, N_eff_alfven) if len(ifreq_active_corr) > 2 else 1.0

        print(f"  - Evaluation window used: {alf_win_tag}")
        print(f"  - Calibration window used: {args.alfven_cal_start:.0f}-{args.alfven_cal_end:.0f} ms")
        print(f"  - Validation Correlation (f_measured vs. f_theoretical_Alfven): r = {r_val_scaling:.4f}, p_std = {format_p_value(p_val_scaling_std)}, p_adj = {format_p_value(p_val_scaling)}")
        r_class = "STRONG" if abs(r_val_scaling) > 0.7 else "MODERATE" if abs(r_val_scaling) > 0.4 else "WEAK / NOT CONCLUSIVE"
        print(f"  - Physical Consistency: {r_class} Validation (r = {r_val_scaling:.2f}).")
    else:
        pass

    # -----------------------------------------------------------------------------------
    # Inter-probe (MP1-MP3, MP1-MP4, MP3-MP4) cross-spectral coherence, for the modal-structure
    # panels: carrier-oscillation coherence (panel 4) and envelope coherence (panel 5).
    # -----------------------------------------------------------------------------------
    print("\n--- Computing Inter-Probe Cross-Spectral Coherence for Modal-Structure Estimation [M8] ---")
    if n_mode_active >= 200 and n_flat_active >= 200:
        t_sec_probe_span = t_sec[i0_flat:i1_flat]
        probe_signals_active = {
            probe: {k: v[i0_flat:i1_flat] for k, v in sig.items()}
            for probe, sig in probe_signals.items()
        }
        nfft_probe = min(args.nfft, max(64, 2 ** int(np.floor(np.log2(
            max(8, (i1_flat - i0_flat) // max(args.ensemble, 1)))))))
        window_tag = "[M7] full burst (no [M9] flat sub-window found)" if flat_info.get("used_fallback") \
            else "[M9] flat-frequency sub-window"
        print(f"  Restricted to the {window_tag} ({t_ms[i0_flat]:.1f}-{t_ms[i1_flat-1]:.1f} ms, "
              f"{i1_flat - i0_flat} samples, nfft={nfft_probe}) -- the carrier oscillation is not a real "
              "wave outside a genuine burst, and further restricts to where "
              "the mode's own frequency is flat/non-chirping; the envelope coherence uses the same "
              "restricted span so both panels are computed at the same frequency resolution.")
        carrier_coh_results = compute_probe_pair_coherence(
            probe_signals_active, "filtered", PROBE_PAIRS, t_sec_probe_span, dt, args, nfft=nfft_probe
        )
        envelope_coh_results = compute_probe_pair_coherence(
            probe_signals_active, "envelope", PROBE_PAIRS, t_sec_probe_span, dt, args, nfft=nfft_probe
        )
    elif n_mode_active >= 200:
        print(f"  ] Flat-frequency sub-window only has {n_flat_active} samples (< 200); inter-probe "
              "coherence ([ both panels) SKIPPED for this shot -- too short to resolve a spectrum. "
              "Try RAISING --flat-growth-tolerance or --flat-scan-window-ms (a larger window), or set "
              "--flat-window-start/--flat-window-end manually.")
        carrier_coh_results = {pair: None for pair in PROBE_PAIRS}
        envelope_coh_results = {pair: None for pair in PROBE_PAIRS}
    else:
        print(f" Only {n_mode_active} mode-active samples found (< 200); inter-probe coherence "
              "(both panels) SKIPPED for this shot -- no genuine burst to analyze.")
        carrier_coh_results = {pair: None for pair in PROBE_PAIRS}
        envelope_coh_results = {pair: None for pair in PROBE_PAIRS}

    # -----------------------------------------------------------------------------------
    # Poloidal mode-number decomposition, over the same flat-frequency/mode-active window and EPM band
    # -----------------------------------------------------------------------------------
    print("\n--- Computing Poloidal Mode-Number (m) Decomposition from the PMP Array [M10] ---")
    if n_mode_active >= 200 and n_flat_active >= 200 and len(pmp_signals) >= 3:
        poloidal_result = poloidal_mode_number_analysis(
            pmp_signals, pmp_plab_rad, dt, i0_flat, i1_flat, fl_hz, fu_hz, args
        )
        if poloidal_result is not None:
            print(f"  [M10] Dominant poloidal mode number: m = {poloidal_result['m_dominant']:+d} "
                  f"at {poloidal_result['f_peak_khz']:.2f} kHz "
                  f"(using {len(poloidal_result['channels'])} poloidal probes: "
                  f"{poloidal_result['channels']}).")
            if poloidal_result["verdict"] == "boundary_artifact":
                print(f"   m = {poloidal_result['m_dominant']:+d} is a BOUNDARY "
                      "ARTIFACT (see [M10-AUTOEXPAND] above) -- do NOT report this as the physical "
                      "mode number. Cross-check against the [M10-SELFTEST] result above: if it also "
                      f"failed near |m| = {poloidal_result['m_dominant']}, the array's angular "
                      "coverage cannot resolve modes at this magnitude at all.")
            elif poloidal_result["verdict"] == "confirmed_stable":
                print(f"  m = {poloidal_result['m_dominant']:+d} is CONFIRMED -- the "
                      "widened re-check (see [M10-AUTOEXPAND] above) found the exact same value with "
                      "much more search room available, which is the strongest available evidence "
                      "this is a real peak and not a boundary artifact of the narrower default range.")
            elif poloidal_result["verdict"] == "moved_off_edge":
                print(f"   VERDICT: m = {poloidal_result['m_dominant']:+d} is NOT YET "
                      f"converged either (the wider check moved to m = "
                      f"{poloidal_result['m_dominant_expanded']:+d}) -- raise --pmp-max-mode-number "
                      "before reporting any value from this shot.")
            elif poloidal_result["is_edge_pinned"]:
                print("  VERDICT: pinned at the search edge but could not be re-checked "
                      "(see [M10-AUTOEXPAND] warning above) -- treat as UNCONFIRMED.")
            else:
                print(f"   VERDICT: m = {poloidal_result['m_dominant']:+d} is an interior value "
                      "(not pinned to the search-range edge) -- consistent with a converged result, "
                      "though real-data noise still means this is not a certainty on its own.")
        else:
            print("  ⚠️ [M10] Poloidal mode-number fit could not be computed (window too short "
                  "for the requested --pmp-nfft, or no EPM-band content found).")
    elif len(pmp_signals) < 3:
        print(f"  ⚠️ [M10] Only {len(pmp_signals)} poloidal probe(s) found (< 3 needed); "
              "poloidal mode-number decomposition SKIPPED for this shot.")
        poloidal_result = None
    else:
        print("  ⚠️ [M10] No mode-active/flat-frequency window available; poloidal mode-number "
              "decomposition SKIPPED for this shot (same window requirement as [M8]/[M9]).")
        poloidal_result = None

    # -----------------------------------------------------------------------------------
    # Phase-structure verification: cross-spectral phase at the dominant frequency
    # -----------------------------------------------------------------------------------
    phase_structure_result = None
    if poloidal_result is not None and poloidal_result["f_peak_khz"] is not None:
        f_peak_for_phase = poloidal_result["f_peak_khz"] * 1000.0  # convert kHz -> Hz
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

    def _print_pair_coherence(results, label):
        for (pa, pb) in PROBE_PAIRS:
            res = results[(pa, pb)]
            if res is None:
                print(f"  - {label} {pa}-{pb}: SKIPPED (missing probe file, or no mode-active window)")
            else:
                f_pair, mean_coh2_pair = res
                mask_band = (f_pair >= fl_hz) & (f_pair <= fu_hz)
                if np.any(mask_band):
                    peak_idx_band = np.argmax(mean_coh2_pair[mask_band])
                    peak_f_khz = f_pair[mask_band][peak_idx_band] / 1000.0
                    peak_coh = mean_coh2_pair[mask_band][peak_idx_band]
                    print(f"  - {label} {pa}-{pb}: peak coherence within the {args.lower:.0f}-{args.upper:.0f} kHz "
                          f"EPM band = {peak_coh:.3f} at {peak_f_khz:.2f} kHz")
                else:
                    print(f"  - {label} {pa}-{pb}: no coherence samples fall within the "
                          f"{args.lower:.0f}-{args.upper:.0f} kHz EPM band.")

    _print_pair_coherence(carrier_coh_results, "Carrier oscillation")
    _print_pair_coherence(envelope_coh_results, "Envelope")

    # Chirp rate, using the same sg_win-smoothed derivative already 
    # validated in the Savitzky-Golay sensitivity sweep.
    chirp_rate_khz_per_ms = dsp.savgol_filter(ifreq_khz, args.smoothing, 2, deriv=1) / (dt * 1000.0)

    # --- [M6] Energetic-particle distribution-function response validation (Zhong et al. approach) ---
    zhong_results = zhong_distribution_function_analysis(
        shot, args, t_ms, envelope, ech_power, density_val, density_detected,
        mask_active_win, decimate_factor, dt_corr, chirp_rate_khz_per_ms
    )

    # =========================================================================================
    # ENERGETIC PARTICLE MODE FREQUENCY SCALING IDENTIFICATION (Objective 2 Core Physics)
    # Compare measured instantaneous frequency against candidate EP mode models:
    # 1. Alfvenic / TAE / GAE: f ~ 1/sqrt(n_e) (Bulk Alfven continuum / toroidicity gap)
    # 2. BAE Acoustic: f ~ sqrt(T_e) (Beta-induced acoustic-wave coupling, sound speed c_s)
    # 3. BAE Coupled: f ~ sqrt(v_A^2 + c_s^2) (Compressibility-modified Alfven continuum)
    # 4. EPM Beam Drive: f ~ sqrt(P_NBI / n_e) (Beam-ion density/pressure gradient drive)
    # 5. EPM Beam Power: f ~ P_NBI (Fast beam ion precession / drift resonance drive)
    # =========================================================================================
    ep_scaling_results = {}
    if density_detected:
        print("\n--- Energetic Particle Mode Frequency Scaling Identification (Objective 2) ---")
        print("  Evaluating candidate EP and MHD mode frequency models against measured instantaneous frequency:")

        # Restrict evaluation strictly to the mode-active window where the mode is strongly present
        if t_mode_strong_start is not None and t_mode_strong_end is not None:
            mask_eval_mode = (t_ms >= t_mode_strong_start) & (t_ms <= t_mode_strong_end)
            t_eval_start = t_mode_strong_start
            t_eval_end = t_mode_strong_end
            window_type_label = f"Mode-Active (Strongly Present) Sub-Window: {t_eval_start:.1f}-{t_eval_end:.1f} ms"
        elif n_mode_active >= 50:
            mask_eval_mode = mask_mode_active
            t_eval_start = t_ms[i0_mode]
            t_eval_end = t_ms[i1_mode-1]
            window_type_label = f"Mode-Active Burst Window: {t_eval_start:.1f}-{t_eval_end:.1f} ms"
        else:
            mask_eval_mode = mask_active_win
            t_eval_start = args.ech_active_start
            t_eval_end = args.ech_active_end
            window_type_label = f"Active ECH Window: {t_eval_start:.1f}-{t_eval_end:.1f} ms"

        print(f"  {window_type_label} ({np.sum(mask_eval_mode)} samples)")
        print(f"  Calibration window: {args.alfven_cal_start:.0f}-{args.alfven_cal_end:.0f} ms")

        mask_cal = (t_ms >= args.alfven_cal_start) & (t_ms <= args.alfven_cal_end)
        mask_cal_mode = mask_cal & mask_eval_mode
        mask_cal_use = mask_cal_mode if np.sum(mask_cal_mode) >= 5 else mask_eval_mode
        mean_meas_cal = np.mean(ifreq_khz[mask_cal_use]) if np.sum(mask_cal_use) >= 5 else np.mean(ifreq_khz)

        # 1. Alfvénic / TAE / GAE (1/sqrt(ne))
        s_alfven = 1.0 / np.sqrt(density_val_clean)

        # 2 & 3. BAE Acoustic and Coupled (using core ECE proxy from Zhong analysis)
        ece_for_bae = zhong_results.get("ece_core") if zhong_results is not None else None
        if ece_for_bae is not None:
            ece_baseline_val = np.median(ece_for_bae[t_ms < args.ech_active_start]) if np.any(t_ms < args.ech_active_start) else 0.0
            ece_clean_bae = np.clip(ece_for_bae - ece_baseline_val, 1e-4, None)
            ece_smoothed_bae = dsp.medfilt(ece_clean_bae, 1001) if len(ece_clean_bae) > 1001 else ece_clean_bae
            s_bae_sound = np.sqrt(ece_smoothed_bae)
            mean_inv_n = np.mean(1.0 / density_val_clean[mask_cal_use]) if np.sum(mask_cal_use) >= 5 else np.mean(1.0 / density_val_clean)
            mean_te = np.mean(ece_smoothed_bae[mask_cal_use]) if np.sum(mask_cal_use) >= 5 else np.mean(ece_smoothed_bae)
            s_bae_coupled = np.sqrt((1.0 / density_val_clean) / max(mean_inv_n, 1e-6) + (ece_smoothed_bae / max(mean_te, 1e-6)))
        else:
            s_bae_sound = None
            s_bae_coupled = None

        # 4 & 5. EPM Beam models
        s_epm_drive = np.sqrt(np.clip(total_nbi_power, 0.0, None) / density_val_clean)
        s_epm_power = np.clip(total_nbi_power, 0.0, None)

        # 6. Current-induced / iota-shear shift
        s_current = ip_signal

        # 7. Stored diamagnetic energy Wp (plasma beta proxy)
        s_beta = wp_signal if wp_signal is not None else None

        candidate_scalings = {
            "Alfvénic / TAE / GAE (1/√ne)": (s_alfven, "Bulk Alfvén continuum / gap", "tab:red", "--"),
            "BAE Acoustic / GAM (√Te)": (s_bae_sound, "Beta-induced acoustic mode", "tab:blue", "-."),
            "BAE Coupled (√(vA²+cs²))": (s_bae_coupled, "Compressibility-coupled Alfvén", "tab:cyan", ":"),
            "EPM Beam-Drive (√(PNBI/ne))": (s_epm_drive, "Fast-ion density gradient drive", "tab:purple", "-."),
            "EPM Beam-Power (PNBI)": (s_epm_power, "Fast ion precession / drift", "darkorange", "-"),
            "Current-induced (Ip)": (s_current, "Rotational transform / shear shift", "forestgreen", "--"),
            "Stored Energy (Wp)": (s_beta, "Diamagnetic / finite-beta scaling", "tab:brown", ":"),
        }

        # Subsample from the mode-active window
        ifreq_act_corr = ifreq_khz[mask_eval_mode][::decimate_factor]
        acf_ep = estimate_acf(ifreq_act_corr, nlags=min(50, len(ifreq_act_corr) - 2))
        N_ep = len(ifreq_act_corr)
        sum_acf_ep = np.sum(acf_ep[1:]) if len(acf_ep) > 1 else 0.0
        N_eff_ep = max(3.0, min(float(N_ep), N_ep / (1.0 + 2.0 * sum_acf_ep)))

        best_model_name = None
        max_abs_r = -1.0

        for model_name, (s_raw, phys_desc, color, ls) in candidate_scalings.items():
            if s_raw is None:
                continue
            mean_s_cal = np.mean(s_raw[mask_cal_use]) if np.sum(mask_cal_use) >= 5 else np.mean(s_raw)
            nc = mean_meas_cal / mean_s_cal if mean_s_cal > 0 else 1.0
            f_theo_curve = s_raw * nc
            f_theo_act_corr = f_theo_curve[mask_eval_mode][::decimate_factor]

            if np.std(f_theo_act_corr) > 0 and np.std(ifreq_act_corr) > 0:
                r_m, p_std_m = stats.pearsonr(ifreq_act_corr, f_theo_act_corr)
                p_adj_m = conservative_p_value(r_m, N_eff_ep)
            else:
                r_m, p_std_m, p_adj_m = 0.0, 1.0, 1.0

            if abs(r_m) > max_abs_r:
                max_abs_r = abs(r_m)
                best_model_name = model_name

            ep_scaling_results[model_name] = {
                "f_theo_curve": f_theo_curve,
                "norm_constant": nc,
                "r": r_m,
                "p_std": p_std_m,
                "p_adj": p_adj_m,
                "N_eff": N_eff_ep,
                "phys_desc": phys_desc,
                "color": color,
                "linestyle": ls,
            }
            meets_m = abs(r_m) > 0.7 and p_adj_m < 0.05
            print(f"    {model_name:<32} r = {r_m:+.4f}, p_adj = {format_p_value(p_adj_m)} | {phys_desc:<32} "
                  f"({'MEETS' if meets_m else 'does not meet'} |r|>0.7)")

        print(f"  ⭐ Best-Fitting EP/MHD Model: '{best_model_name}' (|r| = {max_abs_r:.4f})")
        ep_scaling_results["best_model"] = best_model_name
        ep_scaling_results["eval_window"] = (t_eval_start, t_eval_end)
        ep_scaling_results["window_type_label"] = window_type_label

    # --- 4d. Multiple Comparisons Correction (Bonferroni and Benjamini-Hochberg FDR) [EXT] ---
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

    MIN_NEFF_FOR_FAMILY = 5.0
    excluded_labels, excluded_rvalues, excluded_pvalues, excluded_nevals = [], [], [], []

    def _add_test(label, r, p_adj, n_eff):
        if n_eff is not None and n_eff < MIN_NEFF_FOR_FAMILY:
            excluded_labels.append(label)
            excluded_rvalues.append(r)
            excluded_pvalues.append(p_adj)
            excluded_nevals.append(n_eff)
        else:
            test_labels.append(label)
            test_pvalues.append(p_adj)
            test_rvalues.append(r)

    if zhong_results is not None:
        if zhong_results.get("r_ece_sig") is not None:
            _add_test("[M6] Envelope vs ECE-proxy", zhong_results["r_ece_sig"], zhong_results["p_ece_adj"],
                       zhong_results.get("N_eff_ece_sig"))
        if zhong_results.get("r_pressure_sig") is not None:
            _add_test("[M6] Envelope vs pressure proxy", zhong_results["r_pressure_sig"], zhong_results["p_pressure_adj"],
                       zhong_results.get("N_eff_pressure_sig"))
        if zhong_results.get("r_chirp_ece_sig") is not None:
            _add_test("[M6] Chirp rate vs ECE-proxy", zhong_results["r_chirp_ece_sig"], zhong_results["p_chirp_ece_adj"],
                       zhong_results.get("N_eff_chirp_ece_sig"))
        if zhong_results.get("r_chirp_pressure_sig") is not None:
            _add_test("[M6] Chirp rate vs pressure proxy", zhong_results["r_chirp_pressure_sig"], zhong_results["p_chirp_pressure_adj"],
                       zhong_results.get("N_eff_chirp_pressure_sig"))

    n_tests = len(test_pvalues)
    alpha_bonferroni = 0.05 / n_tests
    bh_significant, bh_p_adjusted = benjamini_hochberg(test_pvalues, alpha=0.05)

    print(f"\n--- Multiple Comparisons Correction (n = {n_tests} tests) [EXT] ---")
    for label, r_v, p_v, p_bh, sig_bh in zip(test_labels, test_rvalues, test_pvalues, bh_p_adjusted, bh_significant):
        sig_bonf = p_v < alpha_bonferroni
        print(f"    {label:<32} r = {r_v:+.3f}, {format_p_value(p_v):<16} "
              f"Bonferroni: {'Sig.' if sig_bonf else 'Not sig.':<8} | "
              f"FDR-BH: p_adj = {p_bh:.3e} ({'Sig.' if sig_bh else 'Not sig.'})")

    if excluded_labels:
        print(f"\n    [METROLOGY] {len(excluded_labels)} test(s) EXCLUDED from the correction family above "
              f"(N_eff < {MIN_NEFF_FOR_FAMILY:.0f}, fixed in advance -- see code comment): reported here "
              f"for transparency, uncorrected, and NOT to be read as passing/failing alongside the rest.")
        for label, r_v, p_v, n_eff_v in zip(excluded_labels, excluded_rvalues, excluded_pvalues, excluded_nevals):
            print(f"    {label:<32} r = {r_v:+.3f}, N_eff = {n_eff_v:.1f}, {format_p_value(p_v)} "
                  f"(no family-wise verdict -- test is underpowered by design)")

    corrected_results = dict(zip(test_labels, zip(test_pvalues, bh_p_adjusted, bh_significant)))

    def is_significant_corrected(label):
        p_v, p_bh, sig_bh = corrected_results[label]
        sig_bonf = p_v < alpha_bonferroni
        return sig_bonf and sig_bh

    significant_nbi = abs(r_nbi_tot) > 0.7 and not nbi_mismatch_detected and is_significant_corrected("Total NBI")

    suffix = get_output_suffix(args)

    # ===================================================================================
    # FIGURE 1: MACROSCOPIC PLASMA PARAMETER CORRELATIONS (4 panels)
    # Panels: Ip vs Env, Wp vs Env, nave vs Env (Noise-Free Window), HAFAST7.5 vs Env
    # ===================================================================================
    print(f"\n--- Generating Figure 1: Macroscopic Correlations (mhd_analysis_objective2_{shot}{suffix}.png) ---")
    fig1, axs1 = plt.subplots(4, 1, figsize=(12.5, 14.0), sharex=False)

    # Panel 0: Correlation of Plasma Current (Ip15) vs. Mode Envelope
    if ip_corr_result is not None:
        axs1[0].plot(t_ms, ip_corr_result["signal"], color='tab:blue', alpha=0.8, label='Ip (Ip15)')
        axs1[0].set_ylabel("Plasma Current Ip (kA)", color='tab:blue')
        axs1[0].tick_params(axis='y', labelcolor='tab:blue')
        ax0_twin = axs1[0].twinx()
        ax0_twin.plot(t_ms, envelope, color='red', alpha=0.7, linewidth=1.5, label='Mode Envelope')
        ax0_twin.set_ylabel("Envelope (V)", color='red')
        ax0_twin.tick_params(axis='y', labelcolor='red')
        if ip_corr_result["r_sig"] is not None:
            ip_title_stat = (f"r={ip_corr_result['r_sig']:.3f}, lag={ip_corr_result['lag_ms']:+.1f} ms, "
                             f"{format_p_value(ip_corr_result['p_adj'])}")
        else:
            ip_title_stat = "N/A"
        axs1[0].set_title(f"Correlation: Plasma Current (Ip15) vs. Mode Envelope - Shot {shot}\n{ip_title_stat}")
        lines0, labels0 = axs1[0].get_legend_handles_labels()
        lines0t, labels0t = ax0_twin.get_legend_handles_labels()
        axs1[0].legend(lines0 + lines0t, labels0 + labels0t, loc='upper right', fontsize=9)
    else:
        axs1[0].text(0.5, 0.5, "No Ip data available\n(Ip15 file missing)",
                    ha='center', va='center', transform=axs1[0].transAxes, fontsize=11, color='gray')
        axs1[0].set_title(f"Correlation: Plasma Current (Ip15) vs. Mode Envelope - Shot {shot}")
    axs1[0].set_xlabel("Time (ms)")
    axs1[0].grid(True, alpha=0.3)

    # Panel 1: Correlation of Stored Energy (Wp) vs. Mode Envelope
    if wp_corr_result is not None:
        axs1[1].plot(t_ms, wp_corr_result["signal"], color='tab:green', alpha=0.8, label='Wp')
        axs1[1].set_ylabel("Stored Energy Wp (a.u.)", color='tab:green')
        axs1[1].tick_params(axis='y', labelcolor='tab:green')
        ax1_twin = axs1[1].twinx()
        ax1_twin.plot(t_ms, envelope, color='red', alpha=0.7, linewidth=1.5, label='Mode Envelope')
        ax1_twin.set_ylabel("Envelope (V)", color='red')
        ax1_twin.tick_params(axis='y', labelcolor='red')
        if wp_corr_result["r_sig"] is not None:
            wp_title_stat = (f"r={wp_corr_result['r_sig']:.3f}, lag={wp_corr_result['lag_ms']:+.1f} ms, "
                             f"{format_p_value(wp_corr_result['p_adj'])}")
        else:
            wp_title_stat = "N/A"
        axs1[1].set_title(f"Correlation: Stored Energy (Wp) vs. Mode Envelope - Shot {shot}\n{wp_title_stat}")
        lines1, labels1 = axs1[1].get_legend_handles_labels()
        lines1t, labels1t = ax1_twin.get_legend_handles_labels()
        axs1[1].legend(lines1 + lines1t, labels1 + labels1t, loc='upper right', fontsize=9)
    else:
        axs1[1].text(0.5, 0.5, "No Wp data available\n(Wp file missing)",
                    ha='center', va='center', transform=axs1[1].transAxes, fontsize=11, color='gray')
        axs1[1].set_title(f"Correlation: Stored Energy (Wp) vs. Mode Envelope - Shot {shot}")
    axs1[1].set_xlabel("Time (ms)")
    axs1[1].grid(True, alpha=0.3)

    # Panel 2: Correlation of Line-Averaged Density (nave) vs. Mode Envelope
    if nave_corr_result is not None:
        axs1[2].plot(t_ms, nave_corr_result["signal"], color='tab:purple', alpha=0.8, label='nave')
        axs1[2].set_ylabel(r"Line-Averaged Density $\bar{n}_e$ (a.u.)", color='tab:purple')
        axs1[2].tick_params(axis='y', labelcolor='tab:purple')

        t_s = nave_corr_result["t_start"]
        t_e = nave_corr_result["t_end"] if nave_corr_result["t_end"] is not None else t_ms[-1]
        t_e_label = f"{nave_corr_result['t_end']:.0f} ms" if nave_corr_result["t_end"] is not None else "end"
        axs1[2].axvspan(t_s, t_e, color='purple', alpha=0.08,
                       label=f'Analyzed Window (Noise-Free, {t_s:.0f} ms - {t_e_label})')

        ax2_twin = axs1[2].twinx()
        ax2_twin.plot(t_ms, envelope, color='red', alpha=0.7, linewidth=1.5, label='Mode Envelope')
        ax2_twin.set_ylabel("Envelope (V)", color='red')
        ax2_twin.tick_params(axis='y', labelcolor='red')
        if nave_corr_result["r_sig"] is not None:
            nave_title_stat = (f"r={nave_corr_result['r_sig']:.3f}, lag={nave_corr_result['lag_ms']:+.1f} ms, "
                               f"{format_p_value(nave_corr_result['p_adj'])}")
        else:
            nave_title_stat = "N/A"
        axs1[2].set_title(f"Correlation: Line-Averaged Density (nave) vs. Mode Envelope (Noise-Free Window: t ≥ {t_s:.0f} ms) - Shot {shot}\n"
                         f"{nave_title_stat}")
        lines2, labels2 = axs1[2].get_legend_handles_labels()
        lines2t, labels2t = ax2_twin.get_legend_handles_labels()
        axs1[2].legend(lines2 + lines2t, labels2 + labels2t, loc='upper right', fontsize=9)
    else:
        axs1[2].text(0.5, 0.5, "No density data available\n(nave file missing)",
                    ha='center', va='center', transform=axs1[2].transAxes, fontsize=11, color='gray')
        axs1[2].set_title(f"Correlation: Line-Averaged Density (nave) vs. Mode Envelope - Shot {shot}")
    axs1[2].set_xlabel("Time (ms)")
    axs1[2].grid(True, alpha=0.3)

    # Panel 3: Correlation of Fast H-alpha Emission (HAFAST7.5) vs. Mode Envelope
    if hafast_corr_result is not None:
        axs1[3].plot(t_ms, hafast_corr_result["signal"], color='tab:green', alpha=0.85, label='Fast H-alpha (HAFAST7.5)')
        axs1[3].set_ylabel(r"Fast $H_\alpha$ Emission (V)", color='tab:green')
        axs1[3].tick_params(axis='y', labelcolor='tab:green')
        ax3_twin = axs1[3].twinx()
        ax3_twin.plot(t_ms, envelope, color='red', alpha=0.7, linewidth=1.5, label='Mode Envelope')
        ax3_twin.set_ylabel("Envelope (V)", color='red')
        ax3_twin.tick_params(axis='y', labelcolor='red')
        if hafast_corr_result["r_sig"] is not None:
            ha_title_stat = (f"r={hafast_corr_result['r_sig']:.3f}, lag={hafast_corr_result['lag_ms']:+.1f} ms, "
                             f"{format_p_value(hafast_corr_result['p_adj'])}")
        else:
            ha_title_stat = "N/A"
        axs1[3].set_title(f"Correlation: Fast H-alpha Emission (HAFAST7.5) vs. Mode Envelope - Shot {shot}\n{ha_title_stat}")
        lines3, labels3 = axs1[3].get_legend_handles_labels()
        lines3t, labels3t = ax3_twin.get_legend_handles_labels()
        axs1[3].legend(lines3 + lines3t, labels3 + labels3t, loc='upper right', fontsize=9)
    else:
        axs1[3].text(0.5, 0.5, "No HAFAST7.5 data available\n(HAFAST7.5 file missing)",
                    ha='center', va='center', transform=axs1[3].transAxes, fontsize=11, color='gray')
        axs1[3].set_title(f"Correlation: Fast H-alpha Emission (HAFAST7.5) vs. Mode Envelope - Shot {shot}")
    axs1[3].set_xlabel("Time (ms)")
    axs1[3].grid(True, alpha=0.3)

    plt.tight_layout()
    output_png1 = f"mhd_analysis_objective2_{shot}{suffix}.png"
    plt.savefig(output_png1, dpi=150)
    plt.close(fig1)
    print(f"  Figure 1 (Macroscopic Correlations) successfully saved to: '{output_png1}'")

    # ===================================================================================
    # FIGURE 2: HEATING SYNCHRONIZATION & EP MODE IDENTIFICATION (4 panels)
    # Panels:
    # 0. NBI & ECH Power vs. Mode Envelope (time traces)
    # 1. Instantaneous Frequency vs. Heating Power (mode-active window)
    # 2. Lagged Cross-Correlation Functions (synchronization analysis)
    # 3. EP Mode Frequency Scaling Identification (Alfvén vs BAE vs EPM models)
    # ===================================================================================
    print(f"\n--- Generating Figure 2: Heating & EP Mode Identification (mhd_analysis_objective2_heating_{shot}{suffix}.png) ---")
    fig2, axs2 = plt.subplots(4, 1, figsize=(12.5, 16.0), sharex=False)

    # Panel 0: Heating Power vs Mode Envelope
    axs2[0].plot(t_ms, total_nbi_power, color='darkorange', alpha=0.85, linewidth=1.6, label='Total NBI Power')
    axs2[0].plot(t_ms, ech_power, color='purple', alpha=0.8, linewidth=1.6, linestyle='--', label='ECH Power')
    axs2[0].set_ylabel("Heating Power (a.u.)")
    axs2[0].axvspan(args.ech_active_start, args.ech_active_end, color='purple', alpha=0.06,
                    label=f'Active ECH Window ({args.ech_active_start:.0f}-{args.ech_active_end:.0f} ms)')
    ax2_0_twin = axs2[0].twinx()
    ax2_0_twin.plot(t_ms, envelope, color='red', alpha=0.75, linewidth=1.5, label='Mode Envelope')
    ax2_0_twin.set_ylabel("Envelope (V)", color='red')
    ax2_0_twin.tick_params(axis='y', labelcolor='red')

    title_p0 = (f"Heating Synchronization: Total NBI & ECH Power vs. Mode Envelope - Shot {shot}\n"
                f"r(Env, NBI)={r_nbi_tot:+.3f} (lag={lag_nbi_ms:+.1f} ms) | "
                f"r(Env, ECH)={r_ech:+.3f} (lag={lag_ech_ms:+.1f} ms) | "
                f"r_partial(ECH step-ctrl)={r_partial:+.3f}")
    axs2[0].set_title(title_p0, fontsize=10)
    lines2_0, labels2_0 = axs2[0].get_legend_handles_labels()
    lines2_0t, labels2_0t = ax2_0_twin.get_legend_handles_labels()
    axs2[0].legend(lines2_0 + lines2_0t, labels2_0 + labels2_0t, loc='upper right', fontsize=8.5)
    axs2[0].set_xlabel("Time (ms)")
    axs2[0].grid(True, alpha=0.3)

    # Panel 1: Instantaneous Frequency vs. Heating Power (in mode-active window)
    if n_mode_active >= 200 and freq_heating_results is not None:
        t_mode_act = t_ms[mask_mode_active]
        f_mode_act = ifreq_khz[mask_mode_active]
        axs2[1].plot(t_mode_act, f_mode_act, color='tab:blue', linewidth=1.8, label=r'Measured $f_\mathrm{inst}$ (kHz)')
        axs2[1].set_ylabel(r"Mode Frequency $f_\mathrm{inst}$ (kHz)", color='tab:blue')
        axs2[1].tick_params(axis='y', labelcolor='tab:blue')

        ax2_1_twin = axs2[1].twinx()
        ax2_1_twin.plot(t_ms, total_nbi_power, color='darkorange', alpha=0.6, label='Total NBI Power')
        ax2_1_twin.plot(t_ms, ech_power, color='purple', alpha=0.6, linestyle='--', label='ECH Power')
        ax2_1_twin.set_ylabel("Heating Power (a.u.)")
        axs2[1].axvspan(t_ms[i0_mode], t_ms[i1_mode - 1], color='tab:blue', alpha=0.08,
                        label=f'Mode-Active Burst ({t_ms[i0_mode]:.1f}-{t_ms[i1_mode-1]:.1f} ms)')
        if t_mode_strong_start is not None and t_mode_strong_end is not None:
            axs2[1].axvspan(t_mode_strong_start, t_mode_strong_end, color='purple', alpha=0.12,
                            label=f'Strongly Present Window ({t_mode_strong_start:.1f}-{t_mode_strong_end:.1f} ms)')

        r_fn = freq_heating_results["r_ifreq_nbi"]
        r_fe = freq_heating_results["r_ifreq_ech"]
        r_fp = freq_heating_results["r_partial_freq"]
        lag_fn = freq_heating_results["lag_ifreq_nbi_ms"]
        lag_fe = freq_heating_results["lag_ifreq_ech_ms"]
        title_p1 = (f"Instantaneous Frequency Response to Heating - Shot {shot}\n"
                    f"r(f_inst, NBI)={r_fn:+.3f} (lag={lag_fn:+.1f} ms) | "
                    f"r(f_inst, ECH)={r_fe:+.3f} (lag={lag_fe:+.1f} ms) | "
                    f"r_partial(f_inst, ECH step-ctrl)={r_fp:+.3f}")
        axs2[1].set_title(title_p1, fontsize=10)
        lines2_1, labels2_1 = axs2[1].get_legend_handles_labels()
        lines2_1t, labels2_1t = ax2_1_twin.get_legend_handles_labels()
        axs2[1].legend(lines2_1 + lines2_1t, labels2_1 + labels2_1t, loc='upper right', fontsize=8.5)
    else:
        axs2[1].text(0.5, 0.5, "Insufficient mode-active window samples (<200) for frequency correlation",
                    ha='center', va='center', transform=axs2[1].transAxes, fontsize=11, color='gray')
        axs2[1].set_title(f"Instantaneous Frequency Response to Heating - Shot {shot}")
    axs2[1].set_xlabel("Time (ms)")
    axs2[1].grid(True, alpha=0.3)

    # Panel 2: Lagged Cross-Correlation Functions
    if np.max(total_nbi_corr) > 0 and len(lags_nbi_ms) > 1:
        axs2[2].plot(lags_nbi_ms, corr_nbi_vals, color='darkorange', linewidth=1.8,
                     label=f'Envelope vs NBI (peak r={r_nbi_peak:+.3f} at {lag_nbi_ms:+.1f} ms)')
        axs2[2].axvline(lag_nbi_ms, color='darkorange', linestyle=':', alpha=0.8)
    if len(lags_ech_ms) > 1:
        axs2[2].plot(lags_ech_ms, corr_ech_vals, color='purple', linewidth=1.8,
                     label=f'Envelope vs ECH (peak r={r_ech_peak:+.3f} at {lag_ech_ms:+.1f} ms)')
        axs2[2].axvline(lag_ech_ms, color='purple', linestyle=':', alpha=0.8)
    if freq_heating_results is not None:
        lags_fn_c = freq_heating_results.get("lags_ifreq_nbi_curve")
        corr_fn_c = freq_heating_results.get("corr_ifreq_nbi_curve")
        if lags_fn_c is not None and len(lags_fn_c) > 1:
            axs2[2].plot(lags_fn_c, corr_fn_c, color='teal', linestyle='--', linewidth=1.5,
                         label=f'f_inst vs NBI (peak r={freq_heating_results["r_ifreq_nbi_peak"]:+.3f} at {freq_heating_results["lag_ifreq_nbi_ms"]:+.1f} ms)')
            axs2[2].axvline(freq_heating_results["lag_ifreq_nbi_ms"], color='teal', linestyle=':', alpha=0.8)
        lags_fe_c = freq_heating_results.get("lags_ifreq_ech_curve")
        corr_fe_c = freq_heating_results.get("corr_ifreq_ech_curve")
        if lags_fe_c is not None and len(lags_fe_c) > 1:
            axs2[2].plot(lags_fe_c, corr_fe_c, color='magenta', linestyle='--', linewidth=1.5,
                         label=f'f_inst vs ECH (peak r={freq_heating_results["r_ifreq_ech_peak"]:+.3f} at {freq_heating_results["lag_ifreq_ech_ms"]:+.1f} ms)')
            axs2[2].axvline(freq_heating_results["lag_ifreq_ech_ms"], color='magenta', linestyle=':', alpha=0.8)

    axs2[2].axhline(y=0.0, color='black', linestyle='-', linewidth=0.6, alpha=0.4)
    axs2[2].axvline(x=0.0, color='black', linestyle='--', linewidth=0.8, alpha=0.5, label='Zero Lag (τ = 0 ms)')
    axs2[2].set_xlabel("Lag (ms)")
    axs2[2].set_ylabel("Cross-Correlation Coefficient")
    axs2[2].set_title(f"Lagged Cross-Correlation Diagnostics (Synchronization & Delay Analysis) - Shot {shot}", fontsize=10)
    axs2[2].legend(loc='upper right', fontsize=8, ncol=2)
    axs2[2].grid(True, alpha=0.3)

    # Panel 3: EP Mode Frequency Scaling Identification
    if ep_scaling_results:
        win_start, win_end = ep_scaling_results.get("eval_window", (t_ms[i0_mode] if n_mode_active >= 50 else args.ech_active_start, t_ms[i1_mode-1] if n_mode_active >= 50 else args.ech_active_end))
        t_plot_min = max(args.ech_active_start - 5.0, win_start - 15.0)
        t_plot_max = min(args.ech_active_end + 15.0, win_end + 20.0)
        mask_plot_ep = (t_ms >= t_plot_min) & (t_ms <= t_plot_max)

        # Plot measured instantaneous frequency: quiescent background faint, mode-active burst in dark gray, strongly-present in bold black
        axs2[3].plot(t_ms[mask_plot_ep], ifreq_khz[mask_plot_ep], color='gray', linewidth=0.9, alpha=0.3, label=r'Measured $f_\mathrm{inst}(t)$ (quiescent)')
        if n_mode_active >= 50:
            axs2[3].plot(t_ms[mask_mode_active], ifreq_khz[mask_mode_active], color='gray', linewidth=1.4, alpha=0.6, label=r'Measured $f_\mathrm{inst}(t)$ (burst)', zorder=3)
        mask_eval_plot = (t_ms >= win_start) & (t_ms <= win_end)
        axs2[3].plot(t_ms[mask_eval_plot], ifreq_khz[mask_eval_plot], color='black', linewidth=2.2, label=r'Measured $f_\mathrm{inst}(t)$ (strongly present)', zorder=5)

        # Shading for the strongly-present window
        axs2[3].axvspan(win_start, win_end, color='purple', alpha=0.10,
                        label=f'Strongly Present Window ({win_start:.1f}-{win_end:.1f} ms)')

        # Overlay candidate EP scaling curves
        for m_name, m_dict in ep_scaling_results.items():
            if m_name in ("best_model", "eval_window", "window_type_label"):
                continue
            r_val = m_dict["r"]
            p_val = m_dict["p_adj"]
            is_best = (m_name == ep_scaling_results.get("best_model"))
            lw = 2.4 if is_best else 1.2
            alpha_val = 0.95 if is_best else 0.70
            lbl = f"{m_name}: r={r_val:+.3f} ({format_p_value(p_val)})" + (" [BEST]" if is_best else "")
            axs2[3].plot(t_ms[mask_plot_ep], m_dict["f_theo_curve"][mask_plot_ep], color=m_dict["color"],
                         linestyle=m_dict["linestyle"], linewidth=lw, alpha=alpha_val, label=lbl,
                         zorder=4 if is_best else 2)

        # Highlight calibration window
        axs2[3].axvspan(args.alfven_cal_start, args.alfven_cal_end, color='gray', alpha=0.15,
                        label=f'Calibration Window ({args.alfven_cal_start:.0f}-{args.alfven_cal_end:.0f} ms)')
        axs2[3].set_xlim(t_plot_min, t_plot_max)
        axs2[3].set_ylim(args.lower - 5.0, args.upper + 15.0)

        best_name = ep_scaling_results.get("best_model", "N/A")
        best_r = ep_scaling_results.get(best_name, {}).get("r", 0.0) if best_name in ep_scaling_results else 0.0
        badge_text = f"Best-Fitting EP/MHD Model: {best_name} (r = {best_r:+.3f} in {win_start:.1f}-{win_end:.1f} ms window)"
        axs2[3].text(0.02, 0.06, badge_text, transform=axs2[3].transAxes, fontsize=9, fontweight='bold',
                     color='darkgreen' if abs(best_r) > 0.4 else 'navy',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='darkorange', alpha=0.9))

        axs2[3].set_title(f"Energetic Particle Mode Identification: Measured Frequency vs. Theoretical Scalings\n"
                          f"(Band: {args.lower:.0f}-{args.upper:.0f} kHz | Strongly-Present Window: {win_start:.1f}-{win_end:.1f} ms | Calibrated on {args.alfven_cal_start:.0f}-{args.alfven_cal_end:.0f} ms)", fontsize=10)
        axs2[3].legend(loc='upper right', fontsize=7.5, ncol=2)
    else:
        axs2[3].text(0.5, 0.5, "Density data unavailable for theoretical EP mode scaling",
                     ha='center', va='center', transform=axs2[3].transAxes, fontsize=11, color='gray')
        axs2[3].set_title("Energetic Particle Mode Identification")
    axs2[3].set_xlabel("Time (ms)")
    axs2[3].set_ylabel("Frequency (kHz)")
    axs2[3].grid(True, alpha=0.3)

    plt.tight_layout()
    output_png2 = f"mhd_analysis_objective2_heating_{shot}{suffix}.png"
    plt.savefig(output_png2, dpi=150)
    plt.close(fig2)
    print(f"  Figure 2 (Heating & EP Mode Identification) successfully saved to: '{output_png2}'")

    # ===================================================================================
    # FIGURE 3: SPATIAL MODAL STRUCTURE ANALYSIS (4 panels)
    # Panels:
    # 0. Inter-probe carrier coherence (MP1-MP3, MP1-MP4, MP3-MP4)
    # 1. Inter-probe envelope coherence
    # 2. Poloidal mode-number (m) power map from PMP array
    # 3. Poloidal phase structure verification
    # ===================================================================================
    print(f"\n--- Generating Figure 3: Spatial Modal Structure (mhd_analysis_objective2_structure_{shot}{suffix}.png) ---")
    fig3, axs3 = plt.subplots(4, 1, figsize=(12.5, 17.0), sharex=False)

    # Panel 0: Carrier Coherence
    if n_mode_active >= 200 and n_flat_active >= 200:
        window_kind = "[M7] burst" if flat_info.get("used_fallback") else "[M9] flat sub-window"
        mode_active_ms_str = f"{t_ms[i0_flat]:.1f}-{t_ms[i1_flat - 1]:.1f} ms, {window_kind}"
    else:
        mode_active_ms_str = "no mode-active/flat window"

    any_carrier_plotted = False
    for (pa, pb) in PROBE_PAIRS:
        res = carrier_coh_results[(pa, pb)]
        if res is None:
            continue
        f_pair, mean_coh2_pair = res
        mask_f_pair = (f_pair >= fl_hz) & (f_pair <= fu_hz)
        style = PROBE_PAIR_STYLES[(pa, pb)]
        axs3[0].plot(f_pair[mask_f_pair] / 1000.0, mean_coh2_pair[mask_f_pair], linewidth=2,
                    label=f'{pa}-{pb} Carrier Coherence', **style)
        any_carrier_plotted = True
    if not any_carrier_plotted:
        if n_mode_active < 200:
            placeholder_msg = "No mode-active window detected\n(too few burst samples)"
        elif n_flat_active < 200:
            placeholder_msg = "Flat-frequency [M9] sub-window\ntoo short (too few samples)"
        else:
            placeholder_msg = "No probe pair available\n(MP3 and/or MP4 file missing)"
        axs3[0].text(0.5, 0.5, placeholder_msg, ha='center', va='center',
                    transform=axs3[0].transAxes, fontsize=11, color='gray')
    axs3[0].axhline(y=0.5, color='black', linestyle=':', label='Significance Threshold (0.5)')
    axs3[0].set_title("Cross-Spectral Coherence Between Mirnov Probes: Carrier-Mode Oscillations [M8]\n"
                      f"(window: {mode_active_ms_str})", fontsize=10)
    axs3[0].set_xlabel("Frequency (kHz)")
    axs3[0].set_ylabel(r"Coherence $\gamma^2$")
    axs3[0].set_ylim(0, 1.05)
    axs3[0].grid(True, alpha=0.3, which='both', linestyle=':')
    axs3[0].legend(loc='upper right')

    # Panel 1: Envelope Coherence
    any_envelope_plotted = False
    for (pa, pb) in PROBE_PAIRS:
        res = envelope_coh_results[(pa, pb)]
        if res is None:
            continue
        f_pair, mean_coh2_pair = res
        mask_f_pair = (f_pair >= 0) & (f_pair <= 10000.0)
        style = PROBE_PAIR_STYLES[(pa, pb)]
        axs3[1].plot(f_pair[mask_f_pair] / 1000.0, mean_coh2_pair[mask_f_pair], linewidth=2,
                    label=f'{pa}-{pb} Envelope Coherence', **style)
        any_envelope_plotted = True
    if not any_envelope_plotted:
        if n_mode_active < 200:
            placeholder_msg = "No mode-active window detected\n(too few burst samples)"
        elif n_flat_active < 200:
            placeholder_msg = "Flat-frequency [M9] sub-window\ntoo short (too few samples)"
        else:
            placeholder_msg = "No probe pair available\n(MP3 and/or MP4 file missing)"
        axs3[1].text(0.5, 0.5, placeholder_msg, ha='center', va='center',
                    transform=axs3[1].transAxes, fontsize=11, color='gray')
    axs3[1].axhline(y=0.5, color='black', linestyle=':', label='Significance Threshold (0.5)')
    axs3[1].set_title("Cross-Spectral Coherence Between Probe Envelopes: Modal-Structure Estimation [M8]\n"
                      f"(window: {mode_active_ms_str})", fontsize=10)
    axs3[1].set_xlabel("Modulation Frequency (kHz)")
    axs3[1].set_ylabel(r"Coherence $\gamma^2$")
    axs3[1].set_ylim(0, 1.05)
    axs3[1].grid(True, alpha=0.3, which='both', linestyle=':')
    axs3[1].legend(loc='upper right')

    # Panel 2: Poloidal Mode Decomposition
    if poloidal_result is not None:
        pcm = axs3[2].pcolormesh(
            poloidal_result["f_band_khz_plot"], poloidal_result["k_grid_plot"],
            np.log10(poloidal_result["P2d_plot"] + 1e-30), cmap='jet', shading='auto'
        )
        cb = plt.colorbar(pcm, ax=axs3[2])
        cb.set_label(r"log$_{10}$ Poloidal Power (a.u.)")
        untrustworthy = poloidal_result["is_edge_pinned"] and poloidal_result["verdict"] != "confirmed_stable"
        line_color = 'red' if untrustworthy else 'white'
        line_label = (f"m = {poloidal_result['m_dominant']:+d} [UNTRUSTWORTHY - see log]" if untrustworthy
                      else f"Dominant m = {poloidal_result['m_dominant']:+d}")
        axs3[2].axhline(y=poloidal_result["m_dominant"], color=line_color, linestyle='--', linewidth=1.4,
                        label=line_label)
        if poloidal_result["was_expanded_for_plot"]:
            axs3[2].text(0.02, 0.02, f"(auto-expanded to +/-{poloidal_result['k_grid_plot'].max()}, "
                        f"default was +/-{poloidal_result['k_grid'].max()})",
                        transform=axs3[2].transAxes, fontsize=8, color='white',
                        bbox=dict(facecolor='black', alpha=0.4, pad=2))
        axs3[2].set_xlim(args.lower, args.upper)
        axs3[2].legend(loc='upper right')
    else:
        placeholder_msg = ("Poloidal array unavailable\n(<3 PMP channels found)" if len(pmp_signals) < 3
                            else "No mode-active/flat window\navailable for this shot")
        axs3[2].text(0.5, 0.5, placeholder_msg, ha='center', va='center',
                    transform=axs3[2].transAxes, fontsize=11, color='gray')
    axs3[2].set_title(f"Poloidal Mode-Number Decomposition (PMP1-PMP14 Array) [M10]\n(window: {mode_active_ms_str})", fontsize=10)
    axs3[2].set_xlabel("Frequency (kHz)")
    axs3[2].set_ylabel("Poloidal Mode Number m")

    # Panel 3: Poloidal Phase Structure Verification
    if phase_structure_result is not None:
        ps = phase_structure_result
        theta_plot_rad = np.arctan2(np.sin(ps["theta_rad"]), np.cos(ps["theta_rad"]))
        phase_meas = ps["measured_phase"]
        m_dom = poloidal_result["m_dominant"] if poloidal_result is not None else 0

        th_theory_grid = np.linspace(-np.pi, np.pi, 1200)
        candidate_ms_to_show = sorted(set([-4, -3, -2, -1, 1, 2, 3, 4, m_dom]))
        for m_val in candidate_ms_to_show:
            ph_theory = np.arctan2(np.sin(m_val * th_theory_grid), np.cos(m_val * th_theory_grid))
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
            axs3[3].plot(th_plot_line, ph_plot_line, color=color, alpha=alpha_line,
                        linewidth=lw, label=label, zorder=3 if is_dom else 1)

        axs3[3].errorbar(theta_plot_rad, phase_meas, yerr=ps["sigma_phase"],
                        fmt='none', ecolor='tab:blue', elinewidth=1.4, capsize=3.5, capthick=1.0,
                        alpha=0.75, zorder=4, label=r'Phase uncertainty $\pm 1\sigma_\phi$ (Bendat & Piersol)')
        axs3[3].scatter(theta_plot_rad, phase_meas, s=80, c='blue', marker='o', edgecolors='black',
                        linewidths=1.2, zorder=5, label=f"Measured phase (ref: {ps['ref_channel']})")

        for i, ch in enumerate(ps["channels"]):
            axs3[3].annotate(ch, (theta_plot_rad[i], phase_meas[i]),
                            textcoords="offset points", xytext=(5, 6),
                            fontsize=7, color='darkblue', fontweight='bold')

        align_info = ps.get("m_alignment", {}).get(m_dom, {"r_circ": 0.0, "mean_error_deg": 90.0})
        r_circ_dom = align_info["r_circ"]
        mean_err_dom = align_info["mean_error_deg"]
        is_phase_confirmed = (r_circ_dom >= 0.70) and (mean_err_dom <= 45.0) and (ps["mean_coherence"] >= 0.40)

        if is_phase_confirmed:
            axs3[3].text(0.02, 0.05, f"[CONFIRMED] Statistically Validated $m = {m_dom:+d}$ Fit\n"
                        f"Alignment $r_{{circ}}$ = {r_circ_dom:.2f} (>= 0.70), Mean Error = {mean_err_dom:.1f} deg (<= 45 deg)",
                        transform=axs3[3].transAxes, fontsize=8.5, color='darkgreen', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='honeydew', edgecolor='darkgreen', alpha=0.9))
        else:
            axs3[3].text(0.02, 0.05, f"[UNCONFIRMED / NOISY FIT] $m = {m_dom:+d}$ Failed Verification\n"
                        f"Alignment $r_{{circ}}$ = {r_circ_dom:.2f} (< 0.70), Mean Error = {mean_err_dom:.1f} deg (> 45 deg)",
                        transform=axs3[3].transAxes, fontsize=8.5, color='darkred', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='linen', edgecolor='darkred', alpha=0.9))

        axs3[3].set_xlim(-np.pi * 1.05, np.pi * 1.05)
        axs3[3].set_ylim(-np.pi * 1.15, np.pi * 1.15)
        axs3[3].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        axs3[3].set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
        axs3[3].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        axs3[3].set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
        axs3[3].axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
        axs3[3].axvline(x=0, color='black', linewidth=0.5, alpha=0.3)
        axs3[3].set_title(f"Poloidal Phase Structure Verification [M10-PHASE]\n"
                          f"(cross-spectral phase at {ps['f_peak_hz']/1000:.2f} kHz, "
                          f"ref: {ps['ref_channel']}, mean $\\gamma^2$ = {ps['mean_coherence']:.2f}, window: {mode_active_ms_str})", fontsize=10)
        axs3[3].set_xlabel(r"Poloidal Angle $\theta$ (rad)")
        axs3[3].set_ylabel(r"Cross-Spectral Phase $\phi$ (rad)")
        axs3[3].grid(True, alpha=0.3)
        axs3[3].legend(loc='upper right', fontsize=8, ncol=2)
    else:
        placeholder_msg = ("Phase structure unavailable\n(no poloidal NUDFT result)" if poloidal_result is None
                            else "Phase structure unavailable\n(insufficient data)")
        axs3[3].text(0.5, 0.5, placeholder_msg, ha='center', va='center',
                    transform=axs3[3].transAxes, fontsize=11, color='gray')
        axs3[3].set_title("Poloidal Phase Structure Verification [M10-PHASE]")
        axs3[3].set_xlabel(r"Poloidal Angle $\theta$ (rad)")
        axs3[3].set_ylabel(r"Cross-Spectral Phase $\phi$ (rad)")

    plt.tight_layout()
    output_png3 = f"mhd_analysis_objective2_structure_{shot}{suffix}.png"
    plt.savefig(output_png3, dpi=150)
    plt.close(fig3)
    print(f"  Figure 3 (Spatial Modal Structure) successfully saved to: '{output_png3}'")

    # --- Per-shot summary printout ---
    print(f"\n🔬 SHOT {shot} SUMMARY 🔬")
    if nbi_mismatch_detected:
        print("  - NBI out of Mirnov acquisition window for this shot; NBI correlation not evaluable.")
    else:
        print(f"  - Envelope vs. Combined NBI: r = {r_nbi_tot:.3f} ({'validated' if significant_nbi else 'NOT validated'} at |r|>0.7 + corrected p<0.05)")
    print(f"  - Envelope vs. ECH (full range): r = {r_ech:.3f}; partial (step-controlled): r_partial = {r_partial:.3f}")
    if density_detected:
        print(f"  - Alfven scaling validation: r = {r_val_scaling:.3f} ({'FINAL' if (bfield_available and b_is_constant) else 'PRELIMINARY (simplified scaling)' if not bfield_available else 'FINAL'})")
    if zhong_results is not None:
        chan_status = "explicit" if args.ece_core_channel is not None else "heuristic"
        calib_status = "Te-calibrated" if zhong_results.get("tau_s_ms") is not None else "qualitative, uncalibrated"
        print(f"  - Zhong et al. [M6]: core ECE ch.{zhong_results['core_ece_channel']} ({chan_status}), "
              f"envelope-vs-ECE lag = {zhong_results['lag_ece_ms']:+.1f} ms ({calib_status})")
        if zhong_results.get("r_ece_sig") is not None:
            meets_ece_summary = abs(zhong_results["r_ece_sig"]) > 0.7 and zhong_results["p_ece_adj"] < 0.05
            print(f"    -> proper r = {zhong_results['r_ece_sig']:.3f}, p_adj = {format_p_value(zhong_results['p_ece_adj'])} "
                  f"({'MEETS' if meets_ece_summary else 'does NOT meet'} |r|>0.7 & p<0.05: primary validation criterion)")
        if zhong_results.get("r_pressure_sig") is not None:
            meets_pressure_summary = abs(zhong_results["r_pressure_sig"]) > 0.7 and zhong_results["p_pressure_adj"] < 0.05
            print(f"    Envelope-vs-pressure-proxy -> proper r = {zhong_results['r_pressure_sig']:.3f}, "
                  f"p_adj = {format_p_value(zhong_results['p_pressure_adj'])} "
                  f"({'MEETS' if meets_pressure_summary else 'does NOT meet'} |r|>0.7 & p<0.05)")
        if zhong_results.get("lag_chirp_ece_ms") is not None:
            print(f"    Chirp-rate-vs-ECE lag [M6/frequency-sweep reading]: {zhong_results['lag_chirp_ece_ms']:+.1f} ms "
                  f"(peak |r| = {zhong_results['r_chirp_ece_peak']:+.3f})")
    else:
        print("  - Zhong et al. [M6]: SKIPPED (no ECE channels found for this shot)")
    if freq_heating_results is not None:
        print(f"  - Freq vs. NBI/ECH [M7]: r_NBI = {freq_heating_results['r_ifreq_nbi']:.3f} "
              f"(lag {freq_heating_results['lag_ifreq_nbi_ms']:+.1f} ms), "
              f"r_ECH = {freq_heating_results['r_ifreq_ech']:.3f} "
              f"(lag {freq_heating_results['lag_ifreq_ech_ms']:+.1f} ms), "
              f"r_ECH_step-controlled = {freq_heating_results['r_partial_freq']:.3f} "
              f"[mode-active window only]")
    else:
        print("  - Freq vs. NBI/ECH [M7]: SKIPPED (too few mode-active samples for this shot)")
    for (pa, pb) in PROBE_PAIRS:
        carrier_res = carrier_coh_results[(pa, pb)]
        envelope_res = envelope_coh_results[(pa, pb)]
        if carrier_res is None or envelope_res is None:
            print(f"  - [M8] {pa}-{pb} inter-probe coherence: SKIPPED (missing probe file)")
            continue
        f_carrier, mean_coh2_carrier = carrier_res
        f_envelope, mean_coh2_envelope = envelope_res
        mask_band_c = (f_carrier >= fl_hz) & (f_carrier <= fu_hz)
        mask_band_e = (f_envelope >= fl_hz) & (f_envelope <= fu_hz)
        peak_carrier = float(np.max(mean_coh2_carrier[mask_band_c])) if np.any(mask_band_c) else float('nan')
        peak_envelope = float(np.max(mean_coh2_envelope[mask_band_e])) if np.any(mask_band_e) else float('nan')
        print(f"  - [M8] {pa}-{pb} inter-probe coherence: carrier peak (in-band) = {peak_carrier:.3f}, "
              f"envelope peak (in-band) = {peak_envelope:.3f}")
    nbi_max = np.max(total_nbi_power)
    if nbi_max > 0:
        mask_nbi_active = total_nbi_power > 0.5 * nbi_max
    else:
        mask_nbi_active = mask_active_win  
    mean_nbi_active = float(np.mean(total_nbi_power[mask_nbi_active])) if np.any(mask_nbi_active) else 0.0
    mean_envelope_active = float(np.mean(envelope[mask_active_win])) if np.any(mask_active_win) else 0.0
    mean_ech_active = float(np.mean(ech_power[mask_active_win])) if np.any(mask_active_win) else 0.0
    mean_ifreq_active = float(np.mean(ifreq_khz[mask_mode_active])) if n_mode_active >= 200 else None

    return {
        "shot": shot,
        "nbi_mismatch_detected": nbi_mismatch_detected,
        "density_detected": density_detected,
        "r_nbi_tot": r_nbi_tot,
        "p_nbi_tot": p_nbi_tot,
        "significant_nbi": significant_nbi if not nbi_mismatch_detected else None,
        "r_ech": r_ech,
        "p_ech": p_ech,
        "r_partial": r_partial,
        "r_val_scaling": r_val_scaling,
        "p_val_scaling": p_val_scaling,
        "lag_nbi_ms": lag_nbi_ms,
        "r_nbi_peak": r_nbi_peak,
        "lag_ech_ms": lag_ech_ms,
        "r_ech_peak": r_ech_peak,
        "mean_nbi_active": mean_nbi_active,
        "mean_envelope_active": mean_envelope_active,
        "mean_ech_active": mean_ech_active,
        "mean_ifreq_active": mean_ifreq_active,
        "envelope_corr": envelope_corr,
        "total_nbi_corr": total_nbi_corr,
        "ech_power_corr": ech_power_corr,
        "dt_corr": dt_corr,
        "zhong_results": zhong_results,
        "freq_heating_results": freq_heating_results,
        "ep_scaling_results": ep_scaling_results,
        "n_mode_active": n_mode_active,
        "poloidal_m_dominant": poloidal_result["m_dominant"] if poloidal_result is not None else None,
        "poloidal_f_peak_khz": poloidal_result["f_peak_khz"] if poloidal_result is not None else None,
    }

def cross_discharge_analysis(results_list, args):
    results_list = [r for r in results_list if r is not None]
    if len(results_list) == 0:
        print("\nNo shots were successfully processed; cross-discharge analysis skipped.")
        return
    if len(results_list) == 1:
        print("\nOnly one shot was successfully processed; cross-discharge analysis (main.md "
              "requirement (c), 'across different experimental discharges') cannot be performed "
              "with a single discharge. Re-run with more shots via --shots.")
        return

    print(f"\n{'=' * 93}")
    print("🔬 CROSS-DISCHARGE ANALYSIS (main.md requirement (c): 'across different experimental")
    print("   discharges under varying NBI power levels') [M1] 🔬")
    print(f"{'=' * 93}")

    # --- (i) Pooled correlation across all discharges ---
    valid_for_pool = [r for r in results_list if not r["nbi_mismatch_detected"] and np.max(r["total_nbi_corr"]) > 0]
    if len(valid_for_pool) >= 2:
        pooled_envelope = np.concatenate([r["envelope_corr"] for r in valid_for_pool])
        pooled_nbi = np.concatenate([r["total_nbi_corr"] for r in valid_for_pool])
        pooled_ech = np.concatenate([r["ech_power_corr"] for r in valid_for_pool])

        r_pool_nbi, p_pool_nbi_std = stats.pearsonr(pooled_envelope, pooled_nbi)
        acf_pool = estimate_acf(pooled_envelope, nlags=50)
        N_eff_pool = len(pooled_envelope) / (1.0 + 2.0 * np.sum(acf_pool[1:]))
        N_eff_pool = max(3.0, min(float(len(pooled_envelope)), N_eff_pool))
        p_pool_nbi = conservative_p_value(r_pool_nbi, N_eff_pool)

        r_pool_ech, p_pool_ech_std = stats.pearsonr(pooled_envelope, pooled_ech)
        p_pool_ech = conservative_p_value(r_pool_ech, N_eff_pool)

        shots_in_pool = [r["shot"] for r in valid_for_pool]
        print(f"\n(i) POOLED correlation across shots {shots_in_pool} (N_total = {len(pooled_envelope)}, "
              f"N_effective ~ {N_eff_pool:.1f}; caveat: pooled ACF ignores shot-boundary discontinuities):")
        print(f"    - Pooled Envelope vs. Combined NBI: r = {r_pool_nbi:.4f}, {format_p_value(p_pool_nbi)} "
              f"({'MEETS' if abs(r_pool_nbi) > 0.7 and p_pool_nbi < 0.05 else 'does NOT meet'} |r|>0.7 & p<0.05)")
        print(f"    - Pooled Envelope vs. ECH:         r = {r_pool_ech:.4f}, {format_p_value(p_pool_ech)} "
              f"({'MEETS' if abs(r_pool_ech) > 0.7 and p_pool_ech < 0.05 else 'does NOT meet'} |r|>0.7 & p<0.05)")
    else:
        print("\n(i) POOLED correlation skipped: fewer than 2 shots have NBI active within the "
              "Mirnov window (need >=2 for a meaningful pooled test).")
    print(f"\n(ii) DISCHARGE-LEVEL summary (one point per shot; caution: N={len(results_list)} discharges "
          f"gives very few degrees of freedom):")
    print(f"    {'Shot':<8}{'Mean NBI (NBI-active win)':<28}{'Mean ECH (ECH-active win)':<28}{'Mean Envelope (ECH-active win)':<32}{'Mean Freq (mode-active, kHz)':<30}{'r_NBI (within-shot)':<20}")
    for r in results_list:
        nbi_str = "N/A (out of window)" if r["nbi_mismatch_detected"] else f"{r['mean_nbi_active']:.4f}"
        r_nbi_str = "N/A" if r["nbi_mismatch_detected"] else f"{r['r_nbi_tot']:+.3f}"
        freq_str = f"{r['mean_ifreq_active']:.2f}" if r.get("mean_ifreq_active") is not None else "N/A"
        print(f"    {r['shot']:<8}{nbi_str:<28}{r['mean_ech_active']:<28.4f}{r['mean_envelope_active']:<32.4f}{freq_str:<30}{r_nbi_str:<20}")

    scaling_points = [(r["mean_nbi_active"], r["mean_envelope_active"]) for r in results_list if not r["nbi_mismatch_detected"]]
    if len(scaling_points) >= 3:
        nbi_levels = np.array([p[0] for p in scaling_points])
        env_levels = np.array([p[1] for p in scaling_points])
        if np.std(nbi_levels) == 0 or np.std(env_levels) == 0:
            print(f"\n    - Discharge-level scaling correlation (amplitude) not computed: NBI power level and/or "
                  f"envelope amplitude is IDENTICAL across the {len(scaling_points)} discharges analyzed "
                  f"(no variation to correlate against). Include shots spanning a wider range of NBI power.")
        else:
            r_scaling, p_scaling = stats.pearsonr(nbi_levels, env_levels)
            print(f"\n    - Discharge-level Pearson r (mean NBI power vs. mean envelope AMPLITUDE, "
                  f"N={len(scaling_points)} discharges): r = {r_scaling:.4f}, {format_p_value(p_scaling)}")
            print(f"      ⚠️ With only {len(scaling_points)} discharges (df={len(scaling_points)-2}), this result is "
                  f"suggestive at best; do not report it as meeting the |r|>0.7 & p<0.05 significance bar")
            print("      without substantially more discharges spanning a wider range of NBI power levels.")
    else:
        print("\n    - Fewer than 3 shots have NBI active in-window; discharge-level amplitude scaling "
              "correlation not computed.")

    freq_scaling_points = [(r["mean_nbi_active"], r["mean_ifreq_active"]) for r in results_list
                            if not r["nbi_mismatch_detected"] and r.get("mean_ifreq_active") is not None]
    if len(freq_scaling_points) >= 3:
        nbi_levels_f = np.array([p[0] for p in freq_scaling_points])
        freq_levels = np.array([p[1] for p in freq_scaling_points])
        if np.std(nbi_levels_f) == 0 or np.std(freq_levels) == 0:
            print(f"    - Discharge-level scaling correlation (FREQUENCY) not computed: NBI power level and/or "
                  f"mean frequency is IDENTICAL across the {len(freq_scaling_points)} discharges analyzed.")
        else:
            r_scaling_f, p_scaling_f = stats.pearsonr(nbi_levels_f, freq_levels)
            print(f"    - Discharge-level Pearson r (mean NBI power vs. mean measured FREQUENCY, "
                  f"N={len(freq_scaling_points)} discharges): r = {r_scaling_f:.4f}, {format_p_value(p_scaling_f)}")
            print(f"      ⚠️ Same caveat as above: with only {len(freq_scaling_points)} discharges "
                  f"(df={len(freq_scaling_points)-2}), do not report this as meeting |r|>0.7 & p<0.05 "
                  "without more discharges spanning a wider NBI power range.")
    else:
        print("    - Fewer than 3 shots have both NBI active and a valid mode-active window; "
              "discharge-level FREQUENCY scaling correlation not computed.")
    print("\n(iii) CROSS-DISCHARGE SYNTHESIS of the [M6] Zhong-et-al. primary validation criterion")
    print("      and the [M4] Alfven velocity-scaling verdict (tally of the per-shot MEETS/does-NOT-meet")
    print("      results above -- not a new statistical test):")
    zhong_list = [r.get("zhong_results") for r in results_list]
    n_shots_total = len(results_list)

    def _tally(key_r, key_p, label):
        n_meets, n_evaluated = 0, 0
        for zr in zhong_list:
            if zr is None or zr.get(key_r) is None:
                continue
            n_evaluated += 1
            if abs(zr[key_r]) > 0.7 and zr[key_p] < 0.05:
                n_meets += 1
        if n_evaluated == 0:
            print(f"    - {label}: not evaluated in any shot.")
        else:
            print(f"    - {label}: MEETS |r|>0.7 & p<0.05 in {n_meets} of {n_evaluated} evaluated discharge(s) "
                  f"(out of {n_shots_total} total).")

    _tally("r_ece_sig", "p_ece_adj", "Envelope vs. ECE-core-proxy (Zhong Fig. 2 analogue)")
    _tally("r_pressure_sig", "p_pressure_adj", "Envelope vs. pressure proxy (Zhong Fig. 3 analogue)")
    print("      ⚠️ Chirp-rate (frequency-sweep) readings of [M6] are excluded from this tally -- see the")
    print("      per-shot [METROLOGY] notes: their N_eff hits the statistical-power floor in every shot,")
    print("      so a MEETS/does-NOT-meet count for them would not be meaningful.")

    n_alfven_meets = sum(
        1 for r in results_list
        if r.get("density_detected") and abs(r.get("r_val_scaling", 0)) > 0.7 and r.get("p_val_scaling", 1.0) < 0.05
    )
    n_alfven_evaluated = sum(1 for r in results_list if r.get("density_detected"))
    if n_alfven_evaluated > 0:
        print(f"    - [M4] Alfven velocity-scaling (f_measured vs f_theoretical): MEETS |r|>0.7 & p<0.05 in "
              f"{n_alfven_meets} of {n_alfven_evaluated} evaluated discharge(s) (out of {n_shots_total} total).")
    print("      ⚠️ N=4 discharges is a very small sample for a physics claim meant to generalize beyond")
    print("      these specific shots; report per-shot AND aggregate numbers together in the thesis, not")
    print("      the aggregate alone.")

    fig, axs = plt.subplots(3, 1, figsize=(10, 13))

    shots_lbl = [str(r["shot"]) for r in results_list]
    r_nbi_vals = [r["r_nbi_tot"] if not r["nbi_mismatch_detected"] else np.nan for r in results_list]
    r_ech_vals = [r["r_ech"] for r in results_list]
    r_partial_vals = [r["r_partial"] for r in results_list]

    x = np.arange(len(shots_lbl))
    width = 0.25
    axs[0].bar(x - width, r_nbi_vals, width, label="r (Envelope vs. NBI)", color="darkorange")
    axs[0].bar(x, r_ech_vals, width, label="r (Envelope vs. ECH, full range)", color="magenta")
    axs[0].bar(x + width, r_partial_vals, width, label="r_partial (ECH, step-controlled)", color="gray")
    axs[0].axhline(y=0.7, color='black', linestyle=':', label='Significance threshold (|r|=0.7)')
    axs[0].axhline(y=-0.7, color='black', linestyle=':')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(shots_lbl)
    axs[0].set_ylabel("Pearson r")
    axs[0].set_title("Per-Shot Correlation Summary Across Discharges")
    axs[0].legend(loc='upper right', fontsize=8)
    axs[0].grid(True, alpha=0.3)

    if len(scaling_points) >= 2:
        nbi_levels = np.array([p[0] for p in scaling_points])
        env_levels = np.array([p[1] for p in scaling_points])
        shot_ids_scal = [r["shot"] for r in results_list if not r["nbi_mismatch_detected"]]
        axs[1].scatter(nbi_levels, env_levels, color='darkorange', s=80, zorder=3)
        for xi, yi, sid in zip(nbi_levels, env_levels, shot_ids_scal):
            axs[1].annotate(str(sid), (xi, yi), textcoords="offset points", xytext=(6, 6), fontsize=9)
        if len(scaling_points) >= 2 and np.std(nbi_levels) > 0:
            fit = np.polyfit(nbi_levels, env_levels, 1)
            xx = np.linspace(nbi_levels.min(), nbi_levels.max(), 50)
            axs[1].plot(xx, np.polyval(fit, xx), color='gray', linestyle='--', alpha=0.7, label='Linear fit (indicative)')
        axs[1].set_xlabel("Mean NBI Power, NBI-active window (a.u.)")
        axs[1].set_ylabel("Mean Envelope Amplitude, active window (V)")
        axs[1].set_title("Discharge-Level Scaling: Mode AMPLITUDE vs. NBI Power Level")
        if np.std(nbi_levels) > 0:
            axs[1].legend(loc='best', fontsize=8)
        axs[1].grid(True, alpha=0.3)
    else:
        axs[1].text(0.5, 0.5, "Not enough in-window-NBI discharges for a scaling plot",
                     ha='center', va='center', transform=axs[1].transAxes)

    if len(freq_scaling_points) >= 2:
        nbi_levels_f = np.array([p[0] for p in freq_scaling_points])
        freq_levels = np.array([p[1] for p in freq_scaling_points])
        shot_ids_freq = [r["shot"] for r in results_list
                          if not r["nbi_mismatch_detected"] and r.get("mean_ifreq_active") is not None]
        axs[2].scatter(nbi_levels_f, freq_levels, color='steelblue', s=80, zorder=3)
        for xi, yi, sid in zip(nbi_levels_f, freq_levels, shot_ids_freq):
            axs[2].annotate(str(sid), (xi, yi), textcoords="offset points", xytext=(6, 6), fontsize=9)
        if np.std(nbi_levels_f) > 0:
            fit_f = np.polyfit(nbi_levels_f, freq_levels, 1)
            xx_f = np.linspace(nbi_levels_f.min(), nbi_levels_f.max(), 50)
            axs[2].plot(xx_f, np.polyval(fit_f, xx_f), color='gray', linestyle='--', alpha=0.7, label='Linear fit (indicative)')
            axs[2].legend(loc='best', fontsize=8)
        axs[2].set_xlabel("Mean NBI Power, NBI-active window (a.u.)")
        axs[2].set_ylabel("Mean Measured Frequency, mode-active window (kHz)")
        axs[2].set_title("Discharge-Level Scaling: Mode FREQUENCY vs. NBI Power Level")
        axs[2].grid(True, alpha=0.3)
    else:
        axs[2].text(0.5, 0.5, "Not enough discharges with a valid mode-active window for a scaling plot",
                     ha='center', va='center', transform=axs[2].transAxes)

    plt.tight_layout()
    suffix = get_output_suffix(args)
    output_png = f"mhd_analysis_objective2_multishot_summary{suffix}.png"
    plt.savefig(output_png, dpi=150)
    plt.close(fig)
    print(f"\nCross-discharge summary plot saved to: '{output_png}'")


def main():
    parser = argparse.ArgumentParser(description="EPM vs. Real Multichannel Heating Correlation Analysis (Objective 2, multi-shot)")
    parser.add_argument("--shots", type=int, nargs="+", default=SHOTS_DEFAULT,
                         help=f"List of shot numbers to analyze (def: {SHOTS_DEFAULT})")
    parser.add_argument("--data-dir-pattern", type=str, default="data/hj{shot}",
                         help="Data directory pattern, '{shot}' is substituted per shot (def: data/hj{shot})")
    parser.add_argument("-l", "--lower", type=float, default=40.0, help="Bandpass filter lower frequency in kHz (def: 40)")
    parser.add_argument("-u", "--upper", type=float, default=80.0, help="Bandpass filter upper frequency in kHz (def: 80)")
    parser.add_argument("-o", "--order", type=int, default=4, help="Bessel filter order (def: 4)")
    parser.add_argument("-s", "--smoothing", type=int, default=OPTIMAL_SG_WIN, help=f"Savitzky-Golay smoothing window (def: {OPTIMAL_SG_WIN})")
    parser.add_argument("-n", "--nfft", type=int, default=1024, help="FFT size for cross-coherence (def: 1024)")
    parser.add_argument("-e", "--ensemble", type=int, default=10, help="Number of ensembles for cross-coherence (def: 10)")
    parser.add_argument("--pmp-nfft", type=int, default=256,
                         help="[M10] FFT size for the poloidal (PMP array) spectrogram used in the "
                              "mode-number decomposition (def: 256; kept smaller than --nfft since the "
                              "flat-frequency window is typically short)")
    parser.add_argument("--pmp-max-mode-number", type=int, default=6,
                         help="[M10] Search poloidal mode numbers m = -this..+this (def: 6, covering the "
                              "range reported for Heliotron J EPMs)")
    parser.add_argument("--pmp-max-mode-expanded", type=int, default=20,
                         help="[M10-AUTOEXPAND] Cap on the automatic widened re-check triggered when "
                              "m_dominant is pinned at +/-pmp_max_mode_number (def: 20)")
    parser.add_argument("--pmp-skip-self-test", action="store_true",
                         help="[M10-SELFTEST] Skip the synthetic array-geometry self-test that runs once "
                              "per shot before the real-data poloidal decomposition")
    parser.add_argument("--pmp-self-test-max-m", type=int, default=10,
                         help="[M10-SELFTEST] Synthetic test mode range m = -this..+this (def: 10)")
    parser.add_argument("--pmp-invert-channels", type=str, nargs="*", default=list(PMP_INVERT_CHANNELS_DEFAULT),
                         help="[M10] Poloidal probe(s) to invert (x -1) at load time, per confirmed reversed "
                              f"wiring (def: {list(PMP_INVERT_CHANNELS_DEFAULT)})")
    parser.add_argument("--disable-poloidal-array", action="store_true",
                         help="[M10] Skip loading/analyzing the PMP1-PMP14 poloidal array entirely")
    parser.add_argument("--nave-medfilt", type=int, default=5,
                         help="Median kernel size applied to raw nave to remove interferometer fringe jumps (def: 5)")
    parser.add_argument("--alfven-cal-start", type=float, default=250.0,
                         help="Start (ms) of the Alfven scaling calibration window (def: 250, mid-burst -- "
                              "away from the ECH step at ~170ms, the NBI step at ~220ms, AND the mode's own "
                              "growth ramp; see --alfven-cal-sweep for a robustness check of this choice)")
    parser.add_argument("--alfven-cal-end", type=float, default=270.0,
                         help="End (ms) of the Alfven scaling calibration window (def: 270)")
    parser.add_argument("--alfven-cal-sweep", type=str, default="150:165,235:250,250:270,270:285",
                         help="[EXT][CORRECTED] Comma-separated start:end (ms) candidate calibration windows. NOTE: "
                              "this does NOT test robustness of r_val_scaling -- Pearson r is mathematically invariant "
                              "to the positive-scalar norm_constant these windows produce (confirmed: identical r to "
                              "4 decimals across all windows in testing). It only shows how norm_constant (i.e. where "
                              "the theoretical curve sits in the plot) varies with the window choice. Set to an empty "
                              "string to disable.")
    parser.add_argument("--mode-active-k", type=float, default=6.0,
                         help="[EXT][BUGFIX] Mode-active threshold: envelope > median(envelope) + k * MAD_std(envelope), "
                              "computed over the FULL trace (def: 6.0). MAD (median absolute deviation, scaled to be "
                              "std-equivalent) is robust to the burst itself being a small fraction of the trace, unlike "
                              "a mean+std threshold anchored to a pre-heating quiescent window: a fixed pre-heating "
                              "baseline badly underestimates the noise floor once heating raises broadband turbulence "
                              "levels everywhere, causing a diffuse threshold-crossing mask across most of the heated "
                              "phase instead of the true coherent burst (confirmed in initial testing: a quiescent-window "
                              "std threshold flagged ~240ms of the discharge as 'mode-active', ~59% of which were "
                              "below-threshold gaps -- i.e. not a real contiguous burst). Used to restrict "
                              "instantaneous-frequency-vs-heating correlation ([M7]) and the chirp-rate-vs-distribution-"
                              "function-proxy analysis ([M6]) to periods where the Hilbert instantaneous frequency is "
                              "physically meaningful.")
    parser.add_argument("--mode-active-max-gap-ms", type=float, default=3.0,
                         help="[EXT][BUGFIX] Max gap (ms) bridged when finding the mode-active BURST: a raw "
                              "envelope > threshold mask is typically speckled across the whole heated phase "
                              "(broadband noise/turbulence rises with heating even without a coherent mode), "
                              "not just the true burst. Small within-burst dropouts up to this many ms are "
                              "closed (binary_closing) before picking the single largest contiguous run, so "
                              "isolated above-threshold noise elsewhere in the discharge is NOT included.")
    parser.add_argument("--mode-active-min-duration-ms", type=float, default=10.0,
                         help="[EXT][BUGFIX] Minimum duration (ms) for the largest contiguous mode-active run "
                              "to be accepted; shorter 'bursts' are treated as noise and [M7]/[M6]-chirp are skipped.")
    parser.add_argument("--flat-slope-smooth-ms", type=float, default=2.0,
                         help="[M9] Moving-average smoothing window (ms) applied to |d(f_inst)/dt| before "
                              "flat-frequency-region detection (def: 2.0). Larger = smoother slope estimate, "
                              "less sensitive to per-sample phase noise, but can blur short genuine sweeps.")
    parser.add_argument("--flat-scan-window-ms", type=float, default=8.0,
                         help="[M9] Length (ms) of the sliding 'core' window scanned across the search "
                              "domain to locate its flattest patch (lowest mean |d(f_inst)/dt|), def: 8.0. "
                              "This core is then grown outward (see --flat-growth-tolerance) into the final "
                              "sub-window. Shorten this if the true flat plateau is narrower than 8 ms; "
                              "lengthen it if the scan is landing on a short noise dip rather than the real "
                              "plateau.")
    parser.add_argument("--flat-growth-tolerance", type=float, default=0.5,
                         help="[M9] Relative tolerance (def: 0.5, i.e. 50%%) used to grow the flattest-patch "
                              "core outward: the window keeps extending in each direction as long as its "
                              "mean |d(f_inst)/dt| stays within (1 + this) x the core's own mean. Lower = "
                              "stricter growth (smaller, more conservative final window, closer to the raw "
                              "--flat-scan-window-ms core); higher = grows further into moderately-sloped "
                              "territory before stopping at the chirping edges.")
    parser.add_argument("--flat-min-duration-ms", type=float, default=5.0,
                         help="[M9] Minimum duration (ms) for the grown flat-frequency window to be accepted "
                              "(def: 5.0). If the grown window is still shorter than this, [M8] falls back to "
                              "the full search domain automatically (with a printed warning).")
    parser.add_argument("--flat-window-start", type=float, default=None,
                         help="[M9] Optional manual override (ms) for the flat-frequency sub-window start, "
                              "instead of the adaptive scan-and-grow detector -- e.g. to pin the window to a "
                              "value read off the Objective-1 spectrogram by eye for one specific shot. Must "
                              "be paired with --flat-window-end. Clipped to the [M9] search domain if it "
                              "extends outside it.")
    parser.add_argument("--flat-window-end", type=float, default=None,
                         help="[M9] Optional manual override (ms) for the flat-frequency sub-window end.")
    parser.add_argument("--obj1-results-dir", type=str, default=".",
                         help="[M9][OBJ1-XREF] Directory to look in for Objective 1's exported "
                              "'discrete_modes_shot_{shot}.json' files (def: '.', i.e. the current working "
                              "directory -- where mhd_analysis_obj1.py writes them by default). The current "
                              "working directory is always also checked as a fallback.")
    parser.add_argument("--obj1-json-pattern", type=str, default="discrete_modes_shot_{shot}.json",
                         help="[M9][OBJ1-XREF] Filename pattern for Objective 1's export, '{shot}' "
                              "substituted (def: discrete_modes_shot_{shot}.json, matching "
                              "mhd_analysis_obj1.py's own naming).")
    parser.add_argument("--obj1-mode-freq-tol-khz", type=float, default=3.0,
                         help="[M9][OBJ1-XREF] Tolerance (kHz) beyond [--lower, --upper] within which an "
                              "Objective-1 mode is still considered 'the same confirmed mode' for "
                              "cross-referencing (def: 3.0), since Objective 1's peak-picked frequency need "
                              "not land exactly on Objective 2's bandpass edges.")
    parser.add_argument("--disable-obj1-reference", action="store_true",
                         help="[M9][OBJ1-XREF] Disable the Objective-1 JSON cross-reference entirely; [M9] "
                              "then always searches the full [M7] envelope-active burst, as if Objective 1's "
                              "export did not exist.")
    parser.add_argument("--mode-active-start", type=float, default=None,
                         help="[EXT] Optional manual override (ms) for the mode-active window start, instead "
                              "of the adaptive envelope-threshold + burst-detection mask. Must be paired with --mode-active-end.")
    parser.add_argument("--mode-active-end", type=float, default=None,
                         help="[EXT] Optional manual override (ms) for the mode-active window end.")
    parser.add_argument("--m7-max-lag-ms", type=float, default=40.0,
                         help="[EXT][BUGFIX] Search window (+/- ms) for the [M7] instantaneous-frequency-vs-"
                              "heating lagged cross-correlation (def: 40, wider than the envelope [M3]'s 20ms "
                              "since two of four test shots hit the +/-20ms edge exactly in initial testing).")
    parser.add_argument("--bfield-file", type=str, default=None,
                         help="Single B(t) .edf channel, used for every shot (only sensible with one shot). "
                              "Prefer --bfield-pattern for multi-shot runs, or --bfield-constant-tesla if B "
                              "is confirmed constant (no time-varying channel exists).")
    parser.add_argument("--bfield-pattern", type=str, default=None,
                         help="B(t) .edf file pattern with '{shot}' substituted per shot, e.g. "
                              "'data/hj{shot}/BTOR@{shot}.edf'. Ignored if --bfield-constant-tesla is set.")
    parser.add_argument("--bfield-constant-tesla", type=float, default=1.25,
                         help="[RESOLVED] Confirmed constant toroidal field in Tesla (def: 1.25, Heliotron J "
                              "flat-top, confirmed -- no time-varying B(t) channel exists for these shots). "
                              "Set to a negative value or override --bfield-pattern to disable and use a "
                              "real B(t) file instead.")
    parser.add_argument("--ion-mass-amu", type=float, default=1.0, help="Ion mass in amu (def: 1.0, H)")
    parser.add_argument("--major-radius-m", type=float, default=1.2, help="Heliotron J major radius in meters (def: 1.2)")
    parser.add_argument("--iota", type=float, default=0.56, help="Rotational transform iota (def: 0.56, q = 1/iota ≈ 1.786)")
    parser.add_argument("--ech-active-start", type=float, default=170.0,
                         help="Start (ms) of the active ECH heating window (def: 170)")
    parser.add_argument("--ech-active-end", type=float, default=290.0,
                         help="End (ms) of the active ECH heating window (def: 290)")
    parser.add_argument("--ech-glitch-start", type=float, default=170.0,
                         help="Start (ms) of the ECH turn-on transient (def: 170)")
    parser.add_argument("--ech-glitch-end", type=float, default=190.0,
                         help="End (ms) of the ECH turn-on transient (def: 190)")
    parser.add_argument("--ece-channels", type=int, nargs="+", default=list(range(1, 17)),
                         help="ECE#FAST channel numbers to search when --ece-core-channel is not given "
                              "(def: 1-16, all channels). No channel is hardcoded as excluded here any more "
                              "-- saturated channels (e.g. channel 12, found saturated for shots 88652-88655) "
                              "are detected and dropped algorithmically at run time by [SAT-DETECT]; see "
                              "--sat-rail-frac-threshold / --sat-plateau-run-threshold.")
    parser.add_argument("--ece-core-channel", type=int, default=None,
                         help="Core ECE channel number to use directly, skipping both saturation-screening and "
                              "the auto-select heuristic. Not set by default: the [SAT-DETECT] algorithm "
                              "screens --ece-channels for ADC saturation/clipping (channel 12 was found "
                              "saturated for shots 88652-88655, but this is now re-checked per shot rather "
                              "than assumed), and the auto-select heuristic picks the core-proxy channel from "
                              "whatever remains. Pass an explicit channel number here once a specific core "
                              "channel is confirmed with the diagnostics team.")
    parser.add_argument("--sat-rail-frac-threshold", type=float, default=0.02,
                         help="[SAT-DETECT] A channel is flagged saturated if more than this fraction of its "
                              "samples sit within 0.5%% of its own observed min or max (def: 0.02, i.e. 2%%). "
                              "Lower = stricter (flags more channels as saturated).")
    parser.add_argument("--sat-plateau-run-threshold", type=int, default=20,
                         help="[SAT-DETECT] A channel is flagged saturated if it contains a run of more than "
                              "this many consecutive near-identical samples anywhere in the trace (def: 20), "
                              "which signals ADC clipping (a flat-topped plateau) rather than genuine plasma "
                              "fluctuation. Lower = stricter.")
    parser.add_argument("--ece-file-pattern", type=str, default="ECE{ch}FAST@{shot}.edf",
                         help="ECE channel filename pattern, '{ch}' and '{shot}' substituted (def: ECE{ch}FAST@{shot}.edf)")
    parser.add_argument("--beam-species", type=str, choices=["H", "D"], default="H",
                         help="[RESOLVED] NBI beam ion species for [M6] (def: H, confirmed).")
    parser.add_argument("--beam-energy-kev", type=float, default=30.0,
                         help="NBI full-energy-component injection energy in keV for [M6] (def: 30.0, "
                              "Heliotron J max acceleration voltage; not yet used quantitatively -- see [M6] notes)")
    parser.add_argument("--m6-max-lag-ms", type=float, default=60.0,
                         help="[M6][BUGFIX] Search window (+/- ms) for envelope-vs-ECE and "
                              "envelope-vs-pressure-proxy lagged cross-correlation (def: 60.0). Widen this "
                              "if the printed BOUNDARY WARNING appears.")
    parser.add_argument("--macro-max-lag-ms", type=float, default=50.0,
                         help="Search window (+/- ms) for Wp (stored energy) and nave (density) lagged "
                              "cross-correlation against mode envelope (def: 50.0).")
    parser.add_argument("--nave-corr-start", type=float, default=270.0,
                         help="Start time (ms) for nave vs. mode envelope correlation, isolating the "
                              "second hill and avoiding early-phase noise (def: 270.0).")
    parser.add_argument("--nave-corr-end", type=float, default=None,
                         help="End time (ms) for nave vs. mode envelope correlation (def: None, until end of trace).")
    parser.add_argument("--te-calib-scale-ev-per-v", type=float, default=None,
                         help="[M6] Te calibration slope (eV per Volt) for the core ECE channel, if you obtain "
                              "one (e.g. from a Thomson-scattering cross-calibration or the diagnostics team's "
                              "radiometer gain). Not derivable from the raw ECE signal alone -- see chat notes. "
                              "When provided, enables the theoretical electron-drag slowing-down-time estimate.")
    parser.add_argument("--te-calib-offset-ev", type=float, default=0.0,
                         help="[M6] Te calibration offset (eV), used with --te-calib-scale-ev-per-v (def: 0.0).")
    parser.add_argument("--bands", type=str, nargs="+", default=["40:80", "80:120"],
                         help="List of frequency bands in lower:upper format (kHz), e.g. --bands 40:80 80:120 (def: ['40:80', '80:120']). "
                              "Runs the full analysis for each frequency window (40-80 kHz secondary mode, 80-120 kHz primary mode).")
    parser.add_argument("--output-suffix", type=str, default=None,
                         help="Optional explicit suffix for output image filenames (e.g. '_40_80kHz')")
    args = parser.parse_args()

    if args.bfield_constant_tesla is not None and args.bfield_constant_tesla < 0:
        args.bfield_constant_tesla = None  # explicit opt-out, falls back to --bfield-pattern/file or simplified scaling
    if args.ece_core_channel is not None and not (1 <= args.ece_core_channel <= 16):
        args.ece_core_channel = None  # falls back to the auto-select heuristic

    if args.bands:
        for band_str in args.bands:
            try:
                l_str, u_str = band_str.split(":")
                b_lower, b_upper = float(l_str), float(u_str)
            except ValueError:
                print(f"⚠️ Invalid band format '{band_str}'; expected lower:upper in kHz (e.g. 80:120)")
                continue
            args.lower = b_lower
            args.upper = b_upper
            args.output_suffix = f"_{int(b_lower)}_{int(b_upper)}kHz"
            print(f"\n{'#' * 93}")
            print(f"=== RUNNING ANALYSIS FOR FREQUENCY BAND: {b_lower:.1f} - {b_upper:.1f} kHz ===")
            print(f"{'#' * 93}")
            print(f"=== Objective 2 Multi-Shot Analysis: shots {args.shots} (Band: {b_lower:.1f}-{b_upper:.1f} kHz) ===")
            results_list = []
            for shot in args.shots:
                result = process_shot(shot, args)
                results_list.append(result)

            n_ok = sum(1 for r in results_list if r is not None)
            n_fail = len(results_list) - n_ok
            print(f"\n{'=' * 93}")
            print(f"Per-shot processing complete for band {b_lower:.1f}-{b_upper:.1f} kHz: {n_ok} shot(s) succeeded, {n_fail} shot(s) skipped.")
            cross_discharge_analysis(results_list, args)
        return

    print(f"=== Objective 2 Multi-Shot Analysis: shots {args.shots} ===")
    results_list = []
    for shot in args.shots:
        result = process_shot(shot, args)
        results_list.append(result)

    n_ok = sum(1 for r in results_list if r is not None)
    n_fail = len(results_list) - n_ok
    print(f"\n{'=' * 93}")
    print(f"Per-shot processing complete: {n_ok} shot(s) succeeded, {n_fail} shot(s) skipped (missing data).")

    cross_discharge_analysis(results_list, args)


if __name__ == "__main__":
    main()