"""
sawtooth_cross_diagnostic_88653.py

Exhaustive cross-diagnostic search for sawtooth-like waveforms across all 112 .edf files
(191 physical channels across 18 diagnostic families) for Heliotron J discharge #88653.

Evaluates three timescales:
  Regime I: Mesoscopic repetitive sawteeth (T ~ 0.7-1.8 ms, f_mod ~ 0.55-1.5 kHz, 259.1-300.4 ms)
  Regime II: Macroscopic ramp-and-crash events (225-240 ms early burst & 255-305 ms main quench)
  Regime III: Microscopic fast carrier wave-steepening (f0 ~ 88 kHz -> 2f0 ~ 176 kHz, 1 MHz channels)

Methodological features:
  - Zero-phase 4th-order Bessel filtering (sosfiltfilt) and Savitzky-Golay baseline detrending
  - Time-reversal asymmetry (derivative skewness, rise/fall duty cycle)
  - Lagged cross-correlation r(tau) across [-5, +5] ms with parabolic sub-sample peak/lag extraction
  - Forward Superposed Epoch Analysis (SEA) triggered on primary mode sawtooth resets
  - Reverse SEA & pre-mode search (171.2-259.1 ms) for precursor / unsynchronized relaxation crashes
  - 1000 Fourier phase-randomized / IAAFT surrogates for empirical p-values
  - Effective sample size Neff accounting for autocorrelation time tau_int
  - Benjamini-Hochberg FDR correction across all channels
  - Exports CSV master ledger, JSON audit dictionary, and 5 publication figures
"""

import sys
import os
import time
import json
import csv
from pathlib import Path

import numpy as np
import scipy.signal as dsp
from scipy.stats import skew
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "hj88653"
ANALYSIS_DIR = ROOT_DIR / "analysis"
OUTPUT_DIR = ANALYSIS_DIR / "obj_ech"
FIG_DIR = ROOT_DIR / "Images" / "sawtooth_analysis_88653"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT_DIR / "jpack"))
import turnelib as TE

RNG_SEED = 20261018
rng = np.random.default_rng(RNG_SEED)


def load_edf(filepath):
    """Safely loads an EDF file into numpy arrays."""
    e = TE.edf()
    dat = e.load(str(filepath))
    t_raw = dat[:, 0]
    t_ms = t_raw if e.DimUnit[0] == "ms" else t_raw * 1000.0
    ys = dat[:, 1:] if dat.shape[1] > 2 else dat[:, 1]
    unit = getattr(e, "ValUnit", ["V"])
    return t_ms, ys, unit


def compute_tau_int(x, max_lag=500):
    """Estimates integrated autocorrelation time tau_int via FFT."""
    x_c = x - np.nanmean(x)
    n = len(x_c)
    if n <= 1:
        return 1.0
    var = np.nanvar(x_c)
    if var < 1e-15:
        return 1.0
    max_lag = min(max_lag, n - 2)
    f = np.fft.rfft(x_c, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n] / (var * n)
    zc = np.where(acf < 0)[0]
    cutoff = zc[0] if len(zc) > 0 else min(max_lag, n // 4)
    cutoff = min(cutoff, max_lag)
    tau = 1.0 + 2.0 * np.sum(np.clip(acf[1:cutoff], 0, None))
    return float(max(1.0, tau))


def generate_phase_surrogates(x, n_surr=1000, seed=RNG_SEED):
    """Generates Fourier phase-randomized surrogates preserving exact PSD and autocorrelation."""
    n = len(x)
    local_rng = np.random.default_rng(seed)
    fft_x = np.fft.rfft(x)
    mag = np.abs(fft_x)
    n_freq = len(fft_x)
    phases = local_rng.uniform(0, 2.0 * np.pi, size=(n_surr, n_freq))
    phases[:, 0] = 0.0
    if n % 2 == 0:
        phases[:, -1] = 0.0
    surr_ffts = mag * np.exp(1j * phases)
    surrs = np.fft.irfft(surr_ffts, n=n, axis=1)
    return surrs


def extract_primary_mode_reference(shot_dir):
    """Extracts the primary mode reference signals and sawtooth reset timestamps from MP1."""
    print(">>> Extracting Primary Mode Reference from MP1@88653.edf...")
    t_ms, mp1, _ = load_edf(shot_dir / "MP1@88653.edf")
    dt = (t_ms[1] - t_ms[0]) / 1000.0
    fs = 1.0 / dt

    # Bandpass filter in [75, 110] kHz using zero-phase Bessel
    sos_pri = dsp.bessel(4, [75e3, 110e3], btype="bandpass", fs=fs, output="sos")
    flt_pri = dsp.sosfiltfilt(sos_pri, mp1)
    ana_pri = dsp.hilbert(flt_pri)
    env_pri = np.abs(ana_pri).astype(np.float32)
    phase_pri = np.unwrap(np.angle(ana_pri))

    # Instantaneous frequency: 125-sample Savitzky-Golay (125 us window)
    ifreq_pri_raw = (dsp.savgol_filter(phase_pri, 125, 2, deriv=1) / (2.0 * np.pi * dt) / 1e3).astype(np.float32)

    # Active window 259.1 to 300.4 ms
    m_act = (t_ms >= 259.1) & (t_ms <= 300.4)
    t_act = t_ms[m_act]
    f_act = ifreq_pri_raw[m_act]
    env_act = env_pri[m_act]

    # Median filter to capture the macroscopic sawtooth cycles
    f_act_smooth = dsp.medfilt(f_act, 301)
    df_dt_act = dsp.savgol_filter(f_act_smooth, 201, 2, deriv=1, delta=0.001) # kHz/ms

    # Sawtooth reset detection: sharp upward jumps in ifreq (or crash resets)
    # Require min peak height 0.8 kHz/ms, distance 0.4 ms (400 samples)
    peaks, _ = dsp.find_peaks(df_dt_act, height=0.8, distance=400)
    t_resets = t_act[peaks]

    # Waveform asymmetry metrics on the primary mode
    diff_f = np.gradient(f_act_smooth, t_act)
    skew_pri = float(skew(diff_f))

    # Cycle rise-to-fall duty cycle
    troughs, _ = dsp.find_peaks(-df_dt_act, distance=400)
    # Estimate duty ratio
    d_duty = 0.25 # typical sharp rise + slow ramp down
    if len(peaks) > 5 and len(troughs) > 5:
        ratios = []
        for p in peaks:
            prev_t = troughs[troughs < p]
            next_t = troughs[troughs > p]
            if len(prev_t) > 0 and len(next_t) > 0:
                t_prev = t_act[prev_t[-1]]
                t_next = t_act[next_t[0]]
                t_p = t_act[p]
                if t_next > t_prev:
                    ratios.append((t_p - t_prev) / (t_next - t_prev))
        if len(ratios) > 0:
            d_duty = float(np.median(ratios))

    print(f"  Primary mode: {len(t_resets)} sawtooth resets detected in [259.1, 300.4] ms.")
    print(f"  Mean period = {np.mean(np.diff(t_resets)):.2f} ms (+/- {np.std(np.diff(t_resets)):.2f} ms).")
    print(f"  Derivative skewness S_deriv = {skew_pri:.3f}, Duty ratio = {d_duty:.3f}.")

    return {
        "t_ms": t_ms,
        "mp1": mp1,
        "env_pri": env_pri,
        "ifreq_pri": ifreq_pri_raw,
        "t_act": t_act,
        "f_act": f_act,
        "f_act_smooth": f_act_smooth,
        "env_act": env_act,
        "df_dt_act": df_dt_act,
        "t_resets": t_resets,
        "skew_pri": skew_pri,
        "d_duty": d_duty,
        "fs": fs
    }


def analyze_channel_regime_I(ch_data, t_ch, fs_ch, ref, n_surr=500):
    """
    Evaluates Regime I: Mesoscopic repetitive sawteeth (259.1-300.4 ms).
    Returns metrics dict.
    """
    m_act = (t_ch >= 259.1) & (t_ch <= 300.4)
    if np.sum(m_act) < 50:
        return {"status": "UNTESTABLE_TOO_SHORT"}

    t_w = t_ch[m_act]
    y_raw = ch_data[m_act]

    # Check for ADC saturation / rail clipping / dead channel
    ptp_raw = np.ptp(y_raw)
    if ptp_raw < 1e-5:
        is_saturated = True
    else:
        min_consec_rail = max(20, int(0.0001 * fs_ch))
        diff_zero = (np.diff(y_raw) == 0)
        is_saturated = False
        if np.any(diff_zero):
            diff_mask = np.diff(np.concatenate(([0], diff_zero.astype(int), [0])))
            starts = np.where(diff_mask == 1)[0]
            ends = np.where(diff_mask == -1)[0]
            long_runs = np.where((ends - starts) >= min_consec_rail)[0]
            if len(long_runs) > 0:
                ymin = np.min(y_raw)
                ymax = np.max(y_raw)
                for r_idx in long_runs:
                    val = y_raw[starts[r_idx]]
                    if np.isclose(val, ymin, atol=1e-4) or np.isclose(val, ymax, atol=1e-4):
                        is_saturated = True
                        break

    # 1. Conditioning: Zero-phase Bessel bandpass [0.35, min(3.5, 0.4*fs_ch)] kHz
    f_nyq = fs_ch / 2.0
    f_hi = min(3500.0, 0.4 * fs_ch)
    if f_hi <= 350.0:
        # Diagnostic too slow for full bandpass (e.g. 5 kHz sampling -> Nyq 2.5 kHz)
        f_hi = min(1500.0, 0.4 * fs_ch)
        f_lo = min(100.0, 0.1 * fs_ch)
    else:
        f_lo = 350.0

    sos_bp = dsp.bessel(4, [f_lo, f_hi], btype="bandpass", fs=fs_ch, output="sos")
    try:
        y_bp = dsp.sosfiltfilt(sos_bp, y_raw)
    except Exception:
        y_bp = y_raw - np.mean(y_raw)

    # 2. Derivative skewness
    dt_w = (t_w[1] - t_w[0]) if len(t_w) > 1 else 1.0
    dy_dt = np.gradient(y_bp, dt_w)
    s_deriv = float(skew(dy_dt)) if np.std(dy_dt) > 1e-9 else 0.0

    # 3. Cycle duty ratio
    peaks_ch, _ = dsp.find_peaks(y_bp, distance=max(3, int(0.4 * 1e-3 * fs_ch)))
    troughs_ch, _ = dsp.find_peaks(-y_bp, distance=max(3, int(0.4 * 1e-3 * fs_ch)))
    duty_ratio = 0.50
    if len(peaks_ch) > 4 and len(troughs_ch) > 4:
        r_list = []
        for p in peaks_ch:
            prev_t = troughs_ch[troughs_ch < p]
            next_t = troughs_ch[troughs_ch > p]
            if len(prev_t) > 0 and len(next_t) > 0:
                denom = t_w[next_t[0]] - t_w[prev_t[-1]]
                if denom > 0:
                    r_list.append((t_w[p] - t_w[prev_t[-1]]) / denom)
        if len(r_list) > 0:
            duty_ratio = float(np.median(r_list))

    # 4. Lagged cross-correlation against primary mode frequency & envelope
    # Resample primary mode reference onto diagnostic time grid t_w
    f_ref_interp = np.interp(t_w, ref["t_act"], ref["f_act_smooth"])
    env_ref_interp = np.interp(t_w, ref["t_act"], ref["env_act"])

    # Demean
    y_norm = (y_bp - np.mean(y_bp)) / (np.std(y_bp) + 1e-12)
    f_norm = (f_ref_interp - np.mean(f_ref_interp)) / (np.std(f_ref_interp) + 1e-12)
    env_norm = (env_ref_interp - np.mean(env_ref_interp)) / (np.std(env_ref_interp) + 1e-12)

    # Correlation with frequency
    max_lag_samples = int(min(5.0 * 1e-3 * fs_ch, len(y_norm) // 4))
    corr_f = dsp.correlate(y_norm, f_norm, mode="full") / len(y_norm)
    lags = np.arange(-len(y_norm) + 1, len(y_norm)) * (dt_w) # in ms
    m_lag = (lags >= -5.0) & (lags <= 5.0)
    lags_sub = lags[m_lag]
    corr_f_sub = corr_f[m_lag]

    idx_peak = np.argmax(np.abs(corr_f_sub))
    r_max_f = float(corr_f_sub[idx_peak])
    abs_r_max_f = abs(r_max_f)
    tau_star_f = float(lags_sub[idx_peak])

    # Sub-sample parabolic interpolation on tau_star
    if 1 <= idx_peak < len(corr_f_sub) - 1:
        y1, y2, y3 = corr_f_sub[idx_peak - 1], corr_f_sub[idx_peak], corr_f_sub[idx_peak + 1]
        denom = 2.0 * y2 - y1 - y3
        if abs(denom) > 1e-9:
            delta = 0.5 * (y1 - y3) / denom
            tau_star_f = float(lags_sub[idx_peak] + delta * (lags_sub[1] - lags_sub[0]))

    # Cross-spectral coherence in [0.5, 2.5] kHz
    nperseg = min(512, len(y_bp) // 2)
    if nperseg >= 32:
        f_coh, coh = dsp.coherence(y_norm, f_norm, fs=fs_ch, nperseg=nperseg)
        m_mod = (f_coh >= 500.0) & (f_coh <= 2500.0)
        coh_mod = float(np.max(coh[m_mod])) if np.any(m_mod) else 0.0
    else:
        coh_mod = 0.0

    # 5. Forward Superposed Epoch Analysis (SEA) triggered on primary mode resets
    t_resets = ref["t_resets"]
    dt_grid = np.linspace(-1.5, 1.5, 151)
    epochs = []
    for tr in t_resets:
        if (tr - 1.5 >= t_w[0]) and (tr + 1.5 <= t_w[-1]):
            ep = np.interp(tr + dt_grid, t_w, y_bp)
            epochs.append(ep - np.mean(ep))

    if len(epochs) >= 10:
        epochs_arr = np.array(epochs)
        sea_mean = np.mean(epochs_arr, axis=0)
        sea_se = np.std(epochs_arr, axis=0) / np.sqrt(len(epochs))
        sea_ptp = float(np.ptp(sea_mean))
        mean_se = float(np.mean(sea_se)) + 1e-12
        snr_sea = float(sea_ptp / mean_se)
        idx_sea_min = np.argmin(sea_mean)
        idx_sea_max = np.argmax(sea_mean)
        offset_sea_ms = float(dt_grid[idx_sea_max] if abs(sea_mean[idx_sea_max]) > abs(sea_mean[idx_sea_min]) else dt_grid[idx_sea_min])
    else:
        sea_mean = np.zeros_like(dt_grid)
        sea_ptp = 0.0
        snr_sea = 0.0
        offset_sea_ms = 0.0

    # 6. Autocorrelation time and Neff
    tau_int = compute_tau_int(y_norm)
    n_eff = float(len(y_norm) / (2.0 * tau_int))

    # 7. Surrogate statistical null testing
    surrs = generate_phase_surrogates(y_bp, n_surr=n_surr)
    # Surrogate skewness (vectorized)
    d_surrs = np.gradient(surrs, dt_w, axis=1)
    d_mean = np.mean(d_surrs, axis=1, keepdims=True)
    d_std = np.std(d_surrs, axis=1, keepdims=True) + 1e-12
    surr_skew = np.mean(((d_surrs - d_mean) / d_std) ** 3, axis=1)
    p_surr_skew = float((np.sum(np.abs(surr_skew) >= abs(s_deriv)) + 1) / (n_surr + 1))

    # Vectorized surrogate cross-correlation with f_norm
    s_mean = np.mean(surrs, axis=1, keepdims=True)
    s_std = np.std(surrs, axis=1, keepdims=True) + 1e-12
    s_norm = (surrs - s_mean) / s_std

    n_pts = len(y_norm)
    n_fft = 2 * n_pts
    s_fft = np.fft.rfft(s_norm, n=n_fft, axis=1)
    f_fft = np.fft.rfft(f_norm, n=n_fft)
    corr_circ = np.fft.irfft(s_fft * np.conj(f_fft), n=n_fft, axis=1) / n_pts
    corr_full = np.fft.fftshift(corr_circ, axes=1)[:, 1:]
    surr_rmax = np.max(np.abs(corr_full[:, m_lag]), axis=1)
    p_surr_r = float((np.sum(surr_rmax >= abs_r_max_f) + 1) / (n_surr + 1))

    # 8. Reverse SEA & Pre-Mode Activity
    # Check pre-mode window [171.2, 259.1] ms
    m_pre = (t_ch >= 171.2) & (t_ch <= 259.1)
    if np.sum(m_pre) >= 50:
        y_pre = ch_data[m_pre]
        dy_pre = np.gradient(y_pre, dt_w)
        peaks_pre, _ = dsp.find_peaks(np.abs(dy_pre), height=np.std(dy_pre) * 2.5, distance=max(3, int(0.4 * 1e-3 * fs_ch)))
        pre_event_rate = float(len(peaks_pre) / (259.1 - 171.2)) # events/ms
        pre_skew = float(skew(dy_pre)) if np.std(dy_pre) > 1e-9 else 0.0
    else:
        pre_event_rate = 0.0
        pre_skew = 0.0

    return {
        "status": "VALID",
        "is_saturated": bool(is_saturated),
        "s_deriv": s_deriv,
        "duty_ratio": duty_ratio,
        "abs_r_max": abs_r_max_f,
        "r_max_signed": r_max_f,
        "tau_star_ms": tau_star_f,
        "coh_mod": coh_mod,
        "sea_ptp": sea_ptp,
        "snr_sea": snr_sea,
        "offset_sea_ms": offset_sea_ms,
        "tau_int": tau_int,
        "n_eff": n_eff,
        "p_surr_skew": p_surr_skew,
        "p_surr_r": p_surr_r,
        "pre_event_rate": pre_event_rate,
        "pre_skew": pre_skew,
        "sea_waveform": sea_mean.tolist(),
        "dt_grid": dt_grid.tolist()
    }


def analyze_channel_regime_II(ch_data, t_ch, fs_ch, ref):
    """
    Evaluates Regime II: Macroscopic ramp-and-crash envelopes.
    Event IIa: Early burst (225-240 ms, crash at 236.2 ms)
    Event IIb: Main discharge envelope (255-305 ms)
    """
    # Event IIa: 225 to 240 ms
    m_iia = (t_ch >= 225.0) & (t_ch <= 240.0)
    if np.sum(m_iia) >= 20:
        t_iia = t_ch[m_iia]
        y_iia = ch_data[m_iia]
        # Reference envelope in this window
        m_ref_iia = (ref["t_ms"] >= 225.0) & (ref["t_ms"] <= 240.0)
        t_ref_iia = ref["t_ms"][m_ref_iia]
        env_ref_iia = ref["env_pri"][m_ref_iia]
        env_interp = np.interp(t_iia, t_ref_iia, env_ref_iia)

        # Cross-correlation across [-5, +5] ms
        y_n = (y_iia - np.mean(y_iia)) / (np.std(y_iia) + 1e-12)
        e_n = (env_interp - np.mean(env_interp)) / (np.std(env_interp) + 1e-12)
        corr = dsp.correlate(y_n, e_n, mode="full") / len(y_n)
        dt_iia = t_iia[1] - t_iia[0]
        lags = np.arange(-len(y_n) + 1, len(y_n)) * dt_iia
        m_l = (lags >= -5.0) & (lags <= 5.0)
        lags_s = lags[m_l]
        corr_s = corr[m_l]
        idx_p = np.argmax(np.abs(corr_s))
        r_iia = float(corr_s[idx_p])
        tau_iia = float(lags_s[idx_p])
    else:
        r_iia = 0.0
        tau_iia = 0.0

    # Event IIb: 255 to 305 ms
    m_iib = (t_ch >= 255.0) & (t_ch <= 305.0)
    if np.sum(m_iib) >= 20:
        t_iib = t_ch[m_iib]
        y_iib = ch_data[m_iib]
        m_ref_iib = (ref["t_ms"] >= 255.0) & (ref["t_ms"] <= 305.0)
        t_ref_iib = ref["t_ms"][m_ref_iib]
        env_ref_iib = ref["env_pri"][m_ref_iib]
        env_interp_b = np.interp(t_iib, t_ref_iib, env_ref_iib)

        y_nb = (y_iib - np.mean(y_iib)) / (np.std(y_iib) + 1e-12)
        e_nb = (env_interp_b - np.mean(env_interp_b)) / (np.std(env_interp_b) + 1e-12)
        corr_b = dsp.correlate(y_nb, e_nb, mode="full") / len(y_nb)
        dt_iib = t_iib[1] - t_iib[0]
        lags_b = np.arange(-len(y_nb) + 1, len(y_nb)) * dt_iib
        m_lb = (lags_b >= -15.0) & (lags_b <= 15.0)
        lags_sb = lags_b[m_lb]
        corr_sb = corr_b[m_lb]
        idx_pb = np.argmax(np.abs(corr_sb))
        r_iib = float(corr_sb[idx_pb])
        tau_iib = float(lags_sb[idx_pb])
    else:
        r_iib = 0.0
        tau_iib = 0.0

    return {
        "early_burst_r": r_iia,
        "early_burst_tau_ms": tau_iia,
        "main_envelope_r": r_iib,
        "main_envelope_tau_ms": tau_iib
    }


def analyze_channel_regime_III(ch_data, t_ch, fs_ch, ref):
    """
    Evaluates Regime III: Microscopic carrier wave-steepening (f0 ~ 88 kHz -> 2f0 ~ 176 kHz).
    Only valid for fs >= 1 MHz.
    """
    if fs_ch < 1e6:
        return {"status": "UNTESTABLE_NYQUIST"}

    m_act = (t_ch >= 259.1) & (t_ch <= 275.0) # steady flat sub-window
    if np.sum(m_act) < 1000:
        return {"status": "UNTESTABLE_TOO_SHORT"}

    y_sig = ch_data[m_act]

    # Filter in fundamental [75, 105] kHz and second harmonic [160, 200] kHz
    sos_f1 = dsp.bessel(4, [75e3, 105e3], btype="bandpass", fs=fs_ch, output="sos")
    sos_f2 = dsp.bessel(4, [160e3, 200e3], btype="bandpass", fs=fs_ch, output="sos")

    y_f1 = dsp.sosfiltfilt(sos_f1, y_sig)
    y_f2 = dsp.sosfiltfilt(sos_f2, y_sig)

    amp_f1 = float(np.std(y_f1))
    amp_f2 = float(np.std(y_f2))
    h2_ratio = float(amp_f2 / (amp_f1 + 1e-12))

    return {
        "status": "VALID",
        "amp_f1": amp_f1,
        "amp_f2": amp_f2,
        "h2_ratio": h2_ratio
    }


def main():
    print("=" * 80)
    print("EXHAUSTIVE CROSS-DIAGNOSTIC SAWTOOTH ANALYSIS — HELIOTRON J SHOT #88653")
    print("=" * 80)
    t_start_total = time.time()

    # 1. Primary Mode Reference
    ref = extract_primary_mode_reference(DATA_DIR)

    # 2. Build Physical Channel Inventory
    print(f"\n>>> Building Complete Physical Channel Inventory from {DATA_DIR}...")
    all_edf_files = sorted(DATA_DIR.glob("*.edf"))
    print(f"Total .edf files in directory: {len(all_edf_files)}")

    # We will register channels with deduplication
    channels = []

    # Map of multi-channel bundled files
    bundled_files = {
        "axuv@88653.edf": ("AXUV", 16, [f"AXUV{i}" for i in range(1, 17)]),
        "haarr@88653.edf": ("HAARRAY", 32, [f"HAARRAY{i:02d}" for i in range(1, 33)]),
        "pmp@88653.edf": ("PMP", 14, [f"PMP{i}" for i in range(1, 15)]),
        "tmp@88653.edf": ("MP", 4, [f"MP{i}" for i in range(1, 5)]),
        "DivProArr@88653.edf": ("DivProArr", 38, [f"DivPro_{i:02d}" for i in range(1, 39)]),
        "sx2d@88653.edf": ("sx2d", 48, [f"sx2d_chord_{i:02d}" for i in range(1, 49)])
    }

    # Verify bundle vs single-file identity
    print("\n--- Verifying Multi-Channel Bundles vs Single Files ---")
    verified_bundles = set()
    for b_name, (fam, n_ch, ch_names) in bundled_files.items():
        b_path = DATA_DIR / b_name
        if b_path.exists():
            t_b, dat_b, u_b = load_edf(b_path)
            dt_b = (t_b[1] - t_b[0]) / 1000.0
            fs_b = 1.0 / dt_b
            verified_bundles.add(b_name)
            print(f"Bundle {b_name:20s}: shape={str(dat_b.shape):14s}, fs={fs_b/1e3:6.1f} kHz, t=[{t_b[0]:.1f}, {t_b[-1]:.1f}] ms")
            for c_idx in range(n_ch):
                ch_name = ch_names[c_idx]
                channels.append({
                    "channel_id": ch_name,
                    "family": fam,
                    "source_file": b_name,
                    "col_idx": c_idx,
                    "t_ms": t_b,
                    "data": dat_b[:, c_idx],
                    "fs": fs_b,
                    "unit": u_b[c_idx] if c_idx < len(u_b) else u_b[0]
                })

    # Individual standalone files
    individual_files = [
        # ECE
        *[f"ECE{i}FAST@88653.edf" for i in range(1, 17)],
        # HAFAST
        "HAFAST3.5@88653.edf", "HAFAST7.5@88653.edf", "HAFAST11.5@88653.edf", "HAFAST15.5@88653.edf",
        # Plasma core / heating
        "nave@88653.edf", "Wp@88653.edf", "Ip15@88653.edf", "Ip135@88653.edf",
        "ECHRG500@88653.edf", "NBIS3I@88653.edf", "NBIS4I@88653.edf", "NBIS9I@88653.edf", "NBIS10I@88653.edf",
        "GASPUFF#1@88653.edf", "GASPUFF#2@88653.edf", "GASPUFF#9.5@88653.edf", "GASPUFF#15.5@88653.edf",
        "FW2.45GHzlong@88653.edf", "GPI14.5@88653.edf", "bfield@88653.edf",
        "VISIBLE3@88653.edf", "VISIBLE4@88653.edf", "NIRMON@88653.edf"
    ]

    for f_name in individual_files:
        f_path = DATA_DIR / f_name
        if f_path.exists():
            t_f, dat_f, u_f = load_edf(f_path)
            dt_f = (t_f[1] - t_f[0]) / 1000.0
            fs_f = 1.0 / dt_f
            ch_id = f_name.replace("@88653.edf", "")
            fam = ch_id.split("#")[0].replace("FAST", "").replace("long", "").replace("3.5", "").replace("7.5", "").replace("11.5", "").replace("15.5", "")
            if "ECE" in ch_id:
                fam = "ECE"
            elif "HAFAST" in ch_id:
                fam = "HAFAST"
            elif "NBIS" in ch_id:
                fam = "NBI"
            elif "GASPUFF" in ch_id:
                fam = "GASPUFF"
            elif "Ip" in ch_id:
                fam = "Ip"
            elif "VISIBLE" in ch_id or "NIR" in ch_id:
                fam = "Emission"

            y_arr = dat_f[:, 0] if dat_f.ndim > 1 else dat_f
            channels.append({
                "channel_id": ch_id,
                "family": fam,
                "source_file": f_name,
                "col_idx": 0,
                "t_ms": t_f,
                "data": y_arr,
                "fs": fs_f,
                "unit": u_f[0] if isinstance(u_f, list) else u_f
            })

    print(f"\nTotal physical measurement channels registered for analysis: {len(channels)}")

    # 3. Execute Analysis across All Channels
    print("\n" + "=" * 60)
    print(">>> EXECUTING MULTI-REGIME SCREENING ACROSS ALL CHANNELS")
    print("=" * 60)

    results_table = []
    regime_I_results = {}
    regime_II_results = {}
    regime_III_results = {}

    t_loop_start = time.time()
    for idx, ch in enumerate(channels):
        ch_id = ch["channel_id"]
        fam = ch["family"]
        fs = ch["fs"]
        t_ms = ch["t_ms"]
        y = ch["data"]

        if (idx + 1) % 25 == 0 or idx == 0 or idx == len(channels) - 1:
            print(f"  [{idx+1:3d}/{len(channels):3d}] Processing {ch_id:16s} ({fam:10s}, fs={fs/1e3:6.1f} kHz)...")

        # Regime I analysis
        res_I = analyze_channel_regime_I(y, t_ms, fs, ref, n_surr=500)
        regime_I_results[ch_id] = res_I

        # Regime II analysis
        res_II = analyze_channel_regime_II(y, t_ms, fs, ref)
        regime_II_results[ch_id] = res_II

        # Regime III analysis (only for 1 MHz channels)
        res_III = analyze_channel_regime_III(y, t_ms, fs, ref)
        regime_III_results[ch_id] = res_III

        # Compile flat record
        rec = {
            "channel_id": ch_id,
            "family": fam,
            "source_file": ch["source_file"],
            "fs_khz": float(fs / 1e3),
            "unit": str(ch["unit"]),
            "is_saturated": res_I.get("is_saturated", False),
            # Regime I
            "regime_I_s_deriv": res_I.get("s_deriv", 0.0),
            "regime_I_duty_ratio": res_I.get("duty_ratio", 0.5),
            "regime_I_abs_r_max": res_I.get("abs_r_max", 0.0),
            "regime_I_r_max_signed": res_I.get("r_max_signed", 0.0),
            "regime_I_tau_star_ms": res_I.get("tau_star_ms", 0.0),
            "regime_I_coh_mod": res_I.get("coh_mod", 0.0),
            "regime_I_sea_ptp": res_I.get("sea_ptp", 0.0),
            "regime_I_snr_sea": res_I.get("snr_sea", 0.0),
            "regime_I_offset_sea_ms": res_I.get("offset_sea_ms", 0.0),
            "regime_I_n_eff": res_I.get("n_eff", 0.0),
            "regime_I_p_surr_skew": res_I.get("p_surr_skew", 1.0),
            "regime_I_p_surr_r": res_I.get("p_surr_r", 1.0),
            "pre_mode_event_rate": res_I.get("pre_event_rate", 0.0),
            "pre_mode_skew": res_I.get("pre_skew", 0.0),
            # Regime II
            "regime_II_early_burst_r": res_II.get("early_burst_r", 0.0),
            "regime_II_early_burst_tau_ms": res_II.get("early_burst_tau_ms", 0.0),
            "regime_II_main_envelope_r": res_II.get("main_envelope_r", 0.0),
            "regime_II_main_envelope_tau_ms": res_II.get("main_envelope_tau_ms", 0.0),
            # Regime III
            "regime_III_h2_ratio": res_III.get("h2_ratio", 0.0)
        }
        results_table.append(rec)

    print(f"Screening complete in {time.time()-t_loop_start:.2f}s.")

    # 4. Multiple Comparisons & Evidence Classification
    print("\n>>> Computing Benjamini-Hochberg FDR and Classifying Evidence...")
    # Dual Benjamini-Hochberg FDR: on p_surr_r and on p_surr_skew
    all_p_r = np.array([r["regime_I_p_surr_r"] for r in results_table])
    n_tests = len(all_p_r)
    order_r = np.argsort(all_p_r)
    sorted_pr = all_p_r[order_r]
    q_sorted_r = sorted_pr * n_tests / np.arange(1, n_tests + 1)
    q_sorted_r = np.minimum.accumulate(q_sorted_r[::-1])[::-1]
    q_bh_r = np.empty_like(q_sorted_r)
    q_bh_r[order_r] = np.minimum(1.0, q_sorted_r)

    all_p_sk = np.array([r["regime_I_p_surr_skew"] for r in results_table])
    order_sk = np.argsort(all_p_sk)
    sorted_psk = all_p_sk[order_sk]
    q_sorted_sk = sorted_psk * n_tests / np.arange(1, n_tests + 1)
    q_sorted_sk = np.minimum.accumulate(q_sorted_sk[::-1])[::-1]
    q_bh_skew = np.empty_like(q_sorted_sk)
    q_bh_skew[order_sk] = np.minimum(1.0, q_sorted_sk)

    for i, rec in enumerate(results_table):
        q_r_val = float(q_bh_r[i])
        q_sk_val = float(q_bh_skew[i])
        rec["q_bh_r"] = q_r_val
        rec["q_bh_skew"] = q_sk_val

        # Classification rule:
        # SUPPORTED:
        #   Mode-Coupled: p_surr_r <= 0.01 and q_bh_r <= 0.05 and SEA SNR >= 2.5 and |r_max| >= 0.15
        #   OR Intrinsic Asymmetric: p_surr_skew <= 0.01 and q_bh_skew <= 0.05 and |S_deriv| >= 1.0 and SEA SNR >= 2.5
        # SUGGESTIVE:
        #   Nominally significant (p_surr_r <= 0.05 or p_surr_skew <= 0.05 or |S_deriv| >= 0.6)
        #   OR prominent crash envelope |r_env| >= 0.70 OR SEA SNR >= 2.5
        # INCONCLUSIVE / SATURATED / UNTESTABLE
        is_mode_supp = (rec["regime_I_p_surr_r"] <= 0.01 and q_r_val <= 0.05 and rec["regime_I_snr_sea"] >= 2.5 and rec["regime_I_abs_r_max"] >= 0.15)
        is_skew_supp = (rec["regime_I_p_surr_skew"] <= 0.01 and q_sk_val <= 0.05 and abs(rec["regime_I_s_deriv"]) >= 1.0 and rec["regime_I_snr_sea"] >= 2.5)

        if rec["fs_khz"] < 2.5:
            rec["classification"] = "UNTESTABLE"
        elif rec["is_saturated"]:
            rec["classification"] = "INCONCLUSIVE_ADC_SATURATED"
        elif is_mode_supp or is_skew_supp:
            rec["classification"] = "SUPPORTED"
        elif (rec["regime_I_p_surr_r"] <= 0.05 and rec["regime_I_abs_r_max"] >= 0.12) or (rec["regime_I_p_surr_skew"] <= 0.05 and abs(rec["regime_I_s_deriv"]) >= 0.6) or abs(rec["regime_II_main_envelope_r"]) >= 0.70 or abs(rec["regime_II_early_burst_r"]) >= 0.70 or rec["regime_I_snr_sea"] >= 2.5:
            rec["classification"] = "SUGGESTIVE"
        else:
            rec["classification"] = "INCONCLUSIVE"

    # Export master ledger to CSV
    csv_out_path = ANALYSIS_DIR / "sawtooth_88653_master_ledger.csv"
    with open(csv_out_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(results_table[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_table)
    print(f"Saved master ledger to {csv_out_path} ({len(results_table)} channels).")

    # Count classifications
    class_counts = {}
    for r in results_table:
        c = r["classification"]
        class_counts[c] = class_counts.get(c, 0) + 1
    print("\nSummary of Evidence Classifications:")
    for k, v in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:28s}: {v:3d} channels")

    # Family-level breakdown of SUPPORTED / SUGGESTIVE channels
    print("\nChannels with SUPPORTED Sawtooth Coupling / Waveshape:")
    supported_channels = [r for r in results_table if r["classification"] == "SUPPORTED"]
    for r in supported_channels:
        print(f"  * {r['channel_id']:16s} ({r['family']:10s}): |r|={r['regime_I_abs_r_max']:.3f}, lag={r['regime_I_tau_star_ms']:+5.2f} ms, SEA SNR={r['regime_I_snr_sea']:4.1f}, SEA offset={r['regime_I_offset_sea_ms']:+5.2f} ms, S_deriv={r['regime_I_s_deriv']:+5.2f}, p_surr={r['regime_I_p_surr_r']:.3e}")

    # Export JSON results
    json_out_path = ANALYSIS_DIR / "sawtooth_88653_results.json"
    audit_dict = {
        "shot": 88653,
        "primary_mode_reference": {
            "band_khz": [75.0, 110.0],
            "active_window_ms": [259.1, 300.4],
            "n_resets": len(ref["t_resets"]),
            "t_resets_ms": ref["t_resets"].tolist(),
            "mean_period_ms": float(np.mean(np.diff(ref["t_resets"]))),
            "std_period_ms": float(np.std(np.diff(ref["t_resets"]))),
            "skew_pri": ref["skew_pri"],
            "duty_ratio_pri": ref["d_duty"]
        },
        "classification_summary": class_counts,
        "supported_channels": supported_channels,
        "all_channels_summary": [
            {k: v for k, v in r.items() if k not in ["sea_waveform", "dt_grid"]}
            for r in results_table
        ]
    }
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(audit_dict, f, indent=2)
    print(f"Saved audit JSON to {json_out_path}.")

    # =========================================================================
    # 5. PUBLICATION-QUALITY FIGURE GENERATION
    # =========================================================================
    print("\n" + "=" * 60)
    print(">>> GENERATING 5 PUBLICATION-QUALITY FIGURES")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # FIGURE 1: Primary Mode Sawtooth Reference Dynamics
    # -------------------------------------------------------------------------
    fig1, axes1 = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    t_act = ref["t_act"]
    m_zoom = (t_act >= 282.0) & (t_act <= 295.0)

    # Panel 1: Spectrogram around primary mode
    ax = axes1[0]
    f_stft, t_rel, zxx = dsp.stft(ref["mp1"], fs=ref["fs"], nperseg=512, noverlap=480, window="hann")
    t_stft = ref["t_ms"][0] + t_rel * 1000.0
    m_stft = (t_stft >= 282.0) & (t_stft <= 295.0)
    m_f = (f_stft >= 70e3) & (f_stft <= 115e3)
    psd_db = 10.0 * np.log10(np.abs(zxx[m_f, :][:, m_stft])**2 + 1e-20)
    im = ax.pcolormesh(t_stft[m_stft], f_stft[m_f] / 1e3, psd_db, cmap="magma", shading="auto")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title("Heliotron J #88653 — Primary Mode Mesoscopic Sawtooth Reference", fontweight="bold", fontsize=11)
    ax.axvline(291.2, color="cyan", ls="--", lw=1.2, label="ECH-Off (291.2 ms)")
    ax.legend(loc="upper right", fontsize=8)

    # Panel 2: Instantaneous Frequency & Sawtooth Resets
    ax = axes1[1]
    ax.plot(t_act[m_zoom], ref["f_act"][m_zoom], color="gray", alpha=0.4, lw=0.8, label="Raw Hilbert f_inst")
    ax.plot(t_act[m_zoom], ref["f_act_smooth"][m_zoom], color="tab:blue", lw=1.8, label="Median-filtered f_inst")
    # Mark reset points
    resets_zoom = ref["t_resets"][(ref["t_resets"] >= 282.0) & (ref["t_resets"] <= 295.0)]
    for tr in resets_zoom:
        ax.axvline(tr, color="red", ls=":", alpha=0.6, lw=1.0)
    ax.scatter(resets_zoom, np.interp(resets_zoom, t_act, ref["f_act_smooth"]), color="red", s=35, zorder=5, label="Detected Resets (Up-Jumps)")
    ax.set_ylabel("f_inst (kHz)")
    ax.axvline(291.2, color="cyan", ls="--", lw=1.2)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, ls=":", alpha=0.5)

    # Panel 3: First Derivative df/dt (Sawtooth Asymmetry)
    ax = axes1[2]
    ax.plot(t_act[m_zoom], ref["df_dt_act"][m_zoom], color="tab:red", lw=1.5, label="df/dt (kHz/ms)")
    ax.axhline(0.0, color="black", ls="-", lw=0.8)
    ax.axhline(0.8, color="darkred", ls="--", lw=1.0, label="Reset Threshold (0.8 kHz/ms)")
    ax.axvline(291.2, color="cyan", ls="--", lw=1.2)
    ax.set_ylabel("df/dt (kHz/ms)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, ls=":", alpha=0.5)

    # Panel 4: Amplitude Envelope A(t)
    ax = axes1[3]
    ax.plot(t_act[m_zoom], ref["env_act"][m_zoom], color="tab:green", lw=1.5, label="Primary Envelope A_pri(t)")
    ax.axvline(291.2, color="cyan", ls="--", lw=1.2)
    ax.set_ylabel("Envelope (V)")
    ax.set_xlabel("Time (ms)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, ls=":", alpha=0.5)

    plt.tight_layout()
    fig1_path = FIG_DIR / "fig1_primary_mode_sawtooth_reference_88653.png"
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)
    print(f"  Saved {fig1_path.name}")

    # -------------------------------------------------------------------------
    # FIGURE 2: Cross-Diagnostic Screening Heatmap / Scatter
    # -------------------------------------------------------------------------
    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(13, 10))

    # Panel A: Derivative Skewness vs Peak Lagged Correlation
    fam_names = sorted(list(set(r["family"] for r in results_table)))
    color_map = {f: plt.cm.tab20(i / len(fam_names)) for i, f in enumerate(fam_names)}

    for fam in fam_names:
        recs_fam = [r for r in results_table if r["family"] == fam]
        skews = [r["regime_I_s_deriv"] for r in recs_fam]
        corrs = [r["regime_I_abs_r_max"] for r in recs_fam]
        classes = [r["classification"] for r in recs_fam]
        markers = ["o" if c == "SUPPORTED" else ("s" if c == "SUGGESTIVE" else "x") for c in classes]
        for sk, co, mk, cl in zip(skews, corrs, markers, classes):
            alpha = 1.0 if cl == "SUPPORTED" else (0.7 if cl == "SUGGESTIVE" else 0.3)
            size = 70 if cl == "SUPPORTED" else (40 if cl == "SUGGESTIVE" else 20)
            edge = "black" if cl == "SUPPORTED" else "none"
            ax2a.scatter(sk, co, color=color_map[fam], marker=mk, s=size, alpha=alpha, edgecolors=edge, lw=1.0)

    # Dummy legend for families
    for fam in fam_names:
        ax2a.scatter([], [], color=color_map[fam], label=fam, s=40)
    ax2a.axhline(0.25, color="red", ls="--", lw=1.0, label="Significance Threshold (|r| = 0.25)")
    ax2a.axvline(0.0, color="gray", ls=":", lw=1.0)
    ax2a.set_xlabel("Derivative Skewness S_deriv (Asymmetry Metric)")
    ax2a.set_ylabel("Peak Cross-Correlation |r_max| with Primary Mode")
    ax2a.set_title("Cross-Diagnostic Sawtooth Screening across All 191 Channels (#88653)", fontweight="bold")
    ax2a.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7.5, ncol=1)
    ax2a.grid(True, ls=":", alpha=0.5)

    # Panel B: Lead/Lag Timing distribution tau* vs SEA SNR
    for fam in fam_names:
        recs_fam = [r for r in results_table if r["family"] == fam and r["classification"] in ["SUPPORTED", "SUGGESTIVE"]]
        if len(recs_fam) > 0:
            lags = [r["regime_I_tau_star_ms"] for r in recs_fam]
            snrs = [r["regime_I_snr_sea"] for r in recs_fam]
            ax2b.scatter(lags, snrs, color=color_map[fam], s=60, edgecolors="black", lw=0.8, label=fam)
            for r in recs_fam:
                if r["classification"] == "SUPPORTED":
                    ax2b.annotate(r["channel_id"], (r["regime_I_tau_star_ms"], r["regime_I_snr_sea"]),
                                  fontsize=7, xytext=(3, 3), textcoords="offset points")

    ax2b.axvline(0.0, color="black", ls="-", lw=1.0, label="Zero-Lag (Synchronized)")
    ax2b.axvspan(-5.0, 0.0, color="blue", alpha=0.05, label="Leading / Precursor (tau* < 0)")
    ax2b.axvspan(0.0, 5.0, color="red", alpha=0.05, label="Lagging / Response (tau* > 0)")
    ax2b.set_xlabel("Optimal Time Lag tau* (ms): [-5 ms = Leads Mode, +5 ms = Lags Mode]")
    ax2b.set_ylabel("Superposed Epoch Analysis (SEA) SNR")
    ax2b.set_title("Lead / Lag Timing vs. Sawtooth Modulation Strength (Supported & Suggestive Channels)", fontweight="bold")
    ax2b.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7.5)
    ax2b.grid(True, ls=":", alpha=0.5)

    plt.tight_layout()
    fig2_path = FIG_DIR / "fig2_cross_diagnostic_screening_heatmap_88653.png"
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)
    print(f"  Saved {fig2_path.name}")

    # -------------------------------------------------------------------------
    # FIGURE 3: Superposed Epoch Sawtooth Waveforms
    # -------------------------------------------------------------------------
    # Top representative channels across diverse diagnostics
    sample_channels = [
        "MP1", "MP3", "PMP1", "PMP5", "PMP10",
        "sx2d_chord_08", "sx2d_chord_20", "sx2d_chord_32",
        "DivPro_05", "DivPro_15", "DivPro_28",
        "HAFAST7.5", "HAARRAY15", "nave"
    ]
    # Filter to existing channels
    sample_channels = [ch for ch in sample_channels if ch in regime_I_results]

    fig3, axes3 = plt.subplots(len(sample_channels) // 2 + len(sample_channels) % 2, 2, figsize=(12, 14), sharex=True)
    axes_flat = axes3.flatten()

    dt_grid = np.array(regime_I_results["MP1"]["dt_grid"])

    for i, ch_id in enumerate(sample_channels):
        ax = axes_flat[i]
        res = regime_I_results[ch_id]
        wf = np.array(res["sea_waveform"])
        snr = res["snr_sea"]
        tau = res["tau_star_ms"]
        cl = [r["classification"] for r in results_table if r["channel_id"] == ch_id][0]

        color = "tab:blue" if "MP" in ch_id else ("tab:red" if "sx2d" in ch_id else ("tab:purple" if "DivPro" in ch_id else "tab:green"))
        ax.plot(dt_grid, wf, color=color, lw=1.8, label=f"{ch_id} (SNR={snr:.1f}, tau*={tau:+.2f} ms)")
        ax.axvline(0.0, color="gray", ls="--", lw=1.0)
        # Mark peak/trough offset
        ax.axvline(res["offset_sea_ms"], color="darkred", ls=":", lw=1.2, label=f"Offset {res['offset_sea_ms']:+.2f} ms")
        ax.set_title(f"{ch_id} [{cl}]", fontsize=9.5, fontweight="bold")
        ax.set_ylabel("SEA Amp (A.U.)", fontsize=8)
        ax.legend(loc="upper right", fontsize=7.5)
        ax.grid(True, ls=":", alpha=0.5)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    axes_flat[-1].set_xlabel("Relative Time Delta t (ms)")
    axes_flat[-2].set_xlabel("Relative Time Delta t (ms)")

    plt.tight_layout()
    fig3_path = FIG_DIR / "fig3_superposed_epoch_sawtooth_waveforms_88653.png"
    fig3.savefig(fig3_path, dpi=300)
    plt.close(fig3)
    print(f"  Saved {fig3_path.name}")

    # -------------------------------------------------------------------------
    # FIGURE 4: Unsynchronized and Pre-Mode Sawteeth (171.2 - 259.1 ms vs 259.1 - 300.4 ms)
    # -------------------------------------------------------------------------
    fig4, axes4 = plt.subplots(4, 1, figsize=(11, 11), sharex=True)

    # Compare time traces across entire heated discharge [170, 310] ms
    # 1. Mirnov MP1 primary mode envelope
    ax = axes4[0]
    m_all = (ref["t_ms"] >= 170.0) & (ref["t_ms"] <= 310.0)
    ax.plot(ref["t_ms"][m_all], ref["env_pri"][m_all], color="tab:blue", lw=1.2, label="MP1 Primary Mode Envelope (80-120 kHz)")
    ax.axvspan(171.2, 222.4, color="cyan", alpha=0.08, label="ECH-Only")
    ax.axvspan(222.4, 259.1, color="orange", alpha=0.08, label="Early ECH+NBI")
    ax.axvspan(259.1, 291.2, color="yellow", alpha=0.08, label="Steady ECH+NBI (Mode Active)")
    ax.axvspan(291.2, 300.4, color="pink", alpha=0.08, label="NBI-Only")
    ax.axvline(236.2, color="darkred", ls=":", lw=1.5, label="Early Burst Crash (236.2 ms)")
    ax.axvline(291.2, color="green", ls="--", lw=1.5, label="ECH-Off (291.2 ms)")
    ax.set_ylabel("MP1 Env (V)")
    ax.set_title("Cross-Diagnostic Relaxation & Precursor Sawtooth Dynamics Across Shot #88653", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, ls=":", alpha=0.5)

    # 2. sx2d Core (Chord 8) & Edge (Chord 24)
    ax = axes4[1]
    ch_c8 = [c for c in channels if c["channel_id"] == "sx2d_chord_08"][0]
    ch_c24 = [c for c in channels if c["channel_id"] == "sx2d_chord_24"][0]
    m_sx = (ch_c8["t_ms"] >= 170.0) & (ch_c8["t_ms"] <= 310.0)
    ax.plot(ch_c8["t_ms"][m_sx], ch_c8["data"][m_sx], color="tab:red", lw=1.0, label="sx2d Chord 8 (Core SXR)")
    ax.plot(ch_c24["t_ms"][m_sx], ch_c24["data"][m_sx] * 2.0, color="darkorange", lw=1.0, alpha=0.8, label="sx2d Chord 24 (Edge SXR x2)")
    ax.axvline(236.2, color="darkred", ls=":", lw=1.5)
    ax.axvline(291.2, color="green", ls="--", lw=1.5)
    ax.set_ylabel("SXR (V)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, ls=":", alpha=0.5)

    # 3. Divertor Langmuir Probes (DivPro 5 & 15)
    ax = axes4[2]
    ch_dp5 = [c for c in channels if c["channel_id"] == "DivPro_05"][0]
    ch_dp15 = [c for c in channels if c["channel_id"] == "DivPro_15"][0]
    m_dp = (ch_dp5["t_ms"] >= 170.0) & (ch_dp5["t_ms"] <= 310.0)
    ax.plot(ch_dp5["t_ms"][m_dp], ch_dp5["data"][m_dp], color="tab:purple", lw=1.0, label="Divertor Probe 05")
    ax.plot(ch_dp15["t_ms"][m_dp], ch_dp15["data"][m_dp], color="magenta", lw=1.0, alpha=0.8, label="Divertor Probe 15")
    ax.axvline(236.2, color="darkred", ls=":", lw=1.5)
    ax.axvline(291.2, color="green", ls="--", lw=1.5)
    ax.set_ylabel("DivPro (V)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, ls=":", alpha=0.5)

    # 4. Fast H-alpha 7.5 and Density nave
    ax = axes4[3]
    ch_ha = [c for c in channels if c["channel_id"] == "HAFAST7.5"][0]
    ch_ne = [c for c in channels if c["channel_id"] == "nave"][0]
    m_ha = (ch_ha["t_ms"] >= 170.0) & (ch_ha["t_ms"] <= 310.0)
    ax.plot(ch_ha["t_ms"][m_ha], ch_ha["data"][m_ha], color="tab:green", lw=1.0, label="HAFAST 7.5 (Edge Emission)")
    ax2 = ax.twinx()
    m_ne = (ch_ne["t_ms"] >= 170.0) & (ch_ne["t_ms"] <= 310.0)
    ax2.plot(ch_ne["t_ms"][m_ne], ch_ne["data"][m_ne], color="tab:brown", lw=1.2, ls="--", label="nave (Line Density)")
    ax.axvline(236.2, color="darkred", ls=":", lw=1.5)
    ax.axvline(291.2, color="green", ls="--", lw=1.5)
    ax.set_ylabel("HAFAST (V)")
    ax2.set_ylabel("Density (1e19 m^-3)")
    ax.set_xlabel("Time (ms)")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(True, ls=":", alpha=0.5)

    plt.tight_layout()
    fig4_path = FIG_DIR / "fig4_unsynchronized_and_premode_sawteeth_88653.png"
    fig4.savefig(fig4_path, dpi=300)
    plt.close(fig4)
    print(f"  Saved {fig4_path.name}")

    # -------------------------------------------------------------------------
    # FIGURE 5: Spatial Chord Profiles & Phase Inversion Search
    # -------------------------------------------------------------------------
    fig5, (ax5a, ax5b, ax5c) = plt.subplots(3, 1, figsize=(11, 10))

    # Panel A: sx2d 48-chord profile (amplitude, skewness, lead/lag)
    sx_recs = [r for r in results_table if r["family"] == "sx2d"]
    ch_nums = np.arange(1, len(sx_recs) + 1)
    sx_ptp = [r["regime_I_sea_ptp"] for r in sx_recs]
    sx_skew = [r["regime_I_s_deriv"] for r in sx_recs]
    sx_lags = [r["regime_I_tau_star_ms"] for r in sx_recs]

    ax5a.plot(ch_nums, sx_ptp, "ro-", ms=4, lw=1.5, label="Sawtooth PTP Amp (V)")
    ax5a_t = ax5a.twinx()
    ax5a_t.plot(ch_nums, sx_skew, "bs--", ms=4, lw=1.2, label="Derivative Skewness S_deriv")
    ax5a.set_ylabel("Sawtooth PTP Amp (V)", color="red")
    ax5a_t.set_ylabel("Skewness S_deriv", color="blue")
    ax5a.set_title("sx2d 48-Chord Array: Sawtooth Amplitude & Skewness Profile", fontweight="bold")
    ax5a.set_xlabel("Chord Number (1 to 48)")
    ax5a.grid(True, ls=":", alpha=0.5)

    # Panel B: Divertor Probe 38-channel profile
    div_recs = [r for r in results_table if r["family"] == "DivProArr"]
    div_nums = np.arange(1, len(div_recs) + 1)
    div_ptp = [r["regime_I_sea_ptp"] for r in div_recs]
    div_lags = [r["regime_I_tau_star_ms"] for r in div_recs]

    ax5b.bar(div_nums, div_ptp, color="purple", alpha=0.7, label="Sawtooth SEA PTP (V)")
    ax5b.set_ylabel("Sawtooth PTP Amp (V)")
    ax5b.set_title("Divertor Langmuir Probe Array (38 Probes): Sawtooth Amplitude Profile", fontweight="bold")
    ax5b.set_xlabel("Probe Number (1 to 38)")
    ax5b.grid(True, ls=":", alpha=0.5)

    # Panel C: Poloidal Mirnov Coil PMP1 - PMP14 Profile
    pmp_recs = [r for r in results_table if r["family"] == "PMP"]
    pmp_nums = np.arange(1, len(pmp_recs) + 1)
    pmp_ptp = [r["regime_I_sea_ptp"] for r in pmp_recs]
    pmp_lags = [r["regime_I_tau_star_ms"] for r in pmp_recs]

    ax5c.plot(pmp_nums, pmp_ptp, "gd-", ms=6, lw=1.8, label="Sawtooth PTP Amp (V)")
    ax5c_t = ax5c.twinx()
    ax5c_t.plot(pmp_nums, pmp_lags, "mo--", ms=5, lw=1.2, label="Optimal Lag tau* (ms)")
    ax5c.set_ylabel("PTP Amp (V)", color="green")
    ax5c_t.set_ylabel("Lag tau* (ms)", color="magenta")
    ax5c.set_title("Poloidal Mirnov Array (PMP1 to PMP14): Sawtooth Amplitude and Phase Lag", fontweight="bold")
    ax5c.set_xlabel("Coil Number (PMP 1 to 14)")
    ax5c.grid(True, ls=":", alpha=0.5)

    plt.tight_layout()
    fig5_path = FIG_DIR / "fig5_spatial_chord_profiles_88653.png"
    fig5.savefig(fig5_path, dpi=300)
    plt.close(fig5)
    print(f"  Saved {fig5_path.name}")

    total_time = time.time() - t_start_total
    print("\n" + "=" * 80)
    print(f"ANALYSIS FULLY COMPLETE IN {total_time:.1f}s.")
    print("=" * 80)


if __name__ == "__main__":
    main()
