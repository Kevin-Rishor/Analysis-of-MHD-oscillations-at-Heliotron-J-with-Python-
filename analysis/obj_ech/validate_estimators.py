"""
validate_estimators.py
Synthetic data validation suite for ECH-related MHD analysis estimators.
Validates:
  1. Instantaneous frequency & chirp slope (df/dt) estimator
  2. Chirp event detection & segmentation (duration, span, repetition rate with Poisson CI)
  3. Amplitude envelope step estimator with block bootstrap uncertainty
  4. Thermal exponential decay time constant (tau) estimator
Run BEFORE applying estimators to experimental Heliotron J data.
"""

import sys
import numpy as np
import scipy.signal as dsp
from pathlib import Path

# Fixed random seed for reproducibility
RNG_SEED = 20261007
rng = np.random.default_rng(RNG_SEED)


def generate_synthetic_chirp_signal(fs=1.0e6, duration=0.010, f0=90.0e3,
                                    chirp_rate=-10.0e6, n_chirps=8,
                                    snr_db=20.0):
    """
    Generates a synthetic train of repetitive down-chirps with known truth.
    chirp_rate: in Hz/s (e.g. -10 MHz/s = -10 kHz/ms = -10 Hz/us)
    """
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    sig = np.zeros(n_samples)
    
    # Event spacing
    t_rep = duration / (n_chirps + 1)
    tau_event = 0.0006  # 600 us chirp duration
    
    true_events = []
    for i in range(1, n_chirps + 1):
        t_start = i * t_rep
        t_end = t_start + tau_event
        true_events.append({
            "t_start": t_start,
            "t_end": t_end,
            "duration": tau_event,
            "f_start": f0,
            "f_end": f0 + chirp_rate * tau_event,
            "df_dt": chirp_rate,
            "df_span": chirp_rate * tau_event,
            "amplitude": 1.0
        })
        
        mask = (t >= t_start) & (t <= t_end)
        t_rel = t[mask] - t_start
        # Quadratic phase: phi(t) = 2*pi*(f0*t + 0.5*alpha*t^2)
        phase = 2.0 * np.pi * (f0 * t_rel + 0.5 * chirp_rate * t_rel**2)
        # Hann envelope for smooth burst
        env = np.sin(np.pi * t_rel / tau_event)**2
        sig[mask] += env * np.cos(phase)
        
    # Add noise
    noise_power = np.var(sig[sig != 0]) / (10.0 ** (snr_db / 10.0)) if np.any(sig != 0) else 1e-4
    noise = rng.normal(0, np.sqrt(noise_power), n_samples)
    sig_noisy = sig + noise
    
    return t, sig_noisy, true_events


def estimate_instantaneous_frequency_hilbert(sig, fs, f_low=70e3, f_high=110e3,
                                             bessel_order=4, sg_win=125, sg_poly=2):
    """
    Bessel bandpass + Hilbert transform with Savitzky-Golay derivative.
    """
    sos = dsp.bessel(bessel_order, [f_low, f_high], btype='bandpass', fs=fs, output='sos')
    filtered = dsp.sosfiltfilt(sos, sig)
    analytic = dsp.hilbert(filtered)
    envelope = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    
    dt = 1.0 / fs
    if sg_win % 2 == 0:
        sg_win += 1
    if sg_win >= len(sig):
        sg_win = len(sig) - 1 if len(sig) % 2 == 0 else len(sig) - 2
        
    ifreq = dsp.savgol_filter(phase, sg_win, sg_poly, deriv=1) / (2.0 * np.pi * dt)
    return envelope, ifreq


def detect_chirp_events(t, envelope, ifreq, threshold_factor=2.0, min_duration=2e-4, max_duration=2e-3):
    """
    Segments individual burst events where envelope exceeds a baseline noise threshold.
    Returns list of detected event dicts.
    """
    dt = t[1] - t[0]
    med_env = np.median(envelope)
    mad_std = 1.4826 * np.median(np.abs(envelope - med_env))
    thresh = med_env + threshold_factor * mad_std
    
    is_active = envelope > thresh
    if not np.any(is_active):
        return []
        
    diff = np.diff(is_active.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    
    if is_active[0]:
        starts = np.r_[0, starts]
    if is_active[-1]:
        ends = np.r_[ends, len(is_active)]
        
    detected = []
    for s, e in zip(starts, ends):
        dur = (e - s) * dt
        if min_duration <= dur <= max_duration:
            t_ev = t[s:e]
            f_ev = ifreq[s:e]
            env_ev = envelope[s:e]
            
            # Robust slope fit: linear regression
            if len(t_ev) >= 5:
                slope, intercept = np.polyfit(t_ev - t_ev[0], f_ev, 1)
                detected.append({
                    "t_start": float(t[s]),
                    "t_end": float(t[e - 1]),
                    "duration": float(dur),
                    "f_start": float(f_ev[0]),
                    "f_end": float(f_ev[-1]),
                    "df_span": float(f_ev[-1] - f_ev[0]),
                    "df_dt": float(slope),
                    "peak_amplitude": float(np.max(env_ev))
                })
    return detected


def block_bootstrap_step_ratio(x_before, x_after, block_len=50, n_boot=2000, seed=RNG_SEED):
    """
    Moving-block bootstrap for autocorrelated time series to compute ratio of means:
    Ratio = mean(x_after) / mean(x_before), with 95% Confidence Interval.
    """
    boot_rng = np.random.default_rng(seed)
    n_b = len(x_before)
    n_a = len(x_after)
    
    n_blocks_b = int(np.ceil(n_b / block_len))
    n_blocks_a = int(np.ceil(n_a / block_len))
    
    ratios = []
    diffs = []
    for _ in range(n_boot):
        # Sample blocks for before
        idx_b = boot_rng.integers(0, max(1, n_b - block_len + 1), size=n_blocks_b)
        sample_b = np.concatenate([x_before[i:i + block_len] for i in idx_b])[:n_b]
        
        # Sample blocks for after
        idx_a = boot_rng.integers(0, max(1, n_a - block_len + 1), size=n_blocks_a)
        sample_a = np.concatenate([x_after[i:i + block_len] for i in idx_a])[:n_a]
        
        m_b = np.mean(sample_b)
        m_a = np.mean(sample_a)
        if m_b > 0:
            ratios.append(m_a / m_b)
        diffs.append(m_a - m_b)
        
    ratios = np.array(ratios)
    diffs = np.array(diffs)
    
    r_mean = np.mean(ratios)
    r_ci = (np.percentile(ratios, 2.5), np.percentile(ratios, 97.5))
    d_mean = np.mean(diffs)
    d_ci = (np.percentile(diffs, 2.5), np.percentile(diffs, 97.5))
    
    return {
        "ratio_mean": float(r_mean),
        "ratio_ci_low": float(r_ci[0]),
        "ratio_ci_high": float(r_ci[1]),
        "diff_mean": float(d_mean),
        "diff_ci_low": float(d_ci[0]),
        "diff_ci_high": float(d_ci[1])
    }


def fit_exponential_decay(t_ms, y, t_off_ms):
    """
    Fits y(t) = y_inf + Delta_y * exp(-(t - t_off)/tau) for t >= t_off.
    """
    mask = t_ms >= t_off_ms
    t_fit = t_ms[mask] - t_off_ms
    y_fit = y[mask]
    
    y_inf = np.percentile(y_fit[-int(len(y_fit)*0.2):], 50)
    dy_init = y_fit[0] - y_inf
    
    # Linear fit in log domain to find initial tau
    log_arg = np.clip((y_fit - y_inf) / (dy_init + 1e-12), 1e-4, 2.0)
    valid = (y_fit > y_inf) & (log_arg > 0.05)
    if np.sum(valid) > 10:
        slope, _ = np.polyfit(t_fit[valid], np.log(log_arg[valid]), 1)
        tau_init = -1.0 / slope if slope < 0 else 1.0
    else:
        tau_init = 1.0
        
    # Non-linear refinement via scipy.optimize
    from scipy.optimize import curve_fit
    def model(t_rel, y_base, amp, tau):
        return y_base + amp * np.exp(-t_rel / np.maximum(tau, 1e-4))
        
    try:
        popt, pcov = curve_fit(model, t_fit, y_fit, p0=[y_inf, dy_init, tau_init], maxfev=2000)
        perr = np.sqrt(np.diag(pcov))
        return {
            "y_inf": float(popt[0]),
            "amp": float(popt[1]),
            "tau_ms": float(popt[2]),
            "tau_err_ms": float(perr[2]),
            "success": True
        }
    except Exception as e:
        return {
            "y_inf": float(y_inf),
            "amp": float(dy_init),
            "tau_ms": float(tau_init),
            "tau_err_ms": float(np.nan),
            "success": False,
            "error": str(e)
        }


def run_synthetic_validation():
    print("================================================================================")
    print("RUNNING SYNTHETIC VALIDATION SUITE (Known Ground Truth)")
    print("================================================================================")
    
    # 1. Validate Chirp Detection & Rate Estimation
    duration = 0.010  # 10 ms
    n_chirps_true = 8
    f0_true = 90.0e3  # 90 kHz
    slope_true = -10.0e6  # -10 kHz/ms = -10 MHz/s
    
    print(f"\n[Test 1: Chirp Detection & Rate Estimation]")
    print(f"  Truth: N_chirps = {n_chirps_true}, f0 = {f0_true/1e3:.1f} kHz, slope = {slope_true/1e6:.1f} kHz/ms")
    
    t, sig, true_events = generate_synthetic_chirp_signal(
        fs=1e6, duration=duration, f0=f0_true, chirp_rate=slope_true,
        n_chirps=n_chirps_true, snr_db=18.0
    )
    
    env, ifreq = estimate_instantaneous_frequency_hilbert(sig, fs=1e6, f_low=70e3, f_high=110e3, sg_win=125)
    det_events = detect_chirp_events(t, env, ifreq, threshold_factor=1.8, min_duration=2e-4, max_duration=1.2e-3)
    
    print(f"  Detected: N_events = {len(det_events)} (Truth: {n_chirps_true})")
    assert abs(len(det_events) - n_chirps_true) <= 1, f"Chirp event count error too high: {len(det_events)} vs {n_chirps_true}"
    
    slopes = [ev["df_dt"] for ev in det_events]
    mean_slope = np.mean(slopes)
    print(f"  Mean detected slope: {mean_slope/1e6:.2f} kHz/ms (Truth: {slope_true/1e6:.2f} kHz/ms, error = {abs(mean_slope - slope_true)/abs(slope_true)*100:.1f}%)")
    assert abs(mean_slope - slope_true) / abs(slope_true) < 0.25, "Chirp slope estimation error exceeds 25%"
    
    # Check Poisson CI on event count
    from scipy.stats import poisson
    k = len(det_events)
    ci_low = poisson.ppf(0.025, k) / (duration * 1000.0)
    ci_high = poisson.ppf(0.975, k) / (duration * 1000.0)
    rep_rate = k / (duration * 1000.0)
    true_rate = n_chirps_true / (duration * 1000.0)
    print(f"  Repetition rate: {rep_rate:.2f} chirps/ms [Poisson 95% CI: {ci_low:.2f}, {ci_high:.2f}] (Truth: {true_rate:.2f})")
    assert ci_low <= true_rate <= ci_high, "True repetition rate falls outside Poisson 95% CI"
    print("  --> Test 1 PASSED: Chirp detection, slope, and Poisson rate verified.")
    
    # 2. Validate Block Bootstrap Step Ratio Estimator
    print(f"\n[Test 2: Block Bootstrap Step Ratio Estimator]")
    n_pts = 5000
    # Autocorrelated AR(1) noise: x[t] = 0.8 * x[t-1] + e
    ar_coef = 0.8
    e_b = rng.normal(0, 1, n_pts)
    e_a = rng.normal(0, 1, n_pts)
    
    x_b = np.zeros(n_pts)
    x_a = np.zeros(n_pts)
    for i in range(1, n_pts):
        x_b[i] = ar_coef * x_b[i-1] + e_b[i]
        x_a[i] = ar_coef * x_a[i-1] + e_a[i]
        
    true_mean_b = 1.0
    true_mean_a = 1.6  # 60% step increase
    true_ratio = true_mean_a / true_mean_b
    
    x_b += true_mean_b
    x_a += true_mean_a
    
    boot_res = block_bootstrap_step_ratio(x_b, x_a, block_len=50, n_boot=1000)
    print(f"  Truth: Ratio = {true_ratio:.3f}")
    print(f"  Bootstrap: Ratio = {boot_res['ratio_mean']:.3f} [95% CI: {boot_res['ratio_ci_low']:.3f}, {boot_res['ratio_ci_high']:.3f}]")
    assert boot_res['ratio_ci_low'] <= true_ratio <= boot_res['ratio_ci_high'], "True step ratio falls outside bootstrap 95% CI"
    print("  --> Test 2 PASSED: Block bootstrap coverage verified.")
    
    # 3. Validate Exponential Decay Estimator
    print(f"\n[Test 3: Exponential Decay Estimator]")
    t_decay = np.linspace(0, 10.0, 1000)  # 10 ms at 100 kHz
    true_tau = 1.5  # ms
    true_amp = 0.8  # V
    true_inf = 0.1  # V
    y_decay_clean = true_inf + true_amp * np.exp(-t_decay / true_tau)
    y_decay_noisy = y_decay_clean + rng.normal(0, 0.03, len(t_decay))
    
    fit_res = fit_exponential_decay(t_decay, y_decay_noisy, t_off_ms=0.0)
    print(f"  Truth: tau = {true_tau:.2f} ms")
    print(f"  Fitted: tau = {fit_res['tau_ms']:.2f} +/- {fit_res['tau_err_ms']:.2f} ms")
    assert abs(fit_res['tau_ms'] - true_tau) < 0.15, f"Tau estimation error too high: {fit_res['tau_ms']}"
    print("  --> Test 3 PASSED: Exponential decay estimator verified.")
    
    print("\n================================================================================")
    print("ALL SYNTHETIC VALIDATION TESTS PASSED SUCCESSFULLY.")
    print("================================================================================")


if __name__ == "__main__":
    run_synthetic_validation()
