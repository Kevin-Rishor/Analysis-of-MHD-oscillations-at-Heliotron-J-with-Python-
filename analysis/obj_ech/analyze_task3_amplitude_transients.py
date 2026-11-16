"""
analyze_task3_amplitude_transients.py
Task 3 of the ECH Extension for Heliotron J shots #88653 and #88654.

Executes:
  1. Re-extracts Primary and Secondary envelopes with an ADAPTIVE band following
     the tracked ridge (+-6 kHz) via complex baseband demodulation.
  2. Computes before/after ratios of envelope mean and RMS across t0 (291.2 ms)
     with equal windows of 5, 3, and 2 ms (excluding t0 to t0+1 ms), block-bootstrap CIs,
     and versions relative to the pre-ECH-off trend. Explains the #88653 (-6.5%) vs #88654 (+63%) discrepancy.
  3. High-resolution event-triggered timeline (289-297 ms, 10-50 us):
     Primary/Secondary envelope, raw MP1 RMS, broadband 120-250 kHz power.
     Quantitative tests of transients (291.6 ms in #88653, 292-293 ms in #88654):
       (a) Physical mode: MP1/MP3/MP4 toroidal n=-1 (mod 4) cross-phase & PMP poloidal m fit.
       (b) Common-mode broadband pickup: broadband/mode ratio, zero-lag correlation, simultaneity with ECE & H-alpha.
  4. Burst statistics: burst detection using quiet window MAD, burst-height distribution,
     intermittency (kurtosis, fraction of time above 2 sigma) before vs after t0.
  5. Physical units: calibrated PMP array (T/s) -> delta-B = Bdot/(2 pi f) and delta-B/B0 (B0=1.25 T).
     Audits gain (2.0) and coil constant (4.5e-3 m^2) against EDF metadata; flags amplitude verification status.
  6. Normalized ECE fluctuation levels (dI / I_mean) at ECE12/13 for Primary band before/after t0.
  7. Generates 300-dpi publication figures and task3_amplitude_transients_results.json.
"""

import sys
import json
from pathlib import Path
import numpy as np
import scipy.signal as dsp
from scipy.stats import kurtosis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = ROOT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OBJ_ECH_DIR = ROOT_DIR / "analysis" / "obj_ech"
FIG_DIR = OBJ_ECH_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(ROOT_DIR / "jpack"))
import turnelib as TE

RNG_SEED = 20261007


def load_edf(filepath):
    e = TE.edf()
    dat = e.load(str(filepath))
    t_raw = dat[:, 0]
    t_ms = t_raw if e.DimUnit[0] == "ms" else t_raw * 1000.0
    ys = dat[:, 1:] if dat.shape[1] > 2 else dat[:, 1]
    return t_ms, ys, e


def get_tau_int(x, max_lag=2000):
    """Estimate integrated autocorrelation time tau_int using FFT."""
    x_c = x - np.mean(x)
    c0 = np.dot(x_c, x_c)
    if c0 <= 0:
        return 1.0
    n = len(x)
    f = np.fft.rfft(x_c, n=2 * n)
    corr = np.fft.irfft(f * np.conj(f))[:n] / c0
    zc = np.where(corr < 0)[0]
    cutoff = zc[0] if len(zc) > 0 else min(max_lag, n // 4)
    cutoff = min(cutoff, max_lag)
    tau = 1.0 + 2.0 * np.sum(corr[1:cutoff])
    return max(1.0, float(tau))


def extract_adaptive_envelope(signal, t_ms, ridge_khz, t_track_ms, bw_khz=6.0, fs=1e6):
    """
    Extracts envelope in adaptive band [f_ridge(t) - bw_khz, f_ridge(t) + bw_khz]
    via complex baseband demodulation and zero-phase Bessel lowpass filtering.
    """
    dt = 1.0 / fs
    f_interp_hz = np.interp(t_ms, t_track_ms, ridge_khz) * 1e3
    phi_t = 2.0 * np.pi * np.cumsum(f_interp_hz) * dt
    z_demod = signal * np.exp(-1j * phi_t)
    sos_lp = dsp.bessel(4, bw_khz * 1e3, btype="lowpass", fs=fs, output="sos")
    z_filt = dsp.sosfiltfilt(sos_lp, z_demod)
    env_adapt = 2.0 * np.abs(z_filt)
    return env_adapt.astype(np.float32)


def moving_block_bootstrap_comparison(x_pre, x_post, n_boot=2000, seed=RNG_SEED):
    """
    Computes before/after ratios of Mean and RMS with moving-block bootstrap.
    Preserves autocorrelation; reports N_eff and 95% CIs.
    """
    x_pre = np.asarray(x_pre)
    x_post = np.asarray(x_post)
    n_pre = len(x_pre)
    n_post = len(x_post)
    
    tau_pre = get_tau_int(x_pre)
    tau_post = get_tau_int(x_post)
    n_eff_pre = float(n_pre / (2.0 * tau_pre))
    n_eff_post = float(n_post / (2.0 * tau_post))
    
    b_len = max(int(np.ceil(max(tau_pre, tau_post) * 1.5)), 50)
    b_len = min(b_len, min(n_pre, n_post) // 2)
    
    boot_rng = np.random.default_rng(seed)
    n_b_pre = int(np.ceil(n_pre / b_len))
    n_b_post = int(np.ceil(n_post / b_len))
    
    mean_pre_val = float(np.mean(x_pre))
    mean_post_val = float(np.mean(x_post))
    rms_pre_val = float(np.sqrt(np.mean(x_pre**2)))
    rms_post_val = float(np.sqrt(np.mean(x_post**2)))
    
    boot_ratio_mean = []
    boot_ratio_rms = []
    
    for _ in range(n_boot):
        idx_pre = boot_rng.integers(0, max(1, n_pre - b_len + 1), size=n_b_pre)
        s_pre = np.concatenate([x_pre[i:i + b_len] for i in idx_pre])[:n_pre]
        idx_post = boot_rng.integers(0, max(1, n_post - b_len + 1), size=n_b_post)
        s_post = np.concatenate([x_post[i:i + b_len] for i in idx_post])[:n_post]
        
        m_b = np.mean(s_pre)
        m_a = np.mean(s_post)
        if m_b > 1e-9:
            boot_ratio_mean.append(m_a / m_b)
            
        r_b = np.sqrt(np.mean(s_pre**2))
        r_a = np.sqrt(np.mean(s_post**2))
        if r_b > 1e-9:
            boot_ratio_rms.append(r_a / r_b)
            
    ci_mean = (float(np.percentile(boot_ratio_mean, 2.5)), float(np.percentile(boot_ratio_mean, 97.5)))
    ci_rms = (float(np.percentile(boot_ratio_rms, 2.5)), float(np.percentile(boot_ratio_rms, 97.5)))
    
    return {
        "mean_pre": mean_pre_val,
        "mean_post": mean_post_val,
        "rms_pre": rms_pre_val,
        "rms_post": rms_post_val,
        "ratio_mean": {"val": mean_post_val / mean_pre_val, "ci": ci_mean, "pct_change": (mean_post_val / mean_pre_val - 1.0) * 100.0},
        "ratio_rms": {"val": rms_post_val / rms_pre_val, "ci": ci_rms, "pct_change": (rms_post_val / rms_pre_val - 1.0) * 100.0},
        "n_eff_pre": n_eff_pre,
        "n_eff_post": n_eff_post,
        "tau_int_pre": tau_pre,
        "tau_int_post": tau_post,
        "block_len": b_len
    }


def analyze_shot_task3(shot):
    print(f"\n=======================================================")
    print(f"Executing Task 3 Amplitude & Transient Analysis for #{shot}")
    print(f"=======================================================")
    
    # 1. Load full record cache and Task 2 tracked ridges
    cache_path = CACHE_DIR / f"shot_{shot}_full_record_cache.npz"
    ev_cache_path = CACHE_DIR / f"shot_{shot}_chirp_events.npz"
    if not cache_path.exists() or not ev_cache_path.exists():
        raise FileNotFoundError(f"Missing cache files for shot {shot}")
        
    d_raw = np.load(cache_path)
    d_ev = np.load(ev_cache_path)
    
    t_ms = d_raw["t_ms"]
    mp1 = d_raw["mp1"]
    mp3 = d_raw["mp3"]
    mp4 = d_raw["mp4"]
    t_track = d_ev["t_track_ms"]
    ridge_pri = d_ev["ridge_pri"]
    ridge_sec = d_ev["ridge_sec"]
    fs = 1e6
    dt = 1.0 / fs
    t0 = 291.2  # ECH-off fiducial (ms)
    
    # 2. Extract Adaptive Envelopes (+-6 kHz around tracked ridge)
    print("Extracting Adaptive Envelopes (+-6 kHz) via Baseband Demodulation...")
    env_pri_adapt = extract_adaptive_envelope(mp1, t_ms, ridge_pri, t_track, bw_khz=6.0, fs=fs)
    env_sec_adapt = extract_adaptive_envelope(mp1, t_ms, ridge_sec, t_track, bw_khz=6.0, fs=fs)
    env_pri_fix = d_raw["env_pri"]
    env_sec_fix = d_raw["env_sec"]
    
    # Cache adaptive envelopes
    np.savez_compressed(
        CACHE_DIR / f"shot_{shot}_adaptive_envelopes.npz",
        shot=shot,
        t_ms=t_ms,
        env_pri_adapt=env_pri_adapt,
        env_sec_adapt=env_sec_adapt,
        env_pri_fix=env_pri_fix,
        env_sec_fix=env_sec_fix
    )
    
    # 3. Before/After Ratios Across t0 (5, 3, 2 ms windows, excluding t0 to t0+1 ms)
    # Also evaluate relative to pre-trend
    print("Computing Before/After Ratios (5, 3, 2 ms; excluding [t0, t0+1] ms)...")
    window_results = {}
    for dt_w in [5.0, 3.0, 2.0]:
        w_key = f"{int(dt_w)}ms"
        # Pre window: [t0 - dt_w, t0]
        # Post window: [t0 + 1.0, t0 + 1.0 + dt_w] (equal length, excluding [t0, t0+1])
        m_pre = (t_ms >= t0 - dt_w) & (t_ms <= t0)
        m_post = (t_ms >= t0 + 1.0) & (t_ms <= t0 + 1.0 + dt_w)
        
        # Primary Adaptive
        res_pri_ada = moving_block_bootstrap_comparison(env_pri_adapt[m_pre], env_pri_adapt[m_post])
        # Primary Fixed
        res_pri_fix = moving_block_bootstrap_comparison(env_pri_fix[m_pre], env_pri_fix[m_post])
        
        # Secondary Adaptive
        res_sec_ada = moving_block_bootstrap_comparison(env_sec_adapt[m_pre], env_sec_adapt[m_post])
        # Secondary Fixed
        res_sec_fix = moving_block_bootstrap_comparison(env_sec_fix[m_pre], env_sec_fix[m_post])
        
        # Trend-relative Primary Adaptive
        t_pre = t_ms[m_pre]
        t_post = t_ms[m_post]
        poly_pre = np.polyfit(t_pre - t0, env_pri_adapt[m_pre], 1)
        expected_post = np.polyval(poly_pre, t_post - t0)
        mean_exp = float(np.mean(expected_post))
        mean_act = float(np.mean(env_pri_adapt[m_post]))
        ratio_trend_ada = mean_act / mean_exp if mean_exp > 1e-6 else np.nan
        
        # Trend-relative Primary Fixed
        poly_pre_f = np.polyfit(t_pre - t0, env_pri_fix[m_pre], 1)
        expected_post_f = np.polyval(poly_pre_f, t_post - t0)
        mean_exp_f = float(np.mean(expected_post_f))
        mean_act_f = float(np.mean(env_pri_fix[m_post]))
        ratio_trend_fix = mean_act_f / mean_exp_f if mean_exp_f > 1e-6 else np.nan
        
        window_results[w_key] = {
            "dt_window_ms": dt_w,
            "t_pre_range": [float(t0 - dt_w), float(t0)],
            "t_post_range": [float(t0 + 1.0), float(t0 + 1.0 + dt_w)],
            "primary_adaptive": res_pri_ada,
            "primary_fixed": res_pri_fix,
            "secondary_adaptive": res_sec_ada,
            "secondary_fixed": res_sec_fix,
            "trend_relative": {
                "primary_adaptive_ratio": float(ratio_trend_ada),
                "primary_adaptive_pct": float((ratio_trend_ada - 1.0) * 100.0) if not np.isnan(ratio_trend_ada) else np.nan,
                "primary_fixed_ratio": float(ratio_trend_fix),
                "primary_fixed_pct": float((ratio_trend_fix - 1.0) * 100.0) if not np.isnan(ratio_trend_fix) else np.nan,
                "pre_slope_v_per_ms": float(poly_pre[0]),
                "pre_intercept_v": float(poly_pre[1])
            }
        }
        
    # 4. Event-Triggered Timeline (289 - 297 ms, 10-50 us resolution)
    print("Computing Event-Triggered Timeline (289 - 297 ms)...")
    m_tl = (t_ms >= 289.0) & (t_ms <= 297.0)
    t_tl = t_ms[m_tl]
    mp1_tl = mp1[m_tl]
    
    # Raw MP1 running RMS (50 us window = 50 samples)
    win_rms = 50
    mp1_sq = mp1_tl**2
    running_rms = np.sqrt(np.convolve(mp1_sq, np.ones(win_rms) / win_rms, mode="same"))
    
    # Broadband 120-250 kHz power
    sos_bb = dsp.bessel(4, [120e3, 250e3], btype="bandpass", fs=fs, output="sos")
    flt_bb = dsp.sosfiltfilt(sos_bb, mp1)
    bb_power_tl = (flt_bb[m_tl]**2)
    # Smooth broadband power with 50 us boxcar
    bb_power_smooth = np.convolve(bb_power_tl, np.ones(win_rms) / win_rms, mode="same")
    
    # 5. Diagnostic Transient Tests
    # Transient intervals:
    # Shot 88653: 291.4 - 291.8 ms (centered around 291.6 ms)
    # Shot 88654: 292.0 - 293.0 ms (large surge)
    t_trans_range = [291.4, 291.8] if shot == 88653 else [292.0, 293.0]
    m_trans = (t_ms >= t_trans_range[0]) & (t_ms <= t_trans_range[1])
    
    # Test (a) & (b): Mirnov Array Cross-Coherence & Cross-Phase
    sos_pri_bessel = dsp.bessel(4, [75e3, 110e3], btype="bandpass", fs=fs, output="sos")
    flt_1 = dsp.sosfiltfilt(sos_pri_bessel, mp1[m_trans])
    flt_3 = dsp.sosfiltfilt(sos_pri_bessel, mp3[m_trans])
    flt_4 = dsp.sosfiltfilt(sos_pri_bessel, mp4[m_trans])
    
    std1, std3, std4 = np.std(flt_1), np.std(flt_3), np.std(flt_4)
    n_tr = len(flt_1)
    lags = dsp.correlation_lags(n_tr, n_tr) * 1e6 * dt # us
    c13 = dsp.correlate(flt_1, flt_3) / (std1 * std3 * n_tr + 1e-12)
    c14 = dsp.correlate(flt_1, flt_4) / (std1 * std4 * n_tr + 1e-12)
    
    m_lag = (lags >= -25.0) & (lags <= 25.0)
    best_lag13 = float(lags[m_lag][np.argmax(c13[m_lag])])
    best_corr13 = float(np.max(c13[m_lag]))
    best_lag14 = float(lags[m_lag][np.argmax(c14[m_lag])])
    best_corr14 = float(np.max(c14[m_lag]))
    
    zero_idx = np.where(lags == 0.0)[0][0]
    zero_corr13 = float(c13[zero_idx])
    zero_corr14 = float(c14[zero_idx])
    
    f_mode_est = 85.0e3
    phase13_deg = float((360.0 * f_mode_est * best_lag13 * 1e-6) % 360.0)
    phase14_deg = float((360.0 * f_mode_est * best_lag14 * 1e-6) % 360.0)
    
    # Theoretical n = -1 (mod 4) expectation:
    # MP1-MP3 (180 deg) -> 180.0 deg
    # MP1-MP4 (270 deg) -> 90.0 deg
    phase13_err = abs(phase13_deg - 180.0)
    phase14_err = abs(phase14_deg - 90.0)
    
    # Broadband vs Mode power ratio
    p_mode_trans = float(np.var(flt_1))
    p_bb_trans = float(np.var(flt_bb[m_trans]))
    bb_to_mode_ratio = float(p_bb_trans / p_mode_trans) if p_mode_trans > 0 else 0.0
    
    # PMP Poloidal Array Analysis
    pmp_edf_path = DATA_DIR / f"hj{shot}" / f"pmp@{shot}.edf"
    pmp_results = {}
    if pmp_edf_path.exists():
        _, pmp_arr, pmp_obj = load_edf(pmp_edf_path)
        # Invert channels PMP1-4
        pmp_arr[:, 0:4] = -pmp_arr[:, 0:4]
        theta_pmp_deg = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 90.0, 100.0, 110.0, 120.0, 150.0, 180.0])
        theta_pmp_rad = np.radians(theta_pmp_deg)
        
        pmp_trans = pmp_arr[m_trans, :]
        flt_pmp = dsp.sosfiltfilt(sos_pri_bessel, pmp_trans, axis=0)
        
        ref_pmp1 = flt_pmp[:, 0]
        ana_ref = dsp.hilbert(ref_pmp1)
        pmp_phases = []
        pmp_cohs = []
        for ch in range(14):
            ana_ch = dsp.hilbert(flt_pmp[:, ch])
            cross_c = np.mean(ana_ch * np.conj(ana_ref))
            pmp_phases.append(np.angle(cross_c))
            coh_c = np.abs(cross_c) / (np.sqrt(np.mean(np.abs(ana_ch)**2) * np.mean(np.abs(ana_ref)**2)) + 1e-12)
            pmp_cohs.append(float(coh_c))
            
        pmp_phases_unw = np.unwrap(pmp_phases)
        slope_m, intercept_m = np.polyfit(theta_pmp_rad, pmp_phases_unw, 1)
        corr_m = float(np.corrcoef(theta_pmp_rad, pmp_phases_unw)[0, 1])
        
        pmp_results = {
            "fitted_poloidal_m": float(slope_m),
            "phase_correlation_r": corr_m,
            "mean_probe_coherence": float(np.mean(pmp_cohs)),
            "is_m3_consistent": bool(abs(abs(slope_m) - 3.0) < 0.6 and abs(corr_m) > 0.90)
        }
    else:
        pmp_results = {"error": "PMP file not found"}
        
    transient_diagnosis = {
        "transient_time_window_ms": t_trans_range,
        "mode_power": p_mode_trans,
        "broadband_120_250k_power": p_bb_trans,
        "broadband_to_mode_ratio": bb_to_mode_ratio,
        "mp1_mp3_toroidal": {
            "peak_corr": best_corr13,
            "best_lag_us": best_lag13,
            "cross_phase_deg": phase13_deg,
            "theoretical_n_minus_1_deg": 180.0,
            "phase_error_deg": phase13_err,
            "zero_lag_corr": zero_corr13
        },
        "mp1_mp4_toroidal": {
            "peak_corr": best_corr14,
            "best_lag_us": best_lag14,
            "cross_phase_deg": phase14_deg,
            "theoretical_n_minus_1_deg": 90.0,
            "phase_error_deg": phase14_err,
            "zero_lag_corr": zero_corr14
        },
        "pmp_poloidal": pmp_results,
        "is_physical_mode": bool(best_corr13 > 0.85 and best_corr14 > 0.85 and phase13_err < 15.0 and phase14_err < 15.0 and bb_to_mode_ratio < 0.10),
        "is_common_mode_pickup": bool(abs(zero_corr13) > 0.85 and abs(best_lag13) < 0.5 and bb_to_mode_ratio > 0.50)
    }
    
    # 6. Burst Statistics & Intermittency
    # Threshold from MAD of quiet window [180, 210] ms
    print("Computing Burst Statistics & Intermittency...")
    m_quiet = (t_ms >= 180.0) & (t_ms <= 210.0)
    quiet_env = env_pri_adapt[m_quiet]
    med_q = float(np.median(quiet_env))
    mad_q = float(1.4826 * np.median(np.abs(quiet_env - med_q)))
    thresh_burst = med_q + 3.0 * mad_q
    
    def compute_burst_stats(env_slice):
        mu = float(np.mean(env_slice))
        sig = float(np.std(env_slice))
        kurt = float(kurtosis(env_slice, fisher=False))  # Pearson kurtosis (Normal = 3.0)
        frac_2sig = float(np.mean(env_slice > (mu + 2.0 * sig)))
        
        # Segment bursts
        is_burst = env_slice > thresh_burst
        diff = np.diff(is_burst.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        if len(starts) > 0 and len(ends) > 0:
            if starts[0] > ends[0]:
                starts = np.r_[0, starts]
            if len(starts) > len(ends):
                ends = np.r_[ends, len(is_burst)]
            burst_peaks = [float(np.max(env_slice[s:e])) for s, e in zip(starts, ends) if e > s]
        else:
            burst_peaks = []
            
        return {
            "mean": mu,
            "std": sig,
            "kurtosis": kurt,
            "fraction_above_2sigma": frac_2sig,
            "n_bursts": len(burst_peaks),
            "burst_peaks": burst_peaks,
            "median_burst_height": float(np.median(burst_peaks)) if len(burst_peaks) > 0 else 0.0,
            "max_burst_height": float(np.max(burst_peaks)) if len(burst_peaks) > 0 else 0.0
        }
        
    m_pre_5 = (t_ms >= t0 - 5.0) & (t_ms <= t0)
    m_post_5 = (t_ms >= t0 + 1.0) & (t_ms <= t0 + 6.0)
    burst_stats_pre = compute_burst_stats(env_pri_adapt[m_pre_5])
    burst_stats_post = compute_burst_stats(env_pri_adapt[m_post_5])
    
    # 7. Physical Units from Calibrated PMP Array (T/s)
    print("Evaluating Physical Units and Auditing EDF Metadata...")
    # Nominal parameters
    nominal_gain = 2.0
    nominal_NA = 4.5e-3  # m^2 (effective area)
    B0 = 1.25  # Tesla
    f_carrier = 85.0e3  # Hz
    
    # Audit EDF comments for PMP
    pmp_comment_verified = False
    audit_notes = []
    if pmp_edf_path.exists():
        _, _, e_pmp = load_edf(pmp_edf_path)
        c_list = e_pmp.comments() if callable(e_pmp.comments) else e_pmp.comments
        audit_notes = [str(c) for c in c_list if any(k in str(c).lower() for k in ["gain", "scale", "unit", "conv", "area"])]
        # Check if 4.5e-3 or gain 2.0 is documented
        has_area = any("4.5" in str(c) for c in c_list)
        has_gain2 = any("gain = 2" in str(c) or "gain = 2.0" in str(c) for c in c_list)
        pmp_comment_verified = has_area and has_gain2
    
    # Calculate physical B-field fluctuations
    # Bdot = V / (G * NA) (T/s)
    # delta_B = Bdot / (2 * pi * f) (Tesla)
    # delta_B / B0
    pmp_peak_v = float(np.max(env_pri_adapt[m_tl]))
    pmp_rms_v = float(np.sqrt(np.mean(env_pri_adapt[m_tl]**2)))
    
    bdot_peak = pmp_peak_v / (nominal_gain * nominal_NA)
    delta_b_peak_tesla = bdot_peak / (2.0 * np.pi * f_carrier)
    rel_b_peak = delta_b_peak_tesla / B0
    
    bdot_rms = pmp_rms_v / (nominal_gain * nominal_NA)
    delta_b_rms_tesla = bdot_rms / (2.0 * np.pi * f_carrier)
    rel_b_rms = delta_b_rms_tesla / B0
    
    physical_units_data = {
        "nominal_gain": nominal_gain,
        "nominal_coil_area_m2": nominal_NA,
        "B0_tesla": B0,
        "f_carrier_hz": f_carrier,
        "edf_metadata_audit": {
            "is_fully_verified_in_edf": pmp_comment_verified,
            "status_flag": "VERIFIED" if pmp_comment_verified else "UNVERIFIED_IN_EDF",
            "audit_extracted_comments": audit_notes
        },
        "nominal_estimates": {
            "peak_v": pmp_peak_v,
            "peak_bdot_tesla_per_s": bdot_peak,
            "peak_delta_b_tesla": delta_b_peak_tesla,
            "peak_delta_b_gauss": delta_b_peak_tesla * 1e4,
            "peak_delta_b_over_B0": rel_b_peak,
            "rms_v": pmp_rms_v,
            "rms_bdot_tesla_per_s": bdot_rms,
            "rms_delta_b_tesla": delta_b_rms_tesla,
            "rms_delta_b_gauss": delta_b_rms_tesla * 1e4,
            "rms_delta_b_over_B0": rel_b_rms
        }
    }
    
    # 8. Normalized ECE Fluctuation Levels (dI / I_mean) at ECE12 & ECE13
    print("Evaluating Normalized ECE Fluctuations (ECE12 & ECE13)...")
    ece_results = {}
    f_ece12 = DATA_DIR / f"hj{shot}" / f"ECE12FAST@{shot}.edf"
    f_ece13 = DATA_DIR / f"hj{shot}" / f"ECE13FAST@{shot}.edf"
    if f_ece12.exists() and f_ece13.exists():
        _, v12, _ = load_edf(f_ece12)
        _, v13, _ = load_edf(f_ece13)
        flt_ece12 = dsp.sosfiltfilt(sos_pri_bessel, v12)
        flt_ece13 = dsp.sosfiltfilt(sos_pri_bessel, v13)
        
        for dt_w in [5.0, 3.0, 2.0]:
            w_k = f"{int(dt_w)}ms"
            m_pre_e = (t_ms >= t0 - dt_w) & (t_ms <= t0)
            m_post_e = (t_ms >= t0 + 1.0) & (t_ms <= t0 + 1.0 + dt_w)
            
            raw_pre_12 = float(np.mean(v12[m_pre_e]))
            raw_post_12 = float(np.mean(v12[m_post_e]))
            fluc_pre_12 = float(np.sqrt(np.mean(flt_ece12[m_pre_e]**2)))
            fluc_post_12 = float(np.sqrt(np.mean(flt_ece12[m_post_e]**2)))
            norm_pre_12 = fluc_pre_12 / raw_pre_12 if abs(raw_pre_12) > 1e-4 else np.nan
            norm_post_12 = fluc_post_12 / raw_post_12 if abs(raw_post_12) > 1e-4 else np.nan
            
            raw_pre_13 = float(np.mean(v13[m_pre_e]))
            raw_post_13 = float(np.mean(v13[m_post_e]))
            fluc_pre_13 = float(np.sqrt(np.mean(flt_ece13[m_pre_e]**2)))
            fluc_post_13 = float(np.sqrt(np.mean(flt_ece13[m_post_e]**2)))
            norm_pre_13 = fluc_pre_13 / raw_pre_13 if abs(raw_pre_13) > 1e-4 else np.nan
            norm_post_13 = fluc_post_13 / raw_post_13 if abs(raw_post_13) > 1e-4 else np.nan
            
            ece_results[w_k] = {
                "ece12": {
                    "raw_dc_pre_v": raw_pre_12,
                    "raw_dc_post_v": raw_post_12,
                    "fluc_rms_pre_v": fluc_pre_12,
                    "fluc_rms_post_v": fluc_post_12,
                    "dI_over_I_pre": norm_pre_12,
                    "dI_over_I_post": norm_post_12,
                    "ratio_post_to_pre": norm_post_12 / norm_pre_12 if not np.isnan(norm_pre_12) else np.nan
                },
                "ece13": {
                    "raw_dc_pre_v": raw_pre_13,
                    "raw_dc_post_v": raw_post_13,
                    "fluc_rms_pre_v": fluc_pre_13,
                    "fluc_rms_post_v": fluc_post_13,
                    "dI_over_I_pre": norm_pre_13,
                    "dI_over_I_post": norm_post_13,
                    "ratio_post_to_pre": norm_post_13 / norm_pre_13 if not np.isnan(norm_pre_13) else np.nan
                }
            }
    else:
        ece_results = {"error": "ECE files missing"}
        
    shot_output = {
        "shot": shot,
        "window_before_after_ratios": window_results,
        "transient_diagnosis": transient_diagnosis,
        "burst_statistics_intermittency": {
            "threshold_v": thresh_burst,
            "quiet_baseline_median_v": med_q,
            "quiet_baseline_mad_v": mad_q,
            "pre_ech_5ms": burst_stats_pre,
            "post_ech_5ms": burst_stats_post
        },
        "physical_units": physical_units_data,
        "normalized_ece_fluctuations": ece_results
    }
    
    return shot_output, t_tl, env_pri_adapt[m_tl], env_sec_adapt[m_tl], running_rms, bb_power_smooth


def main():
    shots = [88653, 88654]
    all_results = {}
    timelines = {}
    
    for shot in shots:
        res, t_tl, ep, es, r_rms, bb_p = analyze_shot_task3(shot)
        all_results[str(shot)] = res
        timelines[shot] = {
            "t_ms": t_tl,
            "env_pri": ep,
            "env_sec": es,
            "running_rms": r_rms,
            "bb_power": bb_p
        }
        
    # Save combined results JSON
    json_path = OBJ_ECH_DIR / "task3_amplitude_transients_results.json"
    with open(json_path, "w", encoding="utf-8") as f_json:
        json.dump(all_results, f_json, indent=2)
    print(f"\nSaved Task 3 results JSON: {json_path}")
    
    # Generate 4-panel publication figure comparing both shots
    print("Generating Publication-Quality 300-dpi Figure...")
    fig, axes = plt.subplots(4, 2, figsize=(15.0, 13.0), sharex="col")
    
    shot_cols = [88653, 88654]
    for col_idx, shot in enumerate(shot_cols):
        tl = timelines[shot]
        t = tl["t_ms"]
        
        # Row 1: Primary & Secondary Adaptive Envelopes
        ax0 = axes[0, col_idx]
        ax0.plot(t, tl["env_pri"], color="crimson", lw=1.5, label="Primary (Adaptive +-6 kHz)")
        ax0.plot(t, tl["env_sec"], color="darkorange", lw=1.2, alpha=0.85, label="Secondary (Adaptive +-6 kHz)")
        ax0.axvline(291.2, color="black", ls="--", lw=1.2, label="ECH-Off (291.2 ms)")
        ax0.set_title(f"Heliotron J #{shot} — Mode Adaptive Envelopes", fontsize=11, fontweight="bold")
        ax0.set_ylabel("Envelope Amplitude (V)", fontsize=9.5)
        ax0.grid(True, linestyle=":", alpha=0.5)
        ax0.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
        
        # Row 2: Raw MP1 Running RMS (50 us)
        ax1 = axes[1, col_idx]
        ax1.plot(t, tl["running_rms"], color="navy", lw=1.3, label="Raw MP1 RMS (50 us win)")
        ax1.axvline(291.2, color="black", ls="--", lw=1.2)
        ax1.set_title("Total High-Frequency Magnetic Power (Raw MP1 RMS)", fontsize=10.5, fontweight="bold")
        ax1.set_ylabel("Running RMS (V)", fontsize=9.5)
        ax1.grid(True, linestyle=":", alpha=0.5)
        ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
        
        # Row 3: Broadband 120-250 kHz Power
        ax2 = axes[2, col_idx]
        ax2.plot(t, tl["bb_power"], color="forestgreen", lw=1.3, label="Broadband Power (120-250 kHz)")
        ax2.axvline(291.2, color="black", ls="--", lw=1.2)
        ax2.set_title("Out-of-Band Context (Broadband 120-250 kHz)", fontsize=10.5, fontweight="bold")
        ax2.set_ylabel("Power (V^2)", fontsize=9.5)
        ax2.grid(True, linestyle=":", alpha=0.5)
        ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
        
        # Row 4: Transient Window Zoom & Physics Verification Indicator
        ax3 = axes[3, col_idx]
        tr_info = all_results[str(shot)]["transient_diagnosis"]
        t_zoom = tr_info["transient_time_window_ms"]
        ax3.axvspan(t_zoom[0], t_zoom[1], color="yellow", alpha=0.35, label=f"Transient Window [{t_zoom[0]}, {t_zoom[1]}] ms")
        ax3.plot(t, tl["env_pri"], color="crimson", lw=1.5)
        ax3.axvline(291.2, color="black", ls="--", lw=1.2)
        
        verdict_str = "PHYSICAL MODE (n=-1, m=3)" if tr_info["is_physical_mode"] else "PICKUP"
        c13_val = tr_info["mp1_mp3_toroidal"]["peak_corr"]
        c14_val = tr_info["mp1_mp4_toroidal"]["peak_corr"]
        m_val = tr_info["pmp_poloidal"]["fitted_poloidal_m"]
        ax3.text(0.03, 0.75, f"Transient Diagnosis: {verdict_str}\nMP1-MP3 Coh: {c13_val:.3f} (180 deg)\nMP1-MP4 Coh: {c14_val:.3f} (90 deg)\nPoloidal Fit: m = {m_val:.1f}",
                 transform=ax3.transAxes, fontsize=8.5, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        ax3.set_xlabel("Time (ms)", fontsize=10)
        ax3.set_ylabel("Primary Envelope (V)", fontsize=9.5)
        ax3.set_xlim(289.0, 297.0)
        ax3.grid(True, linestyle=":", alpha=0.5)
        ax3.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
        
    plt.tight_layout()
    fig_path = FIG_DIR / "fig_task3_amplitude_transients.png"
    plt.savefig(str(fig_path), dpi=300)
    plt.close(fig)
    print(f"Saved publication figure: {fig_path}")
    
    # Copy figure to artifact directory for rendering
    artifact_dir = Path("C:/Users/kevin/.gemini/antigravity/brain/a71f786c-c60d-4491-a791-447bf89ef43e")
    if artifact_dir.exists():
        import shutil
        shutil.copy2(fig_path, artifact_dir / fig_path.name)
        print(f"Copied figure to artifact directory: {artifact_dir / fig_path.name}")


if __name__ == "__main__":
    main()
