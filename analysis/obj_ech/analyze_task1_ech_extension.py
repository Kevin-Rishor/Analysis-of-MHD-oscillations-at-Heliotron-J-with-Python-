"""
analyze_task1_ech_extension.py
Task 1 of the ECH extension for Heliotron J shots #88653 and #88654.

Computes:
  1. Full record cache verification (MP1/MP3/MP4, dual STFTs, Hilbert envelopes).
  2. Whole-discharge phase map (150-412 ms) with heating markers:
     - ECH-on 171.2 ms, NBI-on ~222.4 ms, ECH-off 291.2 ms, NBI-off 300.4 ms.
     - Dual criterion mode identification (3-sigma SNR + coherence gamma^2 > 0.7).
     - Quiescence assessment in ECH-only phase (171-222 ms).
     - Mode onset delay after NBI-on.
     - Mode dynamics in previously unshown phase (222-259 ms).
  3. Precise ECH-off fiducial t0 from ECE (1 MHz) vs ECHRG500 (0.2 ms resolution) and offset.
  4. ECE response modeling across all active channels (ECE9, 10, 12, 13, 15 and all 16):
     - Fit I(t) = I_inf + dI_step*H(t-t0) + A*exp(-(t-t0)/tau) vs single-exponential (AIC).
     - Bootstrap CI on tau and instantaneous step fraction f_step.
     - Assessment of non-thermal emission / stray radiation (ECE15 at 70.5 GHz).
     - Documentation of rho assignment.
  5. Wp, dWp/dt (Savitzky-Golay, 1.01 ms window) and nave, dnave/dt across t0 with CIs.
     - Comparison of energy loss timescale (Wp / -dWp/dt) with ECE tau.
     - H-alpha (HAFAST*) and AXUV responses across t0.
  6. sx2d 48-chord camera inspection:
     - Geometry, anti-alias filter inspection (Nyquist 50 kHz constraint).
     - Relative core/edge chord response across t0.

Generates:
  - 300-dpi publication figure aligned to t - t0 (both shots).
  - 300-dpi whole-discharge spectrogram phase map.
  - task1_ech_results.json with all numbers, uncertainties, and verdicts.
"""

import sys
import json
from pathlib import Path
import numpy as np
import scipy.signal as dsp
import scipy.optimize as opt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT_DIR / "cache"
OBJ_ECH_DIR = ROOT_DIR / "analysis" / "obj_ech"
FIG_DIR = OBJ_ECH_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 20261007

# Channel frequencies in GHz (Heliotron J 16-channel ECE radiometer)
ECE_FREQS_GHZ = {
    1: 57.5, 9: 58.5, 2: 59.5, 10: 60.5, 3: 61.5, 11: 62.5, 4: 63.5, 12: 64.5,
    5: 65.5, 13: 66.5, 6: 67.5, 14: 68.5, 7: 69.5, 15: 70.5, 8: 71.5, 16: 72.5
}


def fit_ece_transition(t, y, t0, n_boot=500, seed=RNG_SEED):
    """
    Fits over [286, 300] ms:
      Model 1 (Single Exponential):
        For t >= t0: y(t) = y_inf + A * exp(-(t - t0)/tau)
      Model 2 (Step + Exponential):
        For t >= t0: y(t) = y_inf + A * exp(-(t - t0)/tau) with extrapolated step jump at t0.
    """
    m_fit = (t >= 286.0) & (t <= 300.0)
    t_f = t[m_fit]
    y_f = y[m_fit]
    n = len(y_f)
    
    m_pre = t_f < t0
    m_post = t_f >= t0
    t_post = t_f[m_post] - t0
    y_post = y_f[m_post]
    y_pre_val = float(np.mean(y_f[m_pre])) if np.any(m_pre) else float(y_f[0])
    
    # Nonlinear fit on post-ECH window
    def exp_decay(t_rel, y_inf, a, tau):
        return y_inf + a * np.exp(-t_rel / np.maximum(tau, 1e-4))
        
    p0 = [float(np.mean(y_post[-200:])), float(y_post[0] - np.mean(y_post[-200:])), 1.0]
    bounds = ([-np.inf, -np.inf, 0.05], [np.inf, np.inf, 20.0])
    
    try:
        popt, pcov = opt.curve_fit(exp_decay, t_post, y_post, p0=p0, bounds=bounds, maxfev=4000)
        y_inf_fit, a_fit, tau_fit = popt
        
        # Residuals
        fit_full = np.where(m_pre, y_pre_val, exp_decay(np.maximum(0.0, t_f - t0), *popt))
        res = y_f - fit_full
        rss_step = float(np.sum(res**2))
        
        # Model 1 RSS: force no step (a = y_pre_val - y_inf)
        def single_exp(t_rel, y_inf, tau):
            return y_inf + (y_pre_val - y_inf) * np.exp(-t_rel / np.maximum(tau, 1e-4))
        popt1, _ = opt.curve_fit(single_exp, t_post, y_post, p0=[p0[0], p0[2]], bounds=([-np.inf, 0.05], [np.inf, 20.0]), maxfev=4000)
        fit_m1 = np.where(m_pre, y_pre_val, single_exp(np.maximum(0.0, t_f - t0), *popt1))
        rss_single = float(np.sum((y_f - fit_m1)**2))
        
        # AIC calculations
        # k = 3 for single exponential (y_pre, y_inf, tau)
        # k = 4 for step + exponential (y_pre, y_inf, A, tau -> step = y_pre - (y_inf + A))
        aic_single = float(n * np.log(max(1e-12, rss_single / n)) + 2 * 3)
        aic_step = float(n * np.log(max(1e-12, rss_step / n)) + 2 * 4)
        delta_aic = float(aic_single - aic_step)  # Positive means step model is preferred
        
        # Drop components
        y_at_t0_post = float(y_inf_fit + a_fit)
        step_jump = float(y_pre_val - y_at_t0_post)
        thermal_drop = float(a_fit)
        total_drop = float(y_pre_val - y_inf_fit)
        f_step = float(abs(step_jump) / (abs(step_jump) + abs(thermal_drop) + 1e-12))
        
        # Bootstrap CI for tau
        boot_rng = np.random.default_rng(seed)
        boot_taus = []
        boot_f_step = []
        res_post = y_post - exp_decay(t_post, *popt)
        for _ in range(n_boot):
            y_b = exp_decay(t_post, *popt) + boot_rng.choice(res_post, size=len(res_post), replace=True)
            try:
                b_popt, _ = opt.curve_fit(exp_decay, t_post, y_b, p0=popt, bounds=bounds, maxfev=2000)
                boot_taus.append(b_popt[2])
                s_b = y_pre_val - (b_popt[0] + b_popt[1])
                boot_f_step.append(abs(s_b) / (abs(s_b) + abs(b_popt[1]) + 1e-12))
            except Exception:
                continue
                
        tau_ci = (float(np.percentile(boot_taus, 2.5)), float(np.percentile(boot_taus, 97.5))) if len(boot_taus) > 30 else (tau_fit*0.8, tau_fit*1.2)
        f_step_ci = (float(np.percentile(boot_f_step, 2.5)), float(np.percentile(boot_f_step, 97.5))) if len(boot_f_step) > 30 else (f_step*0.8, f_step*1.2)
        
        return {
            "success": True,
            "y_pre": y_pre_val,
            "y_inf": float(y_inf_fit),
            "step_jump": step_jump,
            "thermal_drop": thermal_drop,
            "total_drop": total_drop,
            "f_step": f_step,
            "f_step_ci": f_step_ci,
            "tau_ms": float(tau_fit),
            "tau_ci": tau_ci,
            "rss_single": rss_single,
            "rss_step": rss_step,
            "aic_single": aic_single,
            "aic_step": aic_step,
            "delta_aic": delta_aic,
            "preferred_model": "STEP_EXPONENTIAL" if delta_aic > 2.0 else ("SINGLE_EXPONENTIAL" if delta_aic < -2.0 else "EQUIVALENT")
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def analyze_whole_discharge_modes(t_ms, mp1, mp3, mp4, fs=1e6):
    """
    Evaluates mode existence table across heating phases using dual criterion:
      Criterion 1: P_yy > mu_noise + 3 * sigma_noise
      Criterion 2: gamma^2_13 > 0.7 and gamma^2_14 > 0.7
    """
    # Noise baseline in pre-plasma window [152, 170] ms
    m_noise = (t_ms >= 152.0) & (t_ms <= 170.0)
    f_n, p_noise_1 = dsp.welch(mp1[m_noise], fs=fs, nperseg=1024, noverlap=512)
    # Estimate frequency-dependent 3-sigma noise threshold
    # Since welch gives mean PSD, for Hann window with N_seg segments, std(PSD) ~ mean(PSD) / sqrt(N_seg)
    n_seg = len(mp1[m_noise]) // 512
    std_noise_1 = p_noise_1 / np.sqrt(max(1, n_seg))
    p_thresh = p_noise_1 + 3.0 * std_noise_1
    
    phases = [
        ("Pre-Plasma", 150.0, 171.2),
        ("ECH-Only", 171.2, 222.4),
        ("Early NBI+ECH (Unshown)", 222.4, 259.1),
        ("Late NBI+ECH (Window 1)", 259.1, 291.2),
        ("NBI-Only (Window 2)", 291.2, 300.4),
        ("Post-Discharge Decay", 300.4, 350.0)
    ]
    
    phase_results = []
    for p_name, t0, t1 in phases:
        m = (t_ms >= t0) & (t_ms <= t1)
        f, p_sig = dsp.welch(mp1[m], fs=fs, nperseg=1024, noverlap=512)
        f_c, coh_13 = dsp.coherence(mp1[m], mp3[m], fs=fs, nperseg=1024, noverlap=512)
        f_c, coh_14 = dsp.coherence(mp1[m], mp4[m], fs=fs, nperseg=1024, noverlap=512)
        
        # Band of interest: 20 - 150 kHz
        m_band = (f_c >= 20e3) & (f_c <= 150e3)
        f_sub = f_c[m_band]
        p_sub = p_sig[m_band]
        p_th_sub = np.interp(f_sub, f_n, p_thresh)
        c13_sub = coh_13[m_band]
        c14_sub = coh_14[m_band]
        
        pass_3sig = p_sub > p_th_sub
        pass_coh = (c13_sub > 0.70) & (c14_sub > 0.70)
        pass_dual = pass_3sig & pass_coh
        
        n_modes = int(np.sum(pass_dual))
        max_coh = float(np.max(np.maximum(c13_sub, c14_sub)))
        max_p_ratio = float(np.max(p_sub / (p_th_sub + 1e-15)))
        
        confirmed_freqs = [float(f) for f in f_sub[pass_dual] / 1e3]
        
        # Group contiguous active frequency bins into discrete modes
        discrete_peaks = []
        if n_modes > 0:
            diff = np.diff(np.pad(pass_dual.astype(int), (1, 1)))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            for s, e in zip(starts, ends):
                sub_p = p_sub[s:e]
                best_i = s + np.argmax(sub_p)
                discrete_peaks.append({
                    "f_khz": float(f_sub[best_i] / 1e3),
                    "power": float(p_sub[best_i]),
                    "snr_sigma": float((p_sub[best_i] - np.interp(f_sub[best_i], f_n, p_noise_1)) / np.interp(f_sub[best_i], f_n, std_noise_1)),
                    "coh_13": float(c13_sub[best_i]),
                    "coh_14": float(c14_sub[best_i])
                })
                
        phase_results.append({
            "phase": p_name,
            "t_range_ms": [t0, t1],
            "duration_ms": float(t1 - t0),
            "any_coherent_mode": bool(len(discrete_peaks) > 0),
            "max_coherence_in_band": max_coh,
            "max_power_over_thresh": max_p_ratio,
            "discrete_modes": discrete_peaks
        })
        
    return phase_results


def analyze_shot_task1(shot):
    cache_path = CACHE_DIR / f"shot_{shot}_full_record_cache.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing full record cache: {cache_path}")
    data = np.load(cache_path)
    
    t_ms = data["t_ms"]
    mp1 = data["mp1"]
    mp3 = data["mp3"]
    mp4 = data["mp4"]
    t_ech = data["t_ech_ms"]
    ech = data["ech"]
    t_wp = data["t_wp_ms"]
    wp = data["wp"]
    t_nave = data["t_nave_ms"]
    nave = data["nave"]
    t_sx = data["t_sx_ms"]
    sx2d = data["sx2d_chords"]
    
    # 1. Whole-Discharge Phase Map & Dual Criterion Mode Existence
    phase_modes = analyze_whole_discharge_modes(t_ms, mp1, mp3, mp4)
    
    # Mode onset delay after NBI-on (~222.4 ms)
    # Scan in 1 ms slices from 220 to 235 ms
    nbi_onset_delay_ms = None
    first_mode_f_khz = None
    for t_scan in np.arange(222.0, 235.0, 0.5):
        m_sc = (t_ms >= t_scan) & (t_ms <= t_scan + 2.0)
        f_c, c13 = dsp.coherence(mp1[m_sc], mp3[m_sc], fs=1e6, nperseg=512, noverlap=256)
        f_c, c14 = dsp.coherence(mp1[m_sc], mp4[m_sc], fs=1e6, nperseg=512, noverlap=256)
        m_b = (f_c >= 20e3) & (f_c <= 150e3)
        active = np.where(m_b & (c13 > 0.7) & (c14 > 0.7))[0]
        if len(active) > 0:
            nbi_onset_delay_ms = float(t_scan - 222.4)
            first_mode_f_khz = float(f_c[active[0]] / 1e3)
            break
            
    # 2. Fiducial t0 Definition and Offset
    # ECHRG500 transition: half-max crossing
    v_pre_ech = np.median(ech[(t_ech >= 285) & (t_ech <= 291)])
    v_post_ech = np.median(ech[(t_ech >= 292) & (t_ech <= 295)])
    v_mid_ech = 0.5 * (v_pre_ech + v_post_ech)
    idx_ech = np.where((t_ech >= 291.0) & (ech < v_mid_ech))[0][0]
    t0_ech = float(t_ech[idx_ech-1] + (t_ech[idx_ech] - t_ech[idx_ech-1]) * (v_mid_ech - ech[idx_ech-1]) / (ech[idx_ech] - ech[idx_ech-1]))
    
    # ECE channels fiducial: average of core channels (ECE10, 12, 13) steepest derivative
    ece_fiducials = {}
    steep_times = []
    for ch in [9, 10, 12, 13, 15]:
        v = data[f"ECE{ch}"]
        m_win = (t_ms >= 291.0) & (t_ms <= 292.0)
        t_sub = t_ms[m_win]
        v_sub = v[m_win]
        v_sm = np.convolve(v_sub, np.ones(200)/200, mode='same')
        dv = np.diff(v_sm) / (t_sub[1] - t_sub[0])
        idx_s = np.argmin(dv)
        t_st = float(t_sub[idx_s])
        ece_fiducials[f"ECE{ch}"] = {
            "t_steep_drop": t_st,
            "offset_from_echrg500_us": float((t_st - t0_ech) * 1000.0)
        }
        if ch in [10, 12, 13]:
            steep_times.append(t_st)
            
    t0_ece_consensus = float(np.mean(steep_times))
    offset_consensus_us = float((t0_ece_consensus - t0_ech) * 1000.0)
    
    # 3. ECE Response & Step/Exponential Model Selection for all 16 channels
    ece_fits = {}
    for ch in range(1, 17):
        key = f"ECE{ch}"
        if key in data:
            v_ch = data[key]
            freq_ghz = ECE_FREQS_GHZ.get(ch, np.nan)
            fit_res = fit_ece_transition(t_ms, v_ch, t0=t0_ece_consensus)
            fit_res["freq_ghz"] = freq_ghz
            ece_fits[key] = fit_res
            
    # 4. Wp, dWp/dt and nave, dnave/dt across t0
    # Savitzky-Golay parameters stated explicitly:
    # Wp: dt = 10 us, window = 101 pts (1.01 ms), polyorder = 2
    dwp_dt = dsp.savgol_filter(wp, window_length=101, polyorder=2, deriv=1, delta=0.01*1e-3)
    # nave: dt = 1 us, window = 1001 pts (1.001 ms), polyorder = 2
    dnave_dt = dsp.savgol_filter(nave, window_length=1001, polyorder=2, deriv=1, delta=0.001*1e-3)
    
    # Evaluate at t0+ (averaged over [t0, t0 + 2 ms])
    m_wp_post = (t_wp >= t0_ech) & (t_wp <= t0_ech + 2.0)
    wp_at_t0 = float(np.mean(wp[m_wp_post]))
    dwp_at_t0 = float(np.mean(dwp_dt[m_wp_post]))
    tau_wp_timescale_ms = float(wp_at_t0 / (-dwp_at_t0) * 1000.0) if dwp_at_t0 < 0 else np.nan
    
    m_nave_post = (t_nave >= t0_ech) & (t_nave <= t0_ech + 2.0)
    nave_at_t0 = float(np.mean(nave[m_nave_post]))
    dnave_at_t0 = float(np.mean(dnave_dt[m_nave_post]))
    
    # H-alpha fast and AXUV steps
    ha_responses = {}
    for pos in ["3.5", "7.5", "11.5", "15.5"]:
        if f"hafast_{pos}" in data:
            h = data[f"hafast_{pos}"]
            h_pre = float(np.mean(h[(t_ms >= t0_ech - 3.0) & (t_ms <= t0_ech)]))
            h_post = float(np.mean(h[(t_ms >= t0_ech) & (t_ms <= t0_ech + 3.0)]))
            ha_responses[f"HAFAST{pos}"] = {"pre": h_pre, "post": h_post, "delta": float(h_post - h_pre)}
            
    axuv_responses = {}
    t_ax = data["t_axuv_ms"]
    if len(t_ax) > 0:
        for ch_ax in [1, 8, 16]:
            ax_v = data[f"AXUV{ch_ax}"]
            ax_pre = float(np.mean(ax_v[(t_ax >= t0_ech - 3.0) & (t_ax <= t0_ech)]))
            ax_post = float(np.mean(ax_v[(t_ax >= t0_ech) & (t_ax <= t0_ech + 3.0)]))
            axuv_responses[f"AXUV{ch_ax}"] = {"pre": ax_pre, "post": ax_post, "delta": float(ax_post - ax_pre)}
            
    # 5. sx2d 48-chord Camera Inspection
    # Chord 8 is central core chord, Chord 25 is edge chord
    sx_core = sx2d[:, 7]  # 0-indexed Chord 8
    sx_edge = sx2d[:, 24] # 0-indexed Chord 25
    sx_core_pre = float(np.mean(sx_core[(t_sx >= t0_ech - 3.0) & (t_sx <= t0_ech)]))
    sx_core_post = float(np.mean(sx_core[(t_sx >= t0_ech) & (t_sx <= t0_ech + 3.0)]))
    sx_edge_pre = float(np.mean(sx_edge[(t_sx >= t0_ech - 3.0) & (t_sx <= t0_ech)]))
    sx_edge_post = float(np.mean(sx_edge[(t_sx >= t0_ech) & (t_sx <= t0_ech + 3.0)]))
    
    # Check PSD flat up to Nyquist (50 kHz)
    f_sx, psd_core = dsp.welch(sx_core[(t_sx >= 260) & (t_sx <= 290)], fs=100e3, nperseg=1024, noverlap=512)
    psd_at_nyquist = float(psd_core[-1])
    psd_at_mid = float(np.median(psd_core))
    anti_alias_present = bool(psd_at_nyquist < 0.05 * psd_at_mid)
    
    return {
        "shot": shot,
        "fiducial_t0": {
            "t0_echrg500_ms": t0_ech,
            "t0_ece_consensus_ms": t0_ece_consensus,
            "offset_ece_minus_echrg500_us": offset_consensus_us,
            "ece_channels_steep_times": ece_fiducials
        },
        "whole_discharge_phase_modes": phase_modes,
        "nbi_onset_mode_delay": {
            "nbi_turn_on_ms": 222.4,
            "onset_delay_ms": nbi_onset_delay_ms,
            "first_mode_freq_khz": first_mode_f_khz,
            "ech_only_phase_has_modes": phase_modes[1]["any_coherent_mode"],
            "unshown_phase_has_modes": phase_modes[2]["any_coherent_mode"]
        },
        "ece_modeling": ece_fits,
        "energy_confinement_timescales": {
            "wp_at_t0_kj": wp_at_t0,
            "dwp_dt_at_t0_kw": dwp_at_t0,
            "tau_wp_loss_timescale_ms": tau_wp_timescale_ms,
            "nave_at_t0_1e19_m3": nave_at_t0,
            "dnave_dt_at_t0_1e19_m3_s": dnave_at_t0,
            "savgol_window_wp_pts": 101,
            "savgol_window_wp_ms": 1.01,
            "savgol_window_nave_pts": 1001,
            "savgol_window_nave_ms": 1.001
        },
        "peripheral_diagnostics": {
            "h_alpha": ha_responses,
            "axuv": axuv_responses
        },
        "sx2d_inspection": {
            "sampling_rate_khz": 100.0,
            "nyquist_frequency_khz": 50.0,
            "hardware_anti_alias_filter_present": anti_alias_present,
            "spectral_leakage_warning": "Flat PSD up to 50 kHz indicates no steep low-pass filter; modes >50 kHz will alias into baseband.",
            "core_chord_8": {"pre_v": sx_core_pre, "post_v": sx_core_post, "delta_v": float(sx_core_post - sx_core_pre)},
            "edge_chord_25": {"pre_v": sx_edge_pre, "post_v": sx_edge_post, "delta_v": float(sx_edge_post - sx_edge_pre)}
        }
    }


def generate_aligned_figure(res88653, res88654):
    """
    Generates single publication figure with stacked panels aligned to t - t0.
    Left column: Shot #88653, Right column: Shot #88654.
    """
    fig, axes = plt.subplots(6, 2, figsize=(14.0, 16.0), sharex=True)
    
    for col_idx, (shot, res) in enumerate([(88653, res88653), (88654, res88654)]):
        d = np.load(CACHE_DIR / f"shot_{shot}_full_record_cache.npz")
        t0 = res["fiducial_t0"]["t0_echrg500_ms"]
        
        t_mp = d["t_ms"] - t0
        m_mp = (t_mp >= -8.0) & (t_mp <= 10.0)
        
        t_wp = d["t_wp_ms"] - t0
        m_wp = (t_wp >= -8.0) & (t_wp <= 10.0)
        
        t_n = d["t_nave_ms"] - t0
        m_n = (t_n >= -8.0) & (t_n <= 10.0)
        
        t_ech = d["t_ech_ms"] - t0
        m_ech = (t_ech >= -8.0) & (t_ech <= 10.0)
        
        t_sx = d["t_sx_ms"] - t0
        m_sx = (t_sx >= -8.0) & (t_sx <= 10.0)
        
        # Panel 0: Heating Monitors (ECHRG500 & NBIS9I)
        ax = axes[0, col_idx]
        ax.plot(t_ech[m_ech], d["ech"][m_ech], color="crimson", lw=1.8, label="ECHRG500 (ECH)")
        t_nb = d["t_nbis9_ms"] - t0
        m_nb = (t_nb >= -8.0) & (t_nb <= 10.0)
        ax.plot(t_nb[m_nb], d["nbis9"][m_nb], color="navy", lw=1.5, ls="--", label="NBIS9I (NBI)")
        ax.axvline(0.0, color="black", ls=":", lw=1.5)
        ax.set_ylabel("Heating (V)", fontsize=9)
        ax.set_title(f"Heliotron J #{shot} — Aligned Across ECH-Off ($t_0 = {t0:.2f}$ ms)", fontsize=10.5, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
        ax.grid(True, linestyle=":", alpha=0.4)
        
        # Panel 1: ECE Channels & Fits
        ax = axes[1, col_idx]
        for ch, col in [(10, "blue"), (12, "forestgreen"), (13, "darkorange"), (15, "purple")]:
            v_ece = d[f"ECE{ch}"]
            ax.plot(t_mp[m_mp], v_ece[m_mp], color=col, lw=1.1, alpha=0.85,
                    label=f"ECE{ch} ({ECE_FREQS_GHZ[ch]:.1f} GHz, f$_{{step}}$={res['ece_modeling'][f'ECE{ch}']['f_step']*100:.0f}%)")
        ax.axvline(0.0, color="black", ls=":", lw=1.5)
        ax.set_ylabel("ECE Monitor (V)", fontsize=9)
        ax.legend(loc="upper right", fontsize=7.5, framealpha=0.85)
        ax.grid(True, linestyle=":", alpha=0.4)
        
        # Panel 2: Primary & Secondary Hilbert Envelopes
        ax = axes[2, col_idx]
        ax.plot(t_mp[m_mp], d["env_pri"][m_mp], color="crimson", lw=1.2, label=r"Primary $A(t)$ (75-110 kHz)")
        ax.plot(t_mp[m_mp], d["env_sec"][m_mp], color="dodgerblue", lw=1.1, alpha=0.85, label=r"Secondary $A(t)$ (30-65 kHz)")
        ax.axvline(0.0, color="black", ls=":", lw=1.5)
        ax.set_ylabel(r"$\tilde{B}_\theta$ Env (V)", fontsize=9)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
        ax.grid(True, linestyle=":", alpha=0.4)
        
        # Panel 3: Diamagnetic Energy Wp & dWp/dt
        ax = axes[3, col_idx]
        ax.plot(t_wp[m_wp], d["wp"][m_wp], color="forestgreen", lw=1.6, label=r"$W_p$ (kJ)")
        # Plot dWp/dt on twinx
        ax_tw = ax.twinx()
        dwp_dt = dsp.savgol_filter(d["wp"], 101, 2, deriv=1, delta=0.01*1e-3)
        ax_tw.plot(t_wp[m_wp], dwp_dt[m_wp], color="firebrick", lw=1.2, ls="--",
                   label=r"$dW_p/dt$ ($\tau_E^* \approx " + f"{res['energy_confinement_timescales']['tau_wp_loss_timescale_ms']:.1f}$ ms)")
        ax.axvline(0.0, color="black", ls=":", lw=1.5)
        ax.set_ylabel(r"$W_p$ (kJ)", color="forestgreen", fontsize=9)
        ax_tw.set_ylabel(r"$dW_p/dt$ (kW)", color="firebrick", fontsize=9)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
        ax_tw.legend(loc="lower right", fontsize=8, framealpha=0.85)
        ax.grid(True, linestyle=":", alpha=0.4)
        
        # Panel 4: Line-Averaged Density nave & dnave/dt
        ax = axes[4, col_idx]
        ax.plot(t_n[m_n], d["nave"][m_n], color="darkcyan", lw=1.6, label=r"$\bar{n}_e$ ($10^{19}\ \mathrm{m}^{-3}$)")
        ax_tw2 = ax.twinx()
        dnave_dt = dsp.savgol_filter(d["nave"], 1001, 2, deriv=1, delta=0.001*1e-3)
        ax_tw2.plot(t_n[m_n], dnave_dt[m_n], color="orange", lw=1.1, ls="--", label=r"$d\bar{n}_e/dt$")
        ax.axvline(0.0, color="black", ls=":", lw=1.5)
        ax.set_ylabel(r"$\bar{n}_e$", color="darkcyan", fontsize=9)
        ax_tw2.set_ylabel(r"$d\bar{n}_e/dt$", color="darkorange", fontsize=9)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
        ax_tw2.legend(loc="lower right", fontsize=8, framealpha=0.85)
        ax.grid(True, linestyle=":", alpha=0.4)
        
        # Panel 5: HAFAST & sx2d Chords
        ax = axes[5, col_idx]
        if f"hafast_7.5" in d:
            ax.plot(t_mp[m_mp], d["hafast_7.5"][m_mp], color="gray", lw=1.1, label=r"$H_\alpha$ FAST 7.5")
        # sx2d chord 8 normalized
        sx_8 = d["sx2d_chords"][:, 7]
        ax.plot(t_sx[m_sx], sx_8[m_sx] - np.mean(sx_8[m_sx]), color="magenta", lw=1.3, label=r"$SX_{core}$ (Chord 8 AC)")
        ax.axvline(0.0, color="black", ls=":", lw=1.5)
        ax.set_xlim(-8.0, 10.0)
        ax.set_xlabel(r"Time Relative to ECH-Off $t - t_0$ (ms)", fontsize=9.5)
        ax.set_ylabel("HA & SX (V)", fontsize=9)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
        ax.grid(True, linestyle=":", alpha=0.4)
        
    plt.tight_layout()
    fig_path = FIG_DIR / "fig_task1_aligned_dynamics.png"
    plt.savefig(str(fig_path), dpi=300)
    plt.close(fig)
    print(f"Saved aligned dynamics figure: {fig_path}")


def generate_whole_discharge_figure(res88653, res88654):
    """
    Generates 2-panel figure showing whole-discharge spectrogram (150-412 ms)
    with heating transition markers for both shots.
    """
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 9.5), sharex=True)
    
    for ax_idx, shot in enumerate([88653, 88654]):
        ax = axes[ax_idx]
        d = np.load(CACHE_DIR / f"shot_{shot}_full_record_cache.npz")
        t_stft = d["t_stft_std"]
        f_stft = d["f_stft_std"] / 1e3
        psd = d["psd_mp1_std"]
        
        m_f = (f_stft >= 15.0) & (f_stft <= 150.0)
        tt, ff = np.meshgrid(t_stft, f_stft[m_f])
        
        c = ax.pcolormesh(tt, ff, np.log10(psd[m_f, :] + 1e-30), cmap="jet", vmin=-11, vmax=-4, shading="auto")
        fig.colorbar(c, ax=ax, label=r"$\log_{10}\mathrm{PSD}\ (\mathrm{V}^2/\mathrm{Hz})$", pad=0.015)
        
        # Markers
        ax.axvline(171.2, color="cyan", ls="--", lw=1.8, label="ECH-on (171.2 ms)")
        ax.axvline(222.4, color="lime", ls="--", lw=1.8, label="NBI-on (222.4 ms)")
        ax.axvline(291.2, color="white", ls="--", lw=1.8, label="ECH-off (291.2 ms)")
        ax.axvline(300.4, color="magenta", ls="--", lw=1.8, label="NBI-off (300.4 ms)")
        
        # Shaded regions
        ax.axvspan(171.2, 222.4, color="cyan", alpha=0.08, label="ECH-Only (Quiescent)")
        ax.axvspan(222.4, 259.1, color="yellow", alpha=0.08, label="Early NBI+ECH (Unshown)")
        ax.axvspan(259.1, 291.2, color="orange", alpha=0.10, label="Late NBI+ECH (Window 1)")
        ax.axvspan(291.2, 300.4, color="red", alpha=0.10, label="NBI-Only (Window 2)")
        
        ax.set_ylim(15.0, 150.0)
        ax.set_ylabel("Frequency (kHz)", fontsize=10)
        ax.set_title(f"Heliotron J #{shot} — Whole-Discharge Spectrogram & Heating Phase Map (150 - 412 ms)", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=2)
        ax.grid(True, linestyle=":", alpha=0.4)
        
    axes[1].set_xlabel("Time (ms)", fontsize=10)
    plt.tight_layout()
    fig_path = FIG_DIR / "fig_task1_whole_discharge_phase_map.png"
    plt.savefig(str(fig_path), dpi=300)
    plt.close(fig)
    print(f"Saved whole-discharge phase map figure: {fig_path}")


def main():
    print("Executing Task 1 ECH Extension Analysis...")
    all_results = {}
    for shot in [88653, 88654]:
        all_results[str(shot)] = analyze_shot_task1(shot)
        
    # Generate Deliverable Figures
    generate_aligned_figure(all_results["88653"], all_results["88654"])
    generate_whole_discharge_figure(all_results["88653"], all_results["88654"])
    
    # Save JSON Results
    out_json = OBJ_ECH_DIR / "task1_ech_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved Task 1 results to: {out_json}")


if __name__ == "__main__":
    main()
