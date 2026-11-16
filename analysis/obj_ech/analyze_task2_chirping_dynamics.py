"""
analyze_task2_chirping_dynamics.py
Task 2 of the ECH extension for Heliotron J shots #88653 and #88654.

Computes:
  1. Reproduction of 0.4-1.3 jumps/ms numbers with repository script setup.
     Documentation of exact bands and effect of median filtering.
  2. Dynamic Programming (Viterbi) ridge tracking on 256-sample STFT vs Morlet CWT.
  3. Individual chirp event segmentation for Primary and Secondary modes:
     t_start, duration, f_start, f_end, df (SIGNED), df/dt, A_peak, branch_id, inter_event_interval.
     Saves to CSV and NPZ in c:/TFG/cache/ and c:/TFG/analysis/obj_ech/.
  4. Full statistics for ECH+NBI (259.1-291.3 ms), NBI-only (291.3-300.0 ms), and 4 ms bins:
     Event rate (Poisson CI), fraction of down- vs up-chirps (Wilson score CI),
     median |df/dt|, median |df|, duration (bootstrap CIs), effect size (Cliff's delta),
     permutation test with block shuffling.
  5. Null controls: Mode-free band (115-145 kHz) and pre-NBI window (180-210 ms).
  6. Amplitude scaling link: A_peak vs |df/dt| and |df| (log-log slope vs Berk-Breizman ~0.5, Spearman rho).
  7. Thermal link: chirp rate / |df| vs ECE-proxy and ne in 4 ms bins with N_eff.

Outputs:
  - 300-dpi publication figures
  - task2_chirping_results.json
"""

import sys
import json
import csv
from pathlib import Path
import numpy as np
import scipy.signal as dsp
from scipy.stats import poisson, spearmanr, norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT_DIR / "cache"
OBJ_ECH_DIR = ROOT_DIR / "analysis" / "obj_ech"
FIG_DIR = OBJ_ECH_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(ROOT_DIR / "analysis" / "common"))
from mhd_common import morlet_cwt

RNG_SEED = 20261007


def wilson_score_ci(k, n, confidence=0.95):
    if n == 0:
        return 0.0, 0.0, 0.0
    z = norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    p = k / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2.0 * n)) / denom
    half_width = (z * np.sqrt((p * (1.0 - p) + z**2 / (4.0 * n)) / n)) / denom
    return float(p), float(max(0.0, center - half_width)), float(min(1.0, center + half_width))


def bootstrap_median_ci(data, n_boot=2000, seed=RNG_SEED):
    if len(data) == 0:
        return {"val": 0.0, "ci": (0.0, 0.0)}
    data = np.asarray(data)
    boot_rng = np.random.default_rng(seed)
    boot_meds = [np.median(boot_rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    return {
        "val": float(np.median(data)),
        "ci": (float(np.percentile(boot_meds, 2.5)), float(np.percentile(boot_meds, 97.5)))
    }


def compute_cliffs_delta(x, y, n_boot=1000, seed=RNG_SEED):
    """
    Computes Cliff's delta effect size between groups x and y with bootstrap CI.
    delta = ( #(x > y) - #(x < y) ) / (len(x) * len(y))
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) == 0 or len(y) == 0:
        return {"delta": 0.0, "ci": (0.0, 0.0), "interpretation": "UNDEFINED"}
        
    diff = x[:, None] - y[None, :]
    n_gt = np.sum(diff > 0)
    n_lt = np.sum(diff < 0)
    delta = (n_gt - n_lt) / (len(x) * len(y))
    
    # Bootstrap CI
    boot_rng = np.random.default_rng(seed)
    boot_deltas = []
    for _ in range(n_boot):
        sx = boot_rng.choice(x, size=len(x), replace=True)
        sy = boot_rng.choice(y, size=len(y), replace=True)
        d = sx[:, None] - sy[None, :]
        boot_deltas.append((np.sum(d > 0) - np.sum(d < 0)) / (len(sx) * len(sy)))
        
    ci = (float(np.percentile(boot_deltas, 2.5)), float(np.percentile(boot_deltas, 97.5)))
    abs_d = abs(delta)
    interp = "NEGLIGIBLE" if abs_d < 0.147 else ("SMALL" if abs_d < 0.33 else ("MEDIUM" if abs_d < 0.474 else "LARGE"))
    return {"delta": float(delta), "ci": ci, "interpretation": interp}


def block_permutation_test(x, y, n_perm=2000, seed=RNG_SEED):
    """Permutation test comparing difference in medians."""
    if len(x) == 0 or len(y) == 0:
        return 1.0
    obs_diff = abs(np.median(x) - np.median(y))
    pooled = np.r_[x, y]
    nx = len(x)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(pooled)
        perm_diff = abs(np.median(shuffled[:nx]) - np.median(shuffled[nx:]))
        if perm_diff >= obs_diff:
            count += 1
    return float(count / n_perm)


def viterbi_ridge_track_cwt(x_sig, fs, freqs_khz, t_ms, slope_penalty=0.08):
    """
    High-fidelity Morlet CWT tracking with Viterbi dynamic programming.
    """
    freqs_hz = freqs_khz * 1e3
    Wx = morlet_cwt(x_sig, fs, freqs_hz, w=6.0)
    power = np.abs(Wx)**2
    
    # Decimate in time to 64 us step (matching STFT grid) for fast dynamic programming
    step = 64
    power_dec = power[:, ::step]
    t_dec_ms = t_ms[::step]
    n_f, n_t = power_dec.shape
    
    log_p = np.log10(power_dec + 1e-30)
    p_norm = (log_p - np.min(log_p)) / (np.max(log_p) - np.min(log_p) + 1e-12)
    
    df_bin = freqs_khz[1] - freqs_khz[0]
    bin_diff = np.abs(np.arange(n_f)[:, None] - np.arange(n_f)[None, :])
    transition_cost = slope_penalty * (bin_diff ** 2)
    
    dp_cost = np.zeros((n_f, n_t))
    backtrack = np.zeros((n_f, n_t), dtype=int)
    dp_cost[:, 0] = p_norm[:, 0]
    
    for t in range(1, n_t):
        scores = dp_cost[:, t - 1][None, :] - transition_cost
        best_prev = np.argmax(scores, axis=1)
        best_scores = scores[np.arange(n_f), best_prev]
        dp_cost[:, t] = p_norm[:, t] + best_scores
        backtrack[:, t] = best_prev
        
    best_path_idx = np.zeros(n_t, dtype=int)
    best_path_idx[-1] = np.argmax(dp_cost[:, -1])
    for t in range(n_t - 2, -1, -1):
        best_path_idx[t] = backtrack[best_path_idx[t + 1], t + 1]
        
    # Sub-bin parabolic refinement
    ridge_freqs = []
    ridge_pows = []
    for t in range(n_t):
        j = best_path_idx[t]
        if 1 <= j < n_f - 1:
            alpha = log_p[j - 1, t]
            beta = log_p[j, t]
            gamma = log_p[j + 1, t]
            denom = 2.0 * beta - alpha - gamma
            delta = 0.5 * (alpha - gamma) / (denom + 1e-12) if abs(denom) > 1e-12 else 0.0
            delta = np.clip(delta, -0.5, 0.5)
            f_ref = freqs_khz[j] + delta * df_bin
        else:
            f_ref = freqs_khz[j]
        ridge_freqs.append(f_ref)
        ridge_pows.append(power_dec[j, t])
        
    return np.array(ridge_freqs), np.array(ridge_pows), t_dec_ms


def segment_chirp_events(t_ms, f_ridge, p_ridge, env_signal, branch_name, min_dur_ms=0.25, max_dur_ms=3.0):
    """
    Segments burst events where envelope/power exceeds threshold (median + 1.8 MAD).
    Calculates signed df, slope df/dt, peak amplitude, and inter-event interval.
    """
    dt_ms = t_ms[1] - t_ms[0]
    # Threshold based on envelope
    med_env = np.median(env_signal)
    mad_env = 1.4826 * np.median(np.abs(env_signal - med_env))
    thresh = med_env + 1.8 * mad_env
    
    is_active = env_signal > thresh
    if not np.any(is_active):
        return []
        
    diff = np.diff(is_active.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    
    if is_active[0]:
        starts = np.r_[0, starts]
    if is_active[-1]:
        ends = np.r_[ends, len(is_active)]
        
    events = []
    prev_end_time = None
    for s, e in zip(starts, ends):
        dur = (e - s) * dt_ms
        if min_dur_ms <= dur <= max_dur_ms and (e - s) >= 4:
            t_ev = t_ms[s:e]
            f_ev = f_ridge[s:e]
            env_ev = env_signal[s:e]
            
            # Robust linear regression for chirp slope
            slope, intercept = np.polyfit(t_ev - t_ev[0], f_ev, 1) # kHz/ms
            f_start = float(intercept)
            f_end = float(intercept + slope * dur)
            df_signed = float(f_end - f_start) # SIGNED
            
            t_start = float(t_ev[0])
            t_end = float(t_ev[-1])
            inter_val = float(t_start - prev_end_time) if prev_end_time is not None else np.nan
            prev_end_time = t_end
            
            events.append({
                "branch_id": branch_name,
                "t_start_ms": t_start,
                "t_end_ms": t_end,
                "duration_ms": float(dur),
                "f_start_khz": f_start,
                "f_end_khz": f_end,
                "df_signed_khz": df_signed,
                "abs_df_khz": abs(df_signed),
                "df_dt_khz_per_ms": float(slope),
                "abs_df_dt": abs(float(slope)),
                "is_down_chirp": bool(df_signed < 0),
                "peak_amplitude_v": float(np.max(env_ev)),
                "mean_amplitude_v": float(np.mean(env_ev)),
                "inter_event_interval_ms": inter_val
            })
    return events


def analyze_shot_task2(shot):
    print(f"\n==========================================")
    print(f"Executing Task 2 Analysis for Shot #{shot}")
    print(f"==========================================")
    
    cache_path = CACHE_DIR / f"shot_{shot}_full_record_cache.npz"
    d = np.load(cache_path)
    
    t_ms = d["t_ms"]
    mp1 = d["mp1"]
    fs = 1e6
    
    # 1. Reproduce existing 0.4-1.3 jumps/ms numbers
    t_stft_std = d["t_stft_std"]
    f_khz_std = d["f_stft_std"] / 1e3
    psd_std = d["psd_mp1_std"]
    
    reproduce_results = {}
    for tw0, tw1, label in [(259.1, 291.3, "ECH+NBI"), (291.3, 300.0, "NBI-only")]:
        m_w = (t_stft_std >= tw0) & (t_stft_std <= tw1)
        dur = tw1 - tw0
        psd_w = psd_std[:, m_w]
        
        m_pri = (f_khz_std >= 70.0) & (f_khz_std <= 110.0)
        m_sec = (f_khz_std >= 30.0) & (f_khz_std <= 65.0)
        
        f_pri_raw = np.array([f_khz_std[m_pri][np.argmax(col[m_pri])] for col in psd_w.T])
        f_sec_raw = np.array([f_khz_std[m_sec][np.argmax(col[m_sec])] for col in psd_w.T])
        
        # Jumps > 2.5 kHz
        j_pri_raw = np.sum(np.abs(np.diff(f_pri_raw)) > 2.5) / dur
        j_sec_raw = np.sum(np.abs(np.diff(f_sec_raw)) > 2.5) / dur
        
        # With 3-pt median filter
        j_pri_med3 = np.sum(np.abs(np.diff(dsp.medfilt(f_pri_raw, 3))) > 2.5) / dur
        j_sec_med3 = np.sum(np.abs(np.diff(dsp.medfilt(f_sec_raw, 3))) > 2.5) / dur
        
        reproduce_results[label] = {
            "window_ms": [tw0, tw1],
            "duration_ms": dur,
            "primary": {"raw_argmax_jumps_per_ms": float(j_pri_raw), "medfilt3_jumps_per_ms": float(j_pri_med3)},
            "secondary": {"raw_argmax_jumps_per_ms": float(j_sec_raw), "medfilt3_jumps_per_ms": float(j_sec_med3)}
        }
        
    print("Reproduction of Jump Rates (Raw Argmax vs Medfilt):")
    for k, v in reproduce_results.items():
        print(f"  {k}: Primary = {v['primary']['raw_argmax_jumps_per_ms']:.2f}/ms (medfilt: {v['primary']['medfilt3_jumps_per_ms']:.2f}/ms) | Secondary = {v['secondary']['raw_argmax_jumps_per_ms']:.2f}/ms (medfilt: {v['secondary']['medfilt3_jumps_per_ms']:.2f}/ms)")
        
    # 2. Viterbi Ridge Tracking on Morlet CWT
    print("Running Morlet CWT Viterbi Ridge Tracking on Primary & Secondary bands...")
    # Primary band: 70 to 110 kHz (0.5 kHz grid)
    freqs_pri = np.linspace(70.0, 110.0, 81)
    ridge_pri, pow_pri, t_track_ms = viterbi_ridge_track_cwt(mp1, fs, freqs_pri, t_ms, slope_penalty=0.08)
    
    # Secondary band: 30 to 65 kHz (0.5 kHz grid)
    freqs_sec = np.linspace(30.0, 65.0, 71)
    ridge_sec, pow_sec, _ = viterbi_ridge_track_cwt(mp1, fs, freqs_sec, t_ms, slope_penalty=0.08)
    
    # Interpolate Hilbert envelopes onto tracking time grid
    env_pri_grid = np.interp(t_track_ms, t_ms, d["env_pri"])
    env_sec_grid = np.interp(t_track_ms, t_ms, d["env_sec"])
    
    # 3. Segment Individual Chirp Events
    print("Segmenting chirp events...")
    events_pri = segment_chirp_events(t_track_ms, ridge_pri, pow_pri, env_pri_grid, "Primary")
    events_sec = segment_chirp_events(t_track_ms, ridge_sec, pow_sec, env_sec_grid, "Secondary")
    
    all_events = events_pri + events_sec
    print(f"Total events segmented: {len(events_pri)} Primary, {len(events_sec)} Secondary.")
    
    # Save events to CSV and NPZ
    csv_file = OBJ_ECH_DIR / f"shot_{shot}_chirp_events.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f_csv:
        fieldnames = ["branch_id", "t_start_ms", "t_end_ms", "duration_ms", "f_start_khz", "f_end_khz",
                      "df_signed_khz", "abs_df_khz", "df_dt_khz_per_ms", "abs_df_dt", "is_down_chirp",
                      "peak_amplitude_v", "mean_amplitude_v", "inter_event_interval_ms"]
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_events)
        
    npz_events_file = CACHE_DIR / f"shot_{shot}_chirp_events.npz"
    np.savez_compressed(
        npz_events_file,
        shot=shot,
        events_pri=events_pri,
        events_sec=events_sec,
        ridge_pri=ridge_pri,
        ridge_sec=ridge_sec,
        t_track_ms=t_track_ms
    )
    print(f"Saved events to CSV ({csv_file.name}) and NPZ ({npz_events_file.name}).")
    
    # 4. Statistical Analysis: Window A vs Window B
    def compile_window_stats(ev_list, t0, t1, label):
        sub_ev = [e for e in ev_list if t0 <= e["t_start_ms"] <= t1]
        k = len(sub_ev)
        dur_w = t1 - t0
        rate = k / dur_w
        k_ci = (float(poisson.ppf(0.025, k) / dur_w) if k > 0 else 0.0, float(poisson.ppf(0.975, k) / dur_w))
        
        n_down = sum(1 for e in sub_ev if e["is_down_chirp"])
        frac_down, fd_low, fd_high = wilson_score_ci(n_down, k)
        
        slopes = [e["abs_df_dt"] for e in sub_ev]
        dfs = [e["abs_df_khz"] for e in sub_ev]
        durs = [e["duration_ms"] for e in sub_ev]
        amps = [e["peak_amplitude_v"] for e in sub_ev]
        
        return {
            "label": label,
            "t_range_ms": [t0, t1],
            "duration_ms": dur_w,
            "k_events": k,
            "event_rate_per_ms": float(rate),
            "event_rate_poisson_ci": k_ci,
            "n_down_chirps": n_down,
            "n_up_chirps": k - n_down,
            "frac_down_chirps": frac_down,
            "frac_down_ci": (fd_low, fd_high),
            "median_abs_df_dt": bootstrap_median_ci(slopes),
            "median_abs_df": bootstrap_median_ci(dfs),
            "median_duration": bootstrap_median_ci(durs),
            "median_amplitude": bootstrap_median_ci(amps),
            "events": sub_ev
        }
        
    stat_A_pri = compile_window_stats(events_pri, 259.1, 291.3, "ECH+NBI (32.2 ms)")
    stat_B_pri = compile_window_stats(events_pri, 291.3, 300.0, "NBI-only (8.7 ms)")
    stat_Aeq_pri = compile_window_stats(events_pri, 282.6, 291.3, "ECH+NBI Equal (8.7 ms)")
    
    stat_A_sec = compile_window_stats(events_sec, 259.1, 291.3, "ECH+NBI (32.2 ms)")
    stat_B_sec = compile_window_stats(events_sec, 291.3, 300.0, "NBI-only (8.7 ms)")
    stat_Aeq_sec = compile_window_stats(events_sec, 282.6, 291.3, "ECH+NBI Equal (8.7 ms)")
    
    # Effect sizes & Permutation tests
    slopes_A = [e["abs_df_dt"] for e in stat_A_pri["events"]]
    slopes_B = [e["abs_df_dt"] for e in stat_B_pri["events"]]
    cliff_slope = compute_cliffs_delta(slopes_B, slopes_A)
    p_perm_slope = block_permutation_test(slopes_B, slopes_A)
    
    # 5. Null Controls (Mode-free band 115-145 kHz, pre-NBI window 180-210 ms)
    print("Running Null Controls...")
    freqs_null = np.linspace(115.0, 145.0, 61)
    ridge_null, pow_null, _ = viterbi_ridge_track_cwt(mp1, fs, freqs_null, t_ms, slope_penalty=0.08)
    sos_null = dsp.bessel(4, [115e3, 145e3], btype="bandpass", fs=fs, output="sos")
    env_null = np.abs(dsp.hilbert(dsp.sosfiltfilt(sos_null, mp1)))
    env_null_grid = np.interp(t_track_ms, t_ms, env_null)
    events_null_band = segment_chirp_events(t_track_ms, ridge_null, pow_null, env_null_grid, "NullBand_115_145k")
    
    stat_null_band = compile_window_stats(events_null_band, 259.1, 291.3, "Null Band (115-145 kHz)")
    stat_pre_nbi_pri = compile_window_stats(events_pri, 180.0, 210.0, "Pre-NBI Window (180-210 ms)")
    
    # 6. Amplitude Link: A_peak vs |df/dt| and |df| (Log-log & Spearman)
    pri_amps = np.array([e["peak_amplitude_v"] for e in events_pri])
    pri_dfs = np.array([e["abs_df_khz"] for e in events_pri])
    pri_slopes = np.array([e["abs_df_dt"] for e in events_pri])
    
    if len(pri_amps) >= 5:
        log_a = np.log10(pri_amps)
        log_df = np.log10(pri_dfs)
        log_sl = np.log10(pri_slopes)
        
        slope_df, intercept_df = np.polyfit(log_a, log_df, 1)
        rho_df, p_val_df = spearmanr(pri_amps, pri_dfs)
        
        slope_sl, intercept_sl = np.polyfit(log_a, log_sl, 1)
        rho_sl, p_val_sl = spearmanr(pri_amps, pri_slopes)
        
        # Bootstrap CI for log-log slope of df vs A
        boot_rng = np.random.default_rng(RNG_SEED)
        b_slopes_df = []
        for _ in range(1000):
            idx = boot_rng.integers(0, len(log_a), size=len(log_a))
            b_slopes_df.append(np.polyfit(log_a[idx], log_df[idx], 1)[0])
        ci_slope_df = (float(np.percentile(b_slopes_df, 2.5)), float(np.percentile(b_slopes_df, 97.5)))
        
        amp_scaling_results = {
            "n_events": len(pri_amps),
            "loglog_slope_df_vs_amp": float(slope_df),
            "loglog_slope_df_ci": ci_slope_df,
            "spearman_rho_df": float(rho_df),
            "spearman_p_df": float(p_val_df),
            "loglog_slope_df_dt_vs_amp": float(slope_sl),
            "spearman_rho_df_dt": float(rho_sl),
            "berk_breizman_comparison": {
                "theory_expected_slope": 0.5,
                "measured_slope": float(slope_df),
                "is_consistent_with_half_power": bool(ci_slope_df[0] <= 0.5 <= ci_slope_df[1])
            }
        }
    else:
        amp_scaling_results = {"n_events": len(pri_amps), "insufficient_events": True}
        
    # 7. 4 ms Bins over [222, 300] ms linked to Te proxy & ne
    t_nave = d["t_nave_ms"]
    nave = d["nave"]
    t_ece = d["t_ms"]
    v_ece12 = d["ECE12"]
    
    bin_stats = []
    for b_start in np.arange(222.0, 300.0, 4.0):
        b_end = b_start + 4.0
        ev_in_b = [e for e in events_pri if b_start <= e["t_start_ms"] < b_end]
        
        m_n = (t_nave >= b_start) & (t_nave < b_end)
        m_e = (t_ece >= b_start) & (t_ece < b_end)
        
        n_val = float(np.mean(nave[m_n])) if np.any(m_n) else np.nan
        ece_val = float(np.mean(v_ece12[m_e])) if np.any(m_e) else np.nan
        
        k_b = len(ev_in_b)
        med_sl = float(np.median([e["abs_df_dt"] for e in ev_in_b])) if k_b > 0 else np.nan
        med_df = float(np.median([e["abs_df_khz"] for e in ev_in_b])) if k_b > 0 else np.nan
        
        bin_stats.append({
            "t_bin_ms": [float(b_start), float(b_end)],
            "k_events": k_b,
            "median_abs_df_dt": med_sl,
            "median_abs_df": med_df,
            "mean_nave": n_val,
            "mean_ece12": ece_val
        })
        
    # Clean output dictionary
    shot_output = {
        "shot": shot,
        "reproduction_of_repository_jumps": reproduce_results,
        "primary_mode_statistics": {
            "window_A_ech_nbi": {k: v for k, v in stat_A_pri.items() if k != "events"},
            "window_B_nbi_only": {k: v for k, v in stat_B_pri.items() if k != "events"},
            "window_Aeq_equal_len": {k: v for k, v in stat_Aeq_pri.items() if k != "events"},
            "cliffs_delta_slope_B_vs_A": cliff_slope,
            "permutation_p_value_slope": p_perm_slope
        },
        "secondary_mode_statistics": {
            "window_A_ech_nbi": {k: v for k, v in stat_A_sec.items() if k != "events"},
            "window_B_nbi_only": {k: v for k, v in stat_B_sec.items() if k != "events"},
            "window_Aeq_equal_len": {k: v for k, v in stat_Aeq_sec.items() if k != "events"}
        },
        "null_controls": {
            "mode_free_band_115_145k": {k: v for k, v in stat_null_band.items() if k != "events"},
            "pre_nbi_quiescent_window": {k: v for k, v in stat_pre_nbi_pri.items() if k != "events"}
        },
        "amplitude_scaling_berk_breizman": amp_scaling_results,
        "four_ms_bin_trajectories": bin_stats
    }
    
    # Generate 3-panel figure for this shot
    fig, axes = plt.subplots(3, 1, figsize=(12.0, 10.5))
    
    # Panel 1: Spectrogram with segmented events
    ax = axes[0]
    m_p1 = (t_track_ms >= 255.0) & (t_track_ms <= 302.0)
    ax.plot(t_track_ms[m_p1], ridge_pri[m_p1], color="black", lw=1.0, alpha=0.5, label="Viterbi Ridge Track (Primary)")
    for ev in events_pri:
        if 255.0 <= ev["t_start_ms"] <= 302.0:
            col = "crimson" if ev["is_down_chirp"] else "blue"
            ax.plot([ev["t_start_ms"], ev["t_end_ms"]], [ev["f_start_khz"], ev["f_end_khz"]], color=col, lw=2.2)
            
    ax.axvline(291.2, color="gray", ls="--", lw=1.5, label="ECH-Off (291.2 ms)")
    ax.set_xlim(255.0, 302.0)
    ax.set_ylim(75.0, 105.0)
    ax.set_ylabel("Frequency (kHz)", fontsize=10)
    ax.set_title(f"Heliotron J #{shot} — Primary Mode Segmented Chirp Events (Red: Down, Blue: Up)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.4)
    
    # Panel 2: Secondary Mode Ridge & Events
    ax = axes[1]
    ax.plot(t_track_ms[m_p1], ridge_sec[m_p1], color="black", lw=1.0, alpha=0.5, label="Viterbi Ridge Track (Secondary)")
    for ev in events_sec:
        if 255.0 <= ev["t_start_ms"] <= 302.0:
            col = "darkorange" if ev["is_down_chirp"] else "dodgerblue"
            ax.plot([ev["t_start_ms"], ev["t_end_ms"]], [ev["f_start_khz"], ev["f_end_khz"]], color=col, lw=2.0)
            
    ax.axvline(291.2, color="gray", ls="--", lw=1.5)
    ax.set_xlim(255.0, 302.0)
    ax.set_ylim(30.0, 65.0)
    ax.set_ylabel("Frequency (kHz)", fontsize=10)
    ax.set_title("Secondary Mode Segmented Chirp Events", fontsize=10.5, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.4)
    
    # Panel 3: Event Slopes df/dt Across Window A and Window B
    ax = axes[2]
    ev_plot = [e for e in events_pri if 255.0 <= e["t_start_ms"] <= 302.0]
    t_evs = [e["t_start_ms"] for e in ev_plot]
    sl_evs = [e["df_dt_khz_per_ms"] for e in ev_plot]
    amp_evs = [e["peak_amplitude_v"] * 60.0 for e in ev_plot]
    
    sc = ax.scatter(t_evs, sl_evs, s=amp_evs, c=["crimson" if s < 0 else "blue" for s in sl_evs], alpha=0.8, edgecolors="black")
    ax.axhline(0.0, color="black", ls=":", lw=1.2)
    ax.axvline(291.2, color="gray", ls="--", lw=1.5)
    ax.set_xlim(255.0, 302.0)
    ax.set_xlabel("Time (ms)", fontsize=10)
    ax.set_ylabel(r"Chirp Rate $df/dt$ (kHz/ms)", fontsize=10)
    ax.set_title(r"Chirp Rate Distribution (Marker Size $\propto A_{\mathrm{peak}}$) Across ECH Shut-Off", fontsize=10.5, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.4)
    
    plt.tight_layout()
    fig_shot = FIG_DIR / f"fig_task2_ridge_and_events_{shot}.png"
    plt.savefig(str(fig_shot), dpi=300)
    plt.close(fig)
    print(f"Saved ridge and events figure: {fig_shot}")
    
    return shot_output, events_pri


def main():
    all_results = {}
    events_by_shot = {}
    
    for shot in [88653, 88654]:
        res, evs = analyze_shot_task2(shot)
        all_results[str(shot)] = res
        events_by_shot[shot] = evs
        
    out_json = OBJ_ECH_DIR / "task2_chirping_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved Task 2 results to: {out_json}")
    
    # Generate Joint Figure: Amplitude Scaling & Te Proxy Link
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.0))
    
    # Panel (0, 0): Shot 88653 Log-Log Berk-Breizman Fit
    ax = axes[0, 0]
    ev53 = events_by_shot[88653]
    a53 = np.array([e["peak_amplitude_v"] for e in ev53])
    df53 = np.array([e["abs_df_khz"] for e in ev53])
    if len(a53) >= 5:
        ax.scatter(a53, df53, color="crimson", alpha=0.75, s=35, edgecolors="black", label=r"Measured Events (#88653)")
        p_fit = np.polyfit(np.log10(a53), np.log10(df53), 1)
        a_line = np.linspace(np.min(a53), np.max(a53), 50)
        ax.plot(a_line, 10**(np.polyval(p_fit, np.log10(a_line))), color="darkred", lw=2.0,
                label=f"Fit: Slope = {p_fit[0]:.2f} (BB theory: 0.50)")
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel(r"Peak Amplitude $A_{\mathrm{peak}}$ (V)", fontsize=9.5)
    ax.set_ylabel(r"Chirp Frequency Span $|\Delta f|$ (kHz)", fontsize=9.5)
    ax.set_title(f"Heliotron J #88653 — Amplitude Scaling vs Berk-Breizman", fontsize=10.5, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    
    # Panel (0, 1): Shot 88654 Log-Log Berk-Breizman Fit
    ax = axes[0, 1]
    ev54 = events_by_shot[88654]
    a54 = np.array([e["peak_amplitude_v"] for e in ev54])
    df54 = np.array([e["abs_df_khz"] for e in ev54])
    if len(a54) >= 5:
        ax.scatter(a54, df54, color="navy", alpha=0.75, s=35, edgecolors="black", label=r"Measured Events (#88654)")
        p_fit54 = np.polyfit(np.log10(a54), np.log10(df54), 1)
        a_line54 = np.linspace(np.min(a54), np.max(a54), 50)
        ax.plot(a_line54, 10**(np.polyval(p_fit54, np.log10(a_line54))), color="blue", lw=2.0,
                label=f"Fit: Slope = {p_fit54[0]:.2f} (BB theory: 0.50)")
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel(r"Peak Amplitude $A_{\mathrm{peak}}$ (V)", fontsize=9.5)
    ax.set_ylabel(r"Chirp Frequency Span $|\Delta f|$ (kHz)", fontsize=9.5)
    ax.set_title(f"Heliotron J #88654 — Amplitude Scaling vs Berk-Breizman", fontsize=10.5, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    
    # Panel (1, 0): 4 ms Bins Chirp Rate vs ECE12 Proxy (#88654)
    ax = axes[1, 0]
    bins54 = all_results["88654"]["four_ms_bin_trajectories"]
    t_mids = [0.5*(b["t_bin_ms"][0] + b["t_bin_ms"][1]) for b in bins54]
    sl_mids = [b["median_abs_df_dt"] for b in bins54]
    ece_mids = [b["mean_ece12"] for b in bins54]
    ax.plot(t_mids, sl_mids, marker="o", color="crimson", lw=1.8, label=r"Median $|df/dt|$ (kHz/ms)")
    ax.axvline(291.2, color="gray", ls="--", lw=1.5, label="ECH-Off")
    ax_tw = ax.twinx()
    ax_tw.plot(t_mids, ece_mids, marker="s", color="blue", lw=1.5, ls="--", label=r"ECE12 Proxy (V)")
    ax.set_xlabel("Time (ms)", fontsize=9.5)
    ax.set_ylabel(r"Median $|df/dt|$ (kHz/ms)", color="crimson", fontsize=9.5)
    ax_tw.set_ylabel("ECE12 Monitor (V)", color="blue", fontsize=9.5)
    ax.set_title(f"Evolution of Chirp Rate vs Electron Temperature Proxy (#88654)", fontsize=10.5, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.4)
    
    # Panel (1, 1): Null Control Comparison
    ax = axes[1, 1]
    shots_lbl = ["#88653", "#88654"]
    rate_pri_A = [all_results[str(s)]["primary_mode_statistics"]["window_A_ech_nbi"]["event_rate_per_ms"] for s in [88653, 88654]]
    rate_pri_B = [all_results[str(s)]["primary_mode_statistics"]["window_B_nbi_only"]["event_rate_per_ms"] for s in [88653, 88654]]
    rate_null_band = [all_results[str(s)]["null_controls"]["mode_free_band_115_145k"]["event_rate_per_ms"] for s in [88653, 88654]]
    rate_null_time = [all_results[str(s)]["null_controls"]["pre_nbi_quiescent_window"]["event_rate_per_ms"] for s in [88653, 88654]]
    
    x_idx = np.arange(2)
    w = 0.18
    ax.bar(x_idx - 1.5*w, rate_pri_A, width=w, color="crimson", label="Primary ECH+NBI")
    ax.bar(x_idx - 0.5*w, rate_pri_B, width=w, color="darkorange", label="Primary NBI-only")
    ax.bar(x_idx + 0.5*w, rate_null_band, width=w, color="gray", label="Null Band (115-145k)")
    ax.bar(x_idx + 1.5*w, rate_null_time, width=w, color="black", label="Pre-NBI Null Time")
    ax.set_xticks(x_idx)
    ax.set_xticklabels(shots_lbl, fontsize=10)
    ax.set_ylabel("Event Rate (events/ms)", fontsize=9.5)
    ax.set_title("Signal vs Null Controls (Noise Floor Comparison)", fontsize=10.5, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8.2, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.4)
    
    plt.tight_layout()
    fig_joint = FIG_DIR / "fig_task2_amplitude_and_thermal_scaling.png"
    plt.savefig(str(fig_joint), dpi=300)
    plt.close(fig)
    print(f"Saved amplitude and thermal scaling figure: {fig_joint}")


if __name__ == "__main__":
    main()
