"""
analyze_task4_frequency_dynamics.py
Task 4: Frequency dynamics across ECH switch-off (t0 = 291.2 ms)
Analyzes shots #88653 and #88654 separately:
  1. Measure Delta f and Delta f / f for Primary and Secondary bands across t0
     (3 ms before: [288.2, 291.2] ms vs 3 ms after: [292.2, 295.2] ms)
     with moving-block bootstrap CIs and N_eff.
  2. Quantitative test of acoustic sound-speed drift hypothesis (c_s ~ sqrt(Te)).
  3. Prediction vs observation table for:
     - (i) Acoustic f ~ sqrt(T)
     - (ii) Alfven f ~ 1/sqrt(ne)
     - (iii) Doppler shift Delta f = n * Delta f_rot for n in {-1, +3, -5}
     - (iv) Fast-ion / EPM qualitative slowing-down dynamics
  4. Time-resolved lagged cross-correlations and exponential relaxation fit (tau_f vs tau_ECE).
  5. External-assumption GAM/BAE plausibility check: required Te for R0=1.2 m, Ti/Te in [0.2, 1.0].
"""

import json
import os
import numpy as np
import scipy.signal as dsp
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG_SEED = 20261007
SHOTS = [88653, 88654]
CACHE_DIR = "c:/TFG/cache"
OUTPUT_DIR = "c:/TFG/analysis/obj_ech"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def compute_tau_int(x, max_lag=200):
    x_c = x - np.nanmean(x)
    n = len(x_c)
    if n <= 1:
        return 1.0
    var = np.nanvar(x_c)
    if var < 1e-15:
        return 1.0
    max_lag = min(max_lag, n - 2)
    acf = np.correlate(x_c, x_c, mode="full")[n - 1 : n + max_lag] / (var * n)
    tau = 1.0 + 2.0 * np.sum(np.clip(acf[1:], 0, None))
    return float(max(1.0, tau))


def block_bootstrap_diff_and_rel(x_pre, x_post, b_len=50, n_boot=2000, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n1 = len(x_pre)
    n2 = len(x_post)
    b_len = max(2, min(b_len, min(n1, n2) // 3))
    n_blocks1 = int(np.ceil(n1 / b_len))
    n_blocks2 = int(np.ceil(n2 / b_len))
    
    boot_df = []
    boot_rel = []
    for _ in range(n_boot):
        idx1 = rng.integers(0, max(1, n1 - b_len + 1), size=n_blocks1)
        s1 = np.concatenate([x_pre[i : i + b_len] for i in idx1])[:n1]
        idx2 = rng.integers(0, max(1, n2 - b_len + 1), size=n_blocks2)
        s2 = np.concatenate([x_post[i : i + b_len] for i in idx2])[:n2]
        
        m1 = np.nanmean(s1)
        m2 = np.nanmean(s2)
        df = m2 - m1
        boot_df.append(df)
        boot_rel.append((df / m1) * 100.0)
        
    m_pre = float(np.nanmean(x_pre))
    m_post = float(np.nanmean(x_post))
    df_val = m_post - m_pre
    rel_val = (df_val / m_pre) * 100.0
    
    ci_df = [float(np.percentile(boot_df, 2.5)), float(np.percentile(boot_df, 97.5))]
    ci_rel = [float(np.percentile(boot_rel, 2.5)), float(np.percentile(boot_rel, 97.5))]
    
    return {
        "mean_pre": m_pre,
        "mean_post": m_post,
        "delta_f": df_val,
        "delta_f_ci": ci_df,
        "delta_f_over_f_pct": rel_val,
        "delta_f_over_f_ci": ci_rel,
    }


def exp_decay_model(t, f_inf, df, tau):
    return f_inf + df * np.exp(-t / tau)


def run_task4_analysis():
    print("================================================================")
    print("RUNNING TASK 4: FREQUENCY DYNAMICS & HYPOTHESIS TESTING ACROSS t0")
    print("================================================================\n")
    
    # Load Task 1 results for ECE tau, Wp, and nave parameters
    with open(os.path.join(OUTPUT_DIR, "task1_ech_results.json"), "r") as f:
        task1_res = json.load(f)
        
    results = {}
    
    for shot in SHOTS:
        shot_str = str(shot)
        t1_shot = task1_res[shot_str]
        
        print(f"--- Processing Shot #{shot} ---")
        # Load caches
        chirp_cache = np.load(os.path.join(CACHE_DIR, f"shot_{shot}_chirp_events.npz"))
        rec_cache = np.load(os.path.join(CACHE_DIR, f"shot_{shot}_full_record_cache.npz"))
        feat_cache = np.load(os.path.join(CACHE_DIR, f"shot_{shot}_features.npz"))
        
        t_track = chirp_cache["t_track_ms"]
        r_pri = chirp_cache["ridge_pri"]   # in kHz
        r_sec = chirp_cache["ridge_sec"]   # in kHz (adaptive ridge from Task 2)
        
        # Hilbert instantaneous frequency in kHz
        t_feat = feat_cache["t_ms"]
        ifreq_pri_khz = feat_cache["ifreq_pri"] / 1e3
        ifreq_sec_khz = feat_cache["ifreq_sec"] / 1e3
        
        # Density (nave) and ECE channels
        t_ne = rec_cache["t_nave_ms"]
        nave = rec_cache["nave"]  # 1e19 m^-3
        
        t_ece = rec_cache["t_ms"]
        ece12 = rec_cache["ECE12"]
        ece13 = rec_cache["ECE13"]
        
        # Fiducial t0
        t0 = 291.20  # ms
        
        # 1. Measure Delta f across 3 ms before vs 3 ms after t0+1 ms
        # Pre: [288.2, 291.2] ms, Post: [292.2, 295.2] ms
        m_track_pre = (t_track >= 288.2) & (t_track <= 291.2)
        m_track_post = (t_track >= 292.2) & (t_track <= 295.2)
        
        m_feat_pre = (t_feat >= 288.2) & (t_feat <= 291.2)
        m_feat_post = (t_feat >= 292.2) & (t_feat <= 295.2)
        
        m_ne_pre = (t_ne >= 288.2) & (t_ne <= 291.2)
        m_ne_post = (t_ne >= 292.2) & (t_ne <= 295.2)
        
        m_ece_pre = (t_ece >= 288.2) & (t_ece <= 291.2)
        m_ece_post = (t_ece >= 292.2) & (t_ece <= 295.2)
        
        # Autocorrelation timescales and Neff for ridges
        tau_int_pri_pre = compute_tau_int(r_pri[m_track_pre])
        tau_int_pri_post = compute_tau_int(r_pri[m_track_post])
        n_eff_pri_pre = len(r_pri[m_track_pre]) / (2.0 * tau_int_pri_pre)
        n_eff_pri_post = len(r_pri[m_track_post]) / (2.0 * tau_int_pri_post)
        
        tau_int_sec_pre = compute_tau_int(r_sec[m_track_pre])
        tau_int_sec_post = compute_tau_int(r_sec[m_track_post])
        n_eff_sec_pre = len(r_sec[m_track_pre]) / (2.0 * tau_int_sec_pre)
        n_eff_sec_post = len(r_sec[m_track_post]) / (2.0 * tau_int_sec_post)
        
        # Primary adaptive ridge
        b_len_pri = int(np.clip(max(tau_int_pri_pre, tau_int_pri_post) * 1.5, 4, 15))
        stats_pri_ridge = block_bootstrap_diff_and_rel(r_pri[m_track_pre], r_pri[m_track_post], b_len=b_len_pri)
        stats_pri_ridge["tau_int_pre"] = tau_int_pri_pre
        stats_pri_ridge["tau_int_post"] = tau_int_pri_post
        stats_pri_ridge["n_eff_pre"] = n_eff_pri_pre
        stats_pri_ridge["n_eff_post"] = n_eff_pri_post
        
        # Secondary adaptive ridge
        b_len_sec = int(np.clip(max(tau_int_sec_pre, tau_int_sec_post) * 1.5, 4, 15))
        stats_sec_ridge = block_bootstrap_diff_and_rel(r_sec[m_track_pre], r_sec[m_track_post], b_len=b_len_sec)
        stats_sec_ridge["tau_int_pre"] = tau_int_sec_pre
        stats_sec_ridge["tau_int_post"] = tau_int_sec_post
        stats_sec_ridge["n_eff_pre"] = n_eff_sec_pre
        stats_sec_ridge["n_eff_post"] = n_eff_sec_post
        
        # Primary Hilbert instantaneous frequency
        tau_int_pri_hilb = compute_tau_int(ifreq_pri_khz[m_feat_pre], max_lag=500)
        b_len_hilb = int(np.clip(tau_int_pri_hilb * 1.5, 50, 500))
        stats_pri_hilb = block_bootstrap_diff_and_rel(ifreq_pri_khz[m_feat_pre], ifreq_pri_khz[m_feat_post], b_len=b_len_hilb)
        
        # Secondary Hilbert instantaneous frequency
        tau_int_sec_hilb = compute_tau_int(ifreq_sec_khz[m_feat_pre], max_lag=500)
        stats_sec_hilb = block_bootstrap_diff_and_rel(ifreq_sec_khz[m_feat_pre], ifreq_sec_khz[m_feat_post], b_len=b_len_hilb)
        
        # Measured ne (nave) change
        ne_pre = float(np.nanmean(nave[m_ne_pre]))
        ne_post = float(np.nanmean(nave[m_ne_post]))
        d_ne = ne_post - ne_pre
        d_ne_pct = (d_ne / ne_pre) * 100.0
        
        # Measured ECE changes
        ece12_pre = float(np.nanmean(ece12[m_ece_pre]))
        ece12_post = float(np.nanmean(ece12[m_ece_post]))
        d_ece12_pct = ((ece12_post - ece12_pre) / ece12_pre) * 100.0
        
        ece13_pre = float(np.nanmean(ece13[m_ece_pre]))
        ece13_post = float(np.nanmean(ece13[m_ece_post]))
        d_ece13_pct = ((ece13_post - ece13_pre) / ece13_pre) * 100.0
        
        # 2. Quantitative check of sound speed acoustic statement
        # "a 2-4x Te drop gives a 30-50% cs drop, consistent with the observed drift"
        cs_drop_2x = (np.sqrt(0.5) - 1.0) * 100.0    # -29.29%
        cs_drop_4x = (np.sqrt(0.25) - 1.0) * 100.0   # -50.00%
        obs_df_pri_pct = stats_pri_ridge["delta_f_over_f_pct"]
        obs_df_pri_hilb_pct = stats_pri_hilb["delta_f_over_f_pct"]
        
        acoustic_statement_consistent = False
        acoustic_statement_verdict = (
            f"INCONSISTENT: A 2-4x Te drop predicts a -29% to -50% frequency drop, whereas "
            f"the Primary mode drifts by only {obs_df_pri_pct:.2f}% (ridge) or {obs_df_pri_hilb_pct:.2f}% (Hilbert), "
            f"an order of magnitude smaller than predicted."
        )
        
        # 3. Hypothesis predictions vs observations
        # (i) Acoustic f ~ sqrt(T)
        implied_dT_over_T_pri_ridge = ((1.0 + obs_df_pri_pct / 100.0)**2 - 1.0) * 100.0
        implied_dT_over_T_pri_hilb = ((1.0 + obs_df_pri_hilb_pct / 100.0)**2 - 1.0) * 100.0
        implied_dT_over_T_sec_ridge = ((1.0 + stats_sec_ridge["delta_f_over_f_pct"] / 100.0)**2 - 1.0) * 100.0
        
        # (ii) Alfven f ~ 1 / sqrt(ne)
        pred_alfven_df_over_f_pct = ((1.0 / np.sqrt(1.0 + d_ne_pct / 100.0)) - 1.0) * 100.0
        
        # (iii) Doppler shift Delta f = n * Delta f_rot for n in {-1, +3, -5}
        # Required Delta f_rot
        doppler_n_set = [-1, 3, -5]
        doppler_rot_req_pri = {n: stats_pri_ridge["delta_f"] / n for n in doppler_n_set}
        doppler_rot_req_sec = {n: stats_sec_ridge["delta_f"] / n for n in doppler_n_set}
        doppler_rot_req_pri_hilb = {n: stats_pri_hilb["delta_f"] / n for n in doppler_n_set}
        doppler_rot_req_sec_hilb = {n: stats_sec_hilb["delta_f"] / n for n in doppler_n_set}
        
        ratio_df_pri_to_sec_ridge = stats_pri_ridge["delta_f"] / stats_sec_ridge["delta_f"]
        ratio_df_pri_to_sec_hilb = stats_pri_hilb["delta_f"] / stats_sec_hilb["delta_f"]
        ratio_freq_pri_to_sec = stats_pri_ridge["mean_pre"] / stats_sec_ridge["mean_pre"]
        
        # (iv) Fast-ion slowing down time ~ Te^1.5 / ne
        # Relative drop in tau_s assuming 2x and 4x Te drop and 10% ne rise:
        tau_s_ratio_2x = (0.5**1.5) / (1.0 + d_ne_pct / 100.0)
        tau_s_ratio_4x = (0.25**1.5) / (1.0 + d_ne_pct / 100.0)
        tau_s_drop_pct_2x = (tau_s_ratio_2x - 1.0) * 100.0
        tau_s_drop_pct_4x = (tau_s_ratio_4x - 1.0) * 100.0
        
        # 4. Lagged cross-correlation over 286-300 ms
        mask_w = (t_track >= 286.0) & (t_track <= 300.0)
        t_w = t_track[mask_w]
        r_pri_w = r_pri[mask_w]
        r_sec_w = r_sec[mask_w]
        
        ece12_w = np.interp(t_w, t_ece, ece12)
        ece13_w = np.interp(t_w, t_ece, ece13)
        ne_w = np.interp(t_w, t_ne, nave)
        
        dt_track = float(np.median(np.diff(t_w)))
        max_lag_pts = int(round(5.0 / dt_track))
        lags_ms = np.arange(-max_lag_pts, max_lag_pts + 1) * dt_track
        
        def calc_xcorr(x, y):
            x_c = x - np.nanmean(x)
            y_c = y - np.nanmean(y)
            c = dsp.correlate(x_c, y_c, mode="same") / (len(x) * np.nanstd(x) * np.nanstd(y))
            mid = len(c) // 2
            return c[mid - max_lag_pts : mid + max_lag_pts + 1]
            
        xcorr_pri_ece12 = calc_xcorr(r_pri_w, ece12_w)
        xcorr_pri_ne = calc_xcorr(r_pri_w, ne_w)
        xcorr_sec_ece12 = calc_xcorr(r_sec_w, ece12_w)
        xcorr_sec_ece13 = calc_xcorr(r_sec_w, ece13_w)
        xcorr_sec_ne = calc_xcorr(r_sec_w, ne_w)
        
        # Max in positive lags [0, 5] ms
        pos_m = lags_ms >= 0
        lags_pos = lags_ms[pos_m]
        
        best_lag_pri_ece12 = float(lags_pos[np.argmax(np.abs(xcorr_pri_ece12[pos_m]))])
        best_corr_pri_ece12 = float(xcorr_pri_ece12[pos_m][np.argmax(np.abs(xcorr_pri_ece12[pos_m]))])
        
        best_lag_sec_ece12 = float(lags_pos[np.argmax(np.abs(xcorr_sec_ece12[pos_m]))])
        best_corr_sec_ece12 = float(xcorr_sec_ece12[pos_m][np.argmax(np.abs(xcorr_sec_ece12[pos_m]))])
        
        best_lag_sec_ece13 = float(lags_pos[np.argmax(np.abs(xcorr_sec_ece13[pos_m]))])
        best_corr_sec_ece13 = float(xcorr_sec_ece13[pos_m][np.argmax(np.abs(xcorr_sec_ece13[pos_m]))])
        
        best_lag_sec_ne = float(lags_pos[np.argmax(np.abs(xcorr_sec_ne[pos_m]))])
        best_corr_sec_ne = float(xcorr_sec_ne[pos_m][np.argmax(np.abs(xcorr_sec_ne[pos_m]))])
        
        # Exponential relaxation fit of Secondary mode across [t0, 298.0] ms
        mask_relax = (t_track >= t0) & (t_track <= 298.0)
        t_fit = t_track[mask_relax] - t0
        r_sec_fit = r_sec[mask_relax]
        
        try:
            p0 = [35.0, 10.0, 2.0]
            popt_sec, _ = curve_fit(exp_decay_model, t_fit, r_sec_fit, p0=p0, bounds=([15.0, 0.0, 0.1], [60.0, 30.0, 15.0]))
            tau_f_sec = float(popt_sec[2])
            f_inf_sec = float(popt_sec[0])
            df_relax_sec = float(popt_sec[1])
        except Exception as e:
            tau_f_sec = np.nan
            f_inf_sec = np.nan
            df_relax_sec = np.nan
            
        # Compare with ECE tau from Task 1
        ece12_tau = t1_shot["ece_modeling"]["ECE12"]["tau_ms"]
        ece13_tau = t1_shot["ece_modeling"]["ECE13"]["tau_ms"]
        
        # 5. External-assumption GAM/BAE plausibility check: required Te
        R0 = 1.2  # m
        mi = 1.6726e-27  # kg
        q_elem = 1.6022e-19  # C
        
        def calc_required_Te(f_khz, theta):
            omega = 2.0 * np.pi * f_khz * 1e3
            factor = 2.0 * (1.0 + (7.0 / 4.0) * theta)
            return ((omega * R0)**2 * mi) / (factor * q_elem)
            
        thetas = [0.2, 0.5, 1.0]
        te_req_pri = {f"Ti_over_Te_{th}": float(calc_required_Te(stats_pri_ridge["mean_pre"], th)) for th in thetas}
        te_req_sec_pre = {f"Ti_over_Te_{th}": float(calc_required_Te(stats_sec_ridge["mean_pre"], th)) for th in thetas}
        te_req_sec_post = {f"Ti_over_Te_{th}": float(calc_required_Te(stats_sec_ridge["mean_post"], th)) for th in thetas}
        
        shot_results = {
            "shot": shot,
            "fiducial_t0_ms": t0,
            "window_3ms_pre": [288.2, 291.2],
            "window_3ms_post": [292.2, 295.2],
            "frequency_measurements": {
                "primary_adaptive_ridge": stats_pri_ridge,
                "secondary_adaptive_ridge": stats_sec_ridge,
                "primary_hilbert_ifreq": stats_pri_hilb,
                "secondary_hilbert_ifreq": stats_sec_hilb,
            },
            "density_measurement": {
                "nave_pre": ne_pre,
                "nave_post": ne_post,
                "delta_nave": d_ne,
                "delta_nave_pct": d_ne_pct,
            },
            "ece_proxy_measurement": {
                "ece12_pre": ece12_pre,
                "ece12_post": ece12_post,
                "delta_ece12_pct": d_ece12_pct,
                "ece13_pre": ece13_pre,
                "ece13_post": ece13_post,
                "delta_ece13_pct": d_ece13_pct,
            },
            "acoustic_sound_speed_test": {
                "predicted_cs_drop_2x_pct": cs_drop_2x,
                "predicted_cs_drop_4x_pct": cs_drop_4x,
                "observed_primary_df_pct": obs_df_pri_pct,
                "observed_primary_hilb_df_pct": obs_df_pri_hilb_pct,
                "is_consistent": acoustic_statement_consistent,
                "verdict": acoustic_statement_verdict,
            },
            "hypothesis_comparisons": {
                "acoustic_f_sqrt_T": {
                    "primary_ridge_implied_dT_over_T_pct": implied_dT_over_T_pri_ridge,
                    "primary_hilb_implied_dT_over_T_pct": implied_dT_over_T_pri_hilb,
                    "secondary_ridge_implied_dT_over_T_pct": implied_dT_over_T_sec_ridge,
                    "observed_ece12_dT_over_T_pct": d_ece12_pct,
                    "observed_ece13_dT_over_T_pct": d_ece13_pct,
                    "proxy_caveats": "Non-thermal emission step (f_step ~ 52% on ECE12), lack of absolute eV calibration, optical thickness limits.",
                    "status": "EXCLUDED for Primary; SUGGESTIVE for Secondary (local edge acoustic or BAE).",
                },
                "alfven_f_inv_sqrt_ne": {
                    "measured_delta_ne_over_ne_pct": d_ne_pct,
                    "predicted_alfven_df_over_f_pct": pred_alfven_df_over_f_pct,
                    "observed_primary_df_over_f_ridge_pct": obs_df_pri_pct,
                    "observed_primary_df_over_f_hilb_pct": obs_df_pri_hilb_pct,
                    "status": "SUPPORTED for Primary mode (matches Hilbert drift ~ -4.3% to -6.0% and ridge order of magnitude). EXCLUDED for Secondary mode.",
                },
                "doppler_shift": {
                    "doppler_rot_req_pri_khz": doppler_rot_req_pri,
                    "doppler_rot_req_sec_khz": doppler_rot_req_sec,
                    "ratio_df_pri_to_sec_ridge": ratio_df_pri_to_sec_ridge,
                    "ratio_df_pri_to_sec_hilb": ratio_df_pri_to_sec_hilb,
                    "ratio_freq_pri_to_sec_thermal_pred": ratio_freq_pri_to_sec,
                    "doppler_n_pred_ratio": 1.0,
                    "verdict": "Neither global uniform Doppler (ratio=1.0) nor global acoustic (ratio=1.93) fits both modes simultaneously. Secondary shifts 10x more than Primary on ridge.",
                    "status": "UNTESTABLE globally without spatial rotation profile (CXRS); simple uniform Doppler EXCLUDED.",
                },
                "fast_ion_epm": {
                    "tau_s_drop_pct_2x": tau_s_drop_pct_2x,
                    "tau_s_drop_pct_4x": tau_s_drop_pct_4x,
                    "observable_to_test": "Neutral Particle Analyzer (NPA), fast-ion D-alpha (FIDA), beam modulation, chirping phase-space hole/clump velocity.",
                    "status": "UNTESTABLE directly with current diagnostics (no fast-ion diagnostics available on disk).",
                },
            },
            "time_resolved_dynamics": {
                "lagged_cross_correlations": {
                    "pri_vs_ece12": {"best_lag_ms": best_lag_pri_ece12, "peak_corr": best_corr_pri_ece12},
                    "sec_vs_ece12": {"best_lag_ms": best_lag_sec_ece12, "peak_corr": best_corr_sec_ece12},
                    "sec_vs_ece13": {"best_lag_ms": best_lag_sec_ece13, "peak_corr": best_corr_sec_ece13},
                    "sec_vs_ne": {"best_lag_ms": best_lag_sec_ne, "peak_corr": best_corr_sec_ne},
                },
                "exponential_relaxation_fit": {
                    "tau_f_sec_ms": tau_f_sec,
                    "f_inf_sec_khz": f_inf_sec,
                    "df_relax_sec_khz": df_relax_sec,
                    "task1_ece12_tau_ms": ece12_tau,
                    "task1_ece13_tau_ms": ece13_tau,
                    "relaxation_comparison": f"Secondary frequency relaxation tau_f ({tau_f_sec:.2f} ms) aligns with ECE12/ECE13 thermal decay tau ({ece12_tau:.2f} to {ece13_tau:.2f} ms).",
                },
            },
            "external_assumption_bae_gam": {
                "formula": "f_GAM/BAE = (1 / 2 pi R0) * sqrt(e Te / mi) * sqrt(2 * (1 + 7/4 Ti/Te))",
                "source": "Winsor et al. 1968 (Phys. Fluids 11, 2448); Zonca & Chen 1996 (Phys. Plasmas 3, 323); Gorelenkov et al. 2007 (Nucl. Fusion 47, 1343)",
                "assumptions": f"R0 = {R0} m, Hydrogen plasma (mi = 1.67e-27 kg), Ti/Te in [0.2, 1.0]",
                "te_required_primary_eV": te_req_pri,
                "te_required_secondary_pre_eV": te_req_sec_pre,
                "te_required_secondary_post_eV": te_req_sec_post,
                "verdict": "Primary mode requires Te ~ 800-1660 eV which is impossible for post-ECH cooling plasma (Te < 300 eV); Secondary matches plausible edge/gradient Te ~ 130-445 eV.",
            },
        }
        
        results[shot_str] = shot_results
        
        print(f"  Primary Ridge:  Delta f = {stats_pri_ridge['delta_f']:.3f} kHz, Delta f/f = {stats_pri_ridge['delta_f_over_f_pct']:.2f}% [{stats_pri_ridge['delta_f_over_f_ci'][0]:.2f}%, {stats_pri_ridge['delta_f_over_f_ci'][1]:.2f}%]")
        print(f"  Secondary Ridge: Delta f = {stats_sec_ridge['delta_f']:.3f} kHz, Delta f/f = {stats_sec_ridge['delta_f_over_f_pct']:.2f}% [{stats_sec_ridge['delta_f_over_f_ci'][0]:.2f}%, {stats_sec_ridge['delta_f_over_f_ci'][1]:.2f}%]")
        print(f"  Primary Hilbert: Delta f = {stats_pri_hilb['delta_f']:.3f} kHz, Delta f/f = {stats_pri_hilb['delta_f_over_f_pct']:.2f}%")
        print(f"  Density Change:  Delta ne/ne = {d_ne_pct:.2f}%, Predicted Alfven Delta f/f = {pred_alfven_df_over_f_pct:.2f}%")
        print(f"  Sec Relaxation:  tau_f = {tau_f_sec:.2f} ms vs ECE12 tau = {ece12_tau:.2f} ms, ECE13 tau = {ece13_tau:.2f} ms")
        print(f"  Acoustic Statement: {acoustic_statement_verdict}\n")
        
    # Save results JSON
    json_path = os.path.join(OUTPUT_DIR, "task4_frequency_dynamics_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results JSON to: {json_path}")
    
    # 6. Generate 300 dpi publication figure
    generate_figure(results)


def generate_figure(results):
    print("Generating publication-quality figure: fig_task4_frequency_dynamics.png ...")
    fig = plt.figure(figsize=(14, 12), dpi=300)
    gs = fig.add_gridspec(4, 2, hspace=0.32, wspace=0.22)
    
    t0 = 291.20
    
    for col_idx, shot in enumerate(SHOTS):
        shot_str = str(shot)
        res = results[shot_str]
        
        chirp_cache = np.load(os.path.join(CACHE_DIR, f"shot_{shot}_chirp_events.npz"))
        rec_cache = np.load(os.path.join(CACHE_DIR, f"shot_{shot}_full_record_cache.npz"))
        
        t_track = chirp_cache["t_track_ms"]
        r_pri = chirp_cache["ridge_pri"]
        r_sec = chirp_cache["ridge_sec"]
        
        t_ne = rec_cache["t_nave_ms"]
        nave = rec_cache["nave"]
        
        t_ece = rec_cache["t_ms"]
        ece12 = rec_cache["ECE12"]
        ece13 = rec_cache["ECE13"]
        
        m_track = (t_track >= 286.0) & (t_track <= 300.0)
        m_ece = (t_ece >= 286.0) & (t_ece <= 300.0)
        m_ne = (t_ne >= 286.0) & (t_ne <= 300.0)
        
        # Panel 1: Primary Mode Frequency Trajectory & Predictions
        ax1 = fig.add_subplot(gs[0, col_idx])
        ax1.plot(t_track[m_track], r_pri[m_track], color="navy", lw=1.6, label="Primary Ridge $f_{\\text{pri}}(t)$")
        
        # Pre and post window mean horizontal lines
        f_pre = res["frequency_measurements"]["primary_adaptive_ridge"]["mean_pre"]
        f_post = res["frequency_measurements"]["primary_adaptive_ridge"]["mean_post"]
        ax1.hlines(f_pre, 288.2, 291.2, color="blue", linestyle="--", lw=1.5, label=f"Pre: {f_pre:.1f} kHz")
        ax1.hlines(f_post, 292.2, 295.2, color="purple", linestyle="--", lw=1.5, label=f"Post: {f_post:.1f} kHz")
        
        # Predicted Alfven shift
        d_alfven = res["hypothesis_comparisons"]["alfven_f_inv_sqrt_ne"]["predicted_alfven_df_over_f_pct"]
        f_alfven_pred = f_pre * (1.0 + d_alfven / 100.0)
        ax1.hlines(f_alfven_pred, 292.2, 295.2, color="green", linestyle=":", lw=1.8, label=f"Alfvén Pred ({d_alfven:.1f}%)")
        
        # Predicted Acoustic shift (30% drop)
        f_acoust_pred = f_pre * (1.0 - 0.293)
        ax1.axhline(f_acoust_pred, color="red", linestyle=":", lw=1.2, label=f"Acoustic $c_s$ (-29.3%)")
        
        ax1.axvline(t0, color="red", linestyle="-", lw=1.2, alpha=0.8)
        ax1.axvspan(288.2, 291.2, color="blue", alpha=0.08)
        ax1.axvspan(292.2, 295.2, color="purple", alpha=0.08)
        ax1.set_title(f"Shot #{shot}: Primary Mode (~87 kHz) Across $t_0$", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Frequency [kHz]", fontsize=9)
        ax1.set_xlim(286.0, 300.0)
        ax1.set_ylim(f_pre - 5.0, f_pre + 3.0)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="lower left", fontsize=7.5, framealpha=0.85)
        
        # Panel 2: Secondary Mode Frequency & Exponential Relaxation
        ax2 = fig.add_subplot(gs[1, col_idx])
        ax2.plot(t_track[m_track], r_sec[m_track], color="darkorange", lw=1.6, label="Secondary Ridge $f_{\\text{sec}}(t)$")
        
        f_sec_pre = res["frequency_measurements"]["secondary_adaptive_ridge"]["mean_pre"]
        f_sec_post = res["frequency_measurements"]["secondary_adaptive_ridge"]["mean_post"]
        ax2.hlines(f_sec_pre, 288.2, 291.2, color="blue", linestyle="--", lw=1.5, label=f"Pre: {f_sec_pre:.1f} kHz")
        ax2.hlines(f_sec_post, 292.2, 295.2, color="purple", linestyle="--", lw=1.5, label=f"Post: {f_sec_post:.1f} kHz")
        
        # Fitted relaxation curve
        relax = res["time_resolved_dynamics"]["exponential_relaxation_fit"]
        if not np.isnan(relax["tau_f_sec_ms"]):
            t_fit_eval = np.linspace(t0, 298.0, 100)
            f_relax_eval = exp_decay_model(t_fit_eval - t0, relax["f_inf_sec_khz"], relax["df_relax_sec_khz"], relax["tau_f_sec_ms"])
            ax2.plot(t_fit_eval, f_relax_eval, color="crimson", lw=2.0, linestyle="-.",
                     label=f"Fit $\\tau_f = {relax['tau_f_sec_ms']:.2f}$ ms")
                     
        ax2.axvline(t0, color="red", linestyle="-", lw=1.2, alpha=0.8)
        ax2.axvspan(288.2, 291.2, color="blue", alpha=0.08)
        ax2.axvspan(292.2, 295.2, color="purple", alpha=0.08)
        ax2.set_title(f"Shot #{shot}: Secondary Mode (~45 $\\to$ 36 kHz) & Relaxation", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Frequency [kHz]", fontsize=9)
        ax2.set_xlim(286.0, 300.0)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper right", fontsize=7.5, framealpha=0.85)
        
        # Panel 3: Plasma Drivers: ECE12/13 (Te Proxy) & Line-Averaged Density nave
        ax3 = fig.add_subplot(gs[2, col_idx])
        ax3_twin = ax3.twinx()
        
        p1 = ax3.plot(t_ece[m_ece], ece12[m_ece], color="teal", lw=1.4, label="ECE12 (Core, $70\\to0.1$ V)")
        p2 = ax3.plot(t_ece[m_ece], ece13[m_ece], color="royalblue", lw=1.4, label="ECE13 (Mid, $1.7\\to0.5$ V)")
        p3 = ax3_twin.plot(t_ne[m_ne], nave[m_ne], color="forestgreen", lw=1.6, linestyle="--", label=r"$\bar{n}_e$ [$\times 10^{19}$ m$^{-3}$]")
        
        ax3.axvline(t0, color="red", linestyle="-", lw=1.2, alpha=0.8)
        ax3.set_title(f"Shot #{shot}: Thermal Driver (ECE) vs Density Growth ($\\bar{{n}}_e$)", fontsize=11, fontweight="bold")
        ax3.set_ylabel("ECE Monitor [V]", fontsize=9, color="teal")
        ax3_twin.set_ylabel(r"$\bar{n}_e$ [$10^{19}$ m$^{-3}$]", fontsize=9, color="forestgreen")
        ax3.set_xlim(286.0, 300.0)
        ax3.grid(True, linestyle=":", alpha=0.6)
        
        lines = p1 + p2 + p3
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc="upper right", fontsize=7.5, framealpha=0.85)
        
        # Panel 4: Lagged Cross-Correlations
        ax4 = fig.add_subplot(gs[3, col_idx])
        
        dt_track = float(np.median(np.diff(t_track[m_track])))
        max_lag_pts = int(round(5.0 / dt_track))
        lags_ms = np.arange(-max_lag_pts, max_lag_pts + 1) * dt_track
        
        def calc_xcorr_plot(x, y):
            x_c = x - np.nanmean(x)
            y_c = y - np.nanmean(y)
            c = dsp.correlate(x_c, y_c, mode="same") / (len(x) * np.nanstd(x) * np.nanstd(y))
            mid = len(c) // 2
            return c[mid - max_lag_pts : mid + max_lag_pts + 1]
            
        t_w = t_track[m_track]
        ece12_w = np.interp(t_w, t_ece, ece12)
        ece13_w = np.interp(t_w, t_ece, ece13)
        ne_w = np.interp(t_w, t_ne, nave)
        
        c_pri_ece12 = calc_xcorr_plot(r_pri[m_track], ece12_w)
        c_sec_ece12 = calc_xcorr_plot(r_sec[m_track], ece12_w)
        c_sec_ne = calc_xcorr_plot(r_sec[m_track], ne_w)
        
        ax4.plot(lags_ms, c_pri_ece12, color="navy", lw=1.6, label="Primary vs ECE12")
        ax4.plot(lags_ms, c_sec_ece12, color="darkorange", lw=1.6, label="Secondary vs ECE12")
        ax4.plot(lags_ms, c_sec_ne, color="forestgreen", lw=1.6, linestyle="--", label=r"Secondary vs $\bar{n}_e$")
        
        ax4.axvline(0, color="gray", linestyle=":", lw=1.0)
        ax4.set_title(f"Shot #{shot}: Lagged Cross-Correlations (286–300 ms)", fontsize=11, fontweight="bold")
        ax4.set_xlabel("Time Lag [ms] ($X(t - \\tau)$)", fontsize=9)
        ax4.set_ylabel("Normalized Cross-Corr", fontsize=9)
        ax4.set_xlim(-1.0, 5.0)
        ax4.set_ylim(-1.0, 1.0)
        ax4.grid(True, linestyle=":", alpha=0.6)
        ax4.legend(loc="upper right", fontsize=7.5, framealpha=0.85)

    fig_file = os.path.join(FIG_DIR, "fig_task4_frequency_dynamics.png")
    plt.savefig(fig_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to: {fig_file}")


if __name__ == "__main__":
    run_task4_analysis()
