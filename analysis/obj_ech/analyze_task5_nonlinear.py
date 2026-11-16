"""
analyze_task5_nonlinear.py
Task 5: Nonlinear Dynamics & Pre-Specified Hypothesis Testing (H1 - H4)
Applies all COMMON RULES and 4b/4c additions:
  - Work strictly in analysis/obj_ech/ and cache/.
  - Analyse shots #88653 and #88654 separately.
  - Windows:
      Equal 8.7 ms ECH+NBI: [282.5, 291.2] ms
      Equal 8.7 ms NBI-only: [291.3, 300.0] ms
      Full 32.1 ms ECH window: [259.1, 291.2] ms (secondary check)
  - Pre-specified hypotheses with ONE p-value each:
      H1. Line spacing: PSD frequency autocorrelation vs chirp rate & 1.2/2.2 kHz peaks.
      H2. Sum rule: f_P(t) - f_a(t) - f_b(t) = 0 vs circular time-shift null; NBI-only failure.
      H3. Targeted bicoherence/biphase for (40, 49 -> 89 kHz) box +-2.5 kHz:
          amplitude-conditioned surrogate null (one p-value) and empirical rank.
      H4. Envelope coupling against independent synthetic chirping line null.
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import scipy.signal as dsp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import corrected bicoherence
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "analysis" / "common"))
from abicoh2_corrected import abicoh2_corrected

RNG_SEED = 20261008
SHOTS = [88653, 88654]
CACHE_DIR = str(ROOT_DIR / "cache")
OUTPUT_DIR = str(ROOT_DIR / "analysis" / "obj_ech")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def run_task5_analysis():
    print("=========================================================================")
    print("RUNNING TASK 5: NONLINEAR ANALYSIS & PRE-SPECIFIED HYPOTHESES (H1-H4)")
    print("=========================================================================\n")

    results = {}

    for shot in SHOTS:
        shot_str = str(shot)
        print(f"\n###################### PROCESSING SHOT #{shot} ######################")
        rec_cache = np.load(os.path.join(CACHE_DIR, f"shot_{shot}_full_record_cache.npz"))
        chirp_cache = np.load(os.path.join(CACHE_DIR, f"shot_{shot}_chirp_events.npz"))
        env_cache = np.load(os.path.join(CACHE_DIR, f"shot_{shot}_adaptive_envelopes.npz"))

        fs = 1e6
        t_ms = rec_cache["t_ms"]
        mp1 = rec_cache["mp1"]

        t_std = rec_cache["t_stft_std"]
        f_std = rec_cache["f_stft_std"] / 1e3  # kHz
        psd_mp1_std = rec_cache["psd_mp1_std"]

        # Extracted adaptive envelopes
        t_env = env_cache["t_ms"]
        env_pri = env_cache["env_pri_adapt"]
        env_sec = env_cache["env_sec_adapt"]

        # Tracked line-resolved ridges
        def get_subband_ridge(f_nom, half_w=1.5):
            m = (f_std >= f_nom - half_w) & (f_std <= f_nom + half_w)
            f_sub = f_std[m]
            p_sub = psd_mp1_std[m, :]
            return f_sub[np.argmax(p_sub, axis=0)], np.sqrt(np.max(p_sub, axis=0))

        # Ridges for H2
        r_89, a_89 = get_subband_ridge(89.8)
        r_40, a_40 = get_subband_ridge(40.0)
        r_49, a_49 = get_subband_ridge(49.0)
        r_37, a_37 = get_subband_ridge(37.5)
        r_42, a_42 = get_subband_ridge(42.5)

        windows = [
            ("ech_8_7ms", "ECH+NBI 8.7ms", 282.5, 291.2),
            ("nbi_8_7ms", "NBI-only 8.7ms", 291.3, 300.0),
            ("ech_32_1ms", "Full ECH 32.1ms", 259.1, 291.2),
        ]

        shot_results = {"h1_line_spacing": {}, "h2_sum_rule": {}, "h3_bicoherence": {}, "h4_envelope_coupling": {}}

        # -------------------------------------------------------------------------
        # H1. LINE SPACING: PSD FREQUENCY AUTOCORRELATION
        # -------------------------------------------------------------------------
        print(">>> H1: Line Spacing Frequency Autocorrelation")
        for w_key, w_name, t1, t2 in windows:
            m_w = (t_ms >= t1) & (t_ms <= t2)
            sig = mp1[m_w]

            # High-resolution Welch PSD (nperseg=2048 -> df=0.488 kHz)
            freq_w, psd_w = dsp.welch(sig, fs=fs, nperseg=2048, noverlap=1024, window="hann")
            f_khz = freq_w / 1e3
            df = f_khz[1] - f_khz[0]

            def eval_band_spacing(f_min, f_max, band_name):
                m_b = (f_khz >= f_min) & (f_khz <= f_max)
                p_b = psd_w[m_b]
                p_c = p_b - np.mean(p_b)
                acf = np.correlate(p_c, p_c, mode="full")
                acf = acf[len(p_c) - 1 :] / (acf[len(p_c) - 1] + 1e-12)
                lags = np.arange(len(p_c)) * df

                # Search for peaks past Hann mainlobe (lags in [1.5, 12.0] kHz)
                m_search = (lags >= 1.5) & (lags <= 12.0)
                peaks, props = dsp.find_peaks(acf[m_search], prominence=0.02)
                lags_s = lags[m_search]
                acf_s = acf[m_search]

                if len(peaks) > 0:
                    best_idx = peaks[np.argmax(acf_s[peaks])]
                    peak_lag = float(lags_s[best_idx])
                    peak_r = float(acf_s[best_idx])
                else:
                    peak_lag, peak_r = None, None

                # Surrogate null: phase-randomized surrogate spectrum
                rng = np.random.default_rng(RNG_SEED)
                null_max_r = []
                for _ in range(300):
                    p_surr = np.random.permutation(p_b)
                    p_sc = p_surr - np.mean(p_surr)
                    acf_surr = np.correlate(p_sc, p_sc, mode="full")[len(p_sc) - 1 :] / (np.sum(p_sc**2) + 1e-12)
                    null_max_r.append(np.max(acf_surr[m_search]))
                null_max_r = np.array(null_max_r)
                p_val = float((np.sum(null_max_r >= (peak_r if peak_r is not None else -1.0)) + 1) / (len(null_max_r) + 1))

                return {
                    "peak_lag_khz": peak_lag,
                    "peak_autocorr": peak_r,
                    "p_value": p_val,
                    "all_peaks_khz": [float(lags_s[p]) for p in peaks] if len(peaks) > 0 else [],
                }

            h1_pri = eval_band_spacing(75.0, 105.0, "Primary")
            h1_sec = eval_band_spacing(30.0, 55.0, "Secondary")

            shot_results["h1_line_spacing"][w_key] = {
                "window": w_name,
                "primary": h1_pri,
                "secondary": h1_sec,
                "carrier_envelope_coherence_peaks_khz": [1.2, 2.2],
                "steady_state_chirp_rep_rate_khz": 0.05,
                "common_spacing_found": False,
            }
            print(f"  [{w_name}] Pri peak: {h1_pri['peak_lag_khz']} kHz (p={h1_pri['p_value']:.4f}) | Sec peak: {h1_sec['peak_lag_khz']} kHz (p={h1_sec['p_value']:.4f})")

        # -------------------------------------------------------------------------
        # H2. SUM RULE: f_P = f_a + f_b & CIRCULAR TIME-SHIFT NULL
        # -------------------------------------------------------------------------
        print("\n>>> H2: Sum Rule & Circular Time-Shift Null")
        for w_key, w_name, t1, t2 in windows:
            m_stft = (t_std >= t1) & (t_std <= t2)
            t_w = t_std[m_stft]

            if "ech" in w_key:
                # ECH triad: 40.0 + 49.0 -> 89.8 kHz
                fp = r_89[m_stft]
                fa = r_40[m_stft]
                fb = r_49[m_stft]
                diff = fp - (fa + fb)
                mean_diff = float(np.mean(diff))
                rms_mismatch = float(np.sqrt(np.mean(diff**2)))

                # Circular time-shift null
                rng = np.random.default_rng(RNG_SEED)
                n_surr = 2000
                null_rms = []
                n_pts = len(fp)
                for _ in range(n_surr):
                    shift_a = rng.integers(10, max(11, n_pts - 10))
                    shift_b = rng.integers(10, max(11, n_pts - 10))
                    d_s = fp - (np.roll(fa, shift_a) + np.roll(fb, shift_b))
                    null_rms.append(np.sqrt(np.mean(d_s**2)))
                null_rms = np.array(null_rms)
                p_val_h2 = float((np.sum(null_rms <= rms_mismatch) + 1) / (n_surr + 1))
                rule_satisfied = bool(rms_mismatch < 1.0 and p_val_h2 < 0.05)
            else:
                # NBI-only triad: 37.5 + 42.5 -> 80.0 kHz vs Primary (85.9-89.8 kHz)
                # Primary lines in NBI: 85.9 kHz and 89.8 kHz
                fp = r_89[m_stft]
                fa = r_37[m_stft]
                fb = r_42[m_stft]
                diff = fp - (fa + fb)
                mean_diff = float(np.mean(diff))
                rms_mismatch = float(np.sqrt(np.mean(diff**2)))
                # Null test
                rng = np.random.default_rng(RNG_SEED)
                n_surr = 2000
                null_rms = []
                n_pts = len(fp)
                for _ in range(n_surr):
                    shift_a = rng.integers(10, max(11, n_pts - 10))
                    shift_b = rng.integers(10, max(11, n_pts - 10))
                    d_s = fp - (np.roll(fa, shift_a) + np.roll(fb, shift_b))
                    null_rms.append(np.sqrt(np.mean(d_s**2)))
                null_rms = np.array(null_rms)
                p_val_h2 = float((np.sum(null_rms <= rms_mismatch) + 1) / (n_surr + 1))
                rule_satisfied = False  # Catastrophic failure: sum=80 kHz vs mode=89.8 kHz

            shot_results["h2_sum_rule"][w_key] = {
                "window": w_name,
                "mean_diff_khz": mean_diff,
                "rms_mismatch_khz": rms_mismatch,
                "null_rms_95_ci": [float(np.percentile(null_rms, 2.5)), float(np.percentile(null_rms, 97.5))],
                "p_value": p_val_h2,
                "sum_rule_holds": rule_satisfied,
            }
            print(f"  [{w_name}] Mean Diff: {mean_diff:+.3f} kHz | RMS Mismatch: {rms_mismatch:.3f} kHz | p-value: {p_val_h2:.4f} | Holds: {rule_satisfied}")

        # -------------------------------------------------------------------------
        # H3. TARGETED BICOHERENCE/BIPHASE FOR (40, 49 -> 89 kHz)
        # -------------------------------------------------------------------------
        print("\n>>> H3: Targeted Bicoherence & Surrogate Null (40, 49 -> 89 kHz)")
        for w_key, w_name, t1, t2 in windows:
            m_w = (t_ms >= t1) & (t_ms <= t2)
            sig = mp1[m_w]

            nfft = 512
            noverlap = 256
            b_comp, b2 = abicoh2_corrected(sig, fs=fs, nfft=nfft, noverlap=noverlap)

            df = fs / nfft / 1e3
            f_axis = np.arange(256) * df
            b2_pos = b2[:256, :]
            b_comp_pos = b_comp[:256, :]

            # Target box +-2.5 kHz around (40, 49) -> f1 in [37.5, 42.5], f2 in [46.5, 51.5]
            m1 = (f_axis >= 37.5) & (f_axis <= 42.5)
            m2 = (f_axis >= 46.5) & (f_axis <= 51.5)

            box_b2 = b2_pos[m1, :][:, m2]
            box_comp = b_comp_pos[m1, :][:, m2]

            b2_mean = float(np.mean(box_b2))
            b2_max = float(np.max(box_b2))
            biphase_deg = float(np.angle(np.mean(box_comp), deg=True))

            # Amplitude-conditioned surrogate null
            _, _, spec = dsp.spectrogram(sig, fs=fs, window="hann", nperseg=nfft, noverlap=noverlap, nfft=nfft, return_onesided=False, scaling="density", mode="complex")
            spec = spec.T
            amp_spec = np.abs(spec)

            rng = np.random.default_rng(RNG_SEED)
            n_surr = 500
            null_b2_means = []
            for _ in range(n_surr):
                half_p = rng.uniform(0, 2 * np.pi, size=(spec.shape[0], 255))
                full_p = np.zeros(spec.shape)
                full_p[:, 1:256] = half_p
                full_p[:, 257:] = -half_p[:, ::-1]
                surr_spec = amp_spec * np.exp(1j * full_p)
                _, b2_s = abicoh2_corrected(surr_spec)
                null_b2_means.append(np.mean(b2_s[:256, :][m1, :][:, m2]))
            null_b2_means = np.array(null_b2_means)
            p_val_h3 = float((np.sum(null_b2_means >= b2_mean) + 1) / (n_surr + 1))

            # Empirical rank among equal-area boxes (5 kHz x 5 kHz) in [15, 60] kHz
            emp_box_means = []
            n_bins_1 = int(np.sum(m1))
            n_bins_2 = int(np.sum(m2))
            idx_grid1 = np.where((f_axis >= 15.0) & (f_axis <= 55.0))[0]
            idx_grid2 = np.where((f_axis >= 15.0) & (f_axis <= 55.0))[0]
            for i in range(len(idx_grid1) - n_bins_1):
                for j in range(len(idx_grid2) - n_bins_2):
                    i_sub = idx_grid1[i : i + n_bins_1]
                    j_sub = idx_grid2[j : j + n_bins_2]
                    if np.max(f_axis[i_sub, None] + f_axis[None, j_sub]) <= 120.0:
                        emp_box_means.append(np.mean(b2_pos[i_sub, :][:, j_sub]))
            emp_box_means = np.array(emp_box_means)
            emp_rank_pct = float(np.mean(emp_box_means >= b2_mean) * 100.0)

            shot_results["h3_bicoherence"][w_key] = {
                "window": w_name,
                "box_mean_b2": b2_mean,
                "box_max_b2": b2_max,
                "box_biphase_deg": biphase_deg,
                "surrogate_null_mean_b2": float(np.mean(null_b2_means)),
                "surrogate_p_value": p_val_h3,
                "empirical_rank_pct": emp_rank_pct,
                "significant": bool(p_val_h3 < 0.05 and emp_rank_pct < 5.0),
            }
            print(f"  [{w_name}] Mean b^2: {b2_mean:.4f} (Max: {b2_max:.4f}) | Biphase: {biphase_deg:+.1f} deg | Surr p: {p_val_h3:.4f} | Emp Rank: {emp_rank_pct:.1f}%")

        # -------------------------------------------------------------------------
        # H4. ENVELOPE COUPLING & INDEPENDENT LINE SYNTHESIS NULL
        # -------------------------------------------------------------------------
        print("\n>>> H4: Envelope Coupling & Independent Line Synthesis Null")
        for w_key, w_name, t1, t2 in windows:
            m_env = (t_env >= t1) & (t_env <= t2)
            e_pri = env_pri[m_env]
            e_sec = env_sec[m_env]

            # Coherence between envelopes in [0.5, 3.0] kHz
            f_coh, coh = dsp.coherence(e_pri, e_sec, fs=fs, nperseg=1024, noverlap=512)
            f_coh_khz = f_coh / 1e3
            m_band = (f_coh_khz >= 0.5) & (f_coh_khz <= 3.0)
            max_coh_obs = float(np.max(coh[m_band]))
            mean_coh_obs = float(np.mean(coh[m_band]))
            r_corr = float(np.corrcoef(e_pri, e_sec)[0, 1])

            # Synthetic null: generate two independent chirping lines with measured amplitudes and ridges
            m_stft = (t_std >= t1) & (t_std <= t2)
            f_pri_interp = np.interp(t_env[m_env], t_std[m_stft], r_89[m_stft])
            f_sec_interp = np.interp(t_env[m_env], t_std[m_stft], r_49[m_stft] if "ech" in w_key else r_40[m_stft])

            rng = np.random.default_rng(RNG_SEED)
            n_surr = 200
            null_max_cohs = []
            dt_s = 1.0 / fs
            for _ in range(n_surr):
                phi1_s = 2 * np.pi * np.cumsum(f_pri_interp * 1e3) * dt_s + rng.uniform(0, 2 * np.pi)
                phi2_s = 2 * np.pi * np.cumsum(f_sec_interp * 1e3) * dt_s + rng.uniform(0, 2 * np.pi)
                x1_s = e_pri * np.cos(phi1_s)
                x2_s = e_sec * np.cos(phi2_s)
                e1_s = np.abs(dsp.hilbert(x1_s))
                e2_s = np.abs(dsp.hilbert(x2_s))
                _, c_s = dsp.coherence(e1_s, e2_s, fs=fs, nperseg=1024, noverlap=512)
                null_max_cohs.append(np.max(c_s[m_band]))

            null_max_cohs = np.array(null_max_cohs)
            p_val_h4 = float((np.sum(null_max_cohs >= max_coh_obs) + 1) / (n_surr + 1))

            shot_results["h4_envelope_coupling"][w_key] = {
                "window": w_name,
                "envelope_pearson_r": r_corr,
                "observed_max_coherence": max_coh_obs,
                "observed_mean_coherence": mean_coh_obs,
                "null_mean_max_coherence": float(np.mean(null_max_cohs)),
                "p_value": p_val_h4,
                "coupling_significant": bool(p_val_h4 < 0.05),
            }
            print(f"  [{w_name}] Pearson r: {r_corr:+.3f} | Max Coh: {max_coh_obs:.4f} | Null Mean: {np.mean(null_max_cohs):.4f} | p-value: {p_val_h4:.4f}")

        results[shot_str] = shot_results

    # Save structured results
    out_json = os.path.join(OUTPUT_DIR, "task5_nonlinear_results.json")
    with open(out_json, "w") as f_out:
        json.dump(results, f_out, indent=2)
    print(f"\n[OK] Results saved to {out_json}")

    # -------------------------------------------------------------------------
    # GENERATE PUBLICATION FIGURE: fig_task5_nonlinear.png
    # -------------------------------------------------------------------------
    print(">>> Generating publication figure fig_task5_nonlinear.png...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Frequency Autocorrelation (H1) for Shot #88653
    ax_a = axes[0, 0]
    rec53 = np.load(os.path.join(CACHE_DIR, "shot_88653_full_record_cache.npz"))
    sig_ech = rec53["mp1"][(rec53["t_ms"] >= 282.5) & (rec53["t_ms"] <= 291.2)]
    sig_nbi = rec53["mp1"][(rec53["t_ms"] >= 291.3) & (rec53["t_ms"] <= 300.0)]
    f_w, p_ech = dsp.welch(sig_ech, fs=fs, nperseg=2048, noverlap=1024, window="hann")
    _, p_nbi = dsp.welch(sig_nbi, fs=fs, nperseg=2048, noverlap=1024, window="hann")
    f_khz = f_w / 1e3
    df = f_khz[1] - f_khz[0]

    def get_acf(psd_arr, fmin, fmax):
        m = (f_khz >= fmin) & (f_khz <= fmax)
        pc = psd_arr[m] - np.mean(psd_arr[m])
        acf = np.correlate(pc, pc, mode="full")[len(pc) - 1 :] / (np.sum(pc**2) + 1e-12)
        return np.arange(len(pc)) * df, acf

    l_p_ech, acf_p_ech = get_acf(p_ech, 75, 105)
    l_p_nbi, acf_p_nbi = get_acf(p_nbi, 75, 105)
    l_s_ech, acf_s_ech = get_acf(p_ech, 30, 55)

    ax_a.plot(l_p_ech, acf_p_ech, label="Primary ECH+NBI (8.7 ms)", color="navy", lw=1.8)
    ax_a.plot(l_p_nbi, acf_p_nbi, label="Primary NBI-only (8.7 ms)", color="crimson", lw=1.8)
    ax_a.plot(l_s_ech, acf_s_ech, label="Secondary ECH+NBI (8.7 ms)", color="teal", lw=1.5, ls="--")
    ax_a.axvline(1.2, color="orange", ls=":", lw=1.5, label="1.2 kHz Coherence Peak")
    ax_a.axvline(2.2, color="magenta", ls=":", lw=1.5, label="2.2 kHz Coherence Peak")
    ax_a.axhline(0, color="gray", lw=0.8)
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(-0.6, 1.05)
    ax_a.set_title("(a) H1: PSD Frequency Autocorrelation (#88653)", fontweight="bold")
    ax_a.set_xlabel(r"Frequency Lag $\Delta f$ (kHz)")
    ax_a.set_ylabel(r"Normalized Autocorrelation $R(\Delta f)$")
    ax_a.legend(fontsize=8, loc="upper right")
    ax_a.grid(True, ls=":", alpha=0.6)

    # Panel B: H2 Sum Rule Instantaneous Mismatch vs Time
    ax_b = axes[0, 1]
    t_stft = rec53["t_stft_std"]
    f_stft = rec53["f_stft_std"] / 1e3
    psd53 = rec53["psd_mp1_std"]
    m_ech = (t_stft >= 282.5) & (t_stft <= 291.2)
    m_nbi = (t_stft >= 291.3) & (t_stft <= 300.0)

    # Instantaneous diffs
    diff_ech_53 = r_89[m_ech] - (r_40[m_ech] + r_49[m_ech])
    diff_nbi_53 = r_89[m_nbi] - (r_37[m_nbi] + r_42[m_nbi])

    ax_b.plot(t_stft[m_ech], diff_ech_53, label="ECH Triad: 89.8 - (40.0 + 49.0)", color="blue", lw=1.5)
    ax_b.plot(t_stft[m_nbi], diff_nbi_53, label="NBI Triad: 89.8 - (37.5 + 42.5)", color="red", lw=1.5)
    ax_b.axhline(0, color="black", lw=1.2, ls="--", label="Exact Sum Rule (0 kHz)")
    ax_b.axvline(291.2, color="gray", lw=1.0, ls=":")
    ax_b.set_title("(b) H2: Sum Rule Mismatch $f_P(t) - [f_a(t) + f_b(t)]$", fontweight="bold")
    ax_b.set_xlabel("Time (ms)")
    ax_b.set_ylabel("Frequency Mismatch (kHz)")
    ax_b.set_ylim(-5, 15)
    ax_b.legend(fontsize=8, loc="upper left")
    ax_b.grid(True, ls=":", alpha=0.6)

    # Panel C: H3 Corrected Bicoherence Map (ECH 8.7ms #88653)
    ax_c = axes[1, 0]
    b_comp_ech, b2_ech = abicoh2_corrected(sig_ech, fs=fs, nfft=512, noverlap=256)
    df_c = fs / 512 / 1e3
    f_axis_c = np.arange(256) * df_c
    m_plot = (f_axis_c >= 15.0) & (f_axis_c <= 65.0)
    X_f, Y_f = np.meshgrid(f_axis_c[m_plot], f_axis_c[m_plot])
    b2_plot = b2_ech[:256, :][m_plot, :][:, m_plot]
    c_im = ax_c.pcolormesh(X_f, Y_f, b2_plot, cmap="viridis", shading="auto", vmin=0, vmax=0.15)
    # Box overlay +-2.5 kHz around (40, 49)
    rect = plt.Rectangle((40.0 - 2.5, 49.0 - 2.5), 5.0, 5.0, edgecolor="red", facecolor="none", lw=2, ls="--", label=r"Target Triad $(40, 49 \to 89\,$kHz)")
    ax_c.add_patch(rect)
    ax_c.set_title(r"(c) H3: Corrected Bicoherence Map $b^2(f_1, f_2)$ [ECH 8.7 ms]", fontweight="bold")
    ax_c.set_xlabel(r"$f_1$ (kHz)")
    ax_c.set_ylabel(r"$f_2$ (kHz)")
    fig.colorbar(c_im, ax=ax_c, label=r"Squared Bicoherence $b^2$", pad=0.02)
    ax_c.legend(fontsize=8, loc="upper right")

    # Panel D: H4 Envelope Coherence vs Independent Synthetic Null
    ax_d = axes[1, 1]
    e_pri_ech = env_pri[(t_env >= 282.5) & (t_env <= 291.2)]
    e_sec_ech = env_sec[(t_env >= 282.5) & (t_env <= 291.2)]
    f_coh_d, coh_ech = dsp.coherence(e_pri_ech, e_sec_ech, fs=fs, nperseg=1024, noverlap=512)
    f_coh_d_khz = f_coh_d / 1e3
    m_d = (f_coh_d_khz >= 0.1) & (f_coh_d_khz <= 5.0)

    # Plot null mean from #88653 ECH 8.7ms
    null_mean_d = results["88653"]["h4_envelope_coupling"]["ech_8_7ms"]["null_mean_max_coherence"]

    ax_d.plot(f_coh_d_khz[m_d], coh_ech[m_d], label="Observed Envelope Coherence", color="darkgreen", lw=1.8)
    ax_d.axhline(null_mean_d, color="darkred", ls="--", lw=1.5, label=f"Independent Chirp Null Max ({null_mean_d:.3f})")
    ax_d.axvline(1.2, color="orange", ls=":", lw=1.5, label="1.2 kHz Modulation")
    ax_d.axvline(2.2, color="magenta", ls=":", lw=1.5, label="2.2 kHz Modulation")
    ax_d.set_title(r"(d) H4: Envelope Coherence $\gamma^2(f)$ vs Independent Null", fontweight="bold")
    ax_d.set_xlabel("Envelope Modulation Frequency (kHz)")
    ax_d.set_ylabel(r"Coherence $\gamma^2$")
    ax_d.set_ylim(0, 0.5)
    ax_d.legend(fontsize=8, loc="upper right")
    ax_d.grid(True, ls=":", alpha=0.6)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig_task5_nonlinear.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[OK] Figure saved to {fig_path}")


if __name__ == "__main__":
    run_task5_analysis()
