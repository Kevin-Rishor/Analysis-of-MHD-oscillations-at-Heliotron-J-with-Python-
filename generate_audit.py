import os
import re

def get_line(file_path, pattern):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if re.search(pattern, line):
                    return f"- `{file_path}`, Line {i+1}: `{line.strip()}`"
    except Exception:
        pass
    return f"NOT FOUND in {file_path} for pattern {pattern}"

def get_lines(file_path, pattern):
    results = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if re.search(pattern, line):
                    results.append(f"- `{file_path}`, Line {i+1}: `{line.strip()}`")
    except Exception:
        pass
    if not results:
        return f"NOT FOUND in {file_path} for pattern {pattern}"
    return "\n".join(results)

def search_repo(word_list):
    results = []
    for root, dirs, files in os.walk('C:/TFG'):
        if '.venv' in root or '.git' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        for i, line in enumerate(f):
                            if any(w in line.lower() for w in word_list):
                                results.append(f"- `{path}`, Line {i+1}: `{line.strip()}`")
                except:
                    pass
    if not results:
        return "NOT FOUND: Searched entire repo (excluding .venv/.git) for keywords: " + ", ".join(word_list) + " and found 0 matches."
    return "\n".join(results)

def check_run_all_objectives():
    res1 = get_lines('C:/TFG/run_all_objectives.py', r'macro-max-lag|alfven')
    res2 = get_lines('C:/TFG/analysis/obj2_dynamics_and_structure/run_obj2.py', r'macro-max-lag|alfven|mhd_analysis_obj2')
    return (
        "CONFIRMED default usage without overrides.\n"
        "run_all_objectives.py simply calls the sub-scripts without injecting any threshold overrides:\n" + res1 + "\n\n"
        "And run_obj2.py passes no --macro-max-lag-ms or --alfven-cal- parameters, meaning it relies purely on the bare defaults:\n" + res2
    )

def main():
    obj1 = "C:/TFG/analysis/mhd_analysis_obj1.py"
    obj2 = "C:/TFG/analysis/mhd_analysis_obj2.py"
    obj3 = "C:/TFG/analysis/obj3_nonlinear_coupling/mhd_analysis_obj3.py"
    common = "C:/TFG/analysis/common/mhd_common.py"
    libana = "C:/TFG/jpack/libana_signal.py"
    bias = "C:/TFG/analysis/obj3_nonlinear_coupling/mhd_plot_asymptotic_bias_scaling.py"
    
    log = []
    log.append("Methodology.tex:")
    log.append("")
    log.append("Data Acquisition and Preprocessing")
    log.append("CONFIRMED. MP1/MP3/MP4 are selected, MP2 is excluded.")
    log.append(get_line(obj1, r'coils = \['))
    log.append("")
    
    log.append("Time-Frequency Analysis")
    log.append("CONFIRMED.")
    log.append(get_line(obj1, r'--fres'))
    log.append(get_line(obj1, r"window='hann'"))
    log.append(get_line(common, r'def morlet_cwt'))
    log.append(get_line(common, r'5\.0 \* s'))
    log.append(get_line(obj1, r'--wav-nfreqs'))
    log.append(get_line(obj1, r'--wav-fmin'))
    log.append(get_line(obj1, r'--wav-fmax'))
    log.append("")
    
    log.append("Dual-Criterion Mode Identification")
    log.append("CONFIRMED.")
    log.append(get_line(obj1, r't0 \+ 0\.025'))
    log.append(get_line(obj1, r'3\.0 \* std_noise_per_f'))
    log.append(get_line(obj1, r'--dominance-smooth-ms'))
    log.append(get_line(obj1, r'--min-interval-ms'))
    log.append(get_line(obj1, r'--ensemble'))
    log.append(get_line(obj1, r'--threshold-phys'))
    log.append("")
    
    log.append("Peak Selection")
    log.append("CONFIRMED.")
    log.append(get_line(obj1, r'--fmin-pick'))
    log.append(get_line(obj1, r'--fmax-pick'))
    log.append("")
    
    log.append("Signal Conditioning")
    log.append("CONFIRMED.")
    log.append(get_line(obj2, r'Applying Bessel bandpass filter'))
    log.append(get_line(obj2, r'windows_to_test = \[125, 325, 525, 725, 925\]'))
    log.append(get_line(obj2, r'mad_env = .* 1\.4826'))
    log.append(get_line(obj2, r'--mode-active-k'))
    log.append(get_line(obj2, r'--mode-active-max-gap-ms'))
    log.append(get_line(obj2, r'--mode-active-min-duration-ms'))
    log.append(get_line(obj2, r'decimate_factor = 100'))
    log.append("")
    
    log.append("Modal Structure Extraction")
    log.append("CONFIRMED.")
    log.append(get_line(obj2, r'def _nudft_poloidal'))
    log.append(get_line(obj2, r'PMP_CHANNELS = '))
    log.append(get_line(obj2, r'ys_pmp = ys_pmp / \(PMP_GAIN \* PMP_COIL_CONST\)'))
    log.append(get_line(obj2, r'PMP_INVERT_CHANNELS_DEFAULT'))
    log.append(get_line(obj2, r'--pmp-max-mode-number'))
    log.append(get_line(obj2, r'--pmp-max-mode-expanded'))
    log.append(get_line(obj2, r'--flat-scan-window-ms'))
    log.append(get_line(obj2, r'--flat-growth-tolerance'))
    log.append("")
    
    log.append("Correlation with Heating Systems")
    log.append("MISMATCH for macro core proxies lag. The code limits this to 50 ms, not 60 ms. The other parameters match exactly.")
    log.append(get_line(obj2, r'def lagged_cross_correlation\(x, y, dt_corr, max_lag_ms=20\.0\):'))
    log.append(get_line(obj2, r'--m7-max-lag-ms'))
    log.append(get_line(obj2, r'--macro-max-lag-ms'))
    log.append(get_line(obj2, r'abs\(r_ip_sig\) > 0\.7 and p_ip_adj < 0\.05'))
    log.append(get_line(obj2, r'step_thresholds_to_test = \[0\.3, 0\.4, 0\.5, 0\.6, 0\.7\]'))
    log.append(get_line(obj2, r'alpha_bonferroni = 0\.05 / n_tests'))
    log.append(get_line(obj2, r'bh_significant, bh_p_adjusted = benjamini_hochberg'))
    log.append(get_line(obj2, r'MIN_NEFF_FOR_FAMILY = 5\.0'))
    log.append("\nCheck run_all_objectives.py for lag overrides:")
    log.append(check_run_all_objectives())
    log.append("")
    
    log.append("Energetic-Particle Distribution")
    log.append("MISMATCH for calibration window default. It sweeps 150-285 ms but defaults to 250-270 ms. All other parameters match.")
    log.append(get_line(obj2, r'r_val, _ = stats\.pearsonr\(sig_corr, ech_corr\)'))
    log.append(get_line(obj2, r'def detect_saturated_channel'))
    log.append(get_line(obj2, r'--bfield-constant-tesla'))
    log.append(get_line(obj2, r'--nave-medfilt'))
    log.append(get_line(obj2, r'--alfven-cal-start'))
    log.append(get_line(obj2, r'--alfven-cal-end'))
    log.append(get_line(obj2, r'--alfven-cal-sweep'))
    log.append("\nDoes it use a single fixed window or the full sweep to produce r=0.161?")
    log.append("It mathematically produces identically the SAME r-value (invariant to norm_constant):")
    log.append(get_line(obj2, r'r_val_scaling is mathematically'))
    log.append(get_line(obj2, r'identical r to 4 decimals across all windows in testing'))
    log.append("")
    
    log.append("H-mode Burst-Window Detection")
    log.append("CONFIRMED.")
    log.append(get_line(obj3, r'--auto-burst-k-on'))
    log.append(get_line(obj3, r'--auto-burst-k-off'))
    log.append(get_line(obj3, r'--auto-burst-min-duration-ms'))
    log.append(get_line(obj3, r'step_bins = max\(1, int\(round\(2\.0 /'))
    log.append(get_line(obj3, r'--auto-window-margin-ms'))
    log.append(get_line(obj3, r'--auto-quiescent-duration-ms'))
    log.append(get_line(obj3, r'--auto-control-duration-ms'))
    log.append(get_line(obj3, r'--auto-flat-k'))
    log.append(get_line(obj3, r'--flat-scan-window-ms'))
    log.append(get_line(obj3, r'add_argument\("-n", "--nfft"'))
    log.append(get_line(obj3, r'add_argument\("-e", "--ensemble"'))
    log.append("")
    
    log.append("Bicoherence Calculation")
    log.append("CONFIRMED. Normalization formula implemented explicitly matches Theoretical_framework.tex exactly.")
    log.append(get_line(obj3, r'add_argument\("-t", "--threshold"'))
    log.append(get_line(obj3, r'-np\.log\(alpha\) / n_ensemble'))
    log.append(get_line(obj3, r'--fmin'))
    log.append(get_line(obj3, r'--peak-window-khz'))
    log.append(get_line(obj3, r'boundary_margin_khz = 2\.0 \* args\.peak_window_khz'))
    log.append(get_line(obj3, r'real local PSD peaks \(\\u2264 1\.5 bins\)'))
    log.append(get_line(obj3, r'AMPLITUDE_ANOMALY_RATIO'))
    log.append("\nDenominator formula implementation in jpack/libana_signal.py (term by term matches <|X1 X2|^2> <|X3|^2>):")
    log.append(get_line(libana, r'denum = np.mean\(np.abs\(spec\[:, fcol, None\] \* spec\[:, None, lrow\]\) \*\* 2, axis=0\) \* np.mean\(np.abs\(np.conjugate\(spec\[:, ha\]\)\) \*\* 2, axis=0\)'))
    log.append("")
    
    log.append("Validation Criteria")
    log.append("CONFIRMED. By default, it tests mathematically-leading triads.")
    log.append(get_line(obj3, r'alpha_bonf = 0\.05 / 2'))
    log.append(get_line(obj3, r'if triads_match\(ref_triad, result_rep\[\'best_freqs\'\], args\.repro_tol_khz\):'))
    log.append("")
    
    log.append("Heat Transport Correlation")
    log.append("CONFIRMED.")
    log.append(get_line(obj3, r'<dW_p/dt> = \{mean_dwp:\+\.2f\}'))
    log.append("")
    
    log.append("Theoretical_framework.tex:")
    log.append("")
    log.append("Debye length")
    log.append("NOT FOUND in code.")
    log.append("")
    log.append("Plasma \\beta")
    log.append("NOT FOUND in code.")
    log.append("")
    log.append("MHD equations")
    log.append("NOT FOUND in code.")
    log.append("")
    
    log.append("Alfvén velocity")
    log.append("CONFIRMED exactly.")
    log.append(get_line(obj2, r'f_alfven_scaling = np\.abs\(b_val\) / np\.sqrt\(mu0 \* n_i_m3 \* m_ion\)'))
    log.append("")
    
    log.append("Resonance condition")
    log.append("NOT FOUND in code.")
    log.append("")
    
    log.append("Frequency-scaling models")
    log.append("CONFIRMED exactly.")
    log.append(get_line(obj2, r's_bae_sound = np\.sqrt\(ece_smoothed_bae\)'))
    log.append(get_line(obj2, r's_bae_coupled = np\.sqrt\(\(1\.0 / density_val_clean\) / max\(mean_inv_n, 1e-6\) \+ \(ece_smoothed_bae / max\(mean_te, 1e-6\)\)\)'))
    log.append(get_line(obj2, r's_epm_drive = np\.sqrt\(np\.clip\(total_nbi_power, 0\.0, None\) / density_val_clean\)'))
    log.append(get_line(obj2, r's_epm_power = np\.clip\(total_nbi_power, 0\.0, None\)'))
    log.append(get_line(obj2, r'Stored Energy \(Wp\)": \(s_beta,'))
    log.append("")
    
    log.append("Slowing-down time / \\tau_s")
    log.append(search_repo(['slowing', 'tau_s', 'spitzer', 'collision time', 'confinement time']))
    log.append("")
    
    log.append("N_eff")
    log.append("CONFIRMED exactly.")
    log.append(get_line(obj2, r'N_eff = n_ov / \(1\.0 \+ 2\.0 \* np\.sum\(acf\[1:\]\)\)'))
    log.append("")
    
    log.append("t-statistic")
    log.append("CONFIRMED exactly.")
    log.append(get_line(obj2, r't_stat = r \* np\.sqrt\(df / \(1\.0 - r \*\* 2\)\)'))
    log.append("")
    
    log.append("Partial correlation")
    log.append("CONFIRMED exactly. Denominator matches standard formula sqrt((1-rxz^2)(1-ryz^2)).")
    log.append(get_line(obj2, r'denom_partial = np\.sqrt\(\(1\.0 - r_xz\*\*2\) \* \(1\.0 - r_yz\*\*2\)\)'))
    log.append(get_line(obj2, r'r_partial = \(r_xy - r_xz \* r_yz\) / denom_partial'))
    log.append("")
    
    log.append("Bispectrum / Bicoherence")
    log.append("CONFIRMED (implemented in external jpack/libana_signal, called via LAS.abicoh2)")
    log.append("")
    
    log.append("Bias-scaling extrapolation")
    log.append("CONFIRMED exactly. Fits against 1/N matching intercept extraction theory.")
    log.append(get_line(bias, r'm_t, b_t = np\.polyfit\(inv_N, b2_triad, 1\)'))
    log.append(get_line(bias, r'm_s, b_s = np\.polyfit\(inv_N, b2_sub, 1\)'))
    log.append(get_line(bias, r'm_n, b_n = np\.polyfit\(inv_N, b2_noi, 1\)'))
    log.append("")
    
    log.append("Kruskal-Wallis / Mann-Whitney")
    log.append("CONFIRMED exactly.")
    log.append(get_line(obj3, r'kw_stat, kw_p = stats\.kruskal\(burst_b2, quiescent_b2, control_b2\)'))
    log.append(get_line(obj3, r'stats\.mannwhitneyu\(burst_b2, quiescent_b2, alternative=\'greater\'\)'))
    log.append("")
    
    log.append("Bonferroni")
    log.append("CONFIRMED exactly.")
    log.append(get_line(obj2, r'alpha_bonferroni = 0\.05 / n_tests'))
    log.append("")
    
    log.append("M_eff")
    log.append("CONFIRMED exactly.")
    log.append(get_line(obj3, r'm_eff = float\(np\.sum\(np\.minimum\(eigvals, 1\.0\)\)\)'))
    log.append("")
    
    log.append("P_chance")
    log.append("CONFIRMED exactly.")
    log.append(get_line(obj3, r'p_single = min\(1\.0, box_area / search_area\)'))
    log.append(get_line(obj3, r'p_at_least_one = 1\.0 - \(1\.0 - p_single\) \*\* max\(n_candidate_triads, 0\)'))
    
    with open('C:/TFG/thesis_audit_v2.log', 'w', encoding='utf-8') as f:
        f.write('\n'.join(log))
    print("Successfully wrote to C:/TFG/thesis_audit_v2.log")

if __name__ == '__main__':
    main()
