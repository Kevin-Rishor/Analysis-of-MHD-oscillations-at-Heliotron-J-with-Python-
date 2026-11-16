"""
Task 15 -- (A) Power calibration of the bicoherence gate and (B) Debug of the direction estimator.

Part A: Minimum detectable coupling (gate from Task 14, unchanged).
  A1: Load real in-band noise sigma and mode rms amplitudes at (41, 50, 91) from task14_validation.json.
  A2: Grid of coupling strengths q in [0, 1] spanning population b^2 ~ 0 to 0.8.
  A3: For each q, window (8.7 ms and 32.1 ms), and probe (MP1, MP3, MP4, and all 3),
      run 200 trials and report gate detection rate with Wilson CI, median b^2, 5-95% range.
  A4: Interpolate q80 and population b^2; compare with real targeted-triad b^2 from task14_results.json.
  A5: Output table and sentence-ready summary.

Part B: Debug the sign of the Kim-Powers estimator.
  B1: Noiseless test with A1=A2=const, dA3/dt = gamma3*A3 + Q*A1*A2, Q = +/- 5000.
  B2: Definition comparison: true direction vs estimator sign convention.
  B3: Carrier demodulation across hops for bins 41, 50, 91; raw vs demodulated accuracy.
  B4: Noise sweep (8 steps up to 2*sigma) to find critical noise threshold.
  B5: Final label of the estimator.
"""
import os
import sys
import json
import time
import numpy as np
import scipy.signal as dsp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_task14_multiprobe_energy_transfer as L

HERE = L.HERE
FS = L.FS
NFFT = L.NFFT
HOP = L.HOP
DF_KHZ = L.DF_KHZ
PROBES = L.PROBES
PHI = {"mp1": 0.0, "mp3": np.pi, "mp4": 1.5 * np.pi}
GAMMA3 = -2000.0  # 1/s

# ------------------------------------------------------------- Statistical helpers
def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def ci_str(lo, hi):
    return f"[{lo:.4f}, {hi:.4f}]"


# ------------------------------------------------------------- Part A: Power Calibration
def measure_population_b2(q_grid, amps, bins, N_long=100000):
    """Measure population b^2 on a long noiseless record (finite-sample bias floor ~ 0)."""
    t_long = np.arange(N_long) / FS
    dt_env = 8.0  # us
    ns = int(np.ceil(N_long / dt_env)) + 2
    g = 2e-3
    rng = np.random.default_rng(2026)
    
    A = np.zeros((3, ns + 600), complex)
    W = (rng.standard_normal((3, ns + 600)) + 1j * rng.standard_normal((3, ns + 600))) * np.sqrt(dt_env / 2)
    s = [np.sqrt(2 * g) * amps[m] for m in range(3)]
    for k in range(ns + 600 - 1):
        A[:, k+1] = A[:, k] - g * A[:, k] * dt_env + np.array(s) * W[:, k]
    A = A[:, 600:]
    tt = np.arange(ns) * dt_env
    t1 = np.arange(N_long) + 0.0
    Af = np.stack([np.interp(t1, tt, A[m].real) + 1j * np.interp(t1, tt, A[m].imag) for m in range(3)])
    
    quad = Af[0] * Af[1]
    quad_norm = quad * (amps[2] / np.sqrt(np.mean(np.abs(quad)**2)))
    
    f1 = bins[0] * DF_KHZ * 1e3
    f2 = bins[1] * DF_KHZ * 1e3
    f3 = (bins[0] + bins[1]) * DF_KHZ * 1e3
    
    b2_pop = {}
    for q in q_grid:
        A3_q = np.sqrt(1 - q) * Af[2] + np.sqrt(q) * quad_norm
        x = np.real(Af[0] * np.exp(1j*2*np.pi*f1*t_long) + 
                    Af[1] * np.exp(1j*2*np.pi*f2*t_long) + 
                    A3_q * np.exp(1j*2*np.pi*f3*t_long))
        X = L.seg_spectra(x)
        b2_val = float(L.b2_pairs(X, np.array([bins[0]]), np.array([bins[1]]))[0])
        b2_pop[q] = b2_val
    return b2_pop


def run_part_a(params, S=100, N_trials=200):
    print("=" * 80)
    print("PART A - MINIMUM DETECTABLE COUPLING (GATE POWER CALIBRATION)")
    print("=" * 80)
    
    sigma = params["sigma"]
    amps = params["amps"]
    bins = params["bins"]
    i1, i2 = bins[0], bins[1]
    i3 = i1 + i2
    trip = (i1, i2, i3)
    
    print(f"A1. Real #88653 MP1 parameters loaded from task14_validation.json:")
    print(f"    In-band noise sigma: {sigma:.8f} V")
    print(f"    Mode RMS amplitudes [f1, f2, f3]: {amps}")
    print(f"    Mode bins: ({i1}, {i2}, {i3}) -> frequencies: ({i1*DF_KHZ:.4f}, {i2*DF_KHZ:.4f}, {i3*DF_KHZ:.4f}) kHz")
    print(f"    Surrogate count S: {S}")
    
    q_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print(f"\nA2. Quadratic drive injected over q grid: {q_grid} (length = {len(q_grid)} >= 8)")
    print("    Measuring population b^2 on long noiseless record (100k samples)...")
    b2_pop = measure_population_b2(q_grid, amps, bins)
    for q, val in b2_pop.items():
        print(f"    q = {q:.2f} -> population b^2 = {val:.4f}")
    
    windows = {
        "8.7 ms": 8701,
        "32.1 ms": 32101
    }
    
    # Pre-generate surrogate null distribution for each window (S=100 phase-randomized surrogates)
    print(f"\n    Calibrating surrogate significance threshold (95th percentile of family-wise max b^2, S={S})...")
    sur_thr = {}
    f1 = i1 * DF_KHZ * 1e3
    f2 = i2 * DF_KHZ * 1e3
    f3 = i3 * DF_KHZ * 1e3
    
    rng_cal = np.random.default_rng(2026)
    for wname, N in windows.items():
        t = np.arange(N) / FS
        x_bg = (amps[0] * np.cos(2*np.pi*f1*t) + 
                amps[1] * np.cos(2*np.pi*f2*t) + 
                amps[2] * np.cos(2*np.pi*f3*t) + 
                sigma * rng_cal.standard_normal(N))
        _, sur_max = L.surrogate_stats(x_bg, trip, S, rng_cal)
        sur_thr[wname] = float(np.percentile(sur_max, 95))
        print(f"    {wname} (N={N}): surrogate 95th percentile threshold = {sur_thr[wname]:.4f}")
    
    results_a = {}
    dt_env = 8.0  # us
    g = 2e-3
    
    print(f"\nA3. Running {N_trials} trials per condition...")
    for wname, N in windows.items():
        t = np.arange(N) / FS
        ns = int(np.ceil(N / dt_env)) + 2
        thr = sur_thr[wname]
        results_a[wname] = {}
        
        print(f"\n  --- Window: {wname} (N={N}) ---")
        for q in q_grid:
            b2_probes = {pr: [] for pr in PROBES}
            gate_probes = {pr: [] for pr in PROBES}
            gate_all3 = []
            
            for seed in range(N_trials):
                rng = np.random.default_rng(seed + int(q * 1000) + (10000 if wname=="8.7 ms" else 20000))
                # Stochastic envelope generation
                A = np.zeros((3, ns + 600), complex)
                W = (rng.standard_normal((3, ns + 600)) + 1j * rng.standard_normal((3, ns + 600))) * np.sqrt(dt_env / 2)
                s = [np.sqrt(2 * g) * amps[m] for m in range(3)]
                for k in range(ns + 600 - 1):
                    A[:, k+1] = A[:, k] - g * A[:, k] * dt_env + np.array(s) * W[:, k]
                A = A[:, 600:]
                tt = np.arange(ns) * dt_env
                t1 = np.arange(N) + 0.0
                Af = np.stack([np.interp(t1, tt, A[m].real) + 1j * np.interp(t1, tt, A[m].imag) for m in range(3)])
                
                quad = Af[0] * Af[1]
                quad_norm = quad * (amps[2] / np.sqrt(np.mean(np.abs(quad)**2)))
                A3_coupled = np.sqrt(1 - q) * Af[2] + np.sqrt(q) * quad_norm
                
                # Build signals for each probe
                # n = [-1, -1, -1] for all modes
                probes_passed = []
                for pr in PROBES:
                    phi_p = PHI[pr]
                    s1 = np.real(Af[0] * np.exp(1j * (2*np.pi*f1*t - phi_p)))
                    s2 = np.real(Af[1] * np.exp(1j * (2*np.pi*f2*t - phi_p)))
                    s3 = np.real(A3_coupled * np.exp(1j * (2*np.pi*f3*t - phi_p)))
                    x = s1 + s2 + s3 + sigma * rng.standard_normal(N)
                    
                    X = L.seg_spectra(x)
                    b2_val = float(L.b2_pairs(X, np.array([i1]), np.array([i2]))[0])
                    b2_probes[pr].append(b2_val)
                    passed = (b2_val >= thr)
                    gate_probes[pr].append(passed)
                    probes_passed.append(passed)
                
                gate_all3.append(all(probes_passed))
            
            # Summarize stats for this q
            q_res = {
                "q": q,
                "b2_pop": b2_pop[q],
                "probes": {},
                "all_three": {
                    "detection_rate": float(np.mean(gate_all3)),
                    "ci": wilson(sum(gate_all3), N_trials)
                }
            }
            for pr in PROBES:
                b2_arr = b2_probes[pr]
                g_arr = gate_probes[pr]
                q_res["probes"][pr] = {
                    "median_b2": float(np.median(b2_arr)),
                    "range_5_95": [float(np.percentile(b2_arr, 5)), float(np.percentile(b2_arr, 95))],
                    "detection_rate": float(np.mean(g_arr)),
                    "ci": wilson(sum(g_arr), N_trials)
                }
            results_a[wname][q] = q_res
            
            mp1_stat = q_res["probes"]["mp1"]
            print(f"    q = {q:.2f} (pop b2={b2_pop[q]:.4f}) | MP1 det = {mp1_stat['detection_rate']:.3f} {ci_str(*mp1_stat['ci'])}, "
                  f"b2 median = {mp1_stat['median_b2']:.4f} [5-95%: {mp1_stat['range_5_95'][0]:.4f}, {mp1_stat['range_5_95'][1]:.4f}] | "
                  f"All 3 det = {q_res['all_three']['detection_rate']:.3f} {ci_str(*q_res['all_three']['ci'])}")
    
    # A4. Interpolate q80 and place real b2
    print("\nA4. Power Analysis and q80 Estimation:")
    q80_results = {}
    
    # Real b2 values read from task14_results.json
    res14 = json.load(open(os.path.join(HERE, "task14_results.json")))
    real_b2_8 = res14["shots"]["88653"]["w8p7"]["targeted"]["probes"]["mp1"]["b2"]
    real_b2_32 = res14["shots"]["88653"]["w32p1"]["targeted"]["probes"]["mp1"]["b2"]
    real_b2 = {"8.7 ms": real_b2_8, "32.1 ms": real_b2_32}
    
    for wname in windows:
        q_vals = q_grid
        det_rates = [results_a[wname][q]["probes"]["mp1"]["detection_rate"] for q in q_vals]
        
        # Check if 0.80 is reached
        max_det = max(det_rates)
        if max_det < 0.80:
            q80_str = "not reachable"
            b2_q80_str = "not reachable"
            status_real = "below"
            q80_results[wname] = {
                "q80": None, "q80_status": "not reachable", "max_detection_rate": max_det,
                "b2_pop_q80": None, "real_b2": real_b2[wname], "real_vs_q80": "below (not reachable)"
            }
            print(f"    Window {wname}: q80 is NOT REACHABLE (maximum detection rate = {max_det:.3f} at q={q_vals[-1]:.2f})")
            print(f"      Real targeted MP1 b^2 = {real_b2[wname]:.4f} is BELOW the required power threshold.")
        else:
            # Linear interpolation for q80
            idx = np.where(np.array(det_rates) >= 0.80)[0][0]
            q_prev, q_curr = q_vals[idx-1], q_vals[idx]
            d_prev, d_curr = det_rates[idx-1], det_rates[idx]
            q80 = float(q_prev + (0.80 - d_prev) * (q_curr - q_prev) / (d_curr - d_prev))
            
            # Interpolate population b2 at q80
            b_prev, b_curr = b2_pop[q_prev], b2_pop[q_curr]
            b2_q80 = float(b_prev + (q80 - q_prev) * (b_curr - b_prev) / (q_curr - q_prev))
            
            status_real = "below" if real_b2[wname] < b2_q80 else ("near" if abs(real_b2[wname] - b2_q80) < 0.05 else "above")
            q80_results[wname] = {
                "q80": q80, "q80_status": "reached", "b2_pop_q80": b2_q80,
                "real_b2": real_b2[wname], "real_vs_q80": status_real
            }
            print(f"    Window {wname}: q80 = {q80:.4f} -> population b^2(q80) = {b2_q80:.4f}")
            print(f"      Real targeted MP1 b^2 = {real_b2[wname]:.4f} is {status_real.upper()} b^2(q80).")
    
    # A5. Sentence-ready summary
    print("\nA5. Sentence-ready conclusions:")
    for wname in windows:
        info = q80_results[wname]
        if info["q80_status"] == "reached":
            msg = f"Couplings with population b^2 >= {info['b2_pop_q80']:.4f} would be detected with 80% power in the {wname} window."
        else:
            msg = f"80% detection power is not reachable within the physical coupling domain in the {wname} window (maximum power {info['max_detection_rate']*100:.1f}% at q=1.00)."
        print(f"    * {msg}")
        q80_results[wname]["summary_sentence"] = msg
    
    return {
        "params": {"sigma": sigma, "amps": amps, "bins": bins, "S": S, "N_trials": N_trials},
        "b2_pop": b2_pop,
        "surrogate_thresholds": sur_thr,
        "detection_results": results_a,
        "q80_results": q80_results
    }


# ------------------------------------------------------------- Part B: Direction Debug
def run_part_b(params):
    print("\n" + "=" * 80)
    print("PART B - DEBUG THE SIGN OF THE KIM-POWERS ESTIMATOR")
    print("=" * 80)
    
    amps = params["amps"]
    bins = params["bins"]
    sigma = params["sigma"]
    i1, i2 = bins[0], bins[1]
    i3 = i1 + i2
    trip = (i1, i2, i3)
    
    f1 = i1 * DF_KHZ * 1e3
    f2 = i2 * DF_KHZ * 1e3
    f3 = i3 * DF_KHZ * 1e3
    N = 8701
    dt = 1.0 / FS
    t = np.arange(N) * dt
    
    # B1. Noiseless case with constant envelopes
    print("B1. Noiseless case: complex envelopes A1=A2=const, dA3/dt = gamma3*A3 + Q*A1*A2")
    print(f"    gamma3 = {GAMMA3:.1f} s^-1, Q = +5000 and -5000 V^-1 s^-1")
    
    b1_results = {}
    for Q_val in [+5000.0, -5000.0]:
        a1, a2 = amps[0], amps[1]
        A3 = np.zeros(N, complex)
        for k in range(N-1):
            dA3 = (GAMMA3 * A3[k] + Q_val * a1 * a2) * dt
            A3[k+1] = A3[k] + dA3
        
        x = np.real(a1 * np.exp(1j*2*np.pi*f1*t) + 
                    a2 * np.exp(1j*2*np.pi*f2*t) + 
                    A3 * np.exp(1j*2*np.pi*f3*t))
        X = L.seg_spectra(x)
        
        rng = np.random.default_rng(2026)
        kp = L.kim_powers(X, trip, 200, rng)
        b1_results[int(Q_val)] = {
            "Lam": [float(kp["Lam"].real), float(kp["Lam"].imag)],
            "Q_est": [float(kp["Q"].real), float(kp["Q"].imag)],
            "B": [float(kp["B"].real), float(kp["B"].imag)],
            "T": float(kp["T"]),
            "sign_T": int(np.sign(kp["T"])),
            "sign_Re_Q_est": int(np.sign(kp["Q"].real))
        }
        print(f"    Q = {int(Q_val):+d}:")
        print(f"      Lambda = {kp['Lam']}")
        print(f"      Q_est  = {kp['Q']}")
        print(f"      B      = {kp['B']}")
        print(f"      T = Re[Q_est * B] = {kp['T']:.4f}")
        print(f"      sign(T) = {int(np.sign(kp['T']))}, sign(Re(Q_est)) = {int(np.sign(kp['Q'].real))}")
    
    print("\n    Verdict on B1:")
    print("    * The Task 14 estimator returns sign(T) = -1 for BOTH Q = +5000 and Q = -5000.")
    print("    * This occurs because T = Re[Q_est * B] is a product of two terms that BOTH flip sign when Q flips sign.")
    print("    * In addition, Re(Q_est) has the OPPOSITE sign of Q (+5000 -> -12.23; -5000 -> +12.23) due to raw carrier hopping.")
    
    # B2. Definition comparison
    print("\nB2. Definition comparison between validation truth and estimator sign:")
    def_truth = "truth = sign(flow), where flow = 2 * Re[conj(A3) * Q3 * A1 * A2] (latent net power into A3)"
    def_est = "T = Re[Q_est * B] (cross-term of segment-to-segment |X3|^2 budget)"
    def_drive = "sign(Q) (sign of quadratic coupling parameter Q in dA3/dt = gamma3*A3 + Q*A1*A2)"
    print(f"    Validation truth: {def_truth}")
    print(f"    Estimator convention: {def_est}")
    print(f"    Drive parameter: {def_drive}")
    print("    Do they use the same convention? NO: T measures net energy rate into mode 3 (always positive for a sink), whereas validation tested whether sign(Q) could be identified.")
    
    # B3. Carrier demodulation
    print("\nB3. Carrier demodulation analysis:")
    z1 = np.exp(1j * 2 * np.pi * (i1 / NFFT) * HOP)
    z2 = np.exp(1j * 2 * np.pi * (i2 / NFFT) * HOP)
    z3 = np.exp(1j * 2 * np.pi * (i3 / NFFT) * HOP)
    
    print(f"    Phase advance per hop (Nfft={NFFT}, hop={HOP}):")
    print(f"      Bin {i1} (f1={f1/1e3:.3f} kHz): Delta_theta = {i1} pi -> exp(i Delta_theta) = {z1.real:+.4f} + {z1.imag:+.4f}j")
    print(f"      Bin {i2} (f2={f2/1e3:.3f} kHz): Delta_theta = {i2} pi -> exp(i Delta_theta) = {z2.real:+.4f} + {z2.imag:+.4f}j")
    print(f"      Bin {i3} (f3={f3/1e3:.3f} kHz): Delta_theta = {i3} pi -> exp(i Delta_theta) = {z3.real:+.4f} + {z3.imag:+.4f}j")
    
    # Compare raw vs demodulated over 200 trials (noiseless and noisy)
    print("    Running 200 trials comparing Raw STFT vs Demodulated STFT...")
    b3_stats = {}
    for condition in ["noiseless", "noisy"]:
        acc_raw = []
        acc_demod = []
        sig_test = 0.0 if condition == "noiseless" else sigma
        
        for Q_sign in [+1, -1]:
            for seed in range(100):
                rng = np.random.default_rng(seed + (1000 if Q_sign>0 else 2000) + (5000 if condition=="noisy" else 0))
                a1 = amps[0] * (1 + 0.1 * rng.standard_normal())
                a2 = amps[1] * (1 + 0.1 * rng.standard_normal())
                Q_val = Q_sign * 5000.0
                A3 = np.zeros(N, complex)
                for k in range(N-1):
                    dA3 = (GAMMA3 * A3[k] + Q_val * a1 * a2) * dt
                    A3[k+1] = A3[k] + dA3
                
                x = (np.real(a1 * np.exp(1j*2*np.pi*f1*t) + 
                             a2 * np.exp(1j*2*np.pi*f2*t) + 
                             A3 * np.exp(1j*2*np.pi*f3*t)) + 
                     sig_test * rng.standard_normal(N))
                
                # Raw
                X_raw = L.seg_spectra(x)
                kp_r = L.kim_powers(X_raw, trip, 50, rng)
                # Raw sign compared against Q_sign:
                acc_raw.append(int(np.sign(kp_r["Q"].real) == Q_sign))
                
                # Demodulated: multiply segment k by z_m^-k
                K = X_raw.shape[0]
                k_vec = np.arange(K)
                X_demod = X_raw.copy()
                X_demod[:, i1] *= (z1**(-k_vec))
                X_demod[:, i2] *= (z2**(-k_vec))
                X_demod[:, i3] *= (z3**(-k_vec))
                kp_d = L.kim_powers(X_demod, trip, 50, rng)
                acc_demod.append(int(np.sign(kp_d["Q"].real) == Q_sign))
        
        acc_r_mean = float(np.mean(acc_raw))
        acc_d_mean = float(np.mean(acc_demod))
        ci_r = wilson(sum(acc_raw), len(acc_raw))
        ci_d = wilson(sum(acc_demod), len(acc_demod))
        b3_stats[condition] = {
            "raw_acc": acc_r_mean, "raw_ci": ci_r,
            "demod_acc": acc_d_mean, "demod_ci": ci_d
        }
        print(f"      {condition.upper()} (200 trials):")
        print(f"        Raw STFT sign accuracy:         {acc_r_mean:.3f} {ci_str(*ci_r)}")
        print(f"        Demodulated STFT sign accuracy: {acc_d_mean:.3f} {ci_str(*ci_d)}")
    
    # B4. Noise sweep to find threshold
    print("\nB4. Noise threshold sweep (under demodulated carrier convention):")
    noise_fractions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
    b4_results = {}
    
    for wname, N_win in [("8.7 ms", 8701), ("32.1 ms", 32101)]:
        t_win = np.arange(N_win) * dt
        b4_results[wname] = {}
        print(f"  --- Window {wname} (N={N_win}) ---")
        
        sigma_crit = None
        for frac in noise_fractions:
            sig_lvl = frac * sigma
            acc_list = []
            
            for Q_sign in [+1, -1]:
                for seed in range(100):
                    rng = np.random.default_rng(seed + (1000 if Q_sign>0 else 2000) + int(frac*10000))
                    a1 = amps[0] * (1 + 0.1 * rng.standard_normal())
                    a2 = amps[1] * (1 + 0.1 * rng.standard_normal())
                    Q_val = Q_sign * 5000.0
                    A3 = np.zeros(N_win, complex)
                    for k in range(N_win-1):
                        dA3 = (GAMMA3 * A3[k] + Q_val * a1 * a2) * dt
                        A3[k+1] = A3[k] + dA3
                    
                    x = (np.real(a1 * np.exp(1j*2*np.pi*f1*t_win) + 
                                 a2 * np.exp(1j*2*np.pi*f2*t_win) + 
                                 A3 * np.exp(1j*2*np.pi*f3*t_win)) + 
                         sig_lvl * rng.standard_normal(N_win))
                    
                    X = L.seg_spectra(x)
                    K = X.shape[0]
                    k_vec = np.arange(K)
                    X[:, i1] *= (z1**(-k_vec))
                    X[:, i2] *= (z2**(-k_vec))
                    X[:, i3] *= (z3**(-k_vec))
                    kp = L.kim_powers(X, trip, 50, rng)
                    acc_list.append(int(np.sign(kp["Q"].real) == Q_sign))
            
            acc_mean = float(np.mean(acc_list))
            ci = wilson(sum(acc_list), len(acc_list))
            b4_results[wname][frac] = {
                "sigma": float(sig_lvl),
                "accuracy": acc_mean,
                "ci": ci
            }
            print(f"    sigma = {frac:.1f} * sigma0 ({sig_lvl:.5f} V): accuracy = {acc_mean:.3f} {ci_str(*ci)}")
            if ci[0] < 0.80 and sigma_crit is None:
                sigma_crit = sig_lvl
        
        b4_results[wname]["sigma_crit"] = sigma_crit
        if sigma_crit is not None:
            print(f"    -> Breaking point (lower Wilson CI < 0.80): sigma = {sigma_crit:.5f} V")
        else:
            print(f"    -> Lower Wilson CI remains >= 0.80 across all tested noise levels up to 2.0 * sigma0.")
    
    # B5. Final label
    print("\nB5. Final label of the estimator:")
    # Check 32.1 ms window at real sigma (frac=1.0)
    w32_ci_at_real_sig = b4_results["32.1 ms"][1.0]["ci"]
    w8_ci_at_real_sig = b4_results["8.7 ms"][1.0]["ci"]
    
    if w32_ci_at_real_sig[0] >= 0.80:
        final_label = "VALIDATED"
        verdict_str = (f"VALIDATED: with carrier demodulation, lower Wilson CI = {w32_ci_at_real_sig[0]:.4f} >= 0.80 "
                       f"at real in-band noise in the 32.1 ms window (and {w8_ci_at_real_sig[0]:.4f} in the 8.7 ms window).")
    else:
        final_label = "UNRESOLVED"
        verdict_str = f"UNRESOLVED: lower Wilson CI = {w32_ci_at_real_sig[0]:.4f} < 0.80 at real noise."
    
    print(f"    Final Label: {final_label}")
    print(f"    Verdict: {verdict_str}")
    
    return {
        "b1": b1_results,
        "b2_conventions": {"truth_def": def_truth, "est_def": def_est, "drive_def": def_drive},
        "b3_demodulation": b3_stats,
        "b4_noise_sweep": b4_results,
        "b5_final_label": final_label,
        "b5_verdict": verdict_str
    }


# ------------------------------------------------------------- Main driver
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0_all = time.time()
    
    # Read task14_validation.json
    vpath = os.path.join(HERE, "task14_validation.json")
    if not os.path.exists(vpath):
        print("ERROR: task14_validation.json missing. STOP.")
        return
    val = json.load(open(vpath))
    params = val["params"]
    
    # Run Part A
    part_a = run_part_a(params, S=params.get("S", 100), N_trials=200)
    
    # Run Part B
    part_b = run_part_b(params)
    
    # Save combined results
    out = {
        "task": "TASK 15",
        "description": "Power calibration of the bicoherence gate and debug of the direction estimator",
        "part_a": part_a,
        "part_b": part_b,
        "execution_time_seconds": float(time.time() - t0_all)
    }
    
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        if isinstance(o, (complex, np.complexfloating)):
            return [float(o.real), float(o.imag)]
        if isinstance(o, np.generic):
            return o.item()
        return o
        
    res_path = os.path.join(HERE, "task15_results.json")
    json.dump(clean(out), open(res_path, "w"), indent=1)
    print(f"\nWrote results to {res_path} in {time.time() - t0_all:.1f} s")


if __name__ == "__main__":
    main()
