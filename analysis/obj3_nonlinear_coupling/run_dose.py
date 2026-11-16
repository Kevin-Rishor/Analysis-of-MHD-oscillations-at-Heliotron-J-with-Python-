import sys
import time
import hashlib
import traceback
from pathlib import Path
import numpy as np
import scipy.signal as dsp

t_start_wall = time.time()

ROOT_DIR = Path(__file__).resolve().parents[2]
out_dir = ROOT_DIR / "scratch" / "final18"
out_dir.mkdir(parents=True, exist_ok=True)
stdout_file = out_dir / "dose_stdout.txt"
stdout_h = open(stdout_file, "w", encoding="utf-8")

def p_out(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=stdout_h)

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

p_out("========================================================================================================================")
p_out("TASK FINAL18 - STEP 1: MODE POWER DOSE-RESPONSE & HARMONIC CORRELATIONS")
p_out("========================================================================================================================")

try:
    # -------------------------------------------------------------------------
    # Reusing unchanged data loading, spectrogram call, and direct b2 triad helper
    # -------------------------------------------------------------------------
    root_dir = ROOT_DIR
    for p_add in [
        root_dir / "jpack",
        root_dir / "analysis",
        root_dir / "analysis" / "common",
        root_dir / "analysis" / "heating_phase_comparison",
    ]:
        if str(p_add) not in sys.path:
            sys.path.append(str(p_add))

    from comparison_common import load_channel
    staging_common = ROOT_DIR / "analysis" / "common"
    sys.path.insert(0, str(staging_common))

    t_ms_53, ys_mp1_53, dt_53, fs_53 = load_channel(88653, "MP1")
    t_ms_54, ys_mp1_54, dt_54, fs_54 = load_channel(88654, "MP1")

    nfft = 512
    nhalf = 256
    hop = 256
    noverlap = 256
    n_ens = 30
    sig_len = 256 * 29 + 512  # 7936 samples = 7.936 ms
    df_khz = (fs_53 / nfft) / 1e3  # 1.953125 kHz

    def compute_direct_triad(spec, ka, kb):
        kb_idx = kb % nfft
        kc = (ka + kb) % nfft
        z_s = spec[:, ka] * spec[:, kb_idx] * np.conj(spec[:, kc])
        prod_ab = spec[:, ka] * spec[:, kb_idx]
        spec_c = spec[:, kc]
        num = np.abs(np.mean(z_s)) ** 2
        den = np.mean(np.abs(prod_ab) ** 2) * np.mean(np.abs(spec_c) ** 2)
        b2 = float(num / den) if den > 0 else 0.0
        return b2

    def compute_family(spec, ka_min, ka_max):
        ka_arr = np.arange(ka_min, ka_max + 1, dtype=int)
        kc_arr = 2 * ka_arr
        prod_s = spec[:, ka_arr] ** 2
        z_s = prod_s * np.conj(spec[:, kc_arr])
        num = np.abs(np.mean(z_s, axis=0)) ** 2
        den = np.mean(np.abs(prod_s) ** 2, axis=0) * np.mean(np.abs(spec[:, kc_arr]) ** 2, axis=0)
        b2_arr = np.where(den > 0, num / den, 0.0)
        top_i = int(np.argmax(b2_arr))
        return float(b2_arr[top_i]), int(ka_arr[top_i]), b2_arr

    # State classifier based on experimental log constants
    ECH_on, ECH_off = 240.0, 291.2
    NBI_on, NBI_off = 223.2, 300.4

    def classify_state(start_ms, end_ms):
        if end_ms < NBI_on:
            return "pre-NBI"
        elif start_ms >= NBI_on and end_ms <= ECH_on:
            return "NBI-only (pre-ECH)"
        elif start_ms >= ECH_on and end_ms <= ECH_off:
            return "ECH+NBI"
        elif start_ms >= ECH_off and end_ms <= NBI_off:
            return "NBI-only (post-ECH)"
        elif start_ms >= NBI_off:
            return "post-NBI"
        else:
            return "transition"

    # Rank function with average ties implementation
    def rankdata_average(a):
        a = np.asarray(a)
        n = len(a)
        order = np.argsort(a)
        ranks = np.empty(n, dtype=float)
        sorted_a = a[order]
        i = 0
        while i < n:
            j = i
            while j < n and sorted_a[j] == sorted_a[i]:
                j += 1
            avg_rank = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg_rank
            i = j
        return ranks

    def spearman_rho(x, y):
        rx = rankdata_average(x)
        ry = rankdata_average(y)
        rx_m = rx - np.mean(rx)
        ry_m = ry - np.mean(ry)
        denom = np.sqrt(np.sum(rx_m ** 2) * np.sum(ry_m ** 2))
        if denom == 0.0:
            return 0.0
        return float(np.sum(rx_m * ry_m) / denom)

    # Permutation test for single shot
    def run_permutation_test_single(L, Y, n_perm=10000, seed=1):
        obs_rho = spearman_rho(L, Y)
        rL = rankdata_average(L)
        rY = rankdata_average(Y)
        rL_m = rL - np.mean(rL)
        rY_m = rY - np.mean(rY)
        denom = np.sqrt(np.sum(rL_m ** 2) * np.sum(rY_m ** 2))
        if denom == 0.0:
            return obs_rho, 0, 1.0
        rng = np.random.default_rng(seed)
        perm_rL = np.array([rng.permutation(rL) for _ in range(n_perm)])
        perm_rL_m = perm_rL - np.mean(rL)
        perm_rhos = (perm_rL_m @ rY_m) / denom
        k = int(np.sum(perm_rhos >= obs_rho))
        p_val = k / n_perm
        return obs_rho, k, p_val

    # Window analysis processor for a shot and offset
    def process_windows(shot_id, t_ms, ys, fs, offset):
        n_windows = (len(ys) - offset) // sig_len
        records = []
        bg_bins = np.concatenate([np.arange(72, 90), np.arange(95, 113)])
        for w in range(n_windows):
            i_start = offset + w * sig_len
            i_end = i_start + sig_len
            w_start = float(t_ms[i_start])
            w_end = float(t_ms[i_start] + sig_len * dt_53 * 1000.0)
            state_str = classify_state(w_start, w_end)

            y_w = ys[i_start:i_end]
            _, _, spec = dsp.spectrogram(
                y_w, fs=fs, window="hann", nperseg=nfft, noverlap=noverlap, nfft=nfft,
                detrend="linear", return_onesided=False, scaling="density", axis=0, mode="complex"
            )
            spec = np.transpose(spec, [1, 0])[:n_ens, :]

            # P_mode and L
            p_band = np.sum(np.abs(spec[:, 41:51]) ** 2, axis=1)
            P_mode = float(np.mean(p_band))
            L = float(np.log10(P_mode))

            # H, C1, C2
            H, argmax_H, _ = compute_family(spec, 40, 50)
            C1, _, _ = compute_family(spec, 20, 30)
            C2, _, _ = compute_family(spec, 60, 70)

            # S92
            P_k = np.mean(np.abs(spec) ** 2, axis=0)
            P_bg_med = float(np.median(P_k[bg_bins]))
            S92 = float(P_k[92] / P_bg_med) if P_bg_med > 0 else 0.0

            records.append({
                "win": w,
                "start_ms": w_start,
                "end_ms": w_end,
                "state": state_str,
                "L": L,
                "H": H,
                "argmax_H": argmax_H,
                "C1": C1,
                "C2": C2,
                "S92": S92,
            })
        return records

    shots = [
        (88653, t_ms_53, ys_mp1_53, fs_53),
        (88654, t_ms_54, ys_mp1_54, fs_54),
    ]
    tilings = [
        ("Tiling 1 (Offset 0 samples = 0.000 ms)", 0),
        ("Tiling 2 (Offset 3968 samples = 3.968 ms)", 3968),
    ]

    all_data = {}  # (shot_id, offset) -> records

    for shot_id, t_ms, ys, fs in shots:
        for t_name, offset in tilings:
            p_out("\n" + "=" * 120)
            p_out(f"SHOT #{shot_id} | {t_name}")
            p_out("=" * 120)

            recs = process_windows(shot_id, t_ms, ys, fs, offset)
            all_data[(shot_id, offset)] = recs

            p_out(f"Total complete windows: {len(recs)}")
            hdr = (
                f"{'Win':<4} | {'Start (ms)':<10} | {'State':<20} | {'L = log10(P)':<14} | "
                f"{'H (40..50)':<10} | {'argmax(H)':<10} | {'C1 (20..30)':<11} | {'C2 (60..70)':<11} | {'S92':<10}"
            )
            p_out(hdr)
            p_out("-" * len(hdr))

            for r in recs:
                row_str = (
                    f"{r['win']:<4d} | {r['start_ms']:<10.3f} | {r['state']:<20} | {r['L']:<14.4f} | "
                    f"{r['H']:<10.4f} | {r['argmax_H']:<10d} | {r['C1']:<11.4f} | {r['C2']:<11.4f} | {r['S92']:<10.4f}"
                )
                p_out(row_str)

    # -------------------------------------------------------------------------
    # Correlation Analysis per Shot & Tiling
    # -------------------------------------------------------------------------
    p_out("\n" + "=" * 120)
    p_out("SPEARMAN RANK CORRELATIONS & PERMUTATION P-VALUES (PER SHOT & TILING)")
    p_out("=" * 120)

    for shot_id, _, _, _ in shots:
        for t_name, offset in tilings:
            recs = all_data[(shot_id, offset)]
            p_out(f"\n--- Shot #{shot_id} | {t_name} ---")

            # 1. All windows
            N_all = len(recs)
            L_all = np.array([r["L"] for r in recs])
            H_all = np.array([r["H"] for r in recs])
            C1_all = np.array([r["C1"] for r in recs])
            C2_all = np.array([r["C2"] for r in recs])

            p_out(f"\n[A] All Windows (N = {N_all}):")
            p_out(f"{'Variable':<12} | {'Observed rho':<14} | {'Exceedances (k/10000)':<24} | {'One-sided p-value'}")
            p_out("-" * 72)
            for var_name, var_arr in [("H (40..50)", H_all), ("C1 (20..30)", C1_all), ("C2 (60..70)", C2_all)]:
                obs, k, p_val = run_permutation_test_single(L_all, var_arr, n_perm=10000, seed=1)
                p_out(f"{var_name:<12} | {obs:<+14.4f} | {k:<5d} / 10000{' '*10} | {p_val:.6f}")

            # 2. Restricted windows (neither pre-NBI nor post-NBI)
            recs_restr = [r for r in recs if r["state"] not in ("pre-NBI", "post-NBI")]
            N_restr = len(recs_restr)
            L_restr = np.array([r["L"] for r in recs_restr])
            H_restr = np.array([r["H"] for r in recs_restr])
            C1_restr = np.array([r["C1"] for r in recs_restr])
            C2_restr = np.array([r["C2"] for r in recs_restr])

            p_out(f"\n[B] Restricted Windows (Neither 'pre-NBI' nor 'post-NBI', N = {N_restr}):")
            p_out(f"{'Variable':<12} | {'Observed rho':<14} | {'Exceedances (k/10000)':<24} | {'One-sided p-value'}")
            p_out("-" * 72)
            for var_name, var_arr in [("H (40..50)", H_restr), ("C1 (20..30)", C1_restr), ("C2 (60..70)", C2_restr)]:
                obs, k, p_val = run_permutation_test_single(L_restr, var_arr, n_perm=10000, seed=1)
                p_out(f"{var_name:<12} | {obs:<+14.4f} | {k:<5d} / 10000{' '*10} | {p_val:.6f}")

    # -------------------------------------------------------------------------
    # Pooled Statistic Analysis
    # -------------------------------------------------------------------------
    p_out("\n" + "=" * 120)
    p_out("POOLED STATISTIC (SUM OF PER-SHOT RHOS) WITH INDEPENDENT PERMUTATIONS (SEED 2)")
    p_out("=" * 120)

    def run_pooled_permutation_test(L53, Y53, L54, Y54, n_perm=10000, seed=2):
        obs_53 = spearman_rho(L53, Y53)
        obs_54 = spearman_rho(L54, Y54)
        obs_pooled = obs_53 + obs_54

        rL53 = rankdata_average(L53); rL53_m = rL53 - np.mean(rL53)
        rY53 = rankdata_average(Y53); rY53_m = rY53 - np.mean(rY53)
        den53 = np.sqrt(np.sum(rL53_m ** 2) * np.sum(rY53_m ** 2))

        rL54 = rankdata_average(L54); rL54_m = rL54 - np.mean(rL54)
        rY54 = rankdata_average(Y54); rY54_m = rY54 - np.mean(rY54)
        den54 = np.sqrt(np.sum(rL54_m ** 2) * np.sum(rY54_m ** 2))

        rng = np.random.default_rng(seed)
        k = 0
        for _ in range(n_perm):
            p53_rL = rng.permutation(rL53) - np.mean(rL53)
            p54_rL = rng.permutation(rL54) - np.mean(rL54)
            p_rho53 = float(np.sum(p53_rL * rY53_m) / den53) if den53 > 0 else 0.0
            p_rho54 = float(np.sum(p54_rL * rY54_m) / den54) if den54 > 0 else 0.0
            if (p_rho53 + p_rho54) >= obs_pooled:
                k += 1
        p_val = k / n_perm
        return obs_53, obs_54, obs_pooled, k, p_val

    for t_name, offset in tilings:
        p_out(f"\n====================================================================================================")
        p_out(f"POOLED CORRELATIONS | {t_name}")
        p_out(f"====================================================================================================")

        recs53 = all_data[(88653, offset)]
        recs54 = all_data[(88654, offset)]

        # 1. All windows
        N53_all = len(recs53)
        N54_all = len(recs54)
        L53_all = np.array([r["L"] for r in recs53])
        L54_all = np.array([r["L"] for r in recs54])

        p_out(f"\n[A] All Windows: N_88653 = {N53_all}, N_88654 = {N54_all} (Total N = {N53_all + N54_all})")
        p_out(f"{'Variable':<12} | {'rho_88653':<10} | {'rho_88654':<10} | {'Pooled rho':<12} | {'Exceedances (k/10000)':<24} | {'Permutation p'}")
        p_out("-" * 92)

        for var_name, k_field in [("H (40..50)", "H"), ("C1 (20..30)", "C1"), ("C2 (60..70)", "C2")]:
            Y53 = np.array([r[k_field] for r in recs53])
            Y54 = np.array([r[k_field] for r in recs54])
            o53, o54, o_pool, k_pool, p_pool = run_pooled_permutation_test(L53_all, Y53, L54_all, Y54, n_perm=10000, seed=2)
            p_out(f"{var_name:<12} | {o53:<+10.4f} | {o54:<+10.4f} | {o_pool:<+12.4f} | {k_pool:<5d} / 10000{' '*10} | {p_pool:.6f}")

        # 2. Restricted windows
        recs53_restr = [r for r in recs53 if r["state"] not in ("pre-NBI", "post-NBI")]
        recs54_restr = [r for r in recs54 if r["state"] not in ("pre-NBI", "post-NBI")]
        N53_restr = len(recs53_restr)
        N54_restr = len(recs54_restr)
        L53_restr = np.array([r["L"] for r in recs53_restr])
        L54_restr = np.array([r["L"] for r in recs54_restr])

        p_out(f"\n[B] Restricted Windows (Neither 'pre-NBI' nor 'post-NBI'): N_88653 = {N53_restr}, N_88654 = {N54_restr} (Total N = {N53_restr + N54_restr})")
        p_out(f"{'Variable':<12} | {'rho_88653':<10} | {'rho_88654':<10} | {'Pooled rho':<12} | {'Exceedances (k/10000)':<24} | {'Permutation p'}")
        p_out("-" * 92)

        for var_name, k_field in [("H (40..50)", "H"), ("C1 (20..30)", "C1"), ("C2 (60..70)", "C2")]:
            Y53 = np.array([r[k_field] for r in recs53_restr])
            Y54 = np.array([r[k_field] for r in recs54_restr])
            o53, o54, o_pool, k_pool, p_pool = run_pooled_permutation_test(L53_restr, Y53, L54_restr, Y54, n_perm=10000, seed=2)
            p_out(f"{var_name:<12} | {o53:<+10.4f} | {o54:<+10.4f} | {o_pool:<+12.4f} | {k_pool:<5d} / 10000{' '*10} | {p_pool:.6f}")

    wall_elapsed = time.time() - t_start_wall
    p_out("\n" + "=" * 120)
    p_out(f"Total run_dose.py wall-clock execution time: {wall_elapsed:.2f} s")
    p_out("=" * 120)

except Exception as ex:
    p_out(f"\nERROR: Exception occurred during run_dose.py execution:")
    p_out(traceback.format_exc())
    stdout_h.close()
    sys.exit(1)

stdout_h.close()
raw_bytes = stdout_file.read_bytes()
out_sha = hashlib.sha256(raw_bytes).hexdigest()
print(f"Saved stdout to dose_stdout.txt ({len(raw_bytes)} bytes, SHA-256: {out_sha})")
