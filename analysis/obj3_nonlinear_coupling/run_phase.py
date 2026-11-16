import sys
import time
import hashlib
import traceback
from pathlib import Path
import numpy as np
import scipy.signal as dsp

t_start_wall = time.time()

ROOT_DIR = Path(__file__).resolve().parents[2]
out_dir = ROOT_DIR / "scratch" / "final19"
out_dir.mkdir(parents=True, exist_ok=True)
stdout_file = out_dir / "phase_stdout.txt"
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
p_out("TASK FINAL19 - STEP 1: BIPHASE, PHASE COHERENCE R2 & TRIMMED BICOHERENCE ANALYSES")
p_out("========================================================================================================================")

try:
    # -------------------------------------------------------------------------
    # Reusing unchanged data loading, spectrogram call, and helpers from final18
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

    # Single-triad detailed metrics
    def compute_triad_detailed(spec, ka, kb):
        kb_idx = kb % nfft
        kc = (ka + kb) % nfft
        z_s = spec[:, ka] * spec[:, kb_idx] * np.conj(spec[:, kc])
        abs_z = np.abs(z_s)
        prod_ab = spec[:, ka] * spec[:, kb_idx]
        spec_c = spec[:, kc]

        # b2
        num = np.abs(np.mean(z_s)) ** 2
        den = np.mean(np.abs(prod_ab) ** 2) * np.mean(np.abs(spec_c) ** 2)
        b2 = float(num / den) if den > 0 else 0.0

        # R2 and biphase
        unit_z = np.where(abs_z > 0, z_s / abs_z, 0.0)
        mean_unit_z = np.mean(unit_z)
        r2 = float(np.abs(mean_unit_z) ** 2)
        biphase_deg = float(np.degrees(np.angle(mean_unit_z)))

        # N_eff
        sum_abs_z = float(np.sum(abs_z))
        sum_sq_abs_z = float(np.sum(abs_z ** 2))
        n_eff = float((sum_abs_z ** 2) / sum_sq_abs_z) if sum_sq_abs_z > 0 else 0.0

        # trimmed b2 for m=1, 2, 3
        order_desc = np.argsort(-abs_z)
        trimmed_b2 = {}
        for m in [1, 2, 3]:
            kept = order_desc[m:]
            z_kept = z_s[kept]
            prod_kept = prod_ab[kept]
            c_kept = spec_c[kept]
            num_t = np.abs(np.sum(z_kept)) ** 2
            den_t = np.sum(np.abs(prod_kept) ** 2) * np.sum(np.abs(c_kept) ** 2)
            trimmed_b2[m] = float(num_t / den_t) if den_t > 0 else 0.0

        return {
            "ka": ka,
            "kb": kb,
            "kc": kc,
            "z_s": z_s,
            "abs_z": abs_z,
            "b2": b2,
            "r2": r2,
            "biphase_deg": biphase_deg,
            "n_eff": n_eff,
            "trimmed_b2": trimmed_b2,
        }

    # Family vector calculation: R2 max and trimmed b2 (m=1) max
    def compute_family_metrics(spec, ka_min, ka_max):
        ka_arr = np.arange(ka_min, ka_max + 1, dtype=int)
        kc_arr = 2 * ka_arr
        prod_s = spec[:, ka_arr] ** 2
        z_s = prod_s * np.conj(spec[:, kc_arr])
        abs_z = np.abs(z_s)

        # b2
        num = np.abs(np.mean(z_s, axis=0)) ** 2
        den = np.mean(np.abs(prod_s) ** 2, axis=0) * np.mean(np.abs(spec[:, kc_arr]) ** 2, axis=0)
        b2_arr = np.where(den > 0, num / den, 0.0)

        # R2
        unit_z = np.where(abs_z > 0, z_s / abs_z, 0.0)
        r2_arr = np.abs(np.mean(unit_z, axis=0)) ** 2

        # trimmed b2 (m=1)
        b2_trim1_arr = np.zeros(len(ka_arr), dtype=float)
        for idx in range(len(ka_arr)):
            order = np.argsort(-abs_z[:, idx])
            kept = order[1:]
            z_k = z_s[kept, idx]
            p_k = prod_s[kept, idx]
            c_k = spec[kept, kc_arr[idx]]
            n_t = np.abs(np.sum(z_k)) ** 2
            d_t = np.sum(np.abs(p_k) ** 2) * np.sum(np.abs(c_k) ** 2)
            b2_trim1_arr[idx] = n_t / d_t if d_t > 0 else 0.0

        b2_max_idx = int(np.argmax(b2_arr))
        r2_max_idx = int(np.argmax(r2_arr))
        trim1_max_idx = int(np.argmax(b2_trim1_arr))

        return {
            "b2_max": float(b2_arr[b2_max_idx]),
            "b2_argmax": int(ka_arr[b2_max_idx]),
            "r2_max": float(r2_arr[r2_max_idx]),
            "r2_argmax": int(ka_arr[r2_max_idx]),
            "trim1_max": float(b2_trim1_arr[trim1_max_idx]),
            "trim1_argmax": int(ka_arr[trim1_max_idx]),
            "b2_arr": b2_arr,
            "r2_arr": r2_arr,
            "trim1_arr": b2_trim1_arr,
        }

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

    # Spearman rank correlation helpers with average ties
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

    # Permutation test for single shot (seed 1)
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

    # Pooled permutation test (seed 2)
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

    shots = [
        (88653, t_ms_53, ys_mp1_53, fs_53),
        (88654, t_ms_54, ys_mp1_54, fs_54),
    ]

    # =========================================================================
    # (a) TILING 1 WINDOWS 16, 17, 18 DETAILED TRIAD METRICS
    # =========================================================================
    p_out("\n" + "=" * 120)
    p_out("(a) TILING 1 (OFFSET 0) WINDOWS 16, 17, 18 DETAILED METRICS FOR F_h ARGMAX & FIXED (46,46->92)")
    p_out("=" * 120)

    for shot_id, t_ms, ys, fs in shots:
        p_out(f"\n" + "-" * 100)
        p_out(f"--- SHOT #{shot_id} | WINDOWS 16, 17, 18 ---")
        p_out("-" * 100)

        for w in [16, 17, 18]:
            i_start = w * sig_len
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

            # Find argmax of b2 over F_h (40..50)
            f_res = compute_family_metrics(spec, 40, 50)
            top_k = f_res["b2_argmax"]

            triads_to_eval = [
                ("F_h Argmax Triad", top_k),
                ("Fixed Triad (46,46->92)", 46),
            ]

            p_out(f"\n>>> Window {w} | Start: {w_start:.3f} ms | End: {w_end:.3f} ms | State: {state_str}")

            for t_desc, ka in triads_to_eval:
                kc = 2 * ka
                st = compute_triad_detailed(spec, ka, ka)

                p_out(f"\n  [{t_desc}]: ({ka}, {ka} -> {kc}) = ({ka*df_khz:.2f}, {ka*df_khz:.2f} -> {kc*df_khz:.2f}) kHz")
                p_out(f"    b2:                         {st['b2']:.6f}")
                p_out(f"    R2 (phase coherence):       {st['r2']:.6f}")
                p_out(f"    N_eff:                      {st['n_eff']:.2f} / 30")
                p_out(f"    biphase (circular mean):    {st['biphase_deg']:+.2f} deg")
                p_out(f"    Trimmed b2(m=1):            {st['trimmed_b2'][1]:.6f}")
                p_out(f"    Trimmed b2(m=2):            {st['trimmed_b2'][2]:.6f}")
                p_out(f"    Trimmed b2(m=3):            {st['trimmed_b2'][3]:.6f}")

                # Top-5 segments with largest |z_s| in time order
                abs_z = st["abs_z"]
                sum_abs_z = float(np.sum(abs_z))
                top5_indices = sorted(np.argsort(abs_z)[-5:])
                ranks_abs_z = np.argsort(np.argsort(-abs_z)) + 1

                p_out(f"    Top-5 Segments by |z_s| (in chronological time order):")
                p_out(f"      {'Rank':<6} | {'Seg':<4} | {'Start (ms)':<11} | {'|z_s|':<14} | {'|z_s| Share (%)'}")
                p_out(f"      " + "-" * 56)
                for s in top5_indices:
                    seg_start = w_start + s * (hop * dt_53 * 1000.0)
                    share_pct = (abs_z[s] / sum_abs_z * 100.0) if sum_abs_z > 0 else 0.0
                    r_num = ranks_abs_z[s]
                    p_out(f"      #{r_num:<5d} | {s:<4d} | {seg_start:<11.3f} | {abs_z[s]:<14.4e} | {share_pct:6.2f}%")

    # =========================================================================
    # Pre-computation of all window metrics for (b) and (c)
    # =========================================================================
    tilings = [
        ("Tiling 1 (Offset 0 samples = 0.000 ms)", 0),
        ("Tiling 2 (Offset 3968 samples = 3.968 ms)", 3968),
    ]

    # Structure: (shot_id, offset) -> list of record dicts
    all_window_data = {}

    for shot_id, t_ms, ys, fs in shots:
        for t_name, offset in tilings:
            n_windows = (len(ys) - offset) // sig_len
            records = []
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

                # Family metrics for F_h (40..50), F_c1 (20..30), F_c2 (60..70)
                fh_res = compute_family_metrics(spec, 40, 50)
                fc1_res = compute_family_metrics(spec, 20, 30)
                fc2_res = compute_family_metrics(spec, 60, 70)

                records.append({
                    "win": w,
                    "start_ms": w_start,
                    "end_ms": w_end,
                    "state": state_str,
                    "L": L,
                    # Phase-only coherence R2 maxima
                    "HR": fh_res["r2_max"],
                    "C1R": fc1_res["r2_max"],
                    "C2R": fc2_res["r2_max"],
                    # Trimmed b2 (m=1) maxima
                    "H1": fh_res["trim1_max"],
                    "C1T": fc1_res["trim1_max"],
                    "C2T": fc2_res["trim1_max"],
                })
            all_window_data[(shot_id, offset)] = records

    # Helper function to print correlations table for a metric triplet
    def print_correlations_section(var_keys, var_labels, section_title):
        p_out("\n" + "=" * 120)
        p_out(section_title)
        p_out("=" * 120)

        # 1. Per shot and per tiling
        for shot_id, _, _, _ in shots:
            for t_name, offset in tilings:
                recs = all_window_data[(shot_id, offset)]
                p_out(f"\n--- Shot #{shot_id} | {t_name} ---")

                # (A) All windows
                N_all = len(recs)
                L_all = np.array([r["L"] for r in recs])
                p_out(f"\n[A] All Windows (N = {N_all}):")
                p_out(f"{'Variable':<14} | {'Observed rho':<14} | {'Exceedances (k/10000)':<24} | {'One-sided p-value'}")
                p_out("-" * 74)
                for v_key, v_lbl in zip(var_keys, var_labels):
                    v_arr = np.array([r[v_key] for r in recs])
                    obs, k, p_val = run_permutation_test_single(L_all, v_arr, n_perm=10000, seed=1)
                    p_out(f"{v_lbl:<14} | {obs:<+14.4f} | {k:<5d} / 10000{' '*10} | {p_val:.6f}")

                # (B) Restricted windows
                recs_restr = [r for r in recs if r["state"] not in ("pre-NBI", "post-NBI")]
                N_restr = len(recs_restr)
                L_restr = np.array([r["L"] for r in recs_restr])
                p_out(f"\n[B] Restricted Windows (Neither 'pre-NBI' nor 'post-NBI', N = {N_restr}):")
                p_out(f"{'Variable':<14} | {'Observed rho':<14} | {'Exceedances (k/10000)':<24} | {'One-sided p-value'}")
                p_out("-" * 74)
                for v_key, v_lbl in zip(var_keys, var_labels):
                    v_arr = np.array([r[v_key] for r in recs_restr])
                    obs, k, p_val = run_permutation_test_single(L_restr, v_arr, n_perm=10000, seed=1)
                    p_out(f"{v_lbl:<14} | {obs:<+14.4f} | {k:<5d} / 10000{' '*10} | {p_val:.6f}")

        # 2. Pooled correlations
        p_out("\n" + "-" * 100)
        p_out("POOLED CORRELATIONS (SUM OF PER-SHOT RHOS) WITH INDEPENDENT PERMUTATIONS (SEED 2)")
        p_out("-" * 100)

        for t_name, offset in tilings:
            recs53 = all_window_data[(88653, offset)]
            recs54 = all_window_data[(88654, offset)]

            p_out(f"\n>>> {t_name}")

            # [A] All windows
            N53_all = len(recs53)
            N54_all = len(recs54)
            L53_all = np.array([r["L"] for r in recs53])
            L54_all = np.array([r["L"] for r in recs54])

            p_out(f"\n[A] All Windows: N_88653 = {N53_all}, N_88654 = {N54_all} (Total N = {N53_all + N54_all})")
            p_out(f"{'Variable':<14} | {'rho_88653':<10} | {'rho_88654':<10} | {'Pooled rho':<12} | {'Exceedances (k/10000)':<24} | {'Permutation p'}")
            p_out("-" * 94)

            for v_key, v_lbl in zip(var_keys, var_labels):
                Y53 = np.array([r[v_key] for r in recs53])
                Y54 = np.array([r[v_key] for r in recs54])
                o53, o54, o_pool, k_pool, p_pool = run_pooled_permutation_test(L53_all, Y53, L54_all, Y54, n_perm=10000, seed=2)
                p_out(f"{v_lbl:<14} | {o53:<+10.4f} | {o54:<+10.4f} | {o_pool:<+12.4f} | {k_pool:<5d} / 10000{' '*10} | {p_pool:.6f}")

            # [B] Restricted windows
            recs53_restr = [r for r in recs53 if r["state"] not in ("pre-NBI", "post-NBI")]
            recs54_restr = [r for r in recs54 if r["state"] not in ("pre-NBI", "post-NBI")]
            N53_restr = len(recs53_restr)
            N54_restr = len(recs54_restr)
            L53_restr = np.array([r["L"] for r in recs53_restr])
            L54_restr = np.array([r["L"] for r in recs54_restr])

            p_out(f"\n[B] Restricted Windows: N_88653 = {N53_restr}, N_88654 = {N54_restr} (Total N = {N53_restr + N54_restr})")
            p_out(f"{'Variable':<14} | {'rho_88653':<10} | {'rho_88654':<10} | {'Pooled rho':<12} | {'Exceedances (k/10000)':<24} | {'Permutation p'}")
            p_out("-" * 94)

            for v_key, v_lbl in zip(var_keys, var_labels):
                Y53 = np.array([r[v_key] for r in recs53_restr])
                Y54 = np.array([r[v_key] for r in recs54_restr])
                o53, o54, o_pool, k_pool, p_pool = run_pooled_permutation_test(L53_restr, Y53, L54_restr, Y54, n_perm=10000, seed=2)
                p_out(f"{v_lbl:<14} | {o53:<+10.4f} | {o54:<+10.4f} | {o_pool:<+12.4f} | {k_pool:<5d} / 10000{' '*10} | {p_pool:.6f}")

    # =========================================================================
    # (b) Phase-only bicoherence maxima (HR, C1R, C2R) vs L
    # =========================================================================
    print_correlations_section(
        var_keys=["HR", "C1R", "C2R"],
        var_labels=["HR (40..50)", "C1R (20..30)", "C2R (60..70)"],
        section_title="(b) PHASE-ONLY COHERENCE MAXIMA (HR, C1R, C2R) VS MODE POWER L"
    )

    # =========================================================================
    # (c) Trimmed bicoherence maxima (H1, C1T, C2T) vs L
    # =========================================================================
    print_correlations_section(
        var_keys=["H1", "C1T", "C2T"],
        var_labels=["H1 (40..50)", "C1T (20..30)", "C2T (60..70)"],
        section_title="(c) TRIMMED (m=1) BICOHERENCE MAXIMA (H1, C1T, C2T) VS MODE POWER L"
    )

    wall_elapsed = time.time() - t_start_wall
    p_out("\n" + "=" * 120)
    p_out(f"Total run_phase.py wall-clock execution time: {wall_elapsed:.2f} s")
    p_out("=" * 120)

except Exception as ex:
    p_out(f"\nERROR: Exception occurred during run_phase.py execution:")
    p_out(traceback.format_exc())
    stdout_h.close()
    sys.exit(1)

stdout_h.close()
raw_bytes = stdout_file.read_bytes()
out_sha = hashlib.sha256(raw_bytes).hexdigest()
print(f"Saved stdout to phase_stdout.txt ({len(raw_bytes)} bytes, SHA-256: {out_sha})")
