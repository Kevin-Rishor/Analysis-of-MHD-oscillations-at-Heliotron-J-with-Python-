import sys
import time
import hashlib
import traceback
from pathlib import Path
import numpy as np
import scipy.signal as dsp

t_global_start = time.time()

ROOT_DIR = Path(__file__).resolve().parents[2]
out_dir = ROOT_DIR / "scratch" / "final16"
out_dir.mkdir(parents=True, exist_ok=True)
stdout_file = out_dir / "extension_stdout.txt"
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
p_out("TASK FINAL16: EXTENSION AND SENSITIVITY ANALYSIS (SHOTS 88653 & 88654)")
p_out("========================================================================================================================")

try:
    # -------------------------------------------------------------------------
    # 0. Source script integrity and parameter setup
    # -------------------------------------------------------------------------
    src_f15_py = ROOT_DIR / "analysis" / "obj3_nonlinear_coupling" / "run_domain_check.py"
    sha_src_f15 = sha256_file(src_f15_py)
    p_out(f"[Source Copy] {src_f15_py}")
    p_out(f"SHA-256 of final15 source: {sha_src_f15}")

    # Setup paths and libraries
    root_dir = ROOT_DIR
    for p_add in [root_dir / "jpack", root_dir / "analysis", root_dir / "analysis" / "common", root_dir / "analysis" / "heating_phase_comparison"]:
        if str(p_add) not in sys.path:
            sys.path.append(str(p_add))

    from comparison_common import load_channel
    staging_common = ROOT_DIR / "analysis" / "common"
    sys.path.insert(0, str(staging_common))
    from abicoh2_corrected import abicoh2_corrected

    # Load data for shots 88653 and 88654 (MP1)
    t_ms_53, ys_mp1_53, dt_53, fs_53 = load_channel(88653, "MP1")
    t_ms_54, ys_mp1_54, dt_54, fs_54 = load_channel(88654, "MP1")

    nfft = 512
    nhalf = 256
    hop = 256
    noverlap = 256
    n_ens = 30
    sig_len = hop * (n_ens - 1) + nfft
    df_khz = (fs_53 / nfft) / 1e3 # 1.953125 kHz
    f1_vec = np.arange(nhalf) * df_khz
    f2_vec = (np.arange(nfft) - nhalf) * df_khz
    FF1, FF2 = np.meshgrid(f1_vec, f2_vec, indexing="ij")

    # Define mask_B exactly as in final15
    fmin_khz = 5.0
    f1max = 100.0
    f2max = 100.0
    fs_khz = fs_53 / 1e3
    physical_mask_diff = (FF1 > 0) & (FF2 < 0) & (FF1 > -FF2)
    search_diff = (
        physical_mask_diff
        & (FF1 < f1max)
        & (-FF2 < f2max)
        & (FF1 >= fmin_khz)
        & (-FF2 >= fmin_khz)
        & ((FF1 + FF2) >= fmin_khz)
    )
    physical_mask_sum = (
        (FF1 > 0)
        & (FF2 > 0)
        & (FF1 >= FF2)
        & ((FF1 + FF2) <= (fs_khz / 2.0))
    )
    search_sum = (
        physical_mask_sum
        & (FF1 < f1max)
        & (FF2 < f2max)
        & (FF1 >= FF2)
        & (FF1 >= fmin_khz)
        & (FF2 >= fmin_khz)
    )
    mask_B = search_sum | search_diff

    coords_B = np.argwhere(mask_B)
    ka_B = coords_B[:, 0]
    kb_B = coords_B[:, 1] - nhalf
    kc_B = (ka_B + kb_B) % nfft
    kb_idx_B = kb_B % nfft
    M_B = len(ka_B)

    p_out(f"\nMask B bin count M = {M_B}")
    assert M_B == 2306, f"Mask B bin count mismatch: got {M_B}, expected 2306"
    p_out(f"Assertion M == 2306 passed: True")

    # Direct calculation function for a single triad (ka, kb)
    def compute_direct_triad(spec, ka, kb):
        kb_idx = kb % nfft
        kc = (ka + kb) % nfft
        z_s = spec[:, ka] * spec[:, kb_idx] * np.conj(spec[:, kc])
        prod_ab = spec[:, ka] * spec[:, kb_idx]
        spec_c = spec[:, kc]
        num = np.abs(np.mean(z_s)) ** 2
        den = np.mean(np.abs(prod_ab) ** 2) * np.mean(np.abs(spec_c) ** 2)
        b2 = float(num / den) if den > 0 else 0.0
        abs_z = np.abs(z_s)
        unit_z = np.where(abs_z > 0, z_s / abs_z, 0.0)
        r2 = float(np.abs(np.mean(unit_z)) ** 2)
        return z_s, b2, r2

    # Direct vectorized calculation across all M bins of mask_B
    def compute_mask_b2_r2(spec, ka_arr, kb_idx_arr, kc_arr):
        prod_s = spec[:, ka_arr] * spec[:, kb_idx_arr]
        z_s = prod_s * np.conj(spec[:, kc_arr])
        num = np.abs(np.mean(z_s, axis=0)) ** 2
        den = np.mean(np.abs(prod_s) ** 2, axis=0) * np.mean(np.abs(spec[:, kc_arr]) ** 2, axis=0)
        b2_arr = np.where(den > 0, num / den, 0.0)
        abs_z = np.abs(z_s)
        unit_z = np.where(abs_z > 0, z_s / abs_z, 0.0)
        r2_arr = np.abs(np.mean(unit_z, axis=0)) ** 2
        return b2_arr, r2_arr

    # Surrogate generator: per-segment amplitude-preserving phase randomisation
    def generate_surrogates(spec, n_surrogates, rng):
        S, nf = spec.shape
        nh = nf // 2
        amps = np.abs(spec)
        surr_list = []
        for _ in range(n_surrogates):
            rand_phases = rng.uniform(0, 2 * np.pi, size=(S, nh - 1))
            surr = np.zeros_like(spec, dtype=complex)
            surr[:, 0] = spec[:, 0]
            surr[:, nh] = spec[:, nh]
            surr[:, 1:nh] = amps[:, 1:nh] * np.exp(1j * rand_phases)
            surr[:, nh + 1:] = np.conj(surr[:, 1:nh][:, ::-1])
            surr_list.append(surr)
        return surr_list

    # =========================================================================
    # SHOT 88653 ANALYSIS: WINDOWS 259.1 ms AND 291.3 ms
    # =========================================================================
    windows_53 = [
        ("ECH+NBI", 259.1, 42),
        ("NBI-only", 291.3, 123)
    ]

    shot53_results = {}

    for w_name, t_start, seed_base in windows_53:
        p_out("\n" + "=" * 90)
        p_out(f"--- SHOT 88653 | WINDOW: {w_name} (start = {t_start} ms, seed_base = {seed_base}) ---")
        p_out("=" * 90)

        t_sec_w0 = time.time()
        idx_w = np.where(t_ms_53 >= t_start)[0]
        y_w = ys_mp1_53[idx_w][:sig_len]
        _, _, spec_w = dsp.spectrogram(
            y_w, fs=fs_53, window="hann", nperseg=nfft, noverlap=noverlap, nfft=nfft,
            detrend="linear", return_onesided=False, scaling="density", axis=0, mode="complex"
        )
        spec_w = np.transpose(spec_w, [1, 0])[:n_ens, :]

        # Compute b2 over mask_B using abicoh2_corrected and vectorized formula
        _, b2_raw = abicoh2_corrected(spec_w)
        b2_grid = np.transpose(np.fft.fftshift(b2_raw, axes=0), [1, 0])
        b2_obs = b2_grid[coords_B[:, 0], coords_B[:, 1]]

        top_i = int(np.argmax(b2_obs))
        obs_max_b2 = float(b2_obs[top_i])
        ka_max = int(ka_B[top_i])
        kb_max = int(kb_B[top_i])
        kc_max = int(kc_B[top_i])

        # Assert direct formula equals abicoh2_corrected value to 1e-12
        z_s_top, b2_direct, r2_direct = compute_direct_triad(spec_w, ka_max, kb_max)
        diff_direct = abs(b2_direct - obs_max_b2)
        p_out(f"Nominal argmax triad: ({ka_max}, {kb_max} -> {kc_max}) = ({ka_max * df_khz:.2f}, {kb_max * df_khz:.2f} -> {kc_max * df_khz:.2f}) kHz")
        p_out(f"  b2 (abicoh2_corrected): {obs_max_b2:.14f}")
        p_out(f"  b2 (direct formula):    {b2_direct:.14f}")
        p_out(f"  Absolute difference:    {diff_direct:.2e}")
        assert diff_direct < 1e-12, f"Direct b2 mismatch: {diff_direct} >= 1e-12"
        p_out(f"  Assertion abs diff < 1e-12 passed: True")

        shot53_results[w_name] = {
            "spec_w": spec_w,
            "ka_max": ka_max,
            "kb_max": kb_max,
            "kc_max": kc_max,
            "obs_max_b2": obs_max_b2,
            "top_i": top_i
        }

        # ---------------------------------------------------------------------
        # (a) Monte Carlo precision: 5000 surrogates for 3 seeds
        # ---------------------------------------------------------------------
        p_out("\n" + "-" * 80)
        p_out(f"(a) Monte Carlo Precision (5000 Surrogates, Seeds: {seed_base}, {seed_base+1}, {seed_base+2})")
        p_out("-" * 80)
        t_a_start = time.time()

        n_surr_a = 5000
        p_out(f"{'Seed':<8} | {'k (exceed)':<12} | {'n_surr':<8} | {'p-value':<12} | {'Binomial Std Error'}")
        p_out("-" * 65)

        for s_offset in [0, 1, 2]:
            seed_curr = seed_base + s_offset
            rng_curr = np.random.default_rng(seed_curr)

            k_max_a = 0
            S, nf = spec_w.shape
            nh_local = nf // 2
            amps_local = np.abs(spec_w)

            for _ in range(n_surr_a):
                rand_phases = rng_curr.uniform(0, 2 * np.pi, size=(S, nh_local - 1))
                surr = np.zeros_like(spec_w, dtype=complex)
                surr[:, 0] = spec_w[:, 0]
                surr[:, nh_local] = spec_w[:, nh_local]
                surr[:, 1:nh_local] = amps_local[:, 1:nh_local] * np.exp(1j * rand_phases)
                surr[:, nh_local + 1:] = np.conj(surr[:, 1:nh_local][:, ::-1])

                prod_s = surr[:, ka_B] * surr[:, kb_idx_B]
                num_s = np.abs(np.mean(prod_s * np.conj(surr[:, kc_B]), axis=0)) ** 2
                den_s = np.mean(np.abs(prod_s) ** 2, axis=0) * np.mean(np.abs(surr[:, kc_B]) ** 2, axis=0)
                b2_s = num_s / den_s

                if np.max(b2_s) >= obs_max_b2:
                    k_max_a += 1

            p_a = (1.0 + k_max_a) / (1.0 + n_surr_a)
            binom_se = np.sqrt(p_a * (1.0 - p_a) / n_surr_a)
            p_out(f"{seed_curr:<8d} | {k_max_a:<12d} | {n_surr_a:<8d} | {p_a:<12.6f} | {binom_se:.6f}")

        t_a_elapsed = time.time() - t_a_start
        p_out(f"Section (a) wall-clock time: {t_a_elapsed:.2f} s")

        # ---------------------------------------------------------------------
        # (b) Phase-only statistic (R2): observed max-R2, argmax, per-bin & family p
        # ---------------------------------------------------------------------
        p_out("\n" + "-" * 80)
        p_out(f"(b) Phase-Only Statistic R2 (2000 Surrogates, Seed = {seed_base})")
        p_out("-" * 80)
        t_b_start = time.time()

        b2_obs_all, r2_obs_all = compute_mask_b2_r2(spec_w, ka_B, kb_idx_B, kc_B)
        top_i_r2 = int(np.argmax(r2_obs_all))
        obs_max_r2 = float(r2_obs_all[top_i_r2])
        ka_r2 = int(ka_B[top_i_r2])
        kb_r2 = int(kb_B[top_i_r2])
        kc_r2 = int(kc_B[top_i_r2])

        n_surr_b = 2000
        rng_b = np.random.default_rng(seed_base)
        k_per_bin_r2 = np.zeros(M_B, dtype=int)
        k_max_r2 = 0

        S, nf = spec_w.shape
        nh_local = nf // 2
        amps_local = np.abs(spec_w)

        for _ in range(n_surr_b):
            rand_phases = rng_b.uniform(0, 2 * np.pi, size=(S, nh_local - 1))
            surr = np.zeros_like(spec_w, dtype=complex)
            surr[:, 0] = spec_w[:, 0]
            surr[:, nh_local] = spec_w[:, nh_local]
            surr[:, 1:nh_local] = amps_local[:, 1:nh_local] * np.exp(1j * rand_phases)
            surr[:, nh_local + 1:] = np.conj(surr[:, 1:nh_local][:, ::-1])

            prod_s = surr[:, ka_B] * surr[:, kb_idx_B]
            z_s = prod_s * np.conj(surr[:, kc_B])
            abs_z_s = np.abs(z_s)
            unit_z_s = np.where(abs_z_s > 0, z_s / abs_z_s, 0.0)
            r2_s = np.abs(np.mean(unit_z_s, axis=0)) ** 2

            k_per_bin_r2 += (r2_s >= r2_obs_all)
            if np.max(r2_s) >= obs_max_r2:
                k_max_r2 += 1

        p_per_bin_r2 = (1.0 + k_per_bin_r2[top_i_r2]) / (1.0 + n_surr_b)
        p_fw_r2 = (1.0 + k_max_r2) / (1.0 + n_surr_b)

        p_out(f"Observed max-R2 over mask_B: {obs_max_r2:.4f}")
        p_out(f"Argmax R2 triad: ({ka_r2}, {kb_r2} -> {kc_r2}) = ({ka_r2 * df_khz:.2f}, {kb_r2 * df_khz:.2f} -> {kc_r2 * df_khz:.2f}) kHz")
        p_out(f"Argmax per-bin p:             {p_per_bin_r2:.4f} (k = {k_per_bin_r2[top_i_r2]}/{n_surr_b})")
        p_out(f"Family-wise max-R2 p:         {p_fw_r2:.4f} (k_max = {k_max_r2}/{n_surr_b})")

        t_b_elapsed = time.time() - t_b_start
        p_out(f"Section (b) wall-clock time: {t_b_elapsed:.2f} s")

        # ---------------------------------------------------------------------
        # (c) Segment dominance for nominal argmax triad of max-b2
        # ---------------------------------------------------------------------
        p_out("\n" + "-" * 80)
        p_out(f"(c) Segment Dominance for Triad ({ka_max}, {kb_max} -> {kc_max})")
        p_out("-" * 80)
        t_c_start = time.time()

        abs_z = np.abs(z_s_top)
        sum_abs_z = np.sum(abs_z)
        shares = (abs_z / sum_abs_z) * 100.0
        sorted_seg_indices = np.argsort(shares)[::-1]

        p_out("Top-5 segments by |z_s| share (% of sum_s |z_s|):")
        p_out(f"{'Rank':<6} | {'Seg Index':<10} | {'Start Time (ms)':<18} | {'|z_s| Share (%)'}")
        p_out("-" * 55)
        for r_i in range(5):
            seg_idx = sorted_seg_indices[r_i]
            t_seg = t_start + seg_idx * 0.256
            p_out(f"{r_i + 1:<6d} | {seg_idx:<10d} | {t_seg:<18.3f} | {shares[seg_idx]:.2f}%")

        # Removals:
        # (i) largest-share segment removed
        seg_largest = sorted_seg_indices[0]
        mask_rem_i = [s for s in range(n_ens) if s != seg_largest]
        spec_rem_i = spec_w[mask_rem_i, :]
        _, b2_rem_i, _ = compute_direct_triad(spec_rem_i, ka_max, kb_max)

        # 2000 surrogates on reduced ensemble (i)
        rng_ci = np.random.default_rng(seed_base)
        k_ci = 0
        S_i = len(mask_rem_i)
        amps_i = np.abs(spec_rem_i)
        for _ in range(2000):
            rand_phases = rng_ci.uniform(0, 2 * np.pi, size=(S_i, nh_local - 1))
            surr = np.zeros_like(spec_rem_i, dtype=complex)
            surr[:, 0] = spec_rem_i[:, 0]
            surr[:, nh_local] = spec_rem_i[:, nh_local]
            surr[:, 1:nh_local] = amps_i[:, 1:nh_local] * np.exp(1j * rand_phases)
            surr[:, nh_local + 1:] = np.conj(surr[:, 1:nh_local][:, ::-1])

            p_s = surr[:, ka_max] * surr[:, kb_max % nfft]
            num = np.abs(np.mean(p_s * np.conj(surr[:, kc_max]))) ** 2
            den = np.mean(np.abs(p_s) ** 2) * np.mean(np.abs(surr[:, kc_max]) ** 2)
            b2_s = num / den if den > 0 else 0.0
            if b2_s >= b2_rem_i:
                k_ci += 1
        p_ci = (1.0 + k_ci) / (1.0 + 2000)

        p_out(f"\n(i) Largest-share segment removed (seg {seg_largest}, N=29):")
        p_out(f"    b2 = {b2_rem_i:.4f} | per-bin p = {p_ci:.4f} (k = {k_ci}/2000)")

        # (ii) first two segments removed (s=0, s=1)
        mask_rem_ii = list(range(2, n_ens))
        spec_rem_ii = spec_w[mask_rem_ii, :]
        _, b2_rem_ii, _ = compute_direct_triad(spec_rem_ii, ka_max, kb_max)

        # 2000 surrogates on reduced ensemble (ii): per-bin p and reduced-ensemble max b2
        b2_all_ii, _ = compute_mask_b2_r2(spec_rem_ii, ka_B, kb_idx_B, kc_B)
        top_ii_idx = int(np.argmax(b2_all_ii))
        max_b2_ii = float(b2_all_ii[top_ii_idx])
        ka_ii = int(ka_B[top_ii_idx])
        kb_ii = int(kb_B[top_ii_idx])
        kc_ii = int(kc_B[top_ii_idx])

        rng_cii = np.random.default_rng(seed_base)
        k_cii = 0
        k_max_cii = 0
        S_ii = len(mask_rem_ii)
        amps_ii = np.abs(spec_rem_ii)

        for _ in range(2000):
            rand_phases = rng_cii.uniform(0, 2 * np.pi, size=(S_ii, nh_local - 1))
            surr = np.zeros_like(spec_rem_ii, dtype=complex)
            surr[:, 0] = spec_rem_ii[:, 0]
            surr[:, nh_local] = spec_rem_ii[:, nh_local]
            surr[:, 1:nh_local] = amps_ii[:, 1:nh_local] * np.exp(1j * rand_phases)
            surr[:, nh_local + 1:] = np.conj(surr[:, 1:nh_local][:, ::-1])

            prod_s = surr[:, ka_B] * surr[:, kb_idx_B]
            num_s = np.abs(np.mean(prod_s * np.conj(surr[:, kc_B]), axis=0)) ** 2
            den_s = np.mean(np.abs(prod_s) ** 2, axis=0) * np.mean(np.abs(surr[:, kc_B]) ** 2, axis=0)
            b2_s_all = num_s / den_s

            if b2_s_all[top_i] >= b2_rem_ii:
                k_cii += 1
            if np.max(b2_s_all) >= max_b2_ii:
                k_max_cii += 1

        p_cii = (1.0 + k_cii) / (1.0 + 2000)
        p_fw_cii = (1.0 + k_max_cii) / (1.0 + 2000)

        p_out(f"\n(ii) First two segments removed (s=0 and s=1, N=28):")
        p_out(f"    Nominal triad ({ka_max}, {kb_max} -> {kc_max}): b2 = {b2_rem_ii:.4f} | per-bin p = {p_cii:.4f} (k = {k_cii}/2000)")
        p_out(f"    Reduced ensemble max b2 over mask_B:        b2 = {max_b2_ii:.4f}")
        p_out(f"    Reduced ensemble argmax triad:              ({ka_ii}, {kb_ii} -> {kc_ii}) = ({ka_ii * df_khz:.2f}, {kb_ii * df_khz:.2f} -> {kc_ii * df_khz:.2f}) kHz")
        p_out(f"    Reduced ensemble family-wise max-b2 p:       {p_fw_cii:.4f} (k_max = {k_max_cii}/2000)")

        # (iii) two largest-share segments removed
        top2_segs = sorted_seg_indices[:2]
        mask_rem_iii = [s for s in range(n_ens) if s not in top2_segs]
        spec_rem_iii = spec_w[mask_rem_iii, :]
        _, b2_rem_iii, _ = compute_direct_triad(spec_rem_iii, ka_max, kb_max)

        rng_ciii = np.random.default_rng(seed_base)
        k_ciii = 0
        S_iii = len(mask_rem_iii)
        amps_iii = np.abs(spec_rem_iii)
        for _ in range(2000):
            rand_phases = rng_ciii.uniform(0, 2 * np.pi, size=(S_iii, nh_local - 1))
            surr = np.zeros_like(spec_rem_iii, dtype=complex)
            surr[:, 0] = spec_rem_iii[:, 0]
            surr[:, nh_local] = spec_rem_iii[:, nh_local]
            surr[:, 1:nh_local] = amps_iii[:, 1:nh_local] * np.exp(1j * rand_phases)
            surr[:, nh_local + 1:] = np.conj(surr[:, 1:nh_local][:, ::-1])

            p_s = surr[:, ka_max] * surr[:, kb_max % nfft]
            num = np.abs(np.mean(p_s * np.conj(surr[:, kc_max]))) ** 2
            den = np.mean(np.abs(p_s) ** 2) * np.mean(np.abs(surr[:, kc_max]) ** 2)
            b2_s = num / den if den > 0 else 0.0
            if b2_s >= b2_rem_iii:
                k_ciii += 1
        p_ciii = (1.0 + k_ciii) / (1.0 + 2000)

        p_out(f"\n(iii) Two largest-share segments removed (segs {top2_segs[0]} & {top2_segs[1]}, N=28):")
        p_out(f"    b2 = {b2_rem_iii:.4f} | per-bin p = {p_ciii:.4f} (k = {k_ciii}/2000)")

        t_c_elapsed = time.time() - t_c_start
        p_out(f"Section (c) wall-clock time: {t_c_elapsed:.2f} s")

        # ---------------------------------------------------------------------
        # (d) Start-time stability: nominal + k * 0.25 ms (k = 0..6)
        # ---------------------------------------------------------------------
        p_out("\n" + "-" * 80)
        p_out(f"(d) Start-Time Stability (k = 0..6, Shifts of 0.25 ms up to +1.50 ms)")
        p_out("-" * 80)
        t_d_start = time.time()

        p_out(f"{'k':<3} | {'t_start (ms)':<13} | {'Nominal Triad b2':<18} | {'Max b2 (mask_B)':<16} | {'Argmax Triad (index & kHz)'}")
        p_out("-" * 85)

        for k_shift in range(7):
            t_k = t_start + k_shift * 0.25
            idx_k = np.where(t_ms_53 >= t_k)[0]
            y_k = ys_mp1_53[idx_k][:sig_len]
            _, _, spec_k = dsp.spectrogram(
                y_k, fs=fs_53, window="hann", nperseg=nfft, noverlap=noverlap, nfft=nfft,
                detrend="linear", return_onesided=False, scaling="density", axis=0, mode="complex"
            )
            spec_k = np.transpose(spec_k, [1, 0])[:n_ens, :]

            # b2 of nominal argmax triad
            _, b2_nom_k, _ = compute_direct_triad(spec_k, ka_max, kb_max)

            # Max b2 over mask_B
            b2_k_all, _ = compute_mask_b2_r2(spec_k, ka_B, kb_idx_B, kc_B)
            top_k_i = int(np.argmax(b2_k_all))
            max_b2_k = float(b2_k_all[top_k_i])
            ka_k = int(ka_B[top_k_i])
            kb_k = int(kb_B[top_k_i])
            kc_k = int(kc_B[top_k_i])
            triad_k_str = f"({ka_k:2d},{kb_k:3d}->{kc_k:2d}) = ({ka_k * df_khz:5.2f},{kb_k * df_khz:6.2f}->{kc_k * df_khz:5.2f}) kHz"

            p_out(f"{k_shift:<3d} | {t_k:<13.2f} | {b2_nom_k:<18.4f} | {max_b2_k:<16.4f} | {triad_k_str}")

        t_d_elapsed = time.time() - t_d_start
        p_out(f"Section (d) wall-clock time: {t_d_elapsed:.2f} s")

    # =========================================================================
    # SHOT 88654 (MP1) ANALYSIS: SECTIONS (e) AND (f)
    # =========================================================================
    p_out("\n" + "=" * 90)
    p_out("--- SHOT 88654 (MP1) REPLICATION ANALYSIS (SECTIONS e & f) ---")
    p_out("=" * 90)
    t_ef_start = time.time()

    windows_54 = [
        ("ECH+NBI", 259.1, 42),
        ("NBI-only", 291.3, 123)
    ]

    target_triads_88653 = [
        (38, 12, "88653 NBI-only argmax"),
        (39, 37, "88653 ECH+NBI argmax")
    ]

    for w_name, t_start, seed_base in windows_54:
        p_out("\n" + "-" * 80)
        p_out(f"Shot 88654 | Window: {w_name} (start = {t_start} ms, seed_base = {seed_base})")
        p_out("-" * 80)

        idx_w = np.where(t_ms_54 >= t_start)[0]
        y_w = ys_mp1_54[idx_w][:sig_len]
        _, _, spec_w54 = dsp.spectrogram(
            y_w, fs=fs_54, window="hann", nperseg=nfft, noverlap=noverlap, nfft=nfft,
            detrend="linear", return_onesided=False, scaling="density", axis=0, mode="complex"
        )
        spec_w54 = np.transpose(spec_w54, [1, 0])[:n_ens, :]

        # (e) Max b2 over mask_B, argmax triad, per-bin p and family-wise p (2000 surrogates)
        b2_54_all, _ = compute_mask_b2_r2(spec_w54, ka_B, kb_idx_B, kc_B)
        top_54_i = int(np.argmax(b2_54_all))
        obs_max_54 = float(b2_54_all[top_54_i])
        ka_54 = int(ka_B[top_54_i])
        kb_54 = int(kb_B[top_54_i])
        kc_54 = int(kc_B[top_54_i])

        n_surr_54 = 2000
        rng_54 = np.random.default_rng(seed_base)
        k_per_bin_54 = np.zeros(M_B, dtype=int)
        k_max_54 = 0

        S, nf = spec_w54.shape
        nh_local = nf // 2
        amps_54 = np.abs(spec_w54)

        for _ in range(n_surr_54):
            rand_phases = rng_54.uniform(0, 2 * np.pi, size=(S, nh_local - 1))
            surr = np.zeros_like(spec_w54, dtype=complex)
            surr[:, 0] = spec_w54[:, 0]
            surr[:, nh_local] = spec_w54[:, nh_local]
            surr[:, 1:nh_local] = amps_54[:, 1:nh_local] * np.exp(1j * rand_phases)
            surr[:, nh_local + 1:] = np.conj(surr[:, 1:nh_local][:, ::-1])

            prod_s = surr[:, ka_B] * surr[:, kb_idx_B]
            num_s = np.abs(np.mean(prod_s * np.conj(surr[:, kc_B]), axis=0)) ** 2
            den_s = np.mean(np.abs(prod_s) ** 2, axis=0) * np.mean(np.abs(surr[:, kc_B]) ** 2, axis=0)
            b2_s_all = num_s / den_s

            k_per_bin_54 += (b2_s_all >= b2_54_all)
            if np.max(b2_s_all) >= obs_max_54:
                k_max_54 += 1

        p_per_bin_54 = (1.0 + k_per_bin_54[top_54_i]) / (1.0 + n_surr_54)
        p_fw_54 = (1.0 + k_max_54) / (1.0 + n_surr_54)

        p_out(f"(e) Domain Search Results (mask_B, 2000 Surrogates):")
        p_out(f"    Observed max b2:      {obs_max_54:.4f}")
        p_out(f"    Argmax triad:         ({ka_54}, {kb_54} -> {kc_54}) = ({ka_54 * df_khz:.2f}, {kb_54 * df_khz:.2f} -> {kc_54 * df_khz:.2f}) kHz")
        p_out(f"    Argmax per-bin p:     {p_per_bin_54:.4f} (k = {k_per_bin_54[top_54_i]}/{n_surr_54})")
        p_out(f"    Family-wise max-b2 p: {p_fw_54:.4f} (k_max = {k_max_54}/{n_surr_54})")

        # (f) b2 and per-bin p in 88654 at the two 88653 argmax triads: (38,12) and (39,37)
        p_out(f"\n(f) 88654 Values at 88653 Argmax Triads (2000 Surrogates):")
        for ka_tgt, kb_tgt, desc in target_triads_88653:
            kc_tgt = (ka_tgt + kb_tgt) % nfft
            match_idx = np.where((ka_B == ka_tgt) & (kb_B == kb_tgt))[0]
            if len(match_idx) > 0:
                idx_tgt = match_idx[0]
                b2_val_54 = float(b2_54_all[idx_tgt])
                p_val_54 = float((1.0 + k_per_bin_54[idx_tgt]) / (1.0 + n_surr_54))
                k_val_54 = int(k_per_bin_54[idx_tgt])
                p_out(f"    Triad ({ka_tgt}, {kb_tgt} -> {kc_tgt}) = ({ka_tgt * df_khz:.2f}, {kb_tgt * df_khz:.2f} -> {kc_tgt * df_khz:.2f}) kHz [{desc}]:")
                p_out(f"      b2 = {b2_val_54:.4f} | per-bin p = {p_val_54:.4f} (k = {k_val_54}/{n_surr_54})")
            else:
                p_out(f"    Triad ({ka_tgt}, {kb_tgt} -> {kc_tgt}) not in mask_B")

    t_ef_elapsed = time.time() - t_ef_start
    p_out(f"\nSections (e) & (f) wall-clock time: {t_ef_elapsed:.2f} s")

    t_global_elapsed = time.time() - t_global_start
    p_out("\n" + "=" * 90)
    p_out(f"Total run_extension.py wall-clock execution time: {t_global_elapsed:.2f} s")
    p_out("========================================================================================================================")

except Exception as e:
    p_out(f"EXCEPTION in run_extension: {traceback.format_exc()}")
finally:
    stdout_h.close()

# Print SHA-256 of stdout file
sha_out = sha256_file(stdout_file)
print(f"Saved stdout to {stdout_file.name} ({stdout_file.stat().st_size} bytes, SHA-256: {sha_out})")
