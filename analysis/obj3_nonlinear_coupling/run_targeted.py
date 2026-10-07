import sys
import time
import hashlib
import traceback
from pathlib import Path
import numpy as np
import scipy.signal as dsp

t_start_wall = time.time()

base_dir = Path(r"C:\Users\kevin\.gemini\antigravity\brain\5a23ddc3-ffcc-4a9b-a7f0-d135446e46c3")
out_dir = base_dir / "scratch" / "final17"
stdout_file = out_dir / "targeted_stdout.txt"
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
p_out("TASK FINAL17 - STEP 2: TARGETED HARMONIC & SPECTRAL SUPPORT ANALYSES")
p_out("========================================================================================================================")

try:
    # -------------------------------------------------------------------------
    # Helper functions copied unchanged from run_extension.py
    # -------------------------------------------------------------------------
    root_dir = Path("C:/TFG")
    for p_add in [root_dir / "jpack", root_dir / "analysis", root_dir / "analysis" / "common", root_dir / "analysis" / "heating_phase_comparison"]:
        if str(p_add) not in sys.path:
            sys.path.append(str(p_add))

    from comparison_common import load_channel
    staging_common = base_dir / "scratch" / "final11" / "staging" / "analysis" / "common"
    sys.path.insert(0, str(staging_common))
    from abicoh2_corrected import abicoh2_corrected

    t_ms_53, ys_mp1_53, dt_53, fs_53 = load_channel(88653, "MP1")
    t_ms_54, ys_mp1_54, dt_54, fs_54 = load_channel(88654, "MP1")

    nfft = 512
    nhalf = 256
    hop = 256
    noverlap = 256
    n_ens = 30
    sig_len = hop * (n_ens - 1) + nfft
    df_khz = (fs_53 / nfft) / 1e3 # 1.953125 kHz

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

    # Vectorized multi-triad b2 computation
    def compute_triads_b2(spec, ka_arr, kb_arr, kc_arr):
        kb_idx_arr = kb_arr % nfft
        prod_s = spec[:, ka_arr] * spec[:, kb_idx_arr]
        z_s = prod_s * np.conj(spec[:, kc_arr])
        num = np.abs(np.mean(z_s, axis=0)) ** 2
        den = np.mean(np.abs(prod_s) ** 2, axis=0) * np.mean(np.abs(spec[:, kc_arr]) ** 2, axis=0)
        return np.where(den > 0, num / den, 0.0)

    # Four core analyses definitions
    analyses = [
        ("88653 ECH+NBI", 88653, t_ms_53, ys_mp1_53, fs_53, 259.1, 42),
        ("88653 NBI-only", 88653, t_ms_53, ys_mp1_53, fs_53, 291.3, 123),
        ("88654 ECH+NBI", 88654, t_ms_54, ys_mp1_54, fs_54, 259.1, 42),
        ("88654 NBI-only", 88654, t_ms_54, ys_mp1_54, fs_54, 291.3, 123),
    ]

    # Pre-compute spectrograms for the four analyses
    specs_dict = {}
    for label, shot_id, t_ms, ys, fs, t_start, seed in analyses:
        idx_w = np.where(t_ms >= t_start)[0]
        y_w = ys[idx_w][:sig_len]
        _, _, spec = dsp.spectrogram(
            y_w, fs=fs, window="hann", nperseg=nfft, noverlap=noverlap, nfft=nfft,
            detrend="linear", return_onesided=False, scaling="density", axis=0, mode="complex"
        )
        spec = np.transpose(spec, [1, 0])[:n_ens, :]
        specs_dict[label] = spec

    # -------------------------------------------------------------------------
    # (a) Harmonic family F_h = { (ka,kb) : ka = kb, ka in 40..50 } (11 triads)
    # -------------------------------------------------------------------------
    p_out("\n" + "=" * 100)
    p_out("--- (a) HARMONIC FAMILY F_h = { (ka,kb) : ka = kb, ka in 40..50 } (5000 Surrogates) ---")
    p_out("=" * 100)
    t_a_start = time.time()

    ka_harm = np.arange(40, 51) # 40..50 (11 triads)
    kb_harm = ka_harm
    kc_harm = 2 * ka_harm

    for label, shot_id, t_ms, ys, fs, t_start, seed in analyses:
        p_out("\n" + "-" * 90)
        p_out(f"Analysis: {label} (start = {t_start} ms, seed = {seed}, N = 30)")
        p_out("-" * 90)

        spec = specs_dict[label]
        b2_obs_harm = compute_triads_b2(spec, ka_harm, kb_harm, kc_harm)

        # 5000 surrogates
        n_surr = 5000
        rng = np.random.default_rng(seed)
        S, nf = spec.shape
        nh_local = nf // 2
        amps = np.abs(spec)

        k_per_bin_harm = np.zeros(len(ka_harm), dtype=int)
        k_fam_max = 0
        obs_fam_max = float(np.max(b2_obs_harm))

        for _ in range(n_surr):
            rand_phases = rng.uniform(0, 2 * np.pi, size=(S, nh_local - 1))
            surr = np.zeros_like(spec, dtype=complex)
            surr[:, 0] = spec[:, 0]
            surr[:, nh_local] = spec[:, nh_local]
            surr[:, 1:nh_local] = amps[:, 1:nh_local] * np.exp(1j * rand_phases)
            surr[:, nh_local + 1:] = np.conj(surr[:, 1:nh_local][:, ::-1])

            b2_s = compute_triads_b2(surr, ka_harm, kb_harm, kc_harm)
            k_per_bin_harm += (b2_s >= b2_obs_harm)
            if np.max(b2_s) >= obs_fam_max:
                k_fam_max += 1

        p_per_bin_harm = (1.0 + k_per_bin_harm) / (1.0 + n_surr)
        p_fam_max = (1.0 + k_fam_max) / (1.0 + n_surr)

        # Table of 11 harmonic triads
        p_out(f"{'Triad (ka,ka -> 2ka)':<24} | {'Frequencies (kHz)':<36} | {'b2':<8} | {'per-bin p (k/5000)'}")
        p_out("-" * 88)
        for i in range(len(ka_harm)):
            ka_i = ka_harm[i]
            kc_i = kc_harm[i]
            t_str = f"({ka_i:2d}, {ka_i:2d} -> {kc_i:2d})"
            f_str = f"({ka_i*df_khz:5.2f}, {ka_i*df_khz:5.2f} -> {kc_i*df_khz:6.2f}) kHz"
            p_out(f"{t_str:<24} | {f_str:<36} | {b2_obs_harm[i]:<8.4f} | {p_per_bin_harm[i]:.4f} (k={k_per_bin_harm[i]:<4d})")

        argmax_fam_i = int(np.argmax(b2_obs_harm))
        ka_best = ka_harm[argmax_fam_i]
        kc_best = kc_harm[argmax_fam_i]
        p_out(f"\nFamily Maximum b2: {obs_fam_max:.4f} at triad ({ka_best}, {ka_best} -> {kc_best}) = ({ka_best*df_khz:.2f}, {ka_best*df_khz:.2f} -> {kc_best*df_khz:.2f}) kHz")
        p_out(f"Family-wise max-b2 p over F_h: {p_fam_max:.4f} (k_max = {k_fam_max}/{n_surr})")

    t_a_elapsed = time.time() - t_a_start
    p_out(f"\nSection (a) wall-clock time: {t_a_elapsed:.2f} s")

    # -------------------------------------------------------------------------
    # (b) Spectral support at f3 for the five candidate triads
    # -------------------------------------------------------------------------
    p_out("\n" + "=" * 100)
    p_out("--- (b) SPECTRAL SUPPORT AT f3 FOR CANDIDATE TRIADS ---")
    p_out("=" * 100)
    t_b_start = time.time()

    candidate_triads = [(39, 37), (38, 12), (38, 38), (46, 46), (46, 43)]

    for label, shot_id, t_ms, ys, fs, t_start, seed in analyses:
        p_out("\n" + "-" * 90)
        p_out(f"Analysis: {label} (start = {t_start} ms)")
        p_out("-" * 90)

        spec = specs_dict[label]
        # Mean periodogram power across the 30 segments for all bins
        P_all = np.mean(np.abs(spec) ** 2, axis=0) # length 512

        p_out(f"{'Triad (ka,kb)':<16} | {'kc':<4} | {'f3 (kHz)':<9} | {'Mean |spec[:,kc]|^2':<22} | {'Median Bg Power':<18} | {'Support Ratio (P_kc / P_bg)'}")
        p_out("-" * 100)

        for ka_c, kb_c in candidate_triads:
            kc_c = (ka_c + kb_c) % nfft
            P_kc = float(P_all[kc_c])

            # Neighborhood kc-20..kc+20 excluding kc-2..kc+2
            bg_bins = [k for k in range(kc_c - 20, kc_c + 21) if 0 <= k < nfft and abs(k - kc_c) > 2]
            P_bg_med = float(np.median([P_all[k] for k in bg_bins]))
            ratio = float(P_kc / P_bg_med) if P_bg_med > 0 else 0.0

            triad_str = f"({ka_c:2d}, {kb_c:2d})"
            p_out(f"{triad_str:<16} | {kc_c:<4d} | {kc_c*df_khz:<9.2f} | {P_kc:<22.4e} | {P_bg_med:<18.4e} | {ratio:.4f}")

    t_b_elapsed = time.time() - t_b_start
    p_out(f"\nSection (b) wall-clock time: {t_b_elapsed:.2f} s")

    # -------------------------------------------------------------------------
    # (c) Replication stability in shot 88654 for start shifts k=0..6 (b2 only)
    # -------------------------------------------------------------------------
    p_out("\n" + "=" * 100)
    p_out("--- (c) REPLICATION STABILITY IN SHOT 88654 (START SHIFTS k=0..6, b2 ONLY) ---")
    p_out("=" * 100)
    t_c_start = time.time()

    # 1. Triad (39, 37) in 88654 ECH+NBI (nominal start 259.10 ms)
    p_out("\nTriad (39, 37) in Shot 88654 ECH+NBI (starts 259.10 + k*0.25 ms):")
    p_out(f"{'k':<3} | {'Start (ms)':<12} | {'End (ms)':<12} | {'b2 (39, 37)'}")
    p_out("-" * 45)
    for k in range(7):
        t_k = 259.10 + k * 0.25
        idx_k = np.where(t_ms_54 >= t_k)[0]
        y_k = ys_mp1_54[idx_k][:sig_len]
        _, _, sp_k = dsp.spectrogram(
            y_k, fs=fs_54, window="hann", nperseg=nfft, noverlap=noverlap, nfft=nfft,
            detrend="linear", return_onesided=False, scaling="density", axis=0, mode="complex"
        )
        sp_k = np.transpose(sp_k, [1, 0])[:n_ens, :]
        _, b2_k, _ = compute_direct_triad(sp_k, 39, 37)
        p_out(f"{k:<3d} | {t_k:<12.2f} | {t_k + sig_len*dt_54*1000.0:<12.2f} | {b2_k:.4f}")

    # 2. Triad (38, 12) in 88654 NBI-only (nominal start 291.30 ms)
    p_out("\nTriad (38, 12) in Shot 88654 NBI-only (starts 291.30 + k*0.25 ms):")
    p_out(f"{'k':<3} | {'Start (ms)':<12} | {'End (ms)':<12} | {'b2 (38, 12)'}")
    p_out("-" * 45)
    for k in range(7):
        t_k = 291.30 + k * 0.25
        idx_k = np.where(t_ms_54 >= t_k)[0]
        y_k = ys_mp1_54[idx_k][:sig_len]
        _, _, sp_k = dsp.spectrogram(
            y_k, fs=fs_54, window="hann", nperseg=nfft, noverlap=noverlap, nfft=nfft,
            detrend="linear", return_onesided=False, scaling="density", axis=0, mode="complex"
        )
        sp_k = np.transpose(sp_k, [1, 0])[:n_ens, :]
        _, b2_k, _ = compute_direct_triad(sp_k, 38, 12)
        p_out(f"{k:<3d} | {t_k:<12.2f} | {t_k + sig_len*dt_54*1000.0:<12.2f} | {b2_k:.4f}")

    t_c_elapsed = time.time() - t_c_start
    p_out(f"\nSection (c) wall-clock time: {t_c_elapsed:.2f} s")

    # -------------------------------------------------------------------------
    # (d) Per-bin p of triad (39, 37) in shot 88654 ECH+NBI (5000 Surrogates, Seeds 42, 43, 44)
    # -------------------------------------------------------------------------
    p_out("\n" + "=" * 100)
    p_out("--- (d) PER-BIN p OF TRIAD (39, 37) IN SHOT 88654 ECH+NBI (5000 Surrogates) ---")
    p_out("=" * 100)
    t_d_start = time.time()

    spec_54_ech = specs_dict["88654 ECH+NBI"]
    _, b2_54_39_37, _ = compute_direct_triad(spec_54_ech, 39, 37)
    p_out(f"Observed b2(39, 37) in 88654 ECH+NBI (start = 259.10 ms): {b2_54_39_37:.4f}\n")

    p_out(f"{'Seed':<8} | {'k (exceed)':<12} | {'n_surr':<8} | {'p-value':<12} | {'Binomial Std Error'}")
    p_out("-" * 65)

    n_surr_d = 5000
    S, nf = spec_54_ech.shape
    nh_local = nf // 2
    amps = np.abs(spec_54_ech)

    for s_seed in [42, 43, 44]:
        rng_d = np.random.default_rng(s_seed)
        k_d = 0

        for _ in range(n_surr_d):
            rand_phases = rng_d.uniform(0, 2 * np.pi, size=(S, nh_local - 1))
            surr = np.zeros_like(spec_54_ech, dtype=complex)
            surr[:, 0] = spec_54_ech[:, 0]
            surr[:, nh_local] = spec_54_ech[:, nh_local]
            surr[:, 1:nh_local] = amps[:, 1:nh_local] * np.exp(1j * rand_phases)
            surr[:, nh_local + 1:] = np.conj(surr[:, 1:nh_local][:, ::-1])

            p_s = surr[:, 39] * surr[:, 37 % nfft]
            num = np.abs(np.mean(p_s * np.conj(surr[:, (39 + 37) % nfft]))) ** 2
            den = np.mean(np.abs(p_s) ** 2) * np.mean(np.abs(surr[:, (39 + 37) % nfft]) ** 2)
            b2_s = num / den if den > 0 else 0.0

            if b2_s >= b2_54_39_37:
                k_d += 1

        p_val_d = (1.0 + k_d) / (1.0 + n_surr_d)
        binom_se = np.sqrt(p_val_d * (1.0 - p_val_d) / n_surr_d)
        p_out(f"{s_seed:<8d} | {k_d:<12d} | {n_surr_d:<8d} | {p_val_d:<12.6f} | {binom_se:.6f}")

    t_d_elapsed = time.time() - t_d_start
    p_out(f"\nSection (d) wall-clock time: {t_d_elapsed:.2f} s")

    t_global = time.time() - t_start_wall
    p_out("\n" + "=" * 90)
    p_out(f"Total run_targeted.py wall-clock execution time: {t_global:.2f} s")
    p_out("========================================================================================================================")

except Exception as e:
    p_out(f"EXCEPTION in run_targeted: {traceback.format_exc()}")
finally:
    stdout_h.close()

sha_out = sha256_file(stdout_file)
print(f"Saved stdout to {stdout_file.name} ({stdout_file.stat().st_size} bytes, SHA-256: {sha_out})")
