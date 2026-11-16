import sys
import traceback
import subprocess
import difflib
import re
from pathlib import Path
import numpy as np
import scipy.signal as dsp

ROOT_DIR = Path(__file__).resolve().parents[2]
out_dir = ROOT_DIR / "scratch" / "final15"
out_dir.mkdir(parents=True, exist_ok=True)
staging_dir = out_dir / "staging"
staging_common = ROOT_DIR / "analysis" / "common"
stdout_file = out_dir / "domain_check_stdout.txt"
stdout_h = open(stdout_file, "w", encoding="utf-8")

def p_out(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=stdout_h)

p_out("========================================================================================================================")
p_out("STEP 3: DOMAIN-CONSISTENT RERUN (RUN A REGRESSION AND RUN B HEADLINE DOMAIN)")
p_out("========================================================================================================================")

try:
    # -------------------------------------------------------------------------
    # Setup data and libraries
    # -------------------------------------------------------------------------
    root_dir = ROOT_DIR
    for p_add in [root_dir / "jpack", root_dir / "analysis", root_dir / "analysis" / "common", root_dir / "analysis" / "heating_phase_comparison"]:
        if str(p_add) not in sys.path:
            sys.path.append(str(p_add))

    from comparison_common import load_channel
    sys.path.insert(0, str(staging_common))
    from abicoh2_corrected import abicoh2_corrected

    t_ms, ys_mp1, dt, fs = load_channel(88653, "MP1")

    nfft_512 = 512
    nhalf_512 = 256
    hop_512 = 256
    noverlap_512 = 256
    df_khz_512 = (fs / nfft_512) / 1e3 # 1.953125 kHz
    f1_vec_512 = np.arange(nhalf_512) * df_khz_512
    f2_vec_512 = (np.arange(nfft_512) - nhalf_512) * df_khz_512
    FF1_512, FF2_512 = np.meshgrid(f1_vec_512, f2_vec_512, indexing="ij")
    FF3_512 = np.abs(FF1_512 + FF2_512)

    # Original 4800-bin mask definition
    mask_4800 = (FF1_512 > 0) & (FF1_512 > np.abs(FF2_512)) & (FF3_512 >= 40.0) & (FF3_512 <= 80.0)
    coords_4800 = np.argwhere(mask_4800)
    ka_4800 = coords_4800[:, 0]
    kb_4800 = coords_4800[:, 1] - nhalf_512
    kc_4800 = (ka_4800 + kb_4800) % nfft_512
    kb_idx_4800 = kb_4800 % nfft_512
    M_4800 = len(ka_4800)

    n_ens = 30
    sig_len = hop_512 * (n_ens - 1) + nfft_512
    n_surr = 1000

    # =========================================================================
    # RUN A (REGRESSION): ORIGINAL mask_4800 EXACTLY AS IN SOURCE
    # =========================================================================
    p_out("\n" + "=" * 90)
    p_out("--- RUN A (REGRESSION): ORIGINAL mask_4800 (M = 4800) ---")
    p_out("=" * 90)
    p_out(f"Mask A (mask_4800) bin count M = {M_4800}")

    # Expected values from final11 c_presentation_stdout.txt lines 404-414
    expected_f11 = {
        "ECH+NBI": {
            "obs_max": 0.3152,
            "argmax_triad": (75, -36, 39),
            "per_bin_p": 0.0030,
            "p_fw_max": 0.6314,
            "k_max": 631
        },
        "NBI-only": {
            "obs_max": 0.7824,
            "argmax_triad": (154, -126, 28),
            "per_bin_p": 0.2737,
            "p_fw_max": 0.3856,
            "k_max": 385
        }
    }

    for w_name, t_start, seed in [("ECH+NBI", 259.1, 42), ("NBI-only", 291.3, 123)]:
        idx_w = np.where(t_ms >= t_start)[0]
        y_w = ys_mp1[idx_w][:sig_len]
        _, _, spec_w = dsp.spectrogram(
            y_w, fs=fs, window="hann", nperseg=nfft_512, noverlap=noverlap_512, nfft=nfft_512,
            detrend="linear", return_onesided=False, scaling="density", axis=0, mode="complex"
        )
        spec_w = np.transpose(spec_w, [1, 0])[:n_ens, :]

        b_comp, b2_raw = abicoh2_corrected(spec_w)
        b2_grid = np.transpose(np.fft.fftshift(b2_raw, axes=0), [1, 0])
        b2_obs = b2_grid[coords_4800[:, 0], coords_4800[:, 1]]

        top3_ranks = np.argsort(b2_obs)[::-1][:3]
        obs_max = float(np.max(b2_obs))

        rng = np.random.default_rng(seed)
        S, nf = spec_w.shape
        nh = nf // 2
        amps = np.abs(spec_w)

        k_per_bin = np.zeros(M_4800, dtype=int)
        k_max = 0

        for s in range(n_surr):
            rand_phases = rng.uniform(0, 2 * np.pi, size=(S, nh - 1))
            surr_spec = np.zeros_like(spec_w, dtype=complex)
            surr_spec[:, 0] = spec_w[:, 0]
            surr_spec[:, nh] = spec_w[:, nh]
            surr_spec[:, 1:nh] = amps[:, 1:nh] * np.exp(1j * rand_phases)
            surr_spec[:, nh + 1:] = np.conj(surr_spec[:, 1:nh][:, ::-1])

            prod_s = surr_spec[:, ka_4800] * surr_spec[:, kb_idx_4800]
            num_s = np.abs(np.mean(prod_s * np.conj(surr_spec[:, kc_4800]), axis=0)) ** 2
            den_s = np.mean(np.abs(prod_s) ** 2, axis=0) * np.mean(np.abs(surr_spec[:, kc_4800]) ** 2, axis=0)
            b2_s = num_s / den_s

            k_per_bin += (b2_s >= b2_obs)
            if np.max(b2_s) >= obs_max:
                k_max += 1

        p_per_bin = (1.0 + k_per_bin) / (1.0 + n_surr)
        p_fw_max = (1.0 + k_max) / (1.0 + n_surr)

        top_i = top3_ranks[0]
        ka_top = int(ka_4800[top_i])
        kb_top = int(kb_4800[top_i])
        kc_top = int(kc_4800[top_i])
        top_triad = (ka_top, kb_top, kc_top)
        top_p = float(p_per_bin[top_i])

        p_out(f"\nWindow: {w_name} (start={t_start} ms, N={n_ens}, seed={seed}, n_surr={n_surr})")
        p_out(f"  Observed max b^2 over mask: {obs_max:.4f} | k_max = {k_max} | Family-wise max-b^2 p = {p_fw_max:.4f}")
        for r, idx in enumerate(top3_ranks):
            ka = ka_4800[idx]
            kb = kb_4800[idx]
            kc = (ka + kb) % nfft_512
            triad_str = f"({ka:3d}, {kb:3d} -> {kc:3d})"
            khz_str = f"({ka * df_khz_512:6.2f}, {kb * df_khz_512:6.2f} -> {kc * df_khz_512:6.2f}) kHz"
            p_out(f"  Rank {r + 1}: Triad {triad_str} | {khz_str} | b^2 = {b2_obs[idx]:.4f} | per-bin p = {p_per_bin[idx]:.4f} (k={k_per_bin[idx]}/{n_surr})")

        # Programmatic comparison against final11 output
        exp = expected_f11[w_name]
        match_b2 = (abs(obs_max - exp["obs_max"]) < 1e-4)
        match_triad = (top_triad == exp["argmax_triad"])
        match_perp = (abs(top_p - exp["per_bin_p"]) < 1e-4)
        match_fwp = (abs(p_fw_max - exp["p_fw_max"]) < 1e-4)

        p_out("\n  --- Regression Assertions vs final11 Output ---")
        p_out(f"  Observed max b^2:      Run A = {obs_max:.4f} | Expected = {exp['obs_max']:.4f} | Match = {match_b2}")
        p_out(f"  Argmax Triad:          Run A = {top_triad} | Expected = {exp['argmax_triad']} | Match = {match_triad}")
        p_out(f"  Argmax per-bin p:      Run A = {top_p:.4f} | Expected = {exp['per_bin_p']:.4f} | Match = {match_perp}")
        p_out(f"  Family-wise max-b^2 p: Run A = {p_fw_max:.4f} | Expected = {exp['p_fw_max']:.4f} | Match = {match_fwp}")
        assert match_b2 and match_triad and match_perp and match_fwp, f"Regression mismatch in window {w_name}!"

    # =========================================================================
    # RUN B (HEADLINE-STYLE DOMAIN AT nfft 512): mask_B
    # =========================================================================
    p_out("\n" + "=" * 90)
    p_out("--- RUN B (HEADLINE-STYLE DOMAIN AT N_fft=512): mask_B ---")
    p_out("=" * 90)

    fmin_khz = 5.0
    f1max = 100.0
    f2max = 100.0
    fs_khz = fs / 1e3
    physical_mask_diff = (FF1_512 > 0) & (FF2_512 < 0) & (FF1_512 > -FF2_512)
    search_diff = (
        physical_mask_diff
        & (FF1_512 < f1max)
        & (-FF2_512 < f2max)
        & (FF1_512 >= fmin_khz)
        & (-FF2_512 >= fmin_khz)
        & ((FF1_512 + FF2_512) >= fmin_khz)
    )
    physical_mask_sum = (
        (FF1_512 > 0)
        & (FF2_512 > 0)
        & (FF1_512 >= FF2_512)
        & ((FF1_512 + FF2_512) <= (fs_khz / 2.0))
    )
    search_sum = (
        physical_mask_sum
        & (FF1_512 < f1max)
        & (FF2_512 < f2max)
        & (FF1_512 >= FF2_512)
        & (FF1_512 >= fmin_khz)
        & (FF2_512 >= fmin_khz)
    )
    mask_B = search_sum | search_diff

    coords_B = np.argwhere(mask_B)
    ka_B = coords_B[:, 0]
    kb_B = coords_B[:, 1] - nhalf_512
    kc_B = (ka_B + kb_B) % nfft_512
    kb_idx_B = kb_B % nfft_512
    M_B = len(ka_B)

    p_out(f"Mask B bin count M = {M_B}")

    for w_name, t_start, seed in [("ECH+NBI", 259.1, 42), ("NBI-only", 291.3, 123)]:
        idx_w = np.where(t_ms >= t_start)[0]
        y_w = ys_mp1[idx_w][:sig_len]
        _, _, spec_w = dsp.spectrogram(
            y_w, fs=fs, window="hann", nperseg=nfft_512, noverlap=noverlap_512, nfft=nfft_512,
            detrend="linear", return_onesided=False, scaling="density", axis=0, mode="complex"
        )
        spec_w = np.transpose(spec_w, [1, 0])[:n_ens, :]

        b_comp, b2_raw = abicoh2_corrected(spec_w)
        b2_grid = np.transpose(np.fft.fftshift(b2_raw, axes=0), [1, 0])
        b2_obs_B = b2_grid[coords_B[:, 0], coords_B[:, 1]]

        top_i_B = int(np.argmax(b2_obs_B))
        obs_max_B = float(b2_obs_B[top_i_B])

        rng = np.random.default_rng(seed)
        S, nf = spec_w.shape
        nh = nf // 2
        amps = np.abs(spec_w)

        k_per_bin_B = np.zeros(M_B, dtype=int)
        k_max_B = 0

        for s in range(n_surr):
            rand_phases = rng.uniform(0, 2 * np.pi, size=(S, nh - 1))
            surr_spec = np.zeros_like(spec_w, dtype=complex)
            surr_spec[:, 0] = spec_w[:, 0]
            surr_spec[:, nh] = spec_w[:, nh]
            surr_spec[:, 1:nh] = amps[:, 1:nh] * np.exp(1j * rand_phases)
            surr_spec[:, nh + 1:] = np.conj(surr_spec[:, 1:nh][:, ::-1])

            prod_s = surr_spec[:, ka_B] * surr_spec[:, kb_idx_B]
            num_s = np.abs(np.mean(prod_s * np.conj(surr_spec[:, kc_B]), axis=0)) ** 2
            den_s = np.mean(np.abs(prod_s) ** 2, axis=0) * np.mean(np.abs(surr_spec[:, kc_B]) ** 2, axis=0)
            b2_s = num_s / den_s

            k_per_bin_B += (b2_s >= b2_obs_B)
            if np.max(b2_s) >= obs_max_B:
                k_max_B += 1

        p_per_bin_B = (1.0 + k_per_bin_B) / (1.0 + n_surr)
        p_fw_max_B = (1.0 + k_max_B) / (1.0 + n_surr)

        ka_b = int(ka_B[top_i_B])
        kb_b = int(kb_B[top_i_B])
        kc_b = int(kc_B[top_i_B])
        triad_str_B = f"({ka_b}, {kb_b} -> {kc_b})"
        khz_str_B = f"({ka_b * df_khz_512:.2f}, {kb_b * df_khz_512:.2f} -> {kc_b * df_khz_512:.2f}) kHz"

        # Check whether argmax triad of RUN B lies in mask_4800
        in_4800 = bool(mask_4800[coords_B[top_i_B, 0], coords_B[top_i_B, 1]])

        p_out(f"\nWindow: {w_name} (start={t_start} ms, N={n_ens}, seed={seed}, n_surr={n_surr})")
        p_out(f"  Observed max b^2 over mask_B: {obs_max_B:.4f}")
        p_out(f"  Argmax triad: index {triad_str_B} | frequency {khz_str_B}")
        p_out(f"  Argmax per-bin p: {p_per_bin_B[top_i_B]:.4f} (k={k_per_bin_B[top_i_B]}/{n_surr})")
        p_out(f"  Family-wise max-b^2 p: {p_fw_max_B:.4f} (k_max={k_max_B}/{n_surr})")
        p_out("  Phase-only max-R2: R2 not computed by source")
        p_out(f"  Argmax triad lies in mask_4800: {in_4800}")

        # Specific triad check for NBI-only window
        if w_name == "NBI-only":
            target_ka, target_kb, target_kc = 52, -26, 26
            match_52 = np.where((ka_B == target_ka) & (kb_B == target_kb))[0]
            if len(match_52) > 0:
                idx_52 = match_52[0]
                p_out(f"  Specific triad ({target_ka},{target_kb}->{target_kc}): b^2 = {b2_obs_B[idx_52]:.4f}, per-bin p = {p_per_bin_B[idx_52]:.4f}")
            else:
                p_out(f"  Specific triad ({target_ka},{target_kb}->{target_kc}): ({target_ka},{target_kb},{target_kc}) not in mask_B")

except Exception as e:
    p_out(f"EXCEPTION in run_domain_check: {traceback.format_exc()}")
finally:
    stdout_h.close()
