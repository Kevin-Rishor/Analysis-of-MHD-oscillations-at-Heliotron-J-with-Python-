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
stdout_file = out_dir / "scan_stdout.txt"
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
p_out("TASK FINAL17 - STEP 1: REAL-DATA CONTROL SCAN (SHOTS 88653 & 88654)")
p_out("========================================================================================================================")

try:
    # -------------------------------------------------------------------------
    # Reusing unchanged data loading, mask_B, abicoh2_corrected, and triad function
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
    sig_len = 256 * 29 + 512 # 7936 samples
    df_khz = (fs_53 / nfft) / 1e3 # 1.953125 kHz
    f1_vec = np.arange(nhalf) * df_khz
    f2_vec = (np.arange(nfft) - nhalf) * df_khz
    FF1, FF2 = np.meshgrid(f1_vec, f2_vec, indexing="ij")

    # Define mask_B exactly as in final15 and final16
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
    p_out("Assertion M == 2306 passed: True")

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
        return b2

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

    candidate_triads = [(39, 37), (38, 12), (38, 38), (46, 46), (46, 43)]
    states_order = [
        "pre-NBI",
        "NBI-only (pre-ECH)",
        "ECH+NBI",
        "NBI-only (post-ECH)",
        "post-NBI",
        "transition"
    ]

    shots_data = [
        (88653, t_ms_53, ys_mp1_53, dt_53, fs_53),
        (88654, t_ms_54, ys_mp1_54, dt_54, fs_54)
    ]

    for shot_id, t_ms, ys, dt, fs in shots_data:
        p_out("\n" + "=" * 120)
        p_out(f"--- SHOT #{shot_id} (MP1) CONTROL SCAN ---")
        p_out("=" * 120)

        # Start at first sample with t >= 20.0 ms
        i0_candidates = np.where(t_ms >= 20.0)[0]
        assert len(i0_candidates) > 0, "No sample with t >= 20.0 ms found!"
        i0 = int(i0_candidates[0])
        total_complete_windows = (len(ys) - i0) // sig_len

        p_out(f"Start index: {i0} (t = {t_ms[i0]:.3f} ms), Record length: {len(ys)} samples, sig_len = {sig_len} samples ({sig_len * dt * 1000.0:.3f} ms)")
        p_out(f"Total complete non-overlapping windows: {total_complete_windows}\n")

        # Table header
        hdr = (
            f"{'Win':<4} | {'Start(ms)':<9} | {'End(ms)':<9} | {'State':<20} | "
            f"{'Max b2':<8} | {'Argmax Triad (idx & kHz)':<40} | "
            f"{'b2(39,37)':<10} | {'b2(38,12)':<10} | {'b2(38,38)':<10} | {'b2(46,46)':<10} | {'b2(46,43)':<10} | "
            f"{'Mode Pwr (80-98k)'}"
        )
        p_out(hdr)
        p_out("-" * len(hdr))

        window_records = []

        for w in range(total_complete_windows):
            i_start = i0 + w * sig_len
            i_end = i_start + sig_len
            w_start = float(t_ms[i_start])
            w_end = float(t_ms[i_start] + sig_len * dt * 1000.0)
            state_str = classify_state(w_start, w_end)

            y_w = ys[i_start:i_end]
            _, _, spec_w = dsp.spectrogram(
                y_w, fs=fs, window="hann", nperseg=nfft, noverlap=noverlap, nfft=nfft,
                detrend="linear", return_onesided=False, scaling="density", axis=0, mode="complex"
            )
            spec_w = np.transpose(spec_w, [1, 0])[:n_ens, :]

            # Bicoherence over mask_B
            _, b2_raw = abicoh2_corrected(spec_w)
            b2_grid = np.transpose(np.fft.fftshift(b2_raw, axes=0), [1, 0])
            b2_obs = b2_grid[coords_B[:, 0], coords_B[:, 1]]

            top_i = int(np.argmax(b2_obs))
            max_b2_w = float(b2_obs[top_i])
            ka_w = int(ka_B[top_i])
            kb_w = int(kb_B[top_i])
            kc_w = int(kc_B[top_i])
            triad_str = f"({ka_w:2d},{kb_w:3d}->{kc_w:2d})={ka_w*df_khz:5.1f},{kb_w*df_khz:5.1f}->{kc_w*df_khz:5.1f}k"

            # b2 for five candidate triads
            b2_cands = {}
            for c_ka, c_kb in candidate_triads:
                b2_cands[(c_ka, c_kb)] = compute_direct_triad(spec_w, c_ka, c_kb)

            # Mode power: mean over segments of periodogram density summed over bins 41..50 (80-98 kHz)
            p_band_s = np.sum(np.abs(spec_w[:, 41:51]) ** 2, axis=1)
            mode_power = float(np.mean(p_band_s))

            row_str = (
                f"{w:<4d} | {w_start:<9.3f} | {w_end:<9.3f} | {state_str:<20} | "
                f"{max_b2_w:<8.4f} | {triad_str:<40} | "
                f"{b2_cands[(39,37)]:<10.4f} | {b2_cands[(38,12)]:<10.4f} | {b2_cands[(38,38)]:<10.4f} | "
                f"{b2_cands[(46,46)]:<10.4f} | {b2_cands[(46,43)]:<10.4f} | "
                f"{mode_power:12.4e}"
            )
            p_out(row_str)

            window_records.append({
                "win": w,
                "start": w_start,
                "end": w_end,
                "state": state_str,
                "max_b2": max_b2_w,
                "argmax_triad": (ka_w, kb_w, kc_w),
                "b2_cands": b2_cands,
                "mode_power": mode_power
            })

        # Summary statistics per state
        p_out("\n" + "-" * 105)
        p_out(f"SUMMARY BY STATE FOR SHOT #{shot_id}")
        p_out("-" * 105)
        sum_hdr = (
            f"{'State':<22} | {'N_win':<6} | {'Min b2':<8} | {'Median b2':<10} | {'90th% b2':<10} | "
            f"{'Max b2':<8} | {'N(>=0.3351)':<12} | {'N(>=0.7119)'}"
        )
        p_out(sum_hdr)
        p_out("-" * len(sum_hdr))

        for st in states_order:
            st_b2s = [rec["max_b2"] for rec in window_records if rec["state"] == st]
            n_w = len(st_b2s)
            if n_w > 0:
                min_b = float(np.min(st_b2s))
                med_b = float(np.median(st_b2s))
                p90_b = float(np.percentile(st_b2s, 90))
                max_b = float(np.max(st_b2s))
                c_3351 = int(np.sum(np.array(st_b2s) >= 0.3351))
                c_7119 = int(np.sum(np.array(st_b2s) >= 0.7119))
                p_out(
                    f"{st:<22} | {n_w:<6d} | {min_b:<8.4f} | {med_b:<10.4f} | {p90_b:<10.4f} | "
                    f"{max_b:<8.4f} | {c_3351:<12d} | {c_7119:<10d}"
                )
            else:
                p_out(f"{st:<22} | {0:<6d} | {'N/A':<8} | {'N/A':<10} | {'N/A':<10} | {'N/A':<8} | {0:<12d} | {0:<10d}")

        # Total summary across all states
        all_b2s = [rec["max_b2"] for rec in window_records]
        tot_min = float(np.min(all_b2s))
        tot_med = float(np.median(all_b2s))
        tot_p90 = float(np.percentile(all_b2s, 90))
        tot_max = float(np.max(all_b2s))
        tot_c3351 = int(np.sum(np.array(all_b2s) >= 0.3351))
        tot_c7119 = int(np.sum(np.array(all_b2s) >= 0.7119))
        p_out("-" * len(sum_hdr))
        p_out(
            f"{'ALL STATES (TOTAL)':<22} | {len(all_b2s):<6d} | {tot_min:<8.4f} | {tot_med:<10.4f} | {tot_p90:<10.4f} | "
            f"{tot_max:<8.4f} | {tot_c3351:<12d} | {tot_c7119:<10d}"
        )

    t_elapsed = time.time() - t_start_wall
    p_out("\n" + "=" * 90)
    p_out(f"Total run_scan.py execution time: {t_elapsed:.2f} s")
    p_out("========================================================================================================================")

except Exception as e:
    p_out(f"EXCEPTION in run_scan: {traceback.format_exc()}")
finally:
    stdout_h.close()

sha_out = sha256_file(stdout_file)
print(f"Saved stdout to {stdout_file.name} ({stdout_file.stat().st_size} bytes, SHA-256: {sha_out})")
