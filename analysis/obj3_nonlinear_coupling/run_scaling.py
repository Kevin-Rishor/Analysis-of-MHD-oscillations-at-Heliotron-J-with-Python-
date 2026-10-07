import sys
import time
import hashlib
import traceback
from pathlib import Path
import numpy as np
import scipy.signal as dsp

t_start_wall = time.time()

base_dir = Path(r"C:\Users\kevin\.gemini\antigravity\brain\5a23ddc3-ffcc-4a9b-a7f0-d135446e46c3")
out_dir = base_dir / "scratch" / "final19"
out_dir.mkdir(parents=True, exist_ok=True)
stdout_file = out_dir / "scaling_stdout.txt"
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
p_out("TASK FINAL19 - STEP 2: HARMONIC AMPLITUDE SCALING & BIPHASE DYNAMICS (276.0 - 300.8 ms)")
p_out("========================================================================================================================")

try:
    root_dir = Path("C:/TFG")
    for p_add in [
        root_dir / "jpack",
        root_dir / "analysis",
        root_dir / "analysis" / "common",
        root_dir / "analysis" / "heating_phase_comparison",
    ]:
        if str(p_add) not in sys.path:
            sys.path.append(str(p_add))

    from comparison_common import load_channel
    staging_common = base_dir / "scratch" / "final11" / "staging" / "analysis" / "common"
    sys.path.insert(0, str(staging_common))

    t_ms_53, ys_mp1_53, dt_53, fs_53 = load_channel(88653, "MP1")
    t_ms_54, ys_mp1_54, dt_54, fs_54 = load_channel(88654, "MP1")

    nfft = 512
    nhalf = 256
    hop = 256
    noverlap = 256
    df_khz = (fs_53 / nfft) / 1e3  # 1.953125 kHz

    # Triad bins
    ka = 46
    kb = 46
    kc = 92

    # Stretch definition: 276.0 to ~300.8 ms
    # With hop = 256 samples (0.256 ms), (300.8 - 276.0) / 0.256 = 96.875
    # Choosing n_seg = 97 segments:
    # Segments s = 0..96 start from 276.000 ms to 300.576 ms (last segment ends at 301.088 ms; center at 300.832 ms)
    # sig_len_stretch = 256 * (n_seg - 1) + 512 = 25088 samples = 25.088 ms
    n_seg = 97
    sig_len_stretch = hop * (n_seg - 1) + nfft

    p_out(f"\nStretch parameters:")
    p_out(f"  Nominal time range:    276.000 - 300.800 ms")
    p_out(f"  Segment spacing (hop): {hop} samples ({hop * dt_53 * 1000.0:.3f} ms)")
    p_out(f"  FFT window length:     {nfft} samples ({nfft * dt_53 * 1000.0:.3f} ms)")
    p_out(f"  Number of segments:    {n_seg}")
    p_out(f"  Total signal length:   {sig_len_stretch} samples ({sig_len_stretch * dt_53 * 1000.0:.3f} ms)")
    p_out(f"  Triad (ka, ka -> 2ka): ({ka}, {kb} -> {kc}) = ({ka*df_khz:.2f}, {kb*df_khz:.2f} -> {kc*df_khz:.2f}) kHz")

    def analyze_stretch(shot_id, t_ms, ys, fs):
        idx0 = int(np.where(t_ms >= 276.0)[0][0])
        w_start_ms = float(t_ms[idx0])
        y_stretch = ys[idx0 : idx0 + sig_len_stretch]

        _, _, spec = dsp.spectrogram(
            y_stretch, fs=fs, window="hann", nperseg=nfft, noverlap=noverlap, nfft=nfft,
            detrend="linear", return_onesided=False, scaling="density", axis=0, mode="complex"
        )
        spec = np.transpose(spec, [1, 0])[:n_seg, :]

        seg_starts = np.array([w_start_ms + s * (hop * dt_53 * 1000.0) for s in range(n_seg)])
        x46_sq = np.abs(spec[:, ka]) ** 2
        x92_sq = np.abs(spec[:, kc]) ** 2
        z_s = spec[:, ka] ** 2 * np.conj(spec[:, kc])
        abs_z = np.abs(z_s)
        angles_rad = np.angle(z_s)
        angles_deg = np.degrees(angles_rad)

        return {
            "shot_id": shot_id,
            "w_start_ms": w_start_ms,
            "seg_starts": seg_starts,
            "x46_sq": x46_sq,
            "x92_sq": x92_sq,
            "z_s": z_s,
            "abs_z": abs_z,
            "angles_rad": angles_rad,
            "angles_deg": angles_deg,
        }

    res53 = analyze_stretch(88653, t_ms_53, ys_mp1_53, fs_53)
    res54 = analyze_stretch(88654, t_ms_54, ys_mp1_54, fs_54)

    # 1. Print per-segment table for each shot
    for res in [res53, res54]:
        shot_id = res["shot_id"]
        p_out("\n" + "=" * 120)
        p_out(f"SHOT #{shot_id} | PER-SEGMENT TABLE (276.0 - 300.8 ms, n_seg = {n_seg})")
        p_out("=" * 120)

        hdr = (
            f"{'Seg':<4} | {'Start (ms)':<11} | {'|X(46)|^2':<14} | {'|X(92)|^2':<14} | "
            f"{'|z_s|':<14} | {'Angle of z_s (deg)':<18}"
        )
        p_out(hdr)
        p_out("-" * len(hdr))

        for s in range(n_seg):
            row_str = (
                f"{s:<4d} | {res['seg_starts'][s]:<11.3f} | {res['x46_sq'][s]:<14.4e} | "
                f"{res['x92_sq'][s]:<14.4e} | {res['abs_z'][s]:<14.4e} | {res['angles_deg'][s]:+18.2f}"
            )
            p_out(row_str)

    # 2. Statistical analysis per shot: OLS slope, R2, circular mean, circular std
    def circ_metrics(arr_rad):
        R_complex = np.mean(np.exp(1j * arr_rad))
        c_mean_deg = float(np.degrees(np.angle(R_complex)))
        R_mag = np.abs(R_complex)
        c_std_deg = float(np.degrees(np.sqrt(-2.0 * np.log(R_mag)))) if R_mag > 0 else 180.0
        return c_mean_deg, c_std_deg

    def run_regressions_and_circ(res):
        seg_starts = res["seg_starts"]
        x46 = res["x46_sq"]
        x92 = res["x92_sq"]
        ang_rad = res["angles_rad"]

        m_all = np.ones(len(seg_starts), dtype=bool)
        m_pre = seg_starts < 291.2
        m_post = seg_starts >= 291.2

        subsets = [
            ("All Segments (276.0 - 300.8 ms)", m_all),
            ("Restricted: Start < 291.2 ms (ECH+NBI / Transition)", m_pre),
            ("Restricted: Start >= 291.2 ms (NBI-only / Post-ECH)", m_post),
        ]

        stats_list = []
        for name, m in subsets:
            n_sub = int(np.sum(m))
            lx46 = np.log10(x46[m])
            lx92 = np.log10(x92[m])
            # OLS linear regression
            slope, intercept = np.polyfit(lx46, lx92, 1)
            # R2 = square of Pearson correlation
            r_val = float(np.corrcoef(lx46, lx92)[0, 1])
            r2 = r_val ** 2
            # Circular stats
            c_mean, c_std = circ_metrics(ang_rad[m])

            stats_list.append({
                "name": name,
                "N": n_sub,
                "slope": float(slope),
                "intercept": float(intercept),
                "r": r_val,
                "r2": r2,
                "circ_mean": c_mean,
                "circ_std": c_std,
            })
        return stats_list

    p_out("\n" + "=" * 120)
    p_out("OLS SCALING OF log10|X(92)|^2 VS log10|X(46)|^2 & BIPHASE CIRCULAR STATISTICS")
    p_out("=" * 120)

    all_stats = {}
    for res in [res53, res54]:
        shot_id = res["shot_id"]
        st_list = run_regressions_and_circ(res)
        all_stats[shot_id] = st_list

        p_out(f"\n--- Shot #{shot_id} ---")
        p_out(f"{'Subset':<50} | {'N':<4} | {'OLS Slope':<10} | {'R^2':<8} | {'Circ Mean (deg)':<16} | {'Circ Std (deg)'}")
        p_out("-" * 110)
        for s in st_list:
            p_out(
                f"{s['name']:<50} | {s['N']:<4d} | {s['slope']:<+10.4f} | {s['r2']:<8.4f} | "
                f"{s['circ_mean']:<+16.2f} | {s['circ_std']:<14.2f}"
            )

    # 3. Circular mean difference between the two shots (all segments)
    mean53_all = all_stats[88653][0]["circ_mean"]
    mean54_all = all_stats[88654][0]["circ_mean"]
    diff_circ_signed = float(np.degrees(np.angle(np.exp(1j * np.radians(mean53_all - mean54_all)))))
    diff_circ_abs = abs(diff_circ_signed)

    p_out("\n" + "=" * 120)
    p_out("CIRCULAR MEAN BIPHASE DIFFERENCE BETWEEN SHOTS (ALL SEGMENTS, N = 97)")
    p_out("=" * 120)
    p_out(f"Shot 88653 Circular Mean Biphase: {mean53_all:+.2f} deg")
    p_out(f"Shot 88654 Circular Mean Biphase: {mean54_all:+.2f} deg")
    p_out(f"Signed Circular Difference (88653 - 88654): {diff_circ_signed:+.2f} deg")
    p_out(f"Absolute Circular Difference:               {diff_circ_abs:.2f} deg")

    wall_elapsed = time.time() - t_start_wall
    p_out("\n" + "=" * 120)
    p_out(f"Total run_scaling.py wall-clock execution time: {wall_elapsed:.2f} s")
    p_out("=" * 120)

except Exception as ex:
    p_out(f"\nERROR: Exception occurred during run_scaling.py execution:")
    p_out(traceback.format_exc())
    stdout_h.close()
    sys.exit(1)

stdout_h.close()
raw_bytes = stdout_file.read_bytes()
out_sha = hashlib.sha256(raw_bytes).hexdigest()
print(f"Saved stdout to scaling_stdout.txt ({len(raw_bytes)} bytes, SHA-256: {out_sha})")
