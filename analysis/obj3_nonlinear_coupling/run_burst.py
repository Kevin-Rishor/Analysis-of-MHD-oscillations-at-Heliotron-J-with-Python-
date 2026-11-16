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
stdout_file = out_dir / "burst_stdout.txt"
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
p_out("TASK FINAL19 - STEP 3: BURST PROFILE & MULTI-CHANNEL MIRNOV ANALYSIS (264.0 - 268.0 ms)")
p_out("========================================================================================================================")

try:
    root_dir = Path(__file__).resolve().parents[2]
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

    nfft = 512
    nhalf = 256
    hop = 256
    noverlap = 256
    n_ens = 30
    sig_len = 256 * 29 + 512  # 7936 samples = 7.936 ms
    kc_target = 76  # 76 * 1.953125 kHz = 148.4375 kHz
    df_khz = 1000.0 / nfft  # 1.953125 kHz

    # Broadband context bins: 66..86 inclusive excluding 74..78 inclusive
    bb_bins = np.concatenate([np.arange(66, 74), np.arange(79, 87)])

    channels = ["MP1", "MP2", "MP3", "MP4"]
    shots = [88653, 88654]

    for shot_id in shots:
        p_out("\n" + "=" * 120)
        p_out(f"SHOT #{shot_id} | BURST ANALYSIS IN STRETCH 264.0 - 268.0 ms")
        p_out("=" * 120)

        # Load channel data
        ch_data = {}
        t_ms_ref = None
        fs_ref = None
        dt_ref = None

        for ch in channels:
            try:
                t_m, y_s, dt, fs = load_channel(shot_id, ch)
                ch_data[ch] = (t_m, y_s, dt, fs)
                if t_ms_ref is None:
                    t_ms_ref = t_m
                    fs_ref = fs
                    dt_ref = dt
            except Exception as e:
                ch_data[ch] = "MISSING"

        idx0 = int(np.where(t_ms_ref >= 264.0)[0][0])
        w_start_ms = float(t_ms_ref[idx0])

        p_out(f"Window Start: {w_start_ms:.3f} ms, Target bin kc = {kc_target} ({kc_target * df_khz:.2f} kHz)")
        p_out(f"Broadband context: bins 66..86 excluding 74..78 ({len(bb_bins)} bins, {66*df_khz:.1f}-{86*df_khz:.1f} kHz)")

        # Compute spectrograms for available channels
        specs = {}
        for ch in channels:
            if ch_data[ch] == "MISSING":
                specs[ch] = None
            else:
                _, y_s, _, fs = ch_data[ch]
                y_w = y_s[idx0 : idx0 + sig_len]
                _, _, s_raw = dsp.spectrogram(
                    y_w, fs=fs, window="hann", nperseg=nfft, noverlap=noverlap, nfft=nfft,
                    detrend="linear", return_onesided=False, scaling="density", axis=0, mode="complex"
                )
                specs[ch] = np.transpose(s_raw, [1, 0])[:n_ens, :]

        # Segment start times
        seg_starts = [w_start_ms + s * (hop * dt_ref * 1000.0) for s in range(n_ens)]
        # Filter segments in stretch 264.0 - 268.0 ms (start time <= 268.0 ms)
        stretch_indices = [s for s in range(n_ens) if seg_starts[s] <= 268.0]
        n_stretch = len(stretch_indices)

        p_out(f"\n--- Segments in Stretch 264.0 - 268.0 ms (N = {n_stretch} segments, starts {seg_starts[0]:.3f} to {seg_starts[stretch_indices[-1]]:.3f} ms) ---")

        hdr = (
            f"{'Seg':<4} | {'Start (ms)':<11} | "
            f"{'MP1 |X(76)|^2':<15} | {'MP1 BB Context':<15} | "
            f"{'MP2 |X(76)|^2':<15} | {'MP2 BB Context':<15} | "
            f"{'MP3 |X(76)|^2':<15} | {'MP3 BB Context':<15} | "
            f"{'MP4 |X(76)|^2':<15} | {'MP4 BB Context':<15}"
        )
        p_out(hdr)
        p_out("-" * len(hdr))

        for s in stretch_indices:
            t_s = seg_starts[s]
            parts = [f"{s:<4d} | {t_s:<11.3f}"]
            for ch in channels:
                if specs[ch] is None:
                    parts.append(f"{'MISSING':<15} | {'MISSING':<15}")
                else:
                    x76 = np.abs(specs[ch][s, kc_target]) ** 2
                    bb = np.sum(np.abs(specs[ch][s, bb_bins]) ** 2)
                    parts.append(f"{x76:15.4e} | {bb:15.4e}")
            p_out(" | ".join(parts))

        # Channel summary statistics: Pearson correlation vs MP1 and Maximum
        p_out("\n" + "-" * 100)
        p_out(f"CHANNEL SUMMARY: PEASON CORRELATION VS MP1 & BURST PEAK TIMING")
        p_out("-" * 100)

        s_mp1_stretch = np.abs(specs["MP1"][stretch_indices, kc_target]) ** 2
        s_mp1_full = np.abs(specs["MP1"][:, kc_target]) ** 2

        sum_hdr = (
            f"{'Channel':<10} | {'Status':<10} | {'Pearson r vs MP1 (Stretch)':<26} | "
            f"{'Pearson r vs MP1 (Full 30)':<26} | {'Max Start (ms)':<15} | {'Max Seg':<8} | {'Max |X(76)|^2'}"
        )
        p_out(sum_hdr)
        p_out("-" * len(sum_hdr))

        for ch in channels:
            if specs[ch] is None:
                p_out(f"{ch:<10} | {'MISSING':<10} | {'MISSING':<26} | {'MISSING':<26} | {'MISSING':<15} | {'MISSING':<8} | {'MISSING'}")
            else:
                s_ch_stretch = np.abs(specs[ch][stretch_indices, kc_target]) ** 2
                s_ch_full = np.abs(specs[ch][:, kc_target]) ** 2

                # Pearson correlation over stretch
                r_stretch = float(np.corrcoef(s_mp1_stretch, s_ch_stretch)[0, 1])
                # Pearson correlation over full nominal window
                r_full = float(np.corrcoef(s_mp1_full, s_ch_full)[0, 1])

                # Max over stretch
                top_s_stretch = stretch_indices[int(np.argmax(s_ch_stretch))]
                t_max_stretch = seg_starts[top_s_stretch]
                v_max_stretch = float(s_ch_stretch[np.argmax(s_ch_stretch)])

                p_out(
                    f"{ch:<10} | {'OK':<10} | {r_stretch:<+26.6f} | "
                    f"{r_full:<+26.6f} | {t_max_stretch:<15.3f} | {top_s_stretch:<8d} | {v_max_stretch:14.4e}"
                )

        # Extended table showing remaining segments of nominal window (s = 16..29)
        rem_indices = [s for s in range(n_ens) if s not in stretch_indices]
        p_out(f"\n--- Additional Nominal-Window Segments Beyond 268.0 ms (Segments {rem_indices[0]}..{rem_indices[-1]}) ---")
        p_out(hdr)
        p_out("-" * len(hdr))
        for s in rem_indices:
            t_s = seg_starts[s]
            parts = [f"{s:<4d} | {t_s:<11.3f}"]
            for ch in channels:
                if specs[ch] is None:
                    parts.append(f"{'MISSING':<15} | {'MISSING':<15}")
                else:
                    x76 = np.abs(specs[ch][s, kc_target]) ** 2
                    bb = np.sum(np.abs(specs[ch][s, bb_bins]) ** 2)
                    parts.append(f"{x76:15.4e} | {bb:15.4e}")
            p_out(" | ".join(parts))

    wall_elapsed = time.time() - t_start_wall
    p_out("\n" + "=" * 120)
    p_out(f"Total run_burst.py wall-clock execution time: {wall_elapsed:.2f} s")
    p_out("=" * 120)

except Exception as ex:
    p_out(f"\nERROR: Exception occurred during run_burst.py execution:")
    p_out(traceback.format_exc())
    stdout_h.close()
    sys.exit(1)

stdout_h.close()
raw_bytes = stdout_file.read_bytes()
out_sha = hashlib.sha256(raw_bytes).hexdigest()
print(f"Saved stdout to burst_stdout.txt ({len(raw_bytes)} bytes, SHA-256: {out_sha})")
