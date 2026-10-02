import sys
import argparse
import json
import multiprocessing
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as dsp
from scipy.ndimage import uniform_filter1d, median_filter
import matplotlib.patheffects as pe

SHOTS_DEFAULT = [88653]

current = Path(__file__).resolve().parent
root_dir = None
for p in [current] + list(current.parents):
    if (p / "jpack").exists():
        root_dir = p
        break
if root_dir is None:
    root_dir = Path("c:/TFG")

for p_add in [root_dir / "jpack", root_dir / "analysis", root_dir / "analysis" / "common", root_dir / "analysis" / "heating_phase_comparison"]:
    if str(p_add) not in sys.path:
        sys.path.append(str(p_add))

import hashlib
import io
import matplotlib.transforms as transforms
import turnelib as TE
import libana_signal as LAS
from mhd_common import morlet_cwt
from comparison_common import (
    DEFAULT_SHOT,
    T_NOISE_START, T_NOISE_END,
    T_W1_START, T_W1_END,
    T_W2_START, T_W2_END,
)

import logging

log = logging.getLogger("mhd_obj1")

def setup_logging(verbose=False, log_file=None):
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(console)

    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(file_handler)
        log.info(f"(Full diagnostic detail for this run is being written to: {log_file})")

def log_print(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    log.info(msg)

setup_logging()
print = log_print

CONFIRMATION_RULE = "any single coil pair > threshold"


def filter_short_runs(mask, min_len=9):
    """Cleans a boolean mask by dropping runs shorter than min_len consecutive True samples."""
    cleaned = mask.copy()
    in_run = False
    run_start = 0
    runs = []
    for i, val in enumerate(mask):
        if val and not in_run:
            in_run = True
            run_start = i
        elif not val and in_run:
            in_run = False
            runs.append((run_start, i))
    if in_run:
        runs.append((run_start, len(mask)))
    n_dropped = 0
    for start, end in runs:
        rlen = end - start
        if rlen < min_len:
            cleaned[start:end] = False
            n_dropped += rlen
    return cleaned, n_dropped


def track_mode_ridges(signal, fs, time_start_sec=0.0,
                      nperseg=1024, noverlap=960,
                      pri_band=(70.0, 110.0), sec_band=(30.0, 65.0),
                      med_filter_size=17):
    """Tracks primary and secondary mode ridges using STFT peak picking and median filter."""
    f_stft, t_s, Zxx = dsp.stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann')
    t_stft_ms = float(time_start_sec) * 1000.0 + t_s * 1000.0
    psd = np.abs(Zxx) ** 2
    f_khz = f_stft / 1000.0

    m_pri = (f_khz >= pri_band[0]) & (f_khz <= pri_band[1])
    m_sec = (f_khz >= sec_band[0]) & (f_khz <= sec_band[1])

    raw_pri = f_khz[m_pri][np.argmax(psd[m_pri, :], axis=0)]
    raw_sec = f_khz[m_sec][np.argmax(psd[m_sec, :], axis=0)]

    med_pri = median_filter(raw_pri, size=med_filter_size)
    med_sec = median_filter(raw_sec, size=med_filter_size)

    return {
        't_stft_ms': t_stft_ms,
        'f_khz': f_khz,
        'psd': psd,
        'raw_pri': raw_pri,
        'raw_sec': raw_sec,
        'med_pri': med_pri,
        'med_sec': med_sec,
        'm_pri_band': m_pri,
        'm_sec_band': m_sec,
    }


def find_active_intervals(bool_series, time_s, min_duration_ms=0.0):
    """Returns [start_ms, end_ms] active intervals where bool_series is True."""
    bool_series = np.asarray(bool_series, dtype=bool)
    if bool_series.size == 0 or not np.any(bool_series):
        return []

    changes = np.diff(bool_series.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1

    if bool_series[0]:
        starts = np.r_[0, starts]
    if bool_series[-1]:
        ends = np.r_[ends, bool_series.size]

    intervals = []
    for s_idx, e_idx in zip(starts, ends):
        t_start_ms = float(time_s[s_idx]) * 1000.0
        t_end_ms = float(time_s[e_idx - 1]) * 1000.0
        if (t_end_ms - t_start_ms) >= min_duration_ms:
            intervals.append([round(t_start_ms, 3), round(t_end_ms, 3)])
    return intervals


def safe_savefig(fig, out_path, dpi=150):
    out_path = Path(out_path)
    try:
        if out_path.exists():
            try:
                out_path.unlink()
            except Exception:
                pass
        fig.savefig(str(out_path), dpi=dpi)
    except Exception as e:
        import shutil
        tmp_file = out_path.parent / f"tmp_{out_path.name}"
        fig.savefig(str(tmp_file), dpi=dpi)
        try:
            shutil.move(str(tmp_file), str(out_path))
        except Exception:
            pass


def compute_phase_mode_frequencies(ys_mp1, fs, time_s):
    """
    Computes per-phase ridge frequencies for Primary and Secondary modes
    using the exact tracking logic from analysis/heating_phase_comparison/mhd_compare_spectrograms.py
    """
    nperseg = 1024
    noverlap = 960  # fine temporal resolution: step = 64 us
    f_stft, t_s, Zxx = dsp.stft(ys_mp1, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann')
    t_stft_ms = float(time_s[0]) * 1000.0 + t_s * 1000.0
    psd = np.abs(Zxx) ** 2
    f_khz = f_stft / 1000.0

    mask_noise = (t_stft_ms >= T_NOISE_START) & (t_stft_ms <= T_NOISE_END)
    mean_noise = np.mean(psd[:, mask_noise], axis=1)
    std_noise = np.std(psd[:, mask_noise], axis=1)
    thresh_3sigma = mean_noise + 3.0 * std_noise

    windows = [
        ("ECH + NBI Phase", T_W1_START, T_W1_END),
        ("NBI-only Phase", T_W2_START, T_W2_END)
    ]

    m_pri = (f_khz >= 70.0) & (f_khz <= 110.0)
    m_sec = (f_khz >= 30.0) & (f_khz <= 65.0)

    tracked = {}
    for phase_name, tw_start, tw_end in windows:
        mask_win = (t_stft_ms >= tw_start) & (t_stft_ms <= tw_end)
        psd_sub = psd[:, mask_win]

        f_pri_track = []
        f_sec_track = []
        for col in psd_sub.T:
            idx_p = np.argmax(col[m_pri])
            f_pri_track.append(f_khz[m_pri][idx_p])

            idx_s = np.argmax(col[m_sec])
            f_sec_track.append(f_khz[m_sec][idx_s])

        tracked[phase_name] = {
            "tw_start": tw_start,
            "tw_end": tw_end,
            "f_pri_mean": float(np.mean(f_pri_track)),
            "f_sec_mean": float(np.mean(f_sec_track)),
        }
    return tracked


def parse_args():
    p = argparse.ArgumentParser(description="MHD Analysis - Objective 1 (Mode Identification)")
    p.add_argument("-s", "--shots", type=int, nargs="+", default=SHOTS_DEFAULT,
                   help=f"List of shot numbers to analyze (default: {SHOTS_DEFAULT})")
    p.add_argument("-d", "--data-dir-pattern", type=str, default="data/hj{shot}",
                   help="Data directory pattern (default: data/hj{shot})")
    p.add_argument("-o", "--out-dir", type=str, default=str(root_dir / "Images" / "obj1_mode_identification"),
                   help="Output directory for generated figures")
    p.add_argument("--fres", type=float, default=500.0,
                   help="FFT frequency resolution in Hz (default: 500.0)")
    p.add_argument("--wav-nfreqs", type=int, default=150,
                   help="Number of frequencies for Morlet CWT (default: 150)")
    p.add_argument("--wav-fmin", type=float, default=10.0,
                   help="Min frequency in kHz for Morlet CWT (default: 10)")
    p.add_argument("--wav-fmax", type=float, default=150.0,
                   help="Max frequency in kHz for Morlet CWT (default: 150)")
    p.add_argument("--noise-start", type=float, default=None,
                   help="Noise baseline start in seconds")
    p.add_argument("--noise-end", type=float, default=None,
                   help="Noise baseline end in seconds")
    p.add_argument("--dominance-smooth-ms", type=float, default=4.0,
                   help="Smoothing window in ms for power dominance (default: 4.0)")
    p.add_argument("--min-interval-ms", type=float, default=2.0,
                   help="Minimum interval duration in ms (default: 2.0)")
    p.add_argument("--baseline-margin-ms", type=float, default=5.0,
                   help="Minimum baseline margin in ms (default: 5.0)")
    p.add_argument("-e", "--ensemble", type=int, default=100,
                   help="Max Welch ensembles (default: 100)")
    p.add_argument("--fmax", type=float, default=150.0,
                   help="Max frequency analyzed in kHz (default: 150)")
    p.add_argument("--threshold-phys", type=float, default=0.7,
                   help="Physical coherence threshold (default: 0.7)")
    p.add_argument("--fmin-pick", type=float, default=15.0,
                   help="Min frequency for peak searching in kHz")
    p.add_argument("--fmax-pick", type=float, default=100.0,
                   help="Max frequency for peak searching in kHz")
    p.add_argument("--n-peaks", type=int, default=8,
                   help="Max candidate peaks to report (default: 8)")
    p.add_argument("--plot-limit", type=int, default=10000,
                   help="Sample limit for raw plots")
    p.add_argument("--log-file", type=str, default=None,
                   help="Path to output log file")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable verbose output")
    return p.parse_args()


def process_shot(shot, args):
    try:
        print(f"\nProcessing Shot {shot}...")
        try:
            data_dir = Path(args.data_dir_pattern.format(shot=shot))
        except Exception:
            data_dir = Path(args.data_dir_pattern)

        coils = ["MP1", "MP3", "MP4"]
        signals = {}
        time_sec = None
        dt = None
        fs = None

        edf = TE.edf()
        for coil in coils:
            file_path = data_dir / f"{coil}@{shot}.edf"
            if not file_path.exists():
                print(f"Warning: file for {coil} not found at {file_path}")
                continue

            dat = edf.load(str(file_path))
            t = dat[:, 0]
            ys = dat[:, 1]
            t_s = t / 1000.0 if edf.DimUnit[0] == 'ms' else t

            signals[coil] = ys
            if time_sec is None:
                time_sec = t_s
                n_samples = len(t_s)
                sample_idx = min(100, n_samples - 1) if n_samples > 1 else 0
                dt = (t_s[sample_idx] - t_s[0]) / float(sample_idx) if sample_idx > 0 else 1e-6
                fs = 1.0 / dt

        if "MP1" not in signals:
            print("Error: MP1 reference channel missing; aborting shot.")
            return

        t0 = float(time_sec[0])
        fres = args.fres
        nfft = int(np.round(1.0 / (dt * fres)))

        try:
            f, tave, Pyy1 = LAS.running(
                signals["MP1"], time_sec, dt=dt, nfft=nfft, noverlap=nfft // 2,
                window='hann', detrend='constant'
            )
        except TypeError:
            f, tave, Pyy1 = LAS.running(
                signals["MP1"], time_sec, dt=dt, nfft=nfft, noverlap=nfft // 2,
                detrend='constant'
            )

        freqs_wav = np.linspace(args.wav_fmin * 1e3, args.wav_fmax * 1e3, args.wav_nfreqs)
        Wx = morlet_cwt(signals["MP1"], fs, freqs_wav, verbose=False)
        wav_power = np.abs(Wx) ** 2

        noise_start = args.noise_start if args.noise_start is not None else t0
        noise_end = args.noise_end if args.noise_end is not None else t0 + 0.025

        tave_step_ms = float((tave[1] - tave[0]) * 1000.0) if len(tave) > 1 else 1.0
        smooth_bins = max(1, int(round(args.dominance_smooth_ms / tave_step_ms)))
        Pyy1_smooth = uniform_filter1d(Pyy1, size=smooth_bins, axis=1, mode='nearest') if smooth_bins > 1 else Pyy1

        mask_noise_t = (tave >= noise_start) & (tave <= noise_end)
        if not np.any(mask_noise_t):
            mask_noise_t = np.zeros_like(tave, dtype=bool)
            mask_noise_t[0] = True

        mean_noise_per_f = np.mean(Pyy1_smooth[:, mask_noise_t], axis=1)
        std_noise_per_f = np.std(Pyy1_smooth[:, mask_noise_t], axis=1)
        snr_threshold_per_f = mean_noise_per_f + 3.0 * std_noise_per_f
        dominance_map = Pyy1_smooth > snr_threshold_per_f[:, None]

        baseline_dominant_mask = dominance_map[:, mask_noise_t]
        n_baseline_dominant = int(np.sum(baseline_dominant_mask))
        baseline_is_clean = (n_baseline_dominant == 0)
        baseline_contaminated_freqs_hz = f[np.any(baseline_dominant_mask, axis=1)] if not baseline_is_clean else np.array([])

        coherence_spectra = {}
        f_csd = None
        target_pairs = [("MP1", "MP3"), ("MP1", "MP4"), ("MP3", "MP4")]
        for c1, c2 in target_pairs:
            if c1 not in signals or c2 not in signals:
                continue
            f_c, coh2 = dsp.coherence(
                signals[c1], signals[c2], fs=fs, window='hann',
                nperseg=nfft, noverlap=nfft // 2
            )
            pair_key = f"{c1}_{c2}"
            coherence_spectra[pair_key] = coh2
            if f_csd is None:
                f_csd = f_c

        Pyy_avg = np.mean(Pyy1, axis=1)
        mask_f_pick = (f / 1000.0 >= args.fmin_pick) & (f / 1000.0 <= args.fmax_pick)
        df = f[1] - f[0]
        min_dist_bins = max(1, int(5000.0 / df))

        log_Pyy = np.log10(Pyy_avg + 1e-30)
        peak_idx_local, _ = dsp.find_peaks(log_Pyy[mask_f_pick], distance=min_dist_bins)
        idx_map = np.where(mask_f_pick)[0]
        peak_idx = idx_map[peak_idx_local]

        order = np.argsort(Pyy_avg[peak_idx])[::-1][:args.n_peaks]
        peak_idx = peak_idx[order]

        discrete_modes = []
        for p_idx in peak_idx:
            freq_hz = float(f[p_idx])
            amp = float(Pyy_avg[p_idx])
            thresh = float(snr_threshold_per_f[p_idx])
            is_dominant = bool(amp >= thresh)

            coherence_dict = {}
            confirming_pairs = []
            if f_csd is not None:
                idx_csd = int(np.abs(f_csd - freq_hz).argmin())
                for pair_key, spec in coherence_spectra.items():
                    val = float(spec[idx_csd])
                    coherence_dict[pair_key] = val
                    if val > args.threshold_phys:
                        confirming_pairs.append(pair_key)
            coherence_confirmed = len(confirming_pairs) > 0
            dual_pass = bool(is_dominant and coherence_confirmed)

            active_intervals_ms = find_active_intervals(
                dominance_map[p_idx, :], tave, min_duration_ms=args.min_interval_ms
            )
            total_active_duration_ms = round(sum(e - s for s, e in active_intervals_ms), 3)

            baseline_contaminated = bool(
                baseline_contaminated_freqs_hz.size > 0
                and np.any(np.abs(baseline_contaminated_freqs_hz - freq_hz) <= df / 2.0)
            )

            discrete_modes.append({
                "frequency_hz": freq_hz,
                "amplitude": amp,
                "snr_threshold": thresh,
                "is_above_noise_floor": is_dominant,
                "coherence": coherence_dict,
                "confirming_pairs": confirming_pairs,
                "coherence_confirmed": coherence_confirmed,
                "confirmation_rule": CONFIRMATION_RULE,
                "dual_criterion_pass": dual_pass,
                "baseline_calibration_contaminated": baseline_contaminated,
                "active_intervals_ms": active_intervals_ms,
                "total_active_duration_ms": total_active_duration_ms,
            })

        discrete_modes.sort(key=lambda m: m["amplitude"], reverse=True)

        export = {
            "shot": str(shot),
            "confirmation_rule": CONFIRMATION_RULE,
            "threshold_phys": args.threshold_phys,
            "noise_baseline_window_ms": [noise_start * 1000.0, noise_end * 1000.0],
            "discrete_modes": discrete_modes,
        }
        out_json = f"discrete_modes_shot_{shot}.json"
        with open(out_json, "w", encoding="utf-8") as fjson:
            json.dump(export, fjson, indent=2)

        # Plot overview
        plt.rcdefaults()
        fig, axs = plt.subplots(3, 1, figsize=(12, 13))

        tt, ff = np.meshgrid(tave * 1000.0, f / 1000.0)
        mask_fplot = (f / 1000.0) <= args.fmax

        c0 = axs[0].pcolormesh(tt[mask_fplot, :], ff[mask_fplot, :], np.log10(Pyy1[mask_fplot, :] + 1e-30),
                                cmap='jet', shading='auto')
        cb0 = fig.colorbar(c0, ax=axs[0], label="log10 PSD (V^2/Hz)", pad=0.03)

        dmap_plot = dominance_map[mask_fplot, :].astype(float)
        axs[0].contourf(tt[mask_fplot, :], ff[mask_fplot, :], dmap_plot,
                         levels=[0.5, 1.5], colors=['black'], alpha=0.10)
        axs[0].contour(tt[mask_fplot, :], ff[mask_fplot, :], dmap_plot,
                        levels=[0.5], colors='black', linewidths=1.2, linestyles=':')

        # Time window matching heating_vs_mirnov.py
        axs[0].set_xlim(150.0, 350.0)

        # Mode markers for DEFAULT_SHOT only
        if shot == DEFAULT_SHOT:
            # Dynamic tracked ridge lines for Primary and Secondary modes
            tr_res = track_mode_ridges(signals["MP1"], fs, time_start_sec=float(time_sec[0]))
            t_stft_ms_tr = tr_res['t_stft_ms']
            psd_tr = tr_res['psd']
            f_khz_tr = tr_res['f_khz']
            m_pri_band = tr_res['m_pri_band']
            m_sec_band = tr_res['m_sec_band']

            # Dominance map lookup setup
            t_dmap_ms = tave * 1000.0
            f_dmap_khz = f / 1000.0

            # Process per phase: Phase 1 (259.1-291.3 ms) and Phase 2 (291.3-300.0 ms)
            phase_defs = [
                ("Phase 1 (ECH+NBI)", (t_stft_ms_tr >= 259.1) & (t_stft_ms_tr <= 291.3)),
                ("Phase 2 (NBI-only)", (t_stft_ms_tr > 291.3) & (t_stft_ms_tr <= 300.0)),
            ]

            mode_results = {}
            log_print(f"\nTracked ridge mode statistics for Shot {shot} (median filter: 17 slices = 1.088 ms):")

            for pname, mask_ph in phase_defs:
                t_ph = t_stft_ms_tr[mask_ph]
                n_sl = len(t_ph)
                psd_ph = psd_tr[:, mask_ph]

                raw_p = f_khz_tr[m_pri_band][np.argmax(psd_ph[m_pri_band, :], axis=0)]
                raw_s = f_khz_tr[m_sec_band][np.argmax(psd_ph[m_sec_band, :], axis=0)]

                med_p = median_filter(raw_p, size=17)
                med_s = median_filter(raw_s, size=17)

                for mname, raw_arr, med_arr in [("Primary", raw_p, med_p), ("Secondary", raw_s, med_s)]:
                    # Edge trimming: drop first and last 8 slices
                    edge_drop = np.zeros(n_sl, dtype=bool)
                    edge_drop[:8] = True
                    edge_drop[-8:] = True
                    n_edge = int(np.sum(edge_drop))

                    # Outlier rejection: |raw - filtered| > 5.0 kHz
                    diff = np.abs(raw_arr - med_arr)
                    outlier_drop = diff > 5.0
                    n_outliers = int(np.sum(outlier_drop & (~edge_drop)))

                    # 3-sigma dominance map lookup
                    dom_mask = np.zeros(n_sl, dtype=bool)
                    for idx_pt, (ti, fp) in enumerate(zip(t_ph, med_arr)):
                        it = int(np.abs(t_dmap_ms - ti).argmin())
                        ifp = int(np.abs(f_dmap_khz - fp).argmin())
                        dom_mask[idx_pt] = dominance_map[ifp, it]

                    dom_masked = (~edge_drop) & (~outlier_drop) & (~dom_mask)
                    n_dom = int(np.sum(dom_masked))

                    valid_initial = (~edge_drop) & (~outlier_drop) & dom_mask
                    valid_mask, n_short = filter_short_runs(valid_initial, min_len=9)
                    valid_vals = med_arr[valid_mask]
                    mean_val = float(np.mean(valid_vals)) if len(valid_vals) > 0 else float('nan')
                    min_val = float(np.min(valid_vals)) if len(valid_vals) > 0 else float('nan')
                    max_val = float(np.max(valid_vals)) if len(valid_vals) > 0 else float('nan')

                    log_print(f"  {pname} - {mname} Mode:")
                    log_print(f"    Slices dropped by edge trimming: {n_edge}")
                    log_print(f"    Slices dropped as outliers: {n_outliers}")
                    log_print(f"    Slices masked by 3-sigma dominance map: {n_dom}")
                    log_print(f"    Slices dropped as short runs (< 9 slices): {n_short}")
                    log_print(f"    Resulting mean = {mean_val:.2f} kHz, min = {min_val:.2f} kHz, max = {max_val:.2f} kHz")

                    mode_results[(pname, mname)] = {
                        "t": t_ph,
                        "plot_vals": np.where(valid_mask, med_arr, np.nan),
                        "mean": mean_val,
                        "min": min_val,
                        "max": max_val,
                    }

            pri_w1_mean = mode_results[("Phase 1 (ECH+NBI)", "Primary")]["mean"]
            pri_w2_mean = mode_results[("Phase 2 (NBI-only)", "Primary")]["mean"]
            sec_w1_mean = mode_results[("Phase 1 (ECH+NBI)", "Secondary")]["mean"]
            sec_w2_mean = mode_results[("Phase 2 (NBI-only)", "Secondary")]["mean"]

            # Sanity check against flat reference values
            ref_pri_w1, ref_sec_w1 = 87.60, 45.47
            ref_pri_w2, ref_sec_w2 = 86.69, 40.26
            d_p1 = abs(pri_w1_mean - ref_pri_w1)
            d_p2 = abs(pri_w2_mean - ref_pri_w2)
            d_s1 = abs(sec_w1_mean - ref_sec_w1)
            d_s2 = abs(sec_w2_mean - ref_sec_w2)
            log_print(f"  Differences vs flat references: Pri W1: {d_p1:.2f} kHz, Pri W2: {d_p2:.2f} kHz, Sec W1: {d_s1:.2f} kHz, Sec W2: {d_s2:.2f} kHz")
            if d_p1 > 4.0 or d_p2 > 4.0 or d_s1 > 4.0 or d_s2 > 4.0:
                log_print(f"  WARNING: One or more tracked ridge means differ from flat reference values by more than 4 kHz! Continuing.")

            # Plot cleaned 1.6 pt solid lines with dark outline
            outline_effect = [pe.Stroke(linewidth=3.0, foreground='black'), pe.Normal()]
            for pname in ["Phase 1 (ECH+NBI)", "Phase 2 (NBI-only)"]:
                axs[0].plot(mode_results[(pname, "Primary")]["t"],
                            mode_results[(pname, "Primary")]["plot_vals"],
                            color='white', linewidth=1.6, zorder=5, path_effects=outline_effect)
                axs[0].plot(mode_results[(pname, "Secondary")]["t"],
                            mode_results[(pname, "Secondary")]["plot_vals"],
                            color='yellow', linewidth=1.6, zorder=5, path_effects=outline_effect)

            # Mode labels near right end with arrow pointing to the last valid point
            p2_pri_vals = mode_results[("Phase 2 (NBI-only)", "Primary")]["plot_vals"]
            p2_pri_t = mode_results[("Phase 2 (NBI-only)", "Primary")]["t"]
            valid_p2_pri = np.where(~np.isnan(p2_pri_vals))[0]
            last_t_pri = p2_pri_t[valid_p2_pri[-1]]
            last_f_pri = p2_pri_vals[valid_p2_pri[-1]]

            p2_sec_vals = mode_results[("Phase 2 (NBI-only)", "Secondary")]["plot_vals"]
            p2_sec_t = mode_results[("Phase 2 (NBI-only)", "Secondary")]["t"]
            valid_p2_sec = np.where(~np.isnan(p2_sec_vals))[0]
            last_t_sec = p2_sec_t[valid_p2_sec[-1]]
            last_f_sec = p2_sec_vals[valid_p2_sec[-1]]

            bbox_lbl_pri = dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.85, edgecolor='white', lw=1.2)
            bbox_lbl_sec = dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.85, edgecolor='yellow', lw=1.2)

            label_pri = f"Primary Mode\nmean {pri_w1_mean:.1f} \u2192 {pri_w2_mean:.1f} kHz"
            label_sec = f"Secondary Mode\nmean {sec_w1_mean:.1f} \u2192 {sec_w2_mean:.1f} kHz"

            axs[0].annotate(label_pri, xy=(last_t_pri, last_f_pri), xytext=(305.0, last_f_pri),
                            ha='left', va='center', fontsize=8.5, fontweight='bold', color='white',
                            bbox=bbox_lbl_pri, arrowprops=dict(arrowstyle='->', color='white', lw=1.2))

            axs[0].annotate(label_sec, xy=(last_t_sec, last_f_sec), xytext=(305.0, last_f_sec),
                            ha='left', va='center', fontsize=8.5, fontweight='bold', color='yellow',
                            bbox=bbox_lbl_sec, arrowprops=dict(arrowstyle='->', color='yellow', lw=1.2))
        else:
            log_print(f"Notice: Mode markers are only defined for shot {DEFAULT_SHOT}; skipping mode markers for shot {shot}.")

        axs[0].set_title("MP1 Spectrogram with 3-sigma Dominance Region")
        axs[0].set_xlabel("Time (ms)")
        axs[0].set_ylabel("Frequency (kHz)")

        tt_w, ff_w = np.meshgrid(time_sec * 1000.0, freqs_wav / 1000.0)
        c1 = axs[1].pcolormesh(tt_w, ff_w, np.log10(wav_power + 1e-30), cmap='jet', shading='auto')
        cb1 = fig.colorbar(c1, ax=axs[1], label="log10 Wavelet Power (rel.)", pad=0.03)
        axs[1].set_title("Morlet Wavelet Transform (MP1)")
        axs[1].set_xlabel("Time (ms)")
        axs[1].set_ylabel("Frequency (kHz)")

        colors_coil = {"MP1_MP3": "green", "MP1_MP4": "purple", "MP3_MP4": "brown"}
        mask_f_csd_plot = ((f_csd >= 0.0) & (f_csd / 1000.0 <= args.fmax)) if f_csd is not None else None
        for pair_key, spec in coherence_spectra.items():
            label_name = pair_key.replace("_", "-")
            axs[2].plot(f_csd[mask_f_csd_plot] / 1000.0, spec[mask_f_csd_plot],
                        color=colors_coil.get(pair_key, 'gray'), linewidth=1.3, label=label_name)
        axs[2].axhline(y=args.threshold_phys, color='black', linestyle=':', linewidth=1.2,
                       label=f"Threshold ({args.threshold_phys:.1f})")

        for m in discrete_modes:
            color = 'green' if m["dual_criterion_pass"] else 'red'
            axs[2].axvline(x=m["frequency_hz"] / 1000.0, color=color, linestyle=':', alpha=0.8, linewidth=1.4)

        axs[2].set_title("Inter-Probe Spatial Coherence gamma^2(f)")
        axs[2].set_xlabel("Frequency (kHz)")
        axs[2].set_ylabel(r"Coherence $\gamma^2$")
        axs[2].set_ylim(0, 1.05)
        axs[2].grid(True, alpha=0.3)
        axs[2].legend(loc='upper right', fontsize=9)

        plt.tight_layout()
        out_dir = Path(getattr(args, "out_dir", root_dir / "Images" / "obj1_mode_identification"))
        out_dir.mkdir(parents=True, exist_ok=True)
        output_png = out_dir / f"mhd_analysis_objective1_{shot}.png"
        safe_savefig(fig, output_png, dpi=150)
        print(f"Objective 1 overview saved to '{output_png}'.")

        # Re-extract panel 1 and check panels 2 and 3
        panels_dir = out_dir / "panels"
        panels_dir.mkdir(parents=True, exist_ok=True)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        primary_axes = [ax for ax in fig.get_axes() if not (hasattr(ax, '_colorbar_info') or getattr(ax, '_colorbar', False))]

        pad = 0.08
        # Panel 1 (with colorbar)
        ax1 = primary_axes[0]
        bb1 = ax1.get_tightbbox(renderer)
        if bb1 is None: bb1 = ax1.get_window_extent(renderer)
        bb1_cb = cb0.ax.get_tightbbox(renderer)
        if bb1_cb is None: bb1_cb = cb0.ax.get_window_extent(renderer)
        bb1_union = transforms.Bbox.union([bb1, bb1_cb])
        bb1_in = bb1_union.transformed(fig.dpi_scale_trans.inverted())
        bb1_padded = transforms.Bbox.from_extents(bb1_in.x0 - pad, bb1_in.y0 - pad, bb1_in.x1 + pad, bb1_in.y1 + pad)
        p1_path = panels_dir / f"mhd_analysis_objective1_{shot}_panel1_mp1_spectrogram_with_3_sigma_dominance_r.png"
        fig.savefig(str(p1_path), dpi=150, bbox_inches=bb1_padded, facecolor=fig.get_facecolor(), edgecolor='none', pad_inches=0)
        print(f"Panel 1 extracted and saved to '{p1_path}'.")

        # Panel 2 (with colorbar)
        ax2 = primary_axes[1]
        bb2 = ax2.get_tightbbox(renderer)
        if bb2 is None: bb2 = ax2.get_window_extent(renderer)
        bb2_cb = cb1.ax.get_tightbbox(renderer)
        if bb2_cb is None: bb2_cb = cb1.ax.get_window_extent(renderer)
        bb2_union = transforms.Bbox.union([bb2, bb2_cb])
        bb2_in = bb2_union.transformed(fig.dpi_scale_trans.inverted())
        bb2_padded = transforms.Bbox.from_extents(bb2_in.x0 - pad, bb2_in.y0 - pad, bb2_in.x1 + pad, bb2_in.y1 + pad)
        p2_path = panels_dir / f"mhd_analysis_objective1_{shot}_panel2_morlet_wavelet_transform_mp1.png"
        fig.savefig(str(p2_path), dpi=150, bbox_inches=bb2_padded, facecolor=fig.get_facecolor(), edgecolor='none', pad_inches=0)
        print(f"Panel 2 extracted (with colorbar) and saved to '{p2_path}'.")

        # Panel 3 (no colorbar - verify hash match)
        ax3 = primary_axes[2]
        p3_path = panels_dir / f"mhd_analysis_objective1_{shot}_panel3_inter_probe_spatial_coherence_gamma2f.png"
        if p3_path.exists():
            orig_hash = hashlib.sha256(p3_path.read_bytes()).hexdigest()
            bb3 = ax3.get_tightbbox(renderer)
            if bb3 is None: bb3 = ax3.get_window_extent(renderer)
            bb3_in = bb3.transformed(fig.dpi_scale_trans.inverted())
            bb3_padded = transforms.Bbox.from_extents(bb3_in.x0 - pad, bb3_in.y0 - pad, bb3_in.x1 + pad, bb3_in.y1 + pad)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches=bb3_padded, facecolor=fig.get_facecolor(), edgecolor='none', pad_inches=0)
            new_hash = hashlib.sha256(buf.getvalue()).hexdigest()
            match = (orig_hash == new_hash)
            print(f"Panel 3 SHA-256 match: {match} (orig: {orig_hash}, new: {new_hash})")
            if not match:
                print(f"WARNING: Panel 3 hash differs! Leaving existing file untouched.")

        plt.close(fig)

    except Exception as e:
        print(f"Error processing shot {shot}: {e}")
        import traceback
        traceback.print_exc()


def main():
    args = parse_args()
    
    shot_str = str(args.shots[0]) if args.shots else "run"
    log_file = args.log_file if args.log_file is not None else f"mhd_obj1_{shot_str}.log"
    setup_logging(args.verbose, log_file)
    
    num_processes = min(len(args.shots), multiprocessing.cpu_count())
    if num_processes <= 1:
        for shot in args.shots:
            process_shot(shot, args)
    else:
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(process_shot, [(shot, args) for shot in args.shots])


if __name__ == "__main__":
    main()