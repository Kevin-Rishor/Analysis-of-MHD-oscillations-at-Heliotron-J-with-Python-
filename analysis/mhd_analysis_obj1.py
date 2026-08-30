import sys
import argparse
import json
import multiprocessing
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as dsp
from scipy.ndimage import uniform_filter1d

SHOTS_DEFAULT = [88653]

# Add the jpack library path to Python's path
jpack_path = str(Path(__file__).parent.parent.resolve() / "jpack")
if jpack_path not in sys.path:
    sys.path.append(jpack_path)

import turnelib as TE
import libana_signal as LAS
from mhd_common import morlet_cwt

CONFIRMATION_RULE = "any single coil pair > threshold" 


def find_active_intervals(bool_series, time_s, min_duration_ms=0.0):
    """
    Given a boolean time series (True where a mode's smoothed power exceeds
    its 3-sigma threshold, i.e. a row of `dominance_map`) and the matching
    time axis (seconds), return the list of [start_ms, end_ms] intervals
    over which the mode is continuously "active" (dominant).

    Intervals shorter than `min_duration_ms` are discarded, to avoid
    reporting single-bin noise blips as separate activity windows.
    """
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


def parse_args():
    p = argparse.ArgumentParser(description="MHD Analysis - Specific Objective 1 (Multichannel)")
    p.add_argument("-s", "--shots", type=int, nargs="+", default=SHOTS_DEFAULT,
                    help=f"List of shot numbers to analyze (def: {SHOTS_DEFAULT})")
    p.add_argument("-d", "--data-dir-pattern", type=str, default="data/hj{shot}",
                    help="Data directory pattern (def: data/hj{shot})")

    # FFT / spectrogram
    p.add_argument("--fres", type=float, default=500.0,
                    help="Required FFT frequency resolution in Hz, Delta f = 1/T (def: 500.0)")

    # Morlet wavelet
    p.add_argument("--wav-nfreqs", type=int, default=150, help="Number of frequencies for the Morlet CWT (def: 150)")
    p.add_argument("--wav-fmin", type=float, default=10.0, help="Min frequency (kHz) for the Morlet CWT (def: 10)")
    p.add_argument("--wav-fmax", type=float, default=150.0, help="Max frequency (kHz) for the Morlet CWT (def: 150)")

    # Background noise / 3-sigma dominance
    p.add_argument("--noise-start", type=float, default=None,
                    help="Background/baseline window start (s) for the noise floor.")
    p.add_argument("--noise-end", type=float, default=None, help="Background/baseline window end (s)")
    p.add_argument("--dominance-smooth-ms", type=float, default=4.0,
                    help="Time-smoothing window (ms) applied to the power map before evaluating the "
                         "3-sigma dominance mask (def: 4.0).")
    p.add_argument("--min-interval-ms", type=float, default=2.0,
                    help="Minimum duration (ms) for a contiguous 3-sigma-dominant stretch to be reported "
                         "as an active interval for a mode (def: 2.0).")
    p.add_argument("--baseline-margin-ms", type=float, default=5.0,
                    help="Minimum gap (ms) required between the end of the noise baseline window and the "
                         "first detected 3-sigma-dominant activity; a warning is printed if the gap is "
                         "smaller than this, since it suggests the baseline may be contaminated by early "
                         "MHD activity (def: 5.0).")

    # Spatial coherence (gamma^2)
    p.add_argument("-e", "--ensemble", type=int, default=100,
                    help="Max number of Welch ensembles averaged over the whole discharge (def: 100)")
    p.add_argument("--fmax", type=float, default=150.0, help="Max frequency analyzed (kHz) (def: 150)")
    p.add_argument("--threshold-phys", type=float, default=0.7,
                    help="gamma^2 physical coherence threshold (def: 0.7)")

    # Peak picking
    p.add_argument("--fmin-pick", type=float, default=15.0, help="Min frequency (kHz) searched for dominant peaks")
    p.add_argument("--fmax-pick", type=float, default=100.0, help="Max frequency (kHz) searched for dominant peaks")
    p.add_argument("--n-peaks", type=int, default=8, help="Max number of candidate peaks to report (def: 8)")

    p.add_argument("--plot-limit", type=int, default=10000, help="Sample limit for any raw-signal debugging plot")
    return p.parse_args()


def process_shot(shot, args):
    import traceback
    try:
        print(f"\n========================================\nSTARTING ANALYSIS FOR SHOT {shot}\n========================================")
        try:
            data_dir = Path(args.data_dir_pattern.format(shot=shot))
        except Exception:
            data_dir = Path(args.data_dir_pattern)

        # -----------------------------------------------------------------
        # 1. Load Mirnov coil data
        # -----------------------------------------------------------------
        print(f"--- Loading Mirnov Coil data files (Shot {shot}) ---")
        coils = ["MP1", "MP3", "MP4"]
        signals = {}
        time_sec = None
        dt = None
        fs = None

        for coil in coils:
            file_path = data_dir / f"{coil}@{shot}.edf"
            if not file_path.exists():
                print(f"Warning: File for {coil} not found at {file_path}")
                continue

            edf = TE.edf()
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
            print("Error: MP1 (reference channel) could not be loaded; cannot proceed.")
            return

        print(f"Sampling frequency: {fs/1e6:.2f} MHz (dt = {dt*1e6:.3f} us)")
        print(f"Number of data points: {len(time_sec)}")
        print(f"Coils successfully loaded: {list(signals.keys())}")

        t0, t1 = float(time_sec[0]), float(time_sec[-1])

        # -----------------------------------------------------------------
        # 2. Time-frequency representation A: FFT spectrogram (Hann window, Delta f = 1/T)
        # -----------------------------------------------------------------
        print("\n--- Computing FFT Spectrogram (Hann window, running FFT) for MP1 ---")
        fres = args.fres
        nfft = int(np.round(1.0 / (dt * fres)))
        try:
            f, tave, Pyy1 = LAS.running(signals["MP1"], time_sec, dt=dt, nfft=nfft, noverlap=nfft // 2,
                                         window='hann', detrend='constant')
            print("  LAS.running() called with window='hann' (explicit).")
        except TypeError:
            f, tave, Pyy1 = LAS.running(signals["MP1"], time_sec, dt=dt, nfft=nfft, noverlap=nfft // 2,
                                         detrend='constant')
            print("  WARNING: LAS.running() does not accept a 'window' keyword in this library version; "
                  "falling back to its internal default. Confirm in libana_signal's source that this "
                  "default is indeed Hann, per the First Specific Objective methodology.")
        print(f"Segment length T = {nfft * dt * 1e3:.2f} ms (Nfft = {nfft} samples) "
              f"-> achieved Delta f = {f[1]-f[0]:.1f} Hz (target: {fres:.1f} Hz)")

        # -----------------------------------------------------------------
        # 3. Time-frequency representation B: Morlet wavelet (non-stationary case)
        # -----------------------------------------------------------------
        print("\n--- Computing Morlet Wavelet Transform for MP1 ---")
        freqs_wav = np.linspace(args.wav_fmin * 1e3, args.wav_fmax * 1e3, args.wav_nfreqs)
        Wx = morlet_cwt(signals["MP1"], fs, freqs_wav, verbose=True)
        wav_power = np.abs(Wx) ** 2
        print(f"Morlet CWT computed over {args.wav_nfreqs} frequencies "
              f"({args.wav_fmin:.0f}-{args.wav_fmax:.0f} kHz).")

        # -----------------------------------------------------------------
        # 4. Background noise floor and 3-sigma dominance threshold
        # -----------------------------------------------------------------
        noise_start = args.noise_start if args.noise_start is not None else t0
        noise_end = args.noise_end if args.noise_end is not None else t0 + 0.025
        print(f"\n--- Background Noise Floor Estimation ({noise_start*1000:.0f}-{noise_end*1000:.0f} ms baseline) ---")

        tave_step_ms = float((tave[1] - tave[0]) * 1000.0) if len(tave) > 1 else 1.0
        smooth_bins = max(1, int(round(args.dominance_smooth_ms / tave_step_ms)))
        Pyy1_smooth = uniform_filter1d(Pyy1, size=smooth_bins, axis=1, mode='nearest') if smooth_bins > 1 else Pyy1

        mask_noise_t = (tave >= noise_start) & (tave <= noise_end)
        if not np.any(mask_noise_t):
            print("  Warning: baseline window contains no spectrogram samples; using the first available time bin instead.")
            mask_noise_t = np.zeros_like(tave, dtype=bool)
            mask_noise_t[0] = True

        mean_noise_per_f = np.mean(Pyy1_smooth[:, mask_noise_t], axis=1)
        std_noise_per_f = np.std(Pyy1_smooth[:, mask_noise_t], axis=1)
        snr_threshold_per_f = mean_noise_per_f + 3.0 * std_noise_per_f

        # Time-resolved 3-sigma dominance mask (freq x time), used both to extract
        # per-mode active intervals below and to draw the dominance contour in Panel 1.
        dominance_map = Pyy1_smooth > snr_threshold_per_f[:, None]

        # Check 1: to see that the baseline window is not contaminated
        baseline_dominant_mask = dominance_map[:, mask_noise_t]
        n_baseline_dominant = int(np.sum(baseline_dominant_mask))
        baseline_is_clean = (n_baseline_dominant == 0)
        baseline_contaminated_freqs_hz = f[np.any(baseline_dominant_mask, axis=1)] if not baseline_is_clean else np.array([])
        if baseline_is_clean:
            print(f"  Baseline window ({noise_start*1000:.1f}-{noise_end*1000:.1f} ms) is clean: "
                  f"no (freq, time) bin inside it exceeds its own 3-sigma threshold.")
        else:
            n_freqs_affected = int(np.sum(np.any(baseline_dominant_mask, axis=1)))
            affected_khz_str = ", ".join(f"{v/1e3:.1f}" for v in sorted(baseline_contaminated_freqs_hz))
            print(f"  WARNING: baseline window ({noise_start*1000:.1f}-{noise_end*1000:.1f} ms) is NOT clean: "
                  f"{n_baseline_dominant} (freq, time) bin(s) across {n_freqs_affected} frequency bin(s) "
                  f"exceed their own 3-sigma threshold inside the calibration window itself: [{affected_khz_str}] kHz. "
                  f"Note this can arise simply because PSD/periodogram estimates are right-skewed (not Gaussian), "
                  f"so a mean+3sigma threshold is crossed by chance more often than a true Gaussian tail would "
                  f"suggest -- isolated, scattered bins like this are consistent with that, not necessarily real "
                  f"early MHD activity. Whether it matters is checked below against the actual reported modes.")

        # Check 2: no other noise data
        no_earlier_data = np.isclose(noise_start, t0, atol=1e-9)
        if no_earlier_data:
            print(f"  Note: the baseline window already starts at the first available sample (t0 = "
                  f"{t0*1000:.1f} ms) -- there is no earlier data to shift --noise-start into for this shot.")

        # Compute the gap to the first 3-sigma-dominant activity AFTER the baseline window ends. 
        gap_info = {"status": None} 
        mask_after_baseline = tave > noise_end
        if np.any(mask_after_baseline):
            tave_after = tave[mask_after_baseline]
            any_active_after = np.any(dominance_map[:, mask_after_baseline], axis=0)
            if np.any(any_active_after):
                first_active_idx = int(np.argmax(any_active_after))
                first_active_time_ms = float(tave_after[first_active_idx]) * 1000.0
                gap_ms = first_active_time_ms - noise_end * 1000.0
                if gap_ms < args.baseline_margin_ms:
                    gap_info = {"status": "close", "first_active_time_ms": first_active_time_ms,
                                "gap_ms": gap_ms}
                else:
                    gap_info = {"status": "ok", "first_active_time_ms": first_active_time_ms,
                                "gap_ms": gap_ms}
            else:
                gap_info = {"status": "no_activity"}
        else:
            gap_info = {"status": "no_bins_after"}

        # -----------------------------------------------------------------
        # 5. Spatial coherence gamma^2 for all Mirnov coil pairs
        # -----------------------------------------------------------------
        print("\n--- Computing Whole-Discharge Spatial Coherence (gamma^2) for all coil pairs ---")
        n_ensembles_val = args.ensemble
        coherence_spectra = {}
        f_csd = None
        target_pairs = [
            ("MP1", "MP3"), ("MP1", "MP4"), ("MP3", "MP4"),
        ]
        for c1, c2 in target_pairs:
            if c1 not in signals or c2 not in signals:
                continue
            print(f"  Processing pair {c1} vs {c2}...")
            f_c, Pxy, Pyy, Pxx = LAS.csd(signals[c1], time_sec, signals[c2], dt=dt, nfft=nfft,
                                          noverlap=nfft // 2, nensemble=n_ensembles_val,
                                          window='hann', detrend='constant')
            coh2 = LAS.xcoh2(Pxy, Pyy, Pxx)
            pair_key = f"{c1}_{c2}"
            coherence_spectra[pair_key] = np.mean(coh2, axis=1) if coh2.ndim > 1 else coh2
            if f_csd is None:
                f_csd = f_c

        # -----------------------------------------------------------------
        # 6. Dominant + coherence-confirmed discrete mode identification
        # -----------------------------------------------------------------
        print("\n--- Dominant Mode Identification via Peak-Picking ---")
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

            # Time-resolved activity: intervals where THIS mode's frequency bin
            # is continuously above its 3-sigma threshold, read off dominance_map.
            active_intervals_ms = find_active_intervals(
                dominance_map[p_idx, :], tave, min_duration_ms=args.min_interval_ms
            )
            total_active_duration_ms = round(
                sum(e - s for s, e in active_intervals_ms), 3
            )

            # Cross-reference against Check 1
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

        contaminated_confirmed = [m for m in discrete_modes
                                   if m["dual_criterion_pass"] and m["baseline_calibration_contaminated"]]
        if not baseline_is_clean:
            if contaminated_confirmed:
                bad_freqs_str = ", ".join(f'{m["frequency_hz"]/1e3:.1f}' for m in contaminated_confirmed)
                print(f"  WARNING: {len(contaminated_confirmed)} confirmed dominant+coherent mode(s) "
                      f"[{bad_freqs_str}] kHz coincide with a baseline-contaminated frequency bin from Check 1 "
                      f"-- their noise floor may be biased; treat these with extra caution.")
            else:
                print("  Check 1 vs. reported modes: none of the confirmed dominant+coherent modes coincide "
                      "with a baseline-contaminated frequency bin -- the baseline contamination found above does "
                      "not affect the modes this analysis reports.")
        baseline_effectively_ok = baseline_is_clean or not contaminated_confirmed
        if gap_info["status"] in ("close", "ok"):
            first_active_time_ms = gap_info["first_active_time_ms"]
            gap_ms = gap_info["gap_ms"]
            if gap_info["status"] == "ok":
                print(f"  Baseline OK: first 3-sigma-dominant activity after the baseline window at "
                      f"{first_active_time_ms:.1f} ms, {gap_ms:.1f} ms after the baseline window ends "
                      f"({noise_end*1000:.1f} ms).")
            elif baseline_effectively_ok and no_earlier_data:
                print(f"  Note: first 3-sigma-dominant activity after the baseline window at "
                      f"{first_active_time_ms:.1f} ms, only {gap_ms:.1f} ms after it ends "
                      f"({noise_end*1000:.1f} ms) -- below the {args.baseline_margin_ms:.1f} ms margin, but "
                      f"there is no earlier data to shift into, and any baseline contamination has been shown "
                      f"above not to reach a reported dominant+coherent mode. This reflects genuinely fast MHD "
                      f"onset in this discharge rather than an unreliable baseline.")
            else:
                print(f"  WARNING: first 3-sigma-dominant activity after the baseline window detected at "
                      f"{first_active_time_ms:.1f} ms, only {gap_ms:.1f} ms after the noise baseline window "
                      f"ends ({noise_end*1000:.1f} ms). This is below the requested margin "
                      f"({args.baseline_margin_ms:.1f} ms) and, combined with contamination reaching a reported "
                      f"mode, means the baseline should not be trusted as-is -- move/narrow "
                      f"--noise-start/--noise-end to a genuinely quiet stretch.")
        elif gap_info["status"] == "no_activity":
            print("  Note: no 3-sigma-dominant activity detected anywhere after the baseline window.")
        elif gap_info["status"] == "no_bins_after":
            print("  Note: no spectrogram time bins found after the baseline window; cannot check the gap.")

        # -----------------------------------------------------------------
        # 7. Export results
        # -----------------------------------------------------------------
        export = {
            "shot": str(shot),
            "confirmation_rule": CONFIRMATION_RULE,
            "threshold_phys": args.threshold_phys,
            "noise_baseline_window_ms": [noise_start * 1000.0, noise_end * 1000.0],
            "discrete_modes": discrete_modes,
        }
        out_json = f"discrete_modes_shot_{shot}.json"
        with open(out_json, "w") as fjson:
            json.dump(export, fjson, indent=2)
        print(f"Discrete mode results exported to '{out_json}'.")

        # -----------------------------------------------------------------
        # 8. Plotting: (1) Spectrogram, (2) Wavelet, (3) Spatial Coherence
        # -----------------------------------------------------------------
        print("\n--- Plotting results (3 panels) ---")
        fig, axs = plt.subplots(3, 1, figsize=(12, 13))

        # Panel 1: Spectrogram with 3-sigma dominance contour and peaks
        tt, ff = np.meshgrid(tave * 1000.0, f / 1000.0)
        mask_fplot = (f / 1000.0) <= args.fmax
    
        c0 = axs[0].pcolormesh(tt[mask_fplot, :], ff[mask_fplot, :], np.log10(Pyy1[mask_fplot, :] + 1e-30),
                                cmap='jet', shading='auto')
        fig.colorbar(c0, ax=axs[0], label="log10 PSD (V^2/Hz)", pad=0.03)

        dmap_plot = dominance_map[mask_fplot, :].astype(float)
        axs[0].contourf(tt[mask_fplot, :], ff[mask_fplot, :], dmap_plot,
                         levels=[0.5, 1.5], colors=['black'], alpha=0.10)
        axs[0].contour(tt[mask_fplot, :], ff[mask_fplot, :], dmap_plot,
                        levels=[0.5], colors='black', linewidths=1.2, linestyles=':')

        t_max_ms = tave[-1] * 1000.0
        for m in discrete_modes:
            if not m["dual_criterion_pass"]:
                continue
            f_khz = m["frequency_hz"] / 1000.0
        
            axs[0].axhline(y=f_khz, color='black', linestyle='--', alpha=0.9, linewidth=2.2)
        
            axs[0].text(
                t_max_ms * 0.98, f_khz, f"{f_khz:.1f} kHz", 
                color='white', fontsize=9, fontweight='bold', 
                va='bottom', ha='right',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6, edgecolor='white', lw=0.8)
            )

        axs[0].set_title(f"MP1 Spectrogram with 3σ Dominance Region\n"
                          f"(black dotted outline = smoothed 3σ-dominant region, black lines = confirmed dominant+coherent modes)")
        axs[0].set_xlabel("Time (ms)")
        axs[0].set_ylabel("Frequency (kHz)")

        # Panel 2: Morlet wavelet transform
        tt_w, ff_w = np.meshgrid(time_sec * 1000.0, freqs_wav / 1000.0)
    
        c1 = axs[1].pcolormesh(tt_w, ff_w, np.log10(wav_power + 1e-30), cmap='jet', shading='auto')
        fig.colorbar(c1, ax=axs[1], label="log10 Wavelet Power (rel.)", pad=0.03)
        axs[1].set_title(f"Morlet Wavelet Transform for MP1")
        axs[1].set_xlabel("Time (ms)")
        axs[1].set_ylabel("Frequency (kHz)")

        # Panel 3: Coherence spectrum
        colors_coil = {
            "MP1_MP3": "green",
            "MP1_MP4": "purple",
            "MP3_MP4": "brown",
        }
        mask_f_csd_plot = ((f_csd >= 0.0) & (f_csd / 1000.0 <= args.fmax)) if f_csd is not None else None
        for pair_key, spec in coherence_spectra.items():
            label_name = pair_key.replace("_", "-")
            axs[2].plot(f_csd[mask_f_csd_plot] / 1000.0, spec[mask_f_csd_plot],
                        color=colors_coil.get(pair_key, 'gray'), linewidth=1.3, label=label_name)
        axs[2].axhline(y=args.threshold_phys, color='black', linestyle=':', linewidth=1.2,
                       label=f"Physical Threshold ({args.threshold_phys:.1f})")
    
        for m in discrete_modes:
            color = 'green' if m["dual_criterion_pass"] else 'red'
            axs[2].axvline(x=m["frequency_hz"] / 1000.0, color=color, linestyle=':', alpha=0.8, linewidth=1.4)
        
        axs[2].set_title("Whole-Discharge Spatial Coherence gamma^2(f), all coil pairs\n" \
        "(green dotted lines = confirmed dominant+coherent modes, red dotted lines = dominant-only modes)")
        axs[2].set_xlabel("Frequency (kHz)")
        axs[2].set_ylabel(r"Coherence $\gamma^2$")
        axs[2].set_ylim(0, 1.05)
        axs[2].grid(True, alpha=0.3)
        axs[2].legend(loc='upper right', fontsize=9)

        plt.tight_layout()
        output_png = f"mhd_analysis_objective1_{shot}.png"
        plt.savefig(output_png, dpi=150)
        print(f"Results successfully saved to '{output_png}'.")
        print(f"\n========================================\nFINISHED ANALYSIS FOR SHOT {shot}\n========================================")
    except Exception as e:
        print(f"\n[ERROR] Shot {shot} failed with exception: {e}")
        traceback.print_exc()


def main():
    args = parse_args()
    
    # We use a multiprocessing Pool to process shots concurrently
    num_processes = min(len(args.shots), multiprocessing.cpu_count())
    print(f"Running shots {args.shots} concurrently using {num_processes} processes...\n")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(process_shot, [(shot, args) for shot in args.shots])


if __name__ == "__main__":
    main()