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

current = Path(__file__).resolve().parent
root_dir = None
for p in [current] + list(current.parents):
    if (p / "jpack").exists():
        root_dir = p
        break
if root_dir is None:
    root_dir = Path("c:/TFG")

for p_add in [root_dir / "jpack", root_dir / "analysis", root_dir / "analysis" / "common"]:
    if str(p_add) not in sys.path:
        sys.path.append(str(p_add))

import turnelib as TE
import libana_signal as LAS
from mhd_common import morlet_cwt

CONFIRMATION_RULE = "any single coil pair > threshold"


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


def parse_args():
    p = argparse.ArgumentParser(description="MHD Analysis - Objective 1 (Mode Identification)")
    p.add_argument("-s", "--shots", type=int, nargs="+", default=SHOTS_DEFAULT,
                   help=f"List of shot numbers to analyze (default: {SHOTS_DEFAULT})")
    p.add_argument("-d", "--data-dir-pattern", type=str, default="data/hj{shot}",
                   help="Data directory pattern (default: data/hj{shot})")
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
            f_c, Pxy, Pyy, Pxx = LAS.csd(
                signals[c1], time_sec, signals[c2], dt=dt, nfft=nfft,
                noverlap=nfft // 2, nensemble=args.ensemble,
                window='hann', detrend='constant'
            )
            coh2 = LAS.xcoh2(Pxy, Pyy, Pxx)
            pair_key = f"{c1}_{c2}"
            coherence_spectra[pair_key] = np.mean(coh2, axis=1) if coh2.ndim > 1 else coh2
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

        axs[0].set_title("MP1 Spectrogram with 3-sigma Dominance Region")
        axs[0].set_xlabel("Time (ms)")
        axs[0].set_ylabel("Frequency (kHz)")

        tt_w, ff_w = np.meshgrid(time_sec * 1000.0, freqs_wav / 1000.0)
        c1 = axs[1].pcolormesh(tt_w, ff_w, np.log10(wav_power + 1e-30), cmap='jet', shading='auto')
        fig.colorbar(c1, ax=axs[1], label="log10 Wavelet Power (rel.)", pad=0.03)
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
        output_png = f"mhd_analysis_objective1_{shot}.png"
        plt.savefig(output_png, dpi=150)
        plt.close(fig)
        print(f"Objective 1 overview saved to '{output_png}'.")

    except Exception as e:
        print(f"Error processing shot {shot}: {e}")
        import traceback
        traceback.print_exc()


def main():
    args = parse_args()
    num_processes = min(len(args.shots), multiprocessing.cpu_count())
    if num_processes <= 1:
        for shot in args.shots:
            process_shot(shot, args)
    else:
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(process_shot, [(shot, args) for shot in args.shots])


if __name__ == "__main__":
    main()