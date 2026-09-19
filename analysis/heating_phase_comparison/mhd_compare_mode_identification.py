import sys
import json
import argparse
from pathlib import Path
import numpy as np
import scipy.signal as dsp
import matplotlib.pyplot as plt

current = Path(__file__).resolve().parent
root_dir = None
for p in [current] + list(current.parents):
    if (p / "jpack").exists():
        root_dir = p
        break
if root_dir is None:
    root_dir = Path("c:/TFG")

for p_add in [root_dir / "jpack", root_dir / "analysis", root_dir / "analysis" / "common", current]:
    if str(p_add) not in sys.path:
        sys.path.append(str(p_add))

import libana_signal as LAS
from mhd_common import morlet_cwt
from comparison_common import (
    DEFAULT_SHOT, DEFAULT_OUT_DIR, T_NOISE_START, T_NOISE_END,
    T_W1_START, T_W1_END, T_W2_START, T_W2_END,
    load_signals, safe_savefig
)


def run_mode_identification_comparison(shot=DEFAULT_SHOT, out_dir=None, verbose=False):
    if out_dir is None:
        out_dir = DEFAULT_OUT_DIR
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Mode Identification Overview Comparison (Shot {shot}) ---")

    time_ms, dt_s, fs, signals_dict, _ = load_signals(shot=shot)
    if "MP1" not in signals_dict or time_ms is None:
        print("Error: MP1 signal not available.")
        return

    ys_mp1 = signals_dict["MP1"]

    windows = [
        ("ECH + NBI Phase (259.1 - 291.3 ms)", T_W1_START, T_W1_END, "mhd_analysis_objective1_88653_ECH_NBI.png", "discrete_modes_ECH_NBI.json"),
        ("Just NBI Phase (291.3 - 300.0 ms)", T_W2_START, T_W2_END, "mhd_analysis_objective1_88653_NBI_only.png", "discrete_modes_NBI_only.json")
    ]

    for title_win, tw_start, tw_end, out_png, out_json in windows:
        print(f"  Processing {title_win}...")
        idx_win = np.where((time_ms >= tw_start) & (time_ms <= tw_end))[0]
        t_win_ms = time_ms[idx_win]
        ys_win_mp1 = ys_mp1[idx_win]

        # Running Spectrogram
        fres = 500.0
        nfft = int(np.round(1.0 / (dt_s * fres)))
        f_spec, tave, Pyy1 = LAS.running(ys_win_mp1, t_win_ms / 1000.0, dt=dt_s, nfft=nfft, noverlap=nfft // 2, window='hann', detrend='constant')

        # Morlet Wavelet Transform
        freqs_wav = np.linspace(10.0 * 1e3, 150.0 * 1e3, 150)
        Wx = morlet_cwt(ys_win_mp1, fs, freqs_wav, verbose=False)
        wav_power = np.abs(Wx) ** 2

        # Noise Baseline
        idx_noise = np.where((time_ms >= T_NOISE_START) & (time_ms <= T_NOISE_END))[0]
        _, _, Pyy_noise = LAS.running(ys_mp1[idx_noise], time_ms[idx_noise] / 1000.0, dt=dt_s, nfft=nfft, noverlap=nfft // 2, window='hann', detrend='constant')
        mean_noise = np.mean(Pyy_noise, axis=1)
        std_noise = np.std(Pyy_noise, axis=1)
        snr_threshold = mean_noise + 3.0 * std_noise
        dominance_map = Pyy1 > snr_threshold[:, None]

        # Multi-probe Coherence
        coherence_spectra = {}
        target_pairs = [("MP1", "MP3"), ("MP1", "MP4"), ("MP3", "MP4")]
        f_csd = None
        for c1, c2 in target_pairs:
            if c1 not in signals_dict or c2 not in signals_dict:
                continue
            f_c, Pxy, Pyy, Pxx = LAS.csd(
                signals_dict[c1][idx_win], t_win_ms / 1000.0, signals_dict[c2][idx_win],
                dt=dt_s, nfft=nfft, noverlap=nfft // 2, nensemble=30, window='hann', detrend='constant'
            )
            coh2 = LAS.xcoh2(Pxy, Pyy, Pxx)
            pair_key = f"{c1}_{c2}"
            coherence_spectra[pair_key] = np.mean(coh2, axis=1) if coh2.ndim > 1 else coh2
            if f_csd is None:
                f_csd = f_c

        # Peak Identification
        Pyy_avg = np.mean(Pyy1, axis=1)
        mask_f_pick = (f_spec / 1000.0 >= 15.0) & (f_spec / 1000.0 <= 120.0)
        df = f_spec[1] - f_spec[0]
        min_dist = max(1, int(4000.0 / df))
        peak_idx_local, _ = dsp.find_peaks(np.log10(Pyy_avg[mask_f_pick] + 1e-30), distance=min_dist)
        idx_map = np.where(mask_f_pick)[0]
        peak_idx = idx_map[peak_idx_local]
        order = np.argsort(Pyy_avg[peak_idx])[::-1][:8]
        peak_idx = peak_idx[order]

        discrete_modes = []
        for p_idx in peak_idx:
            freq_hz = float(f_spec[p_idx])
            amp = float(Pyy_avg[p_idx])
            thresh = float(snr_threshold[p_idx])
            is_dom = bool(amp >= thresh)
            coherence_dict = {}
            confirming = []
            if f_csd is not None:
                idx_c = int(np.abs(f_csd - freq_hz).argmin())
                for pk, spec in coherence_spectra.items():
                    val = float(spec[idx_c])
                    coherence_dict[pk] = val
                    if val > 0.7:
                        confirming.append(pk)
            dual_pass = bool(is_dom and len(confirming) > 0)
            discrete_modes.append({
                "frequency_hz": freq_hz,
                "amplitude": amp,
                "snr_threshold": thresh,
                "is_above_noise_floor": is_dom,
                "coherence": coherence_dict,
                "confirming_pairs": confirming,
                "dual_criterion_pass": dual_pass
            })
            if verbose:
                print(f"    Mode {freq_hz/1000.0:5.1f} kHz | Amp: {amp:.2e} | 3-sigma: {'PASS' if is_dom else 'FAIL'} | Coh > 0.7: {confirming} | Dual: {'CONFIRMED' if dual_pass else 'UNCONFIRMED'}")

        out_json_path = out_dir / out_json
        with open(out_json_path, "w", encoding="utf-8") as f_out:
            json.dump({"shot": shot, "window_ms": [tw_start, tw_end], "discrete_modes": discrete_modes}, f_out, indent=2)
        print(f"    Saved JSON: {out_json_path.name}")

        # Plot 3 Panels
        plt.rcdefaults()
        fig, axs = plt.subplots(3, 1, figsize=(12, 13))
        fig.suptitle(f"Heliotron J #{shot} — Objective 1 Mode Identification\n({title_win})", fontsize=13, fontweight='bold')

        # Panel 0: Spectrogram
        tt, ff = np.meshgrid(tave * 1000.0, f_spec / 1000.0)
        mask_fplot = (f_spec / 1000.0) <= 140.0
        c0 = axs[0].pcolormesh(tt[mask_fplot, :], ff[mask_fplot, :], np.log10(Pyy1[mask_fplot, :] + 1e-30), cmap='jet', shading='auto')
        fig.colorbar(c0, ax=axs[0], label=r"$\log_{10}\mathrm{PSD}\ (\mathrm{V}^2/\mathrm{Hz})$", pad=0.03)
        dmap_plot = dominance_map[mask_fplot, :].astype(float)
        axs[0].contourf(tt[mask_fplot, :], ff[mask_fplot, :], dmap_plot, levels=[0.5, 1.5], colors=['black'], alpha=0.12)
        axs[0].contour(tt[mask_fplot, :], ff[mask_fplot, :], dmap_plot, levels=[0.5], colors='black', linewidths=1.2, linestyles=':')

        for m in discrete_modes:
            if not m["dual_criterion_pass"]:
                continue
            f_khz_val = m["frequency_hz"] / 1000.0
            axs[0].axhline(y=f_khz_val, color='black', linestyle='--', alpha=0.85, linewidth=2.0)
            axs[0].text(tw_end * 0.99, f_khz_val, f"{f_khz_val:.1f} kHz", color='white', fontsize=9, fontweight='bold',
                        va='bottom', ha='right', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

        axs[0].set_title("MP1 Spectrogram with 3-sigma Dominance Region")
        axs[0].set_xlabel("Time (ms)")
        axs[0].set_ylabel("Frequency (kHz)")
        axs[0].set_ylim(0, 140.0)

        # Panel 1: Wavelet
        tt_w, ff_w = np.meshgrid(t_win_ms, freqs_wav / 1000.0)
        c1 = axs[1].pcolormesh(tt_w, ff_w, np.log10(wav_power + 1e-30), cmap='jet', shading='auto')
        fig.colorbar(c1, ax=axs[1], label=r"$\log_{10}\mathrm{Power}$", pad=0.03)
        axs[1].set_title("Morlet Wavelet Transform (MP1)")
        axs[1].set_xlabel("Time (ms)")
        axs[1].set_ylabel("Frequency (kHz)")
        axs[1].set_ylim(0, 140.0)

        # Panel 2: Coherence
        colors_coil = {"MP1_MP3": "tab:blue", "MP1_MP4": "tab:green", "MP3_MP4": "tab:purple"}
        mask_f_csd = (f_csd >= 0) & (f_csd <= 140000.0)
        for pair_key, spec in coherence_spectra.items():
            axs[2].plot(f_csd[mask_f_csd] / 1000.0, spec[mask_f_csd], color=colors_coil.get(pair_key, 'gray'), lw=1.8, label=pair_key.replace("_", "-"))
        axs[2].axhline(0.7, color='black', ls=':', lw=1.3, label='Physical Threshold (0.7)')
        for m in discrete_modes:
            color = 'green' if m["dual_criterion_pass"] else 'red'
            axs[2].axvline(m["frequency_hz"] / 1000.0, color=color, ls=':', alpha=0.8, lw=1.5)

        axs[2].set_title(r"Inter-Probe Spatial Coherence $\gamma^2(f)$")
        axs[2].set_xlabel("Frequency (kHz)")
        axs[2].set_ylabel(r"Coherence $\gamma^2$")
        axs[2].set_ylim(0, 1.05)
        axs[2].set_xlim(0, 140.0)
        axs[2].grid(True, alpha=0.3)
        axs[2].legend(loc='upper right', fontsize=9)

        plt.tight_layout()
        out_path = out_dir / out_png
        safe_savefig(fig, out_path, dpi=150)
        plt.close(fig)
        print(f"    Saved figure: {out_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mode Identification Comparison (ECH+NBI vs Just NBI)")
    parser.add_argument("-s", "--shots", type=int, default=DEFAULT_SHOT, help="Shot number to analyze")
    parser.add_argument("-o", "--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode candidates")
    args = parser.parse_args()

    run_mode_identification_comparison(shot=args.shots, out_dir=args.out_dir, verbose=args.verbose)
