"""
Apples-to-apples: run the same windowed dual-criterion check on 88653 at 259.1-275.0 ms.
This gives us same-window reference amplitudes to compare against 88654.
"""
import sys, os
import numpy as np
import scipy.signal as dsp
from pathlib import Path

root_dir = Path("c:/TFG")
for p_add in [root_dir / "jpack", root_dir / "analysis", root_dir / "analysis" / "common"]:
    if str(p_add) not in sys.path:
        sys.path.append(str(p_add))

import turnelib as TE
import libana_signal as LAS

COILS = ["MP1", "MP3", "MP4"]
FRES = 500.0
THRESHOLD_PHYS = 0.7
FMIN_PICK = 15.0
FMAX_PICK = 100.0
N_PEAKS = 8
ANALYSIS_START_MS = 259.1
ANALYSIS_END_MS = 275.0
NOISE_START_MS = 150.0
NOISE_END_MS = 175.0

for SHOT in [88653, 88654]:
    DATA_DIR = root_dir / "data" / f"hj{SHOT}"
    edf = TE.edf()
    signals = {}
    time_sec = None
    dt = None
    fs = None
    for coil in COILS:
        fpath = DATA_DIR / f"{coil}@{SHOT}.edf"
        dat = edf.load(str(fpath))
        t = dat[:, 0]
        ys = dat[:, 1]
        t_s = t / 1000.0 if edf.DimUnit[0] == 'ms' else t
        signals[coil] = ys
        if time_sec is None:
            time_sec = t_s
            sample_idx = min(100, len(t_s) - 1)
            dt = (t_s[sample_idx] - t_s[0]) / float(sample_idx)
            fs = 1.0 / dt

    nfft = int(np.round(1.0 / (dt * FRES)))
    t_ms = time_sec * 1000.0
    analysis_mask = (t_ms >= ANALYSIS_START_MS) & (t_ms <= ANALYSIS_END_MS)
    noise_mask = (t_ms >= NOISE_START_MS) & (t_ms <= NOISE_END_MS)

    signals_win = {c: s[analysis_mask] for c, s in signals.items()}
    signals_noise = {c: s[noise_mask] for c, s in signals.items()}
    time_win = time_sec[analysis_mask]
    time_noise = time_sec[noise_mask]

    try:
        f_run, t_run_noise, Pyy_run_noise = LAS.running(
            signals_noise["MP1"], time_noise, dt=dt, nfft=nfft, noverlap=nfft//2,
            window='hann', detrend='constant')
    except TypeError:
        f_run, t_run_noise, Pyy_run_noise = LAS.running(
            signals_noise["MP1"], time_noise, dt=dt, nfft=nfft, noverlap=nfft//2,
            detrend='constant')

    mean_noise_per_f = np.mean(Pyy_run_noise, axis=1)
    std_noise_per_f = np.std(Pyy_run_noise, axis=1)
    snr_threshold_per_f = mean_noise_per_f + 3.0 * std_noise_per_f

    try:
        f_run_a, t_run_a, Pyy_run_a = LAS.running(
            signals_win["MP1"], time_win, dt=dt, nfft=nfft, noverlap=nfft//2,
            window='hann', detrend='constant')
    except TypeError:
        f_run_a, t_run_a, Pyy_run_a = LAS.running(
            signals_win["MP1"], time_win, dt=dt, nfft=nfft, noverlap=nfft//2,
            detrend='constant')

    Pyy_avg = np.mean(Pyy_run_a, axis=1)

    coherence_spectra = {}
    for c1, c2 in [("MP1","MP3"),("MP1","MP4"),("MP3","MP4")]:
        f_c, coh2 = dsp.coherence(signals_win[c1], signals_win[c2], fs=fs,
                                   window='hann', nperseg=nfft, noverlap=nfft//2)
        coherence_spectra[f"{c1}_{c2}"] = (f_c, coh2)

    mask_f_pick = (f_run / 1000.0 >= FMIN_PICK) & (f_run / 1000.0 <= FMAX_PICK)
    df = f_run[1] - f_run[0]
    min_dist_bins = max(1, int(5000.0 / df))
    log_Pyy = np.log10(Pyy_avg + 1e-30)
    peak_idx_local, _ = dsp.find_peaks(log_Pyy[mask_f_pick], distance=min_dist_bins)
    idx_map = np.where(mask_f_pick)[0]
    peak_idx = idx_map[peak_idx_local]
    order = np.argsort(Pyy_avg[peak_idx])[::-1][:N_PEAKS]
    peak_idx = peak_idx[order]

    print(f"\nSHOT {SHOT} (259.1-275.0 ms window):")
    print(f"{'Rank':<5} {'Freq(kHz)':<10} {'Amplitude':<13} {'3s thr':<13} "
          f"{'3s?':<5} {'g12':<8} {'g14':<8} {'g34':<8} {'DUAL'}")
    print("-" * 95)
    for rank, p_idx in enumerate(peak_idx, 1):
        freq_khz = float(f_run[p_idx]) / 1000.0
        amp = float(Pyy_avg[p_idx])
        thresh = float(snr_threshold_per_f[p_idx]) if p_idx < len(snr_threshold_per_f) else 0.0
        is_dom = amp >= thresh

        coh_vals = {}
        confirming = []
        for pk, (fc, c2) in coherence_spectra.items():
            idx_c = int(np.abs(fc - f_run[p_idx]).argmin())
            v = float(c2[idx_c])
            coh_vals[pk] = v
            if v > THRESHOLD_PHYS:
                confirming.append(pk)

        dual = is_dom and len(confirming) > 0
        g12 = coh_vals.get("MP1_MP3", 0)
        g14 = coh_vals.get("MP1_MP4", 0)
        g34 = coh_vals.get("MP3_MP4", 0)
        d_str = "YES" if dual else "no"
        s_str = "Y" if is_dom else "n"
        print(f"{rank:<5} {freq_khz:<10.1f} {amp:<13.3e} {thresh:<13.3e} "
              f"{s_str:<5} {g12:<8.3f} {g14:<8.3f} {g34:<8.3f} {d_str}")
