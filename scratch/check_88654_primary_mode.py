"""
Phase 1: Primary mode identification for shot 88654, windowed to 259.1-275.0 ms.
Replicates the Objective 1 dual-criterion detector (3-sigma + gamma^2 > 0.7)
on the same time window and with the same parameters used for 88653.
"""
import sys, os, json
import numpy as np
import scipy.signal as dsp
from pathlib import Path

# Setup imports
root_dir = Path("c:/TFG")
for p_add in [root_dir / "jpack", root_dir / "analysis", root_dir / "analysis" / "common"]:
    if str(p_add) not in sys.path:
        sys.path.append(str(p_add))

import turnelib as TE

# --- Parameters (same as 88653 run) ---
SHOT = 88654
DATA_DIR = root_dir / "data" / f"hj{SHOT}"
COILS = ["MP1", "MP3", "MP4"]
FRES = 500.0          # Hz frequency resolution
THRESHOLD_PHYS = 0.7  # coherence threshold
FMIN_PICK = 15.0      # kHz
FMAX_PICK = 100.0     # kHz
N_PEAKS = 8

# Windows (ms)
ANALYSIS_START_MS = 259.1
ANALYSIS_END_MS = 275.0
# Noise baseline: use pre-burst quiet period (same as 88653: 150-175 ms)
NOISE_START_MS = 150.0
NOISE_END_MS = 175.0

# --- Load data ---
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
        n_samples = len(t_s)
        sample_idx = min(100, n_samples - 1) if n_samples > 1 else 0
        dt = (t_s[sample_idx] - t_s[0]) / float(sample_idx) if sample_idx > 0 else 1e-6
        fs = 1.0 / dt

nfft = int(np.round(1.0 / (dt * FRES)))

# --- Slice to analysis window ---
t_ms = time_sec * 1000.0
analysis_mask = (t_ms >= ANALYSIS_START_MS) & (t_ms <= ANALYSIS_END_MS)
noise_mask_full = (t_ms >= NOISE_START_MS) & (t_ms <= NOISE_END_MS)

signals_win = {c: s[analysis_mask] for c, s in signals.items()}
signals_noise = {c: s[noise_mask_full] for c, s in signals.items()}
time_win = time_sec[analysis_mask]
time_noise = time_sec[noise_mask_full]

n_win = len(time_win)
n_noise = len(time_noise)

out = []
def P(s=""):
    out.append(s)
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode('ascii', 'replace').decode('ascii'))

P("=" * 90)
P(f"PRIMARY MODE IDENTIFICATION: Shot {SHOT}")
P(f"Analysis window: {ANALYSIS_START_MS:.1f} -- {ANALYSIS_END_MS:.1f} ms")
P(f"Noise baseline:  {NOISE_START_MS:.1f} -- {NOISE_END_MS:.1f} ms")
P(f"Samples in analysis window: {n_win}  ({n_win*dt*1000:.1f} ms)")
P(f"Samples in noise window:    {n_noise}  ({n_noise*dt*1000:.1f} ms)")
P(f"NFFT: {nfft}, fs: {fs/1e6:.2f} MHz, df: {FRES:.0f} Hz")
P(f"Dual criterion: 3-sigma above noise baseline AND gamma^2 > {THRESHOLD_PHYS}")
P("=" * 90)

# --- Compute Welch PSD on analysis window and noise window separately ---
f_welch, Pyy_win = dsp.welch(signals_win["MP1"], fs=fs, window='hann',
                              nperseg=nfft, noverlap=nfft//2, detrend='constant')
_, Pyy_noise = dsp.welch(signals_noise["MP1"], fs=fs, window='hann',
                          nperseg=nfft, noverlap=nfft//2, detrend='constant')

# 3-sigma threshold from noise baseline
mean_noise = Pyy_noise
# For a single Welch spectrum the "noise" is just the PSD itself; use the
# noise-window PSD as the reference level and assume 3x is the threshold
# (consistent with how the obj1 pipeline uses per-frequency noise stats).
# Actually, the obj1 pipeline uses a running spectrogram and computes
# mean + 3*std across time bins. With a single Welch estimate we just have
# one noise spectrum; treat it as the mean and use a factor for threshold.
# Let's be faithful: compute a running PSD on the noise window too.

# Running spectrogram on noise window (for mean + 3*std per frequency)
try:
    from scipy.ndimage import uniform_filter1d
    import libana_signal as LAS
    f_run, t_run_noise, Pyy_run_noise = LAS.running(
        signals_noise["MP1"], time_noise, dt=dt, nfft=nfft, noverlap=nfft//2,
        window='hann', detrend='constant'
    )
except TypeError:
    f_run, t_run_noise, Pyy_run_noise = LAS.running(
        signals_noise["MP1"], time_noise, dt=dt, nfft=nfft, noverlap=nfft//2,
        detrend='constant'
    )

mean_noise_per_f = np.mean(Pyy_run_noise, axis=1)
std_noise_per_f = np.std(Pyy_run_noise, axis=1)
snr_threshold_per_f = mean_noise_per_f + 3.0 * std_noise_per_f

# Compute running spectrogram on analysis window too (for dominance map)
try:
    f_run_a, t_run_a, Pyy_run_a = LAS.running(
        signals_win["MP1"], time_win, dt=dt, nfft=nfft, noverlap=nfft//2,
        window='hann', detrend='constant'
    )
except TypeError:
    f_run_a, t_run_a, Pyy_run_a = LAS.running(
        signals_win["MP1"], time_win, dt=dt, nfft=nfft, noverlap=nfft//2,
        detrend='constant'
    )

# Time-averaged PSD in the analysis window
Pyy_avg = np.mean(Pyy_run_a, axis=1)

# Coherence in analysis window
coherence_spectra = {}
target_pairs = [("MP1", "MP3"), ("MP1", "MP4"), ("MP3", "MP4")]
for c1, c2 in target_pairs:
    f_c, coh2 = dsp.coherence(
        signals_win[c1], signals_win[c2], fs=fs, window='hann',
        nperseg=nfft, noverlap=nfft//2
    )
    coherence_spectra[f"{c1}_{c2}"] = (f_c, coh2)

# --- Find peaks ---
mask_f_pick = (f_run / 1000.0 >= FMIN_PICK) & (f_run / 1000.0 <= FMAX_PICK)
df = f_run[1] - f_run[0]
min_dist_bins = max(1, int(5000.0 / df))

log_Pyy = np.log10(Pyy_avg + 1e-30)
peak_idx_local, _ = dsp.find_peaks(log_Pyy[mask_f_pick], distance=min_dist_bins)
idx_map = np.where(mask_f_pick)[0]
peak_idx = idx_map[peak_idx_local]

order = np.argsort(Pyy_avg[peak_idx])[::-1][:N_PEAKS]
peak_idx = peak_idx[order]

# --- Evaluate each candidate ---
P(f"\n{'='*90}")
P("CANDIDATE MODES (sorted by amplitude)")
P(f"{'='*90}")
P(f"{'Rank':<5} {'Freq (kHz)':<12} {'Amplitude':<14} {'3-sigma thr':<14} "
  f"{'Above 3s?':<10} {'MP1-MP3':<10} {'MP1-MP4':<10} {'MP3-MP4':<10} "
  f"{'Any>0.7?':<10} {'DUAL PASS?':<10}")
P("-" * 120)

modes_88654 = []
for rank, p_idx in enumerate(peak_idx, 1):
    freq_hz = float(f_run[p_idx])
    freq_khz = freq_hz / 1000.0
    amp = float(Pyy_avg[p_idx])
    thresh = float(snr_threshold_per_f[p_idx]) if p_idx < len(snr_threshold_per_f) else 0.0
    is_dominant = bool(amp >= thresh)

    coh_vals = {}
    confirming_pairs = []
    for pair_key, (fc, coh2) in coherence_spectra.items():
        idx_csd = int(np.abs(fc - freq_hz).argmin())
        val = float(coh2[idx_csd])
        coh_vals[pair_key] = val
        if val > THRESHOLD_PHYS:
            confirming_pairs.append(pair_key)

    coherence_confirmed = len(confirming_pairs) > 0
    dual_pass = bool(is_dominant and coherence_confirmed)

    modes_88654.append({
        "rank": rank,
        "freq_khz": freq_khz,
        "amplitude": amp,
        "threshold": thresh,
        "is_dominant": is_dominant,
        "coherence": coh_vals,
        "confirming_pairs": confirming_pairs,
        "dual_pass": dual_pass,
    })

    g12 = coh_vals.get("MP1_MP3", 0)
    g14 = coh_vals.get("MP1_MP4", 0)
    g34 = coh_vals.get("MP3_MP4", 0)
    dom_str = "YES" if is_dominant else "no"
    coh_str = "YES" if coherence_confirmed else "no"
    dual_str = "** YES **" if dual_pass else "no"

    P(f"{rank:<5} {freq_khz:<12.1f} {amp:<14.3e} {thresh:<14.3e} "
      f"{dom_str:<10} {g12:<10.3f} {g14:<10.3f} {g34:<10.3f} "
      f"{coh_str:<10} {dual_str:<10}")

# --- Comparison with 88653 ---
P(f"\n{'='*90}")
P("COMPARISON WITH SHOT 88653")
P(f"{'='*90}")

ref_modes_passing = [
    {"freq_khz": 89.0, "amp": 8.508e-07, "coh_range": "0.96-0.98"},
    {"freq_khz": 40.0, "amp": 4.119e-07, "coh_range": "0.87-0.90"},
    {"freq_khz": 49.0, "amp": 3.837e-07, "coh_range": "0.88-0.92"},
    {"freq_khz": 95.5, "amp": 2.722e-07, "coh_range": "0.89-0.94"},
    {"freq_khz": 83.5, "amp": 2.265e-07, "coh_range": "0.80-0.88"},
]

P(f"\n88653 dual-criterion passing modes (full-discharge, for reference):")
for m in ref_modes_passing:
    P(f"  {m['freq_khz']:.1f} kHz,  amp={m['amp']:.3e},  gamma2={m['coh_range']}")

passing_88654 = [m for m in modes_88654 if m['dual_pass']]
P(f"\n88654 dual-criterion passing modes (259.1-275.0 ms window):")
if passing_88654:
    for m in passing_88654:
        g12 = m['coherence'].get("MP1_MP3", 0)
        g14 = m['coherence'].get("MP1_MP4", 0)
        g34 = m['coherence'].get("MP3_MP4", 0)
        P(f"  {m['freq_khz']:.1f} kHz,  amp={m['amplitude']:.3e},  "
          f"gamma2: MP1-MP3={g12:.3f}, MP1-MP4={g14:.3f}, MP3-MP4={g34:.3f}")
else:
    P("  *** NO MODES PASS THE DUAL CRITERION ***")

# --- Summary ---
P(f"\n{'='*90}")
P("SUMMARY")
P(f"{'='*90}")
n_above_3s = sum(1 for m in modes_88654 if m['is_dominant'])
n_above_coh = sum(1 for m in modes_88654 if len(m['confirming_pairs']) > 0)
n_dual = len(passing_88654)
P(f"  Candidates above 3-sigma noise floor: {n_above_3s}/{len(modes_88654)}")
P(f"  Candidates with gamma^2 > 0.7 on any pair: {n_above_coh}/{len(modes_88654)}")
P(f"  Candidates passing BOTH criteria: {n_dual}/{len(modes_88654)}")

if n_dual == 0:
    P(f"\n  Shot 88654 does NOT exhibit any physical mode clearing the dual criterion")
    P(f"  in the 259.1-275.0 ms window. This is a distinct finding from the bicoherence")
    P(f"  result: the primary mode itself is absent or below detection threshold,")
    P(f"  not just the nonlinear coupling.")

# Write output
output_path = root_dir / "shot_88654_primary_mode_check.log"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")

P(f"\nDone. Output written to {output_path}")
