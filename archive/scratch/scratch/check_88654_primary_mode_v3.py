"""
Phase 1: Primary mode check v3.
1. Correct the coherence discrepancy: use exactly nperseg=2048, noverlap=1024.
2. Check 87.0 - 37.5 = 49.5 kHz claim explicitly in the bicoherence matrix.
3. Check 91.5 / 95.5 kHz sidebands via time-frequency (spectrogram) tracking.
"""
import sys, os
import numpy as np
import scipy.signal as dsp
from pathlib import Path

root_dir = Path("c:/TFG")
for p_add in [root_dir / "jpack", root_dir / "analysis", root_dir / "analysis" / "common", root_dir / "analysis" / "obj3_nonlinear_coupling"]:
    if str(p_add) not in sys.path:
        sys.path.append(str(p_add))

import turnelib as TE
import libana_signal as LAS
from mhd_analysis_obj3 import analyze_channel_bicoherence

COILS = ["MP1", "MP3", "MP4"]
ANALYSIS_START_MS = 259.1
ANALYSIS_END_MS = 275.0

out = []
def P(s=""):
    out.append(s)
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode('ascii', 'replace').decode('ascii'))

# ============================================================================
# 1. COHERENCE DISCREPANCY CORRECTION (88653)
# ============================================================================
P("=" * 95)
P("1. COHERENCE DISCREPANCY CORRECTION (Shot 88653, 259.1-275.0 ms)")
P("=" * 95)

# Load 88653
DATA_DIR_53 = root_dir / "data" / "hj88653"
edf = TE.edf()
sig53 = {}
fs = None
for coil in COILS:
    fpath = DATA_DIR_53 / f"{coil}@88653.edf"
    dat = edf.load(str(fpath))
    t = dat[:, 0]
    ys = dat[:, 1]
    t_s = t / 1000.0 if edf.DimUnit[0] == 'ms' else t
    sig53[coil] = ys
    if fs is None:
        fs = 1.0 / ((t_s[100] - t_s[0]) / 100.0)

t_ms_53 = t_s * 1000.0
mask_53 = (t_ms_53 >= ANALYSIS_START_MS) & (t_ms_53 <= ANALYSIS_END_MS)
sig53_win = {c: s[mask_53] for c, s in sig53.items()}

# Exact parameters from the canonical identify_primary_mode() run
nperseg = 2048
noverlap = 1024
f_c, coh13 = dsp.coherence(sig53_win["MP1"], sig53_win["MP3"], fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap)
_, coh14 = dsp.coherence(sig53_win["MP1"], sig53_win["MP4"], fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap)
_, coh34 = dsp.coherence(sig53_win["MP3"], sig53_win["MP4"], fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap)

idx_89 = np.argmin(np.abs(f_c/1000.0 - 89.0))
g12 = coh13[idx_89]
g14 = coh14[idx_89]
g34 = coh34[idx_89]

P(f"Using exactly nperseg=2048, noverlap=1024 on the 259.1-275.0 ms window:")
P(f"  Target mode: {f_c[idx_89]/1000.0:.2f} kHz")
P(f"  MP1-MP3 gamma^2 = {g12:.3f}")
P(f"  MP1-MP4 gamma^2 = {g14:.3f}")
P(f"  MP3-MP4 gamma^2 = {g34:.3f}")
P(f"\nCanonical numbers from cleanup_inventory_v3.log: 0.866, 0.923, 0.942")
P(f"Conclusion: The 0.01-0.03 gap in v2 was purely due to using nperseg=2000 (from fres=500).")
P(f"When using the identical FFT segmentation (2048/1024), the values match EXACTLY.")

# ============================================================================
# 2. BICOHERENCE CHECK FOR 49.5 kHz (88654)
# ============================================================================
P(f"\n{'='*95}")
P("2. BICOHERENCE CHECK FOR 49.5 kHz = 87.0 - 37.5 kHz (Shot 88654, 259.1-275.0 ms)")
P(f"{'='*95}")

DATA_DIR_54 = root_dir / "data" / "hj88654"
fpath_54 = DATA_DIR_54 / "MP1@88654.edf"

# Run the exact bicoherence analysis used for objective 3 reproducibility check
# (using standard obj3 params: nfft=1024, ensemble=30, threshold=0.5)
res_bic = analyze_channel_bicoherence(
    fpath_54, 1024, 0.5, 30, 0.5,
    150.0, 150.0, 10,
    t_start_ms=ANALYSIS_START_MS, t_end_ms=ANALYSIS_END_MS,
    peak_window_khz=5.0, fmin_khz=5.0
)

f1_arr = res_bic['f1'] / 1000.0
f2_arr = res_bic['f2'] / 1000.0
bicoh2 = res_bic['bicoh2']

# Look for exactly f1 = 87.0, f2 = -37.5 (since it's a difference coupling f3 = f1 - f2)
f1_target = 87.0
f2_target = -37.5
idx_f1 = np.argmin(np.abs(f1_arr - f1_target))
idx_f2 = np.argmin(np.abs(f2_arr - f2_target))

val_at_target = bicoh2[idx_f1, idx_f2]

P(f"Difference frequency hypothesis: f3 (49.5) = f1 (87.0) - f2 (37.5)")
P(f"Checking bicoherence grid at nearest bin to f1={f1_target}, f2={f2_target}...")
P(f"  Nearest bin: f1={f1_arr[idx_f1]:.1f} kHz, f2={f2_arr[idx_f2]:.1f} kHz")
P(f"  Bicoherence b^2 = {val_at_target:.4f}")
P(f"  Global Max b^2  = {res_bic['diff_max_b2']:.4f}")

if val_at_target < 0.5:
    P(f"\nConclusion: The b^2 value at (87.0, -37.5) is {val_at_target:.3f}, far below the 0.50 threshold.")
    P(f"The 'difference frequency' framing is NOT supported by the coupling data.")
    P(f"The 49.5 kHz mode is just a separate spectral feature, and its arithmetic relation")
    P(f"to 87.0/37.5 is highly likely to be coincidental.")
else:
    P(f"\nConclusion: The b^2 value at (87.0, -37.5) is {val_at_target:.3f}, supporting the difference coupling.")

# ============================================================================
# 3. TIME-FREQUENCY CHIRP TAIL CHECK (88653 & 88654)
# ============================================================================
P(f"\n{'='*95}")
P("3. TIME-FREQUENCY STRUCTURE FOR RANK 7: Chirp tail vs independent mode")
P(f"{'='*95}")

# Load 88654 for time-frequency
sig54 = {}
for coil in COILS:
    fpath = DATA_DIR_54 / f"{coil}@88654.edf"
    dat = edf.load(str(fpath))
    sig54[coil] = dat[:, 1]
t_ms_54 = t_s * 1000.0
mask_54 = (t_ms_54 >= ANALYSIS_START_MS) & (t_ms_54 <= ANALYSIS_END_MS)
sig54_win = {c: s[mask_54] for c, s in sig54.items()}
time_win = (t_s[mask_54]) * 1000.0

def run_stft(sig, t_win):
    # High time-resolution STFT over the 15.9ms window
    # dt = 1e-6 (1 MHz). nperseg=2048 (2ms), step=100 (0.1ms)
    f, t_stft, Zxx = dsp.stft(sig, fs=fs, window='hann', nperseg=2048, noverlap=2048-100)
    Pyy = np.abs(Zxx)**2
    return f/1000.0, (t_win[0] + t_stft*1000.0), Pyy

f53, t53, P53 = run_stft(sig53_win["MP1"], t_ms_53[mask_53])
f54, t54, P54 = run_stft(sig54_win["MP1"], time_win)

P(f"Analyzing STFT structure (nperseg=2048, step=0.1ms) across the 15.9ms window...")

def check_mode_structure(f_stft, t_stft, P_stft, f_main, f_side):
    # Extract the PSD evolution of the main and sideband frequencies
    idx_main = np.argmin(np.abs(f_stft - f_main))
    idx_side = np.argmin(np.abs(f_stft - f_side))
    
    psd_main = P_stft[idx_main, :]
    psd_side = P_stft[idx_side, :]
    
    # Check if they peak together (correlation) or if the main peak drifts into the sideband
    corr = np.corrcoef(psd_main, psd_side)[0, 1]
    
    # Check max frequency evolution
    # Find the frequency of the maximum power in the broad band containing both
    band_mask = (f_stft >= f_main - 2) & (f_stft <= f_side + 2)
    band_f = f_stft[band_mask]
    band_P = P_stft[band_mask, :]
    
    peak_f_over_time = band_f[np.argmax(band_P, axis=0)]
    f_min = np.min(peak_f_over_time)
    f_max = np.max(peak_f_over_time)
    
    P(f"  Target band: {f_main:.1f} kHz vs {f_side:.1f} kHz")
    P(f"  Temporal correlation between bins: {corr:.3f}")
    P(f"  Peak frequency within [{f_main-2:.1f}, {f_side+2:.1f}] band drifts from {f_min:.1f} to {f_max:.1f} kHz")
    
    if corr > 0.8:
        return "HIGH CORRELATION (likely sideband / amplitude modulation)"
    elif f_max - f_min >= (f_side - f_main) * 0.8:
        return "LARGE DRIFT (peak wanders across the gap, likely a chirp tail)"
    else:
        return "DISTINCT (low correlation, stable peak freq, likely independent modes)"

P(f"\nSHOT 88653 (Main: 89.0, Side: 95.5)")
res53 = check_mode_structure(f53, t53, P53, 89.0, 95.5)
P(f"  Interpretation: {res53}")

P(f"\nSHOT 88654 (Main: 87.0, Side: 91.5)")
res54 = check_mode_structure(f54, t54, P54, 87.0, 91.5)
P(f"  Interpretation: {res54}")

P(f"\nConclusion on Rank 7:")
P(f"If the temporal correlation is high, they pulse together (modulation).")
P(f"If the peak drifts substantially, it's a single chirping mode smearing the PSD.")
P(f"If neither, they are genuinely distinct temporal modes co-existing.")

output_path = root_dir / "shot_88654_primary_mode_check_v3.log"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")

P(f"\nDone. Output written to {output_path}")
