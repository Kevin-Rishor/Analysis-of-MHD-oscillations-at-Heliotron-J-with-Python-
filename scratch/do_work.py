import sys
from pathlib import Path
import numpy as np
import scipy.signal as dsp

sys.path.append("c:/TFG/jpack")
import turnelib as TE

data_dir = Path("c:/TFG/data/hj88653")
edf = TE.edf()
shot = 88653

# Windows
t_active_start, t_active_end = 259.1, 275.0
t_quiet_start, t_quiet_end = 400.0, 410.0

results = []

for i in range(1, 17):
    f_ece = data_dir / f"ECE{i}FAST@{shot}.edf"
    if not f_ece.exists():
        continue
    dat = edf.load(str(f_ece))
    t_ms = dat[:, 0]
    if edf.DimUnit[0] != "ms":
        t_ms = t_ms * 1000.0
    ys = dat[:, 1]
    
    dt = (t_ms[1] - t_ms[0]) / 1000.0
    fs = 1.0 / dt
    
    # Filter 80-100 kHz using Bessel 4th order
    b_band, a_band = dsp.bessel(4, [80000.0 / (fs / 2.0), 100000.0 / (fs / 2.0)], btype="bandpass")
    
    y_fluc = dsp.filtfilt(b_band, a_band, ys)
    
    # Active window
    idx_active = np.where((t_ms >= t_active_start) & (t_ms <= t_active_end))[0]
    rms_active = np.std(y_fluc[idx_active])
    
    # Quiet window
    idx_quiet = np.where((t_ms >= t_quiet_start) & (t_ms <= t_quiet_end))[0]
    rms_quiet = np.std(y_fluc[idx_quiet])
    
    ratio = rms_active / rms_quiet if rms_quiet > 0 else np.nan
    results.append({"ch": i, "active": rms_active, "quiet": rms_quiet, "ratio": ratio})

with open("c:/TFG/scratch/out1.txt", "w") as f:
    f.write("1. ECE RMS\n")
    for r in results:
        f.write(f"ECE{r['ch']:02d}: Active={r['active']:.4f}, Quiet={r['quiet']:.4f}, Ratio={r['ratio']:.2f}\n")
    
    f.write("\nRatio > 10 (real):\n")
    for r in results:
        if r['ratio'] > 10:
            f.write(f"ECE{r['ch']:02d}\n")
            
    f.write("\nRatio < 3 (dead/outside):\n")
    for r in results:
        if r['ratio'] < 3:
            f.write(f"ECE{r['ch']:02d}\n")

# 2. Coherence with MP1 for ECE9/10/12/13
mp1_file = data_dir / f"MP1@{shot}.edf"
dat_mp1 = edf.load(str(mp1_file))
t_mp1 = dat_mp1[:, 0]
if edf.DimUnit[0] != "ms":
    t_mp1 = t_mp1 * 1000.0
dt = (t_mp1[1] - t_mp1[0]) / 1000.0
fs = 1.0 / dt
idx_active_mp1 = np.where((t_mp1 >= t_active_start) & (t_mp1 <= t_active_end))[0]
mp1_active = dat_mp1[:, 1][idx_active_mp1]

nperseg = 2048
noverlap = 1024
nd = len(mp1_active) / (nperseg - noverlap) - 1 # approximate dof, formula for welch
alpha = 0.05
floor = 1 - alpha ** (1 / (nd - 1)) if nd > 1 else np.nan

target_ece = [9, 10, 12, 13]
with open("c:/TFG/scratch/out2.txt", "w") as f:
    f.write(f"2. Coherence (nd={nd:.1f}, floor={floor:.4f})\n")
    ece_signals = {}
    for ch in target_ece:
        f_ece = data_dir / f"ECE{ch}FAST@{shot}.edf"
        dat = edf.load(str(f_ece))
        ece_active = dat[:, 1][idx_active_mp1]
        ece_signals[ch] = ece_active
        
        freqs, coh = dsp.coherence(mp1_active, ece_active, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
        idx_band = np.where((freqs >= 80000.0) & (freqs <= 100000.0))[0]
        # band integrated coherence
        avg_coh = np.mean(coh[idx_band])
        f.write(f"ECE{ch}-MP1 band-integrated coherence (80-100 kHz): {avg_coh:.4f}\n")
        
    freqs_1213, coh_1213 = dsp.coherence(ece_signals[12], ece_signals[13], fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
    idx_89 = np.argmin(np.abs(freqs_1213 - 89000.0))
    f.write(f"ECE12-ECE13 coherence at 89 kHz: {coh_1213[idx_89]:.4f}\n")

