import sys
import numpy as np
import scipy.signal as dsp
import scipy.fft as fft
import scipy.linalg as la
from pathlib import Path

root_dir = Path("c:/TFG")
sys.path.append(str(root_dir / "analysis" / "common"))
sys.path.append(str(root_dir / "analysis" / "heating_phase_comparison"))

from comparison_common import load_channel

def custom_abicoh2_complex(sig, dt, nfft=512, nensemble=30):
    noverlap = int(nfft * 0.5)
    f, _, spec = dsp.spectrogram(sig, fs=1.0 / dt, window='hann', nperseg=nfft, noverlap=noverlap, nfft=nfft,
                                 return_onesided=False, scaling='density', axis=0, mode='complex')
    spec = np.transpose(spec, [1, 0])
    if spec.shape[0] > nensemble:
        spec = spec[:nensemble, :]
    nf = f.size
    nhalf = nf // 2 if nf % 2 == 0 else (nf + 1) // 2
    fcol = np.arange(nf, dtype=int)
    lrow = np.roll(np.arange(nhalf, dtype=int), 1)
    ha = la.hankel(fcol, lrow)

    num = np.mean(spec[:, fcol, None] * spec[:, None, lrow] * np.conjugate(spec[:, ha]), axis=0)
    denum = np.mean(np.abs(spec[:, fcol, None] * spec[:, None, lrow]) ** 2, axis=0) * np.mean(np.abs(np.conjugate(spec[:, ha])) ** 2, axis=0)
    bicoh2 = np.abs(num) ** 2 / (denum + 1e-30)

    bicoh2 = np.transpose(fft.fftshift(bicoh2, axes=0), [1, 0])
    f1 = f[0:nhalf]
    f2 = fft.fftshift(f)
    return f1, f2, bicoh2

def run():
    shot = 88653
    t_ms, ys, dt, fs = load_channel(shot, "MP1")

    win_len = 8.0
    step = 0.25
    
    t_start_global = 259.1
    t_end_global = 300.6
    
    triad_peaks = []
    
    # 1. Sliding window
    b2_sliding_for_fixed = []
    
    for t_s in np.arange(t_start_global, t_end_global - win_len + 1e-5, step):
        t_e = t_s + win_len
        idx = np.where((t_ms >= t_s) & (t_ms <= t_e))[0]
        y_sub = ys[idx]
        if len(y_sub) < 512:
            continue
            
        f1, f2, b2 = custom_abicoh2_complex(y_sub, dt=dt, nfft=512, nensemble=30)
        ff1_k, ff2_k = np.meshgrid(f1 / 1000.0, f2 / 1000.0, indexing='ij')
        
        mask = (ff1_k > 95) & (ff1_k < 105) & (ff2_k < -40) & (np.abs(ff2_k / ff1_k + 0.5) < 0.1)
        if np.any(mask):
            best_c = np.argwhere(b2 == np.max(b2[mask]))[0]
            triad_peaks.append((best_c[0], best_c[1]))
            
    # Find most common triad
    from collections import Counter
    c = Counter(triad_peaks)
    fixed_triad = c.most_common(1)[0][0]
    
    # How many within 2 bins?
    count_within_2 = 0
    for p in triad_peaks:
        if abs(p[0] - fixed_triad[0]) <= 2 and abs(p[1] - fixed_triad[1]) <= 2:
            count_within_2 += 1
            
    frac = count_within_2 / len(triad_peaks)
    print(f"Fraction within 2 bins of max triad {fixed_triad}: {frac:.2%} ({count_within_2}/{len(triad_peaks)})")
    
    # Mean b2 in 8 ms windows for the fixed triad
    b2_windows = []
    for t_s in np.arange(t_start_global, t_end_global - win_len + 1e-5, step):
        t_e = t_s + win_len
        idx = np.where((t_ms >= t_s) & (t_ms <= t_e))[0]
        y_sub = ys[idx]
        if len(y_sub) < 512: continue
        f1, f2, b2 = custom_abicoh2_complex(y_sub, dt=dt, nfft=512, nensemble=30)
        b2_windows.append(b2[fixed_triad[0], fixed_triad[1]])
        
    mean_b2_8ms = np.mean(b2_windows)
    
    # 2. Long window (whole 259.1-300.6 ms)
    idx_long = np.where((t_ms >= t_start_global) & (t_ms <= t_end_global))[0]
    y_long = ys[idx_long]
    n_seg = len(y_long) // 256 - 1
    f1_l, f2_l, b2_l = custom_abicoh2_complex(y_long, dt=dt, nfft=512, nensemble=100000) # all segments
    long_b2 = b2_l[fixed_triad[0], fixed_triad[1]]
    
    print(f"Mean b2 of fixed triad in 8ms windows: {mean_b2_8ms:.4f}")
    print(f"Long-window b2 of fixed triad: {long_b2:.4f}")
    
    # Explanation
    if frac > 0.9 and long_b2 < mean_b2_8ms:
        print("Drop is NOT explained by triad hopping (since it stays mostly fixed). It is likely due to random phase slips.")
    else:
        print("Drop IS explained by triad hopping (frequency moving around).")

if __name__ == "__main__":
    run()
