import sys
import numpy as np
import scipy.signal as dsp
import scipy.fft as fft
import scipy.linalg as la
from pathlib import Path
from scipy.stats import circstd

root_dir = Path("c:/TFG")
sys.path.append(str(root_dir / "analysis" / "common"))
sys.path.append(str(root_dir / "analysis" / "heating_phase_comparison"))

from comparison_common import load_channel

def custom_bispec_complex(sig, dt, nfft=512, nensemble=30):
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
    num_shift = np.transpose(fft.fftshift(num, axes=0), [1, 0])
    
    f1 = f[0:nhalf]
    f2 = fft.fftshift(f)
    return f1, f2, bicoh2, np.angle(num_shift)

def run():
    shot = 88653
    t_ms, ys, dt, fs = load_channel(shot, "MP1")

    win_len = 8.0
    step = 0.25 # Use 0.25 ms steps as original script
    
    t_start_global = 259.1
    t_end_global = 300.6
    
    print("=== Task 3: Phase-mixing ===")
    
    # 1. First pass: find max triad in each window
    triad_peaks = []
    times = []
    
    for t_s in np.arange(t_start_global, t_end_global - win_len + 1e-5, step):
        t_e = t_s + win_len
        idx = np.where((t_ms >= t_s) & (t_ms <= t_e))[0]
        y_sub = ys[idx]
        if len(y_sub) < 512:
            continue
            
        f1, f2, b2, phase = custom_bispec_complex(y_sub, dt=dt, nfft=512, nensemble=30)
        ff1_k, ff2_k = np.meshgrid(f1 / 1000.0, f2 / 1000.0, indexing='ij')
        
        # Look for modes near 100kHz and -50kHz
        with np.errstate(divide='ignore', invalid='ignore'):
            mask = (ff1_k > 80) & (ff1_k < 110) & (ff2_k < -30) & (np.abs(ff2_k / ff1_k + 0.5) < 0.2)
        
        if np.any(mask):
            best_c = np.argwhere(b2 == np.max(b2[mask]))[0]
            triad_peaks.append((best_c[0], best_c[1]))
            times.append((t_s, t_e))
        else:
            best_c = np.unravel_index(np.argmax(b2), b2.shape)
            triad_peaks.append((best_c[0], best_c[1]))
            times.append((t_s, t_e))
            
    from collections import Counter
    c = Counter(triad_peaks)
    fixed_triad = c.most_common(1)[0][0]
    
    idx_f1, idx_f2 = fixed_triad
    f1_val = f1[idx_f1] / 1000.0
    f2_val = f2[idx_f2] / 1000.0
    f3_val = f1_val + f2_val
    print(f"Most common triad: ({f1_val:.2f}, {f2_val:.2f}, {f3_val:.2f}) kHz")
    
    strongest_phases = []
    
    for i, max_t in enumerate(triad_peaks):
        t_s, t_e = times[i]
        
        idx = np.where((t_ms >= t_s) & (t_ms <= t_e))[0]
        y_sub = ys[idx]
        f1, f2, b2, phase = custom_bispec_complex(y_sub, dt=dt, nfft=512, nensemble=30)
        
        b2_fixed = b2[fixed_triad[0], fixed_triad[1]]
        phase_fixed = phase[fixed_triad[0], fixed_triad[1]]
        
        dist = max(abs(max_t[0] - fixed_triad[0]), abs(max_t[1] - fixed_triad[1]))
        
        if dist <= 2:
            print(f"Window {t_s:.2f}-{t_e:.2f}ms: b2 = {b2_fixed:.4f}, phase = {np.degrees(phase_fixed):+7.2f} deg")
        
        if dist == 0:
            strongest_phases.append(phase_fixed)
            
    circ_sd = circstd(strongest_phases)
    print(f"\nCircular std of biphase when strongest triad: {np.degrees(circ_sd):.2f} deg")
    
    if np.degrees(circ_sd) > 45:
        print("Biphase drifts (phase slips).")
    else:
        print("Biphase is stable (hopping explains the loss).")
        
if __name__ == "__main__":
    run()
