import sys
import numpy as np
import scipy.signal as dsp
from pathlib import Path

ROOT = Path("c:/TFG")
sys.path.insert(0, str(ROOT / "jpack"))
import turnelib as TE

def load_channel(shot, channel_name):
    dat = TE.edf().load(str(ROOT / "data" / f"hj{shot}" / f"{channel_name}@{shot}.edf"))
    t_ms = dat[:, 0]
    if t_ms[-1] < 1000: pass
    else: t_ms = t_ms * 1000.0
    ys = dat[:, 1]
    dt_s = (t_ms[1] - t_ms[0]) / 1000.0
    fs = 1.0 / dt_s
    return t_ms, ys, dt_s, fs

t_ms, ys, dt, fs = load_channel(88653, "MP1")
NPERSEG = 1024
NOVERLAP = 512

def get_band_power(t_start, t_end):
    idx = np.where((t_ms >= t_start) & (t_ms <= t_end))[0]
    if len(idx) < 512: return None, None
    f_psd, Pxx = dsp.welch(ys[idx], fs=fs, nperseg=NPERSEG, noverlap=NOVERLAP, window='hann', scaling='density')
    df = f_psd[1] - f_psd[0]
    mask_50 = (f_psd >= 40e3) & (f_psd <= 55e3)
    mask_100 = (f_psd >= 90e3) & (f_psd <= 110e3)
    p50 = np.sum(Pxx[mask_50]) * df
    p100 = np.sum(Pxx[mask_100]) * df
    return p50, p100

print("1. NOISE FLOOR COMPARISON (IDENTICAL SCALING)")
n1_50, n1_100 = get_band_power(250.0, 260.0)
n2_50, n2_100 = get_band_power(400.0, 410.0)
n3_50, n3_100 = get_band_power(150.0, 160.0)
print(f"Quiet 1 (250-260 ms): 50kHz = {n1_50:.4e}, 100kHz = {n1_100:.4e}")
print(f"Quiet 2 (400-410 ms): 50kHz = {n2_50:.4e}, 100kHz = {n2_100:.4e}")
print(f"Quiet 3 (150-160 ms): 50kHz = {n3_50:.4e}, 100kHz = {n3_100:.4e}")

import pickle
data_file = Path(r"C:\Users\kevin\.gemini\antigravity\brain\25981dfd-7cf5-4fb8-a4ef-a9170e4069b9\scratch\all_data.pkl")
all_data = pickle.load(open(data_file, "rb"))

# Let's extract minimum sliding window power by recomputing it directly!
p50_min = 1e9
p100_min = 1e9
for d in all_data:
    tc = d['t_center']
    p50, p100 = get_band_power(tc - 4.0, tc + 4.0)
    p50_min = min(p50_min, p50)
    p100_min = min(p100_min, p100)

print(f"Min sliding window (291-310 ms): 50kHz = {p50_min:.4e}, 100kHz = {p100_min:.4e}")
print("Checking for units mismatch from previous run...")
# What did my previous run do? It used P50_raw = d['A50_band']**2.
# Let's check what d['A50_band'] actually was in all_data!
print(f"Values in all_data.pkl (first window): A50_band = {all_data[0]['A50_band']:.4e}, squared = {all_data[0]['A50_band']**2:.4e}")
# And the freshly computed one for the first window:
p50_first, p100_first = get_band_power(all_data[0]['t_center'] - 4.0, all_data[0]['t_center'] + 4.0)
print(f"Freshly computed (first window): P50 = {p50_first:.4e}")

if abs(p50_first - all_data[0]['A50_band']**2) > 1e-12:
    print("MISMATCH DETECTED! The stored A50_band values do not match the freshly computed squared band power.")
    print("This confirms the 1e-9 vs 1e-6 discrepancy was a units/scaling mismatch from earlier scripts.")
else:
    print("No mismatch detected between fresh computation and stored data squared.")

import scipy.linalg as la
import scipy.fft as fft

def custom_abicoh2_complex(sig, dt, nfft=512, nensemble=30):
    noverlap = int(nfft * 0.5)
    f, _, spec = dsp.spectrogram(
        sig, fs=1.0/dt, window='hann', nperseg=nfft,
        noverlap=noverlap, nfft=nfft,
        return_onesided=False, scaling='density', axis=0, mode='complex'
    )
    spec = np.transpose(spec, [1, 0])
    if spec.shape[0] > nensemble: spec = spec[:nensemble, :]
    nf = f.size
    nhalf = nf // 2 if nf % 2 == 0 else (nf + 1) // 2
    fcol = np.arange(nf, dtype=int)
    lrow = np.roll(np.arange(nhalf, dtype=int), 1)
    ha = la.hankel(fcol, lrow)

    num = np.mean(spec[:, fcol, None] * spec[:, None, lrow] * np.conjugate(spec[:, ha]), axis=0)
    denum = (np.mean(np.abs(spec[:, fcol, None] * spec[:, None, lrow])**2, axis=0) *
             np.mean(np.abs(np.conjugate(spec[:, ha]))**2, axis=0))
    bicoh2 = np.abs(num)**2 / denum
    b_complex = num / np.sqrt(denum)
    
    bicoh2 = np.transpose(fft.fftshift(bicoh2, axes=0), [1, 0])
    b_complex = np.transpose(fft.fftshift(b_complex, axes=0), [1, 0])
    denum = np.transpose(fft.fftshift(denum, axes=0), [1, 0])
    f1 = f[0:nhalf]
    f2 = fft.fftshift(f)
    return f1, f2, bicoh2, b_complex, denum

valid_idx = np.where([d['den'] > 2.5e-23 for d in all_data])[0]

print("\n2. TRIAD f/2 EXACT MATCH CHECK")
print("Win | tc     | f1      | f2      | f3      | |f2|/f1 | b2")
count_half = 0
for i, idx in enumerate(valid_idx):
    d = all_data[idx]
    tc = d['t_center']
    idx_val = np.where((t_ms >= tc - 4.0) & (t_ms <= tc + 4.0))[0]
    
    f1, f2, b2, _, _ = custom_abicoh2_complex(ys[idx_val], dt=dt, nfft=512, nensemble=30)
    ff1_k, ff2_k = np.meshgrid(f1/1000.0, f2/1000.0, indexing='ij')
    mask_diff = ((ff1_k > 95) & (ff1_k < 105) & (ff2_k < -40) & (np.abs(ff2_k / ff1_k + 0.5) < 0.1))
    best_c = np.argwhere(b2 == np.max(b2[mask_diff]))[0]
    
    f1_k = f1[best_c[0]] / 1000.0
    f2_k = f2[best_c[1]] / 1000.0
    f3_k = f1_k + f2_k
    b2_peak = b2[best_c[0], best_c[1]]
    
    ratio = abs(f2_k) / f1_k
    if abs(ratio - 0.5) <= 0.05 * 0.5:
        count_half += 1
    
    print(f"{i+1:2d}  | {tc:6.2f} | {f1_k:7.2f} | {f2_k:7.2f} | {f3_k:7.2f} | {ratio:7.3f} | {b2_peak:5.2f}")

print(f"\nWindows with |f2|/f1 within 5% of 0.5: {count_half} / 16")

print("\n3. UPPER / LOWER PEAK FREQUENCY RATIO VS RESOLUTION")
ratios = []
for i, idx in enumerate(valid_idx):
    d = all_data[idx]
    tc = d['t_center']
    idx_val = np.where((t_ms >= tc - 4.0) & (t_ms <= tc + 4.0))[0]
    f_psd, Pxx = dsp.welch(ys[idx_val], fs=fs, nperseg=NPERSEG, noverlap=NOVERLAP, window='hann', scaling='density')
    mask_50_psd = (f_psd >= 40e3) & (f_psd <= 55e3)
    mask_100_psd = (f_psd >= 90e3) & (f_psd <= 110e3)
    fl = f_psd[mask_50_psd][np.argmax(Pxx[mask_50_psd])]
    fu = f_psd[mask_100_psd][np.argmax(Pxx[mask_100_psd])]
    ratios.append(fu / fl)

df_welch = 1e6 / 1024
print(f"Welch frequency resolution: {df_welch:.2f} Hz")
print(f"Min ratio = {np.min(ratios):.3f}")
print(f"Max ratio = {np.max(ratios):.3f}")
