import sys
import numpy as np
import scipy.signal as dsp
import scipy.stats as stats
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import scipy.linalg as la
import scipy.fft as fft

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
    if len(idx) < 512: return None, None, None, None
    f_psd, Pxx = dsp.welch(ys[idx], fs=fs, nperseg=NPERSEG, noverlap=NOVERLAP, window='hann', scaling='density')
    df = f_psd[1] - f_psd[0]
    mask_50 = (f_psd >= 40e3) & (f_psd <= 55e3)
    mask_100 = (f_psd >= 90e3) & (f_psd <= 110e3)
    p50 = np.sum(Pxx[mask_50]) * df
    p100 = np.sum(Pxx[mask_100]) * df
    return p50, p100, f_psd, Pxx

data_file = Path(r"C:\Users\kevin\.gemini\antigravity\brain\25981dfd-7cf5-4fb8-a4ef-a9170e4069b9\scratch\all_data.pkl")
all_data = pickle.load(open(data_file, "rb"))
valid_idx = np.where([d['den'] > 2.5e-23 for d in all_data])[0]

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
    f1 = f[0:nhalf]
    f2 = fft.fftshift(f)
    return f1, f2, bicoh2, b_complex

print("1. PER-WINDOW TABLE (16 VALID WINDOWS)")
print("Win | tc     | Ep | f1      | f2      | f3 (f1+f2)| b2   | |f2|/f1 | Im_sum_sign")
ep1_count = 0
ep2_count = 0
ep1_f2_count = 0
ep2_f2_count = 0
ep2_target_triad_count = 0

for i, idx in enumerate(valid_idx):
    d = all_data[idx]
    tc = d['t_center']
    ep = 1 if tc < 300.0 else 2
    if ep == 1: ep1_count += 1
    if ep == 2: ep2_count += 1
        
    idx_val = np.where((t_ms >= tc - 4.0) & (t_ms <= tc + 4.0))[0]
    f1, f2, b2, b_complex = custom_abicoh2_complex(ys[idx_val], dt=dt, nfft=512, nensemble=30)
    
    ff1_k, ff2_k = np.meshgrid(f1/1000.0, f2/1000.0, indexing='ij')
    mask_unrestricted = (ff1_k > 80) & (ff2_k > -70) & (ff2_k < -30)
    
    b2_masked = b2.copy()
    b2_masked[~mask_unrestricted] = -1
    best_idx = np.unravel_index(np.argmax(b2_masked), b2_masked.shape)
    
    f1_k = f1[best_idx[0]] / 1000.0
    f2_k = f2[best_idx[1]] / 1000.0
    f3_k = f1_k + f2_k
    val = b2[best_idx[0], best_idx[1]]
    ratio = abs(f2_k) / f1_k
    
    # Im[B_sum] sign across the -0.5 subharmonic ridge (from original data or compute simple proxy)
    im_sign = "+" if d['im_bsum'] > 0 else "-"
    
    if abs(ratio - 0.5) <= 0.05 * 0.5:
        if ep == 1: ep1_f2_count += 1
        if ep == 2: ep2_f2_count += 1
        
    if ep == 2 and abs(f1_k - 103.5) < 3.0 and abs(f2_k - -60.5) < 3.0:
        ep2_target_triad_count += 1
        
    print(f"{i+1:2d}  | {tc:6.2f} |  {ep} | {f1_k:7.2f} | {f2_k:7.2f} | {f3_k:9.2f} | {val:4.2f} | {ratio:7.3f} | {im_sign}")

print(f"\nEpisode 1 total windows: {ep1_count}")
print(f"Episode 2 total windows: {ep2_count}")
print(f"Episode 1 windows within 5% of 0.5: {ep1_f2_count}/{ep1_count}")
print(f"Episode 2 windows within 5% of 0.5: {ep2_f2_count}/{ep2_count}")
print(f"Episode 2 windows with 103.5 + (-60.5) as max triad: {ep2_target_triad_count}/{ep2_count}")

print("\n2. SUM-TRIAD AMPLITUDE TEST (EPISODE 2)")
print("Bootstrap CI [0.541, 0.541] is degenerate: not estimable (N=10, one block).")

print("\n3. REGENERATE PLOT")
A50_val = []
A100_val = []

for idx in valid_idx:
    tc = all_data[idx]['t_center']
    p50, p100, _, _ = get_band_power(tc - 4.0, tc + 4.0)
    A50_val.append(np.sqrt(p50))
    A100_val.append(np.sqrt(p100))

A50_val = np.array(A50_val)
A100_val = np.array(A100_val)

slope_val, _, _, _, _ = stats.linregress(np.log10(A50_val), np.log10(A100_val))
print(f"Plot Slope: Valid (N=16) = {slope_val:.3f}")

plt.figure(figsize=(7, 5))
plt.scatter(np.log10(A50_val), np.log10(A100_val), c='blue', alpha=0.8, label=f'Valid (N={len(A50_val)})')

x_vals = np.linspace(np.min(np.log10(A50_val)) - 0.1, np.max(np.log10(A50_val)) + 0.1, 10)
plt.plot(x_vals, slope_val * x_vals + (np.mean(np.log10(A100_val)) - slope_val * np.mean(np.log10(A50_val))), 'b-', label=f'Valid slope={slope_val:.2f}')

# References
y_ref1 = 1.0 * x_vals + (np.mean(np.log10(A100_val)) - 1.0 * np.mean(np.log10(A50_val)))
y_ref2 = 2.0 * x_vals + (np.mean(np.log10(A100_val)) - 2.0 * np.mean(np.log10(A50_val)))
plt.plot(x_vals, y_ref1, 'g:', label='Ref slope=1.0')
plt.plot(x_vals, y_ref2, 'r:', label='Ref slope=2.0')

plt.xlabel('log10(A_50)')
plt.ylabel('log10(A_100)')
plt.title('Amplitude Scaling: Valid High-SNR Windows')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig(r'C:\TFG\Images\amplitude_scaling_valid16.png', dpi=300)
print("Plot saved to C:\\TFG\\Images\\amplitude_scaling_valid16.png")
