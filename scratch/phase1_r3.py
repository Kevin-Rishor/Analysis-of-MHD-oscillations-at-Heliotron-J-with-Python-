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
    
    bicoh2 = np.transpose(fft.fftshift(bicoh2, axes=0), [1, 0])
    f1 = f[0:nhalf]
    f2 = fft.fftshift(f)
    return f1, f2, bicoh2

print("1. UNRESTRICTED TRIAD SEARCH")
print("Win | tc     | Rank | f1      | f2      | f3 (f1+f2) | |f2|/f1 | b2")
for i, idx in enumerate(valid_idx):
    d = all_data[idx]
    tc = d['t_center']
    idx_val = np.where((t_ms >= tc - 4.0) & (t_ms <= tc + 4.0))[0]
    f1, f2, b2 = custom_abicoh2_complex(ys[idx_val], dt=dt, nfft=512, nensemble=30)
    
    ff1_k, ff2_k = np.meshgrid(f1/1000.0, f2/1000.0, indexing='ij')
    mask_unrestricted = (ff1_k > 80) & (ff2_k > -70) & (ff2_k < -30)
    
    b2_masked = b2.copy()
    b2_masked[~mask_unrestricted] = -1
    
    flat_indices = np.argsort(b2_masked.flatten())[::-1]
    
    for rank in range(3):
        best_idx = np.unravel_index(flat_indices[rank], b2_masked.shape)
        f1_k = f1[best_idx[0]] / 1000.0
        f2_k = f2[best_idx[1]] / 1000.0
        f3_k = f1_k + f2_k
        val = b2[best_idx[0], best_idx[1]]
        ratio = abs(f2_k) / f1_k
        print(f"{i+1:2d}  | {tc:6.2f} | {rank+1:4d} | {f1_k:7.2f} | {f2_k:7.2f} | {f3_k:10.2f} | {ratio:7.3f} | {val:5.2f}")

print("\n2. SUM-TRIAD AMPLITUDE TEST (EPISODE 2)")
ep2_idx = [idx for idx in valid_idx if all_data[idx]['t_center'] > 300.0]
df_welch = 1e6 / 1024
A_103 = []
A_43 = []
A_60 = []

for idx in ep2_idx:
    tc = all_data[idx]['t_center']
    _, _, f_psd, Pxx = get_band_power(tc - 4.0, tc + 4.0)
    # Find closest bins
    i103 = np.argmin(np.abs(f_psd - 103.5e3))
    i43 = np.argmin(np.abs(f_psd - 43.0e3))
    i60 = np.argmin(np.abs(f_psd - 60.5e3))
    
    A_103.append(np.sqrt(Pxx[i103] * df_welch))
    A_43.append(np.sqrt(Pxx[i43] * df_welch))
    A_60.append(np.sqrt(Pxx[i60] * df_welch))

A_103 = np.array(A_103)
A_43 = np.array(A_43)
A_60 = np.array(A_60)
X = np.log10(A_43) + np.log10(A_60)
Y = np.log10(A_103)

slope, intercept, r_val, p_val, std_err = stats.linregress(X, Y)

def block_boot(x, y, block_size, n_boot=10000):
    n = len(x)
    rng = np.random.default_rng(42)
    slopes = []
    blocks = [np.arange(i, i+block_size) % n for i in range(n)]
    for _ in range(n_boot):
        chosen = rng.choice(blocks, size=max(1, n//block_size + 1))
        res = np.concatenate(chosen)[:n]
        try:
            slopes.append(stats.linregress(x[res], y[res])[0])
        except: pass
    return np.percentile(slopes, [2.5, 97.5])

ci = block_boot(X, Y, block_size=10)
print(f"Regression log10(A_103.5) ~ log10(A_43.0) + log10(A_60.5) for N=10 (Episode 2)")
print(f"Slope = {slope:.3f}, R2 = {r_val**2:.3f}, CI (block=10) = [{ci[0]:.3f}, {ci[1]:.3f}]")

print("\n3. REGENERATE PLOT")
A50_all = []
A100_all = []
is_valid = []

for d in all_data:
    tc = d['t_center']
    p50, p100, _, _ = get_band_power(tc - 4.0, tc + 4.0)
    A50_all.append(np.sqrt(p50))
    A100_all.append(np.sqrt(p100))
    is_valid.append(d['den'] > 2.5e-23)

A50_all = np.array(A50_all)
A100_all = np.array(A100_all)
is_valid = np.array(is_valid)

p50_q1, p100_q1, _, _ = get_band_power(150.0, 160.0)
p50_q2, p100_q2, _, _ = get_band_power(400.0, 410.0)
Aq1 = (np.sqrt(p50_q1), np.sqrt(p100_q1))
Aq2 = (np.sqrt(p50_q2), np.sqrt(p100_q2))

slope_all, _, _, _, _ = stats.linregress(np.log10(A50_all), np.log10(A100_all))
slope_val, _, _, _, _ = stats.linregress(np.log10(A50_all[is_valid]), np.log10(A100_all[is_valid]))

print(f"Plot Slopes: All (N=75) = {slope_all:.3f}, Valid (N=16) = {slope_val:.3f}")

plt.figure(figsize=(7, 5))
plt.scatter(np.log10(A50_all[~is_valid]), np.log10(A100_all[~is_valid]), c='grey', alpha=0.5, label=f'Low-SNR (N={np.sum(~is_valid)})')
plt.scatter(np.log10(A50_all[is_valid]), np.log10(A100_all[is_valid]), c='blue', alpha=0.8, label=f'Valid (N={np.sum(is_valid)})')
plt.scatter(np.log10(Aq1[0]), np.log10(Aq1[1]), c='red', marker='*', s=150, label='Quiet 1 (150-160 ms)')
plt.scatter(np.log10(Aq2[0]), np.log10(Aq2[1]), c='orange', marker='*', s=150, label='Quiet 2 (400-410 ms)')

x_vals = np.linspace(np.min(np.log10(A50_all)), np.max(np.log10(A50_all)), 10)
plt.plot(x_vals, slope_all * x_vals + (np.mean(np.log10(A100_all)) - slope_all * np.mean(np.log10(A50_all))), 'k--', label=f'All slope={slope_all:.2f}')
plt.plot(x_vals, slope_val * x_vals + (np.mean(np.log10(A100_all[is_valid])) - slope_val * np.mean(np.log10(A50_all[is_valid]))), 'b-', label=f'Valid slope={slope_val:.2f}')

# References
y_ref1 = 1.0 * x_vals + (np.mean(np.log10(A100_all[is_valid])) - 1.0 * np.mean(np.log10(A50_all[is_valid])))
y_ref2 = 2.0 * x_vals + (np.mean(np.log10(A100_all[is_valid])) - 2.0 * np.mean(np.log10(A50_all[is_valid])))
plt.plot(x_vals, y_ref1, 'g:', label='Ref slope=1.0')
plt.plot(x_vals, y_ref2, 'r:', label='Ref slope=2.0')

plt.xlabel('log10(A_50)')
plt.ylabel('log10(A_100)')
plt.title('Amplitude Scaling: All vs Valid Windows')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig(r'C:\TFG\Images\amplitude_scaling_all_vs_valid.png', dpi=300)
print("Plot saved to C:\\TFG\\Images\\amplitude_scaling_all_vs_valid.png")
