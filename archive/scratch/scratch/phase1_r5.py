import sys
import numpy as np
import scipy.signal as dsp
from pathlib import Path
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

def scan_interval(t_start_interval, t_end_interval):
    tcs = np.arange(t_start_interval + 4.0, t_end_interval - 4.0 + 0.25, 0.25)
    max_b2s = []
    im_bsum_signs = []
    
    for tc in tcs:
        idx_val = np.where((t_ms >= tc - 4.0) & (t_ms <= tc + 4.0))[0]
        if len(idx_val) < 512: continue
            
        f1, f2, b2, b_comp = custom_abicoh2_complex(ys[idx_val], dt=dt, nfft=512, nensemble=30)
        ff1_k, ff2_k = np.meshgrid(f1/1000.0, f2/1000.0, indexing='ij')
        
        # Unrestricted mask for max b2
        mask_unrestricted = (ff1_k > 80) & (ff2_k > -70) & (ff2_k < -30)
        b2_masked = b2.copy()
        b2_masked[~mask_unrestricted] = -1
        max_b2s.append(np.max(b2_masked))
        
        # Original mask for Im[B_sum] (from verify script)
        mask_im = ((ff1_k > 95) & (ff1_k < 105) & (ff2_k < -40) & (np.abs(ff2_k / ff1_k + 0.5) < 0.1))
        if np.any(mask_im):
            im_val = np.sum(np.imag(b_comp[mask_im]))
            im_bsum_signs.append(im_val > 0)
            
    return max_b2s, im_bsum_signs

print("Running empirical null on 150-160 ms...")
b2_1, signs_1 = scan_interval(150.0, 160.0)
print("Running empirical null on 400-410 ms...")
b2_2, signs_2 = scan_interval(400.0, 410.0)

all_b2 = np.array(b2_1 + b2_2)
all_signs = np.array(signs_1 + signs_2)

print("\n--- NULL DISTRIBUTION ---")
print(f"Total windows evaluated: {len(all_b2)}")
print(f"b2 Min: {np.min(all_b2):.3f}")
print(f"b2 Median: {np.median(all_b2):.3f}")
print(f"b2 Max: {np.max(all_b2):.3f}")
print(f"b2 95th percentile (empirical FWER): {np.percentile(all_b2, 95):.3f}")
print(f"Im[B_sum] > 0 fraction: {np.sum(all_signs)}/{len(all_signs)} ({np.mean(all_signs)*100:.1f}%)")

valid_16_b2 = [0.88, 0.82, 0.68, 0.71, 0.80, 0.92, 0.95, 0.96, 0.97, 0.97, 0.98, 0.98, 0.98, 0.99, 0.99, 0.99]
cleared = sum([v > np.percentile(all_b2, 95) for v in valid_16_b2])
print(f"\n16 Valid windows clearing 95th percentile FWER: {cleared}/16")

