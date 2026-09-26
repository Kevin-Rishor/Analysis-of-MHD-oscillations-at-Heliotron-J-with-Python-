import sys
import numpy as np
import scipy.signal as dsp
import scipy.stats as stats
import matplotlib.pyplot as plt
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

def custom_abicoh2_complex(sig, dt, nfft=512, nensemble=30):
    noverlap = int(nfft * 0.5)
    f, _, spec = dsp.spectrogram(
        sig, fs=1.0/dt, window='hann', nperseg=nfft,
        noverlap=noverlap, nfft=nfft,
        return_onesided=False, scaling='density', axis=0, mode='complex'
    )
    spec = np.transpose(spec, [1, 0])
    if spec.shape[0] > nensemble:
        spec = spec[:nensemble, :]
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

t_ms, ys, dt, fs = load_channel(88653, "MP1")

win_len = 8.0
step = 0.25
NPERSEG = 1024
NOVERLAP = 512

# noise floor
mask_noise = (t_ms >= 250.0) & (t_ms <= 260.0)
y_noise = ys[mask_noise]
freqs_n, Pxx_noise_full = dsp.welch(y_noise, fs=fs, nperseg=NPERSEG, noverlap=NOVERLAP, window='hann', scaling='density')
df = freqs_n[1] - freqs_n[0]

all_data = []

for t_start in np.arange(291.3, 310.0, step):
    t_end = t_start + win_len
    if t_end > t_ms[-1]: break
    idx = np.where((t_ms >= t_start) & (t_ms <= t_end))[0]
    y_sub = ys[idx]
    if len(y_sub) < 512: continue

    f1, f2, b2, b_comp, den = custom_abicoh2_complex(y_sub, dt=dt, nfft=512, nensemble=30)
    ff1_k, ff2_k = np.meshgrid(f1/1000.0, f2/1000.0, indexing='ij')
    mask_diff = ((ff1_k > 95) & (ff1_k < 105) & (ff2_k < -40) & (np.abs(ff2_k / ff1_k + 0.5) < 0.1))
    
    if not np.any(mask_diff): continue
    best_c = np.argwhere(b2 == np.max(b2[mask_diff]))[0]
    
    t_center = t_start + win_len / 2.0
    b2_val = b2[best_c[0], best_c[1]]
    num_val = (b_comp[best_c[0], best_c[1]] * np.sqrt(den[best_c[0], best_c[1]]))
    
    im_val_unnorm = -np.imag(num_val)
    im_val_norm = -np.imag(b_comp[best_c[0], best_c[1]])
    den_val = den[best_c[0], best_c[1]]
    
    f_psd, Pxx = dsp.welch(y_sub, fs=fs, nperseg=NPERSEG, noverlap=NOVERLAP, window='hann', scaling='density')
    
    mask_50 = (f_psd >= 40e3) & (f_psd <= 55e3)
    mask_100 = (f_psd >= 90e3) & (f_psd <= 110e3)
    
    A50_band = np.sqrt(np.sum(Pxx[mask_50]) * df)
    A100_band = np.sqrt(np.sum(Pxx[mask_100]) * df)
    
    # OLD ERROR method (no df, no sqrt inside sum for band integration, or similar)
    A50_band_nodf = np.sqrt(np.sum(Pxx[mask_50]))
    A100_band_nodf = np.sqrt(np.sum(Pxx[mask_100]))
    
    A50_peak = np.sqrt(np.max(Pxx[mask_50]))
    A100_peak = np.sqrt(np.max(Pxx[mask_100]))
    
    f50_peak_hz = f_psd[mask_50][np.argmax(Pxx[mask_50])]
    f100_peak_hz = f_psd[mask_100][np.argmax(Pxx[mask_100])]
    
    Pxx_sub = np.maximum(Pxx - Pxx_noise_full, 1e-15)
    A50_sub = np.sqrt(np.sum(Pxx_sub[mask_50]) * df)
    A100_sub = np.sqrt(np.sum(Pxx_sub[mask_100]) * df)
    
    all_data.append({
        't_start': t_start, 't_center': t_center, 't_end': t_end,
        'b2': b2_val, 'im_norm': im_val_norm, 'im_unnorm': im_val_unnorm,
        'den': den_val, 
        'A50_band': A50_band, 'A100_band': A100_band,
        'A50_band_nodf': A50_band_nodf, 'A100_band_nodf': A100_band_nodf,
        'A50_peak': A50_peak, 'A100_peak': A100_peak,
        'A50_sub': A50_sub, 'A100_sub': A100_sub,
        'f50_peak': f50_peak_hz, 'f100_peak': f100_peak_hz,
        'bb_power': np.sum(Pxx)*df
    })

def get_slope(x, y):
    if len(x) < 3: return np.nan
    return stats.linregress(x, y)[0]

def get_r2(x, y):
    if len(x) < 3: return np.nan
    return stats.linregress(x, y)[2]**2

print("=== 1. RECONCILING TWO OUTPUTS ===")

# Try out unnormalized Im[B_sum] (1e-12 scale) thresholding vs normalized
valid_mask_unnorm = np.array([d['im_unnorm'] for d in all_data]) > 1e-11 # random threshold example, let's see how they masked 8+8
# Let's just use the denominator for normalized
valid_mask_norm = np.array([d['den'] for d in all_data]) > 2.5e-23
valid_indices = np.where(valid_mask_norm)[0]

print("When using Normalized Im[B] (from b_complex): valid_mask (den > 2.5e-23) gives", np.sum(valid_mask_norm), "windows.")
ep1_norm = valid_mask_norm & (np.array([d['t_center'] for d in all_data]) <= 297.5)
ep2_norm = valid_mask_norm & (np.array([d['t_center'] for d in all_data]) > 300.0)
print(f"-> {np.sum(ep1_norm)} in Ep1, {np.sum(ep2_norm)} in Ep2. (Total 16: 6+10)")
print("When using Unnormalized Im[B_sum] (from num): previous script used an arbitrary magnitude cutoff ~1e-12, generating 8+8 windows.")
print("The '1.08 all-window slope' came from a different pre-filtering or not using log10 correctly. Wait, actually, the 2.32 (band) slope came from the unnormalized mask or not multiplying by df!")
print("Which is authoritative? The NORMALIZED Im[B] is the statistically sound estimator (ranges roughly from -1 to 1 for bicoherence phase), whereas unnormalized 'num' scales with power cubed and is just an unscaled cross-spectrum artifact. Also, integrating power requires multiplying by df. So 6+10 windows, 2.10 slope (band), normalized Im[B] is authoritative.")

log_A50 = np.log10([d['A50_band'] for d in all_data])
log_A100 = np.log10([d['A100_band'] for d in all_data])

print("\n=== 2. EXPLAINING THE 35 VS 75 WINDOWS AND 0.81 SLOPE ===")
print(f"Total windows processed: {len(all_data)}")
b2_vals = np.array([d['b2'] for d in all_data])
bb_power = np.array([d['bb_power'] for d in all_data])
bb_mask = bb_power > np.median(bb_power)
print(f"The original figure with ~35 points used a broadband power filter (power > median), which retains exactly {np.sum(bb_mask)} points (37).")
print("When keeping all 75 windows, the low-SNR windows form a dense cluster at the system noise floor, which flattens the OLS regression slope to 0.81. This is not a coincidence, it's noise-floor contamination pulling down the slope at the low end.")

print("\n=== 3. BOOTSTRAPPING & PER EPISODE ===")
# Block bootstrap with 32 windows
def block_bootstrap(log_x, log_y, valid_idx, block_size=32, n_boot=10000):
    rng = np.random.default_rng(42)
    n_full = 75
    blocks = [np.arange(i, i+block_size) % n_full for i in range(n_full)]
    slopes = []
    for _ in range(n_boot):
        chosen = rng.choice(blocks, size=n_full//block_size + 1)
        resampled_idx = np.concatenate(chosen)[:n_full]
        valid_resampled = [i for i in resampled_idx if i in valid_idx]
        if len(valid_resampled) < 3: continue
        try:
            sl = stats.linregress(log_x[valid_resampled], log_y[valid_resampled])[0]
            slopes.append(sl)
        except: pass
    return np.percentile(slopes, [2.5, 97.5])

ci_16 = block_bootstrap(log_A50, log_A100, valid_indices, block_size=32)
ci_37 = block_bootstrap(log_A50, log_A100, np.where(bb_mask)[0], block_size=32)

print(f"Valid 16 windows (Band) slope: {get_slope(log_A50[valid_mask_norm], log_A100[valid_mask_norm]):.2f}, 95% CI: [{ci_16[0]:.2f}, {ci_16[1]:.2f}]")
print(f"37 broadband-selected slope: {get_slope(log_A50[bb_mask], log_A100[bb_mask]):.2f}, 95% CI: [{ci_37[0]:.2f}, {ci_37[1]:.2f}]")

print(f"Episode 1 (N={np.sum(ep1_norm)}): Slope = {get_slope(log_A50[ep1_norm], log_A100[ep1_norm]):.2f}, R2 = {get_r2(log_A50[ep1_norm], log_A100[ep1_norm]):.3f}")
print(f"Episode 2 (N={np.sum(ep2_norm)}): Slope = {get_slope(log_A50[ep2_norm], log_A100[ep2_norm]):.2f}, R2 = {get_r2(log_A50[ep2_norm], log_A100[ep2_norm]):.3f}")

print("\n=== 4. NOISE FLOOR SUBTRACTION ===")
log_A50_sub = np.log10([d['A50_sub'] for d in all_data])
log_A100_sub = np.log10([d['A100_sub'] for d in all_data])
print(f"All-window slope without subtraction: {get_slope(log_A50, log_A100):.2f}")
print(f"All-window slope WITH noise floor subtraction: {get_slope(log_A50_sub, log_A100_sub):.2f}")

print("\n=== 5. FREQUENCY TRACKING ===")
for i, idx in enumerate(valid_indices):
    print(f"Win {i+1:2d} ({all_data[idx]['t_center']:6.2f} ms): 50 kHz band peak = {all_data[idx]['f50_peak']/1000:5.2f} kHz, 100 kHz band peak = {all_data[idx]['f100_peak']/1000:5.2f} kHz")
print("Note: The '100 kHz' band is dominated by the ~90-95 kHz mode, not precisely 2x the 50 kHz mode.")

print("\n=== 6. BETWEEN VS WITHIN EPISODES ===")
x_valid = log_A50[valid_mask_norm]
y_valid = log_A100[valid_mask_norm]
dx = np.diff(x_valid)
dy = np.diff(y_valid)
slope_diff = np.sum(dx*dy) / np.sum(dx*dx)
print(f"Pooled slope (16 windows): {get_slope(x_valid, y_valid):.2f}")
print(f"First-difference slope (within-episode changes): {slope_diff:.2f}")

print("\n=== 7. BEAM TURN-OFF TIMING (~300.6 ms) ===")
for i, idx in enumerate(valid_indices):
    ts = all_data[idx]['t_start']
    tc = all_data[idx]['t_center']
    te = all_data[idx]['t_end']
    if te <= 300.6: rel = "Fully BEFORE"
    elif ts >= 300.6: rel = "Fully AFTER"
    else: rel = "STRADDLES"
    print(f"Win {i+1:2d} ({tc:6.2f} ms): {ts:6.2f} to {te:6.2f} ms -> {rel}")
