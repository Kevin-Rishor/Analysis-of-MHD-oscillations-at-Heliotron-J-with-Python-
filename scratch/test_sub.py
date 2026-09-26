import numpy as np
import scipy.stats as stats
import sys, os

sys.path.insert(0, "c:/TFG/jpack")
import turnelib as TE
import scipy.signal as dsp

def load_channel(shot, channel_name):
    dat = TE.edf().load(f"c:/TFG/data/hj{shot}/{channel_name}@{shot}.edf")
    t_ms = dat[:, 0]
    if t_ms[-1] < 1000:
        pass
    else:
        t_ms = t_ms * 1000.0
    ys = dat[:, 1]
    dt_s = (t_ms[1] - t_ms[0]) / 1000.0
    fs = 1.0 / dt_s
    return t_ms, ys, dt_s, fs

t_ms, ys, dt, fs = load_channel(88653, "MP1")
mask_noise = (t_ms >= 250.0) & (t_ms <= 260.0)
y_noise = ys[mask_noise]
freqs_n, Pxx_noise_full = dsp.welch(y_noise, fs=fs, nperseg=1024, noverlap=512, window='hann', scaling='density')
df = freqs_n[1] - freqs_n[0]
win_len = 8.0
step = 0.25

all_data = []
for t_start in np.arange(291.3, 310.0, step):
    t_end = t_start + win_len
    if t_end > t_ms[-1]: break
    idx = np.where((t_ms >= t_start) & (t_ms <= t_end))[0]
    y_sub = ys[idx]
    if len(y_sub) < 512: continue
    
    f_psd, Pxx = dsp.welch(y_sub, fs=fs, nperseg=1024, noverlap=512, window='hann', scaling='density')
    
    mask_50 = (f_psd >= 40e3) & (f_psd <= 55e3)
    mask_100 = (f_psd >= 90e3) & (f_psd <= 110e3)
    
    A50_band = np.sqrt(np.sum(Pxx[mask_50]) * df)
    A100_band = np.sqrt(np.sum(Pxx[mask_100]) * df)
    
    Pxx_sub = np.maximum(Pxx - Pxx_noise_full, 1e-15)
    A50_sub = np.sqrt(np.sum(Pxx_sub[mask_50]) * df)
    A100_sub = np.sqrt(np.sum(Pxx_sub[mask_100]) * df)
    
    all_data.append({
        'A50_band': A50_band, 'A100_band': A100_band,
        'A50_sub': A50_sub, 'A100_sub': A100_sub
    })

log_A50 = np.log10([d['A50_band'] for d in all_data])
log_A100 = np.log10([d['A100_band'] for d in all_data])
log_A50_sub = np.log10([d['A50_sub'] for d in all_data])
log_A100_sub = np.log10([d['A100_sub'] for d in all_data])

slope = stats.linregress(log_A50, log_A100)[0]
slope_sub = stats.linregress(log_A50_sub, log_A100_sub)[0]
print(f"Slope: {slope:.6f}")
print(f"Slope Sub: {slope_sub:.6f}")
