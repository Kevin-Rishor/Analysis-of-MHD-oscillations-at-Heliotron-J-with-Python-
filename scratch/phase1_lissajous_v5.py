import sys
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from pathlib import Path

# Add directories to path
sys.path.append(r"c:\TFG\jpack")
sys.path.append(r"c:\TFG\analysis")
sys.path.append(r"c:\TFG\analysis\common")

import turnelib as TE
from mhd_common import extract_instantaneous_frequency

# Configuration
SHOT = 88653
PROBE = "MP1"
EP1_WIN = (295.3, 297.1)
QUIET_WIN = (160.0, 161.8) # Same duration for surrogate

LF_BAND = (36e3, 44e3)
HF_BAND = (84e3, 94e3)
ORDER = 4

data_dir = Path(r"c:\TFG\data\hj88653")
out_dir = Path(r"c:\TFG\scratch")
log_file = out_dir / "lissajous_check_v5.log"

with open(log_file, "w") as f_log:
    f_log.write(f"--- Lissajous Check Phase 1 (V5 - Controls & Fixes) ---\n\n")

# Load data
fname = data_dir / f"{PROBE}@{SHOT}.edf"
edf = TE.edf()
dat = edf.load(str(fname))
t_full = dat[:, 0]

if edf.DimUnit[0] == 's':
    t_ms_full = t_full * 1000.0
else:
    t_ms_full = t_full

fs = 1.0 / (t_full[1] - t_full[0])
if edf.DimUnit[0] == 'ms':
    fs = fs * 1000.0

ys_full = dat[:, 1]

def get_padded_data(t_start, t_end, pad=5.0):
    mask = (t_ms_full >= t_start - pad) & (t_ms_full <= t_end + pad)
    return t_ms_full[mask], ys_full[mask], mask

# 1. Prepare base data
t_ep1_pad, ys_ep1_pad, _ = get_padded_data(*EP1_WIN)
t_quiet_pad, ys_quiet_pad, _ = get_padded_data(*QUIET_WIN)

# Phase randomization function
def phase_randomize(ys):
    Y = np.fft.rfft(ys)
    amps = np.abs(Y)
    phases = np.angle(Y)
    rand_phases = np.random.uniform(-np.pi, np.pi, len(Y))
    rand_phases[0] = 0 # keep DC real
    if len(ys) % 2 == 0:
        rand_phases[-1] = 0 # keep Nyquist real
    Y_rand = amps * np.exp(1j * rand_phases)
    return np.fft.irfft(Y_rand, n=len(ys))

ys_ep1_rand_pad = phase_randomize(ys_ep1_pad)

# 2. Extract components
def extract_components(t_pad, ys_pad, t_start, t_end):
    lf_env, lf_ph, lf_filt, _ = extract_instantaneous_frequency(ys_pad, fs, LF_BAND[0], LF_BAND[1], order=ORDER)
    hf_env, hf_ph, hf_filt, _ = extract_instantaneous_frequency(ys_pad, fs, HF_BAND[0], HF_BAND[1], order=ORDER)
    
    mask = (t_pad >= t_start) & (t_pad <= t_end)
    return {
        't': t_pad[mask],
        'lf_filt': lf_filt[mask], 'hf_filt': hf_filt[mask],
        'lf_env': lf_env[mask], 'hf_env': hf_env[mask],
        'lf_ph': lf_ph[mask], 'hf_ph': hf_ph[mask]
    }

data_real = extract_components(t_ep1_pad, ys_ep1_pad, *EP1_WIN)
data_quiet = extract_components(t_quiet_pad, ys_quiet_pad, *QUIET_WIN)
data_rand = extract_components(t_ep1_pad, ys_ep1_rand_pad, *EP1_WIN)

# Build surrogates dict
datasets = {
    "Real_Ep1": data_real,
    "Surrogate_A_QuietHF": {
        't': data_real['t'],
        'lf_filt': data_real['lf_filt'], 'lf_env': data_real['lf_env'], 'lf_ph': data_real['lf_ph'],
        'hf_filt': data_quiet['hf_filt'], 'hf_env': data_quiet['hf_env'], 'hf_ph': data_quiet['hf_ph']
    },
    "Surrogate_B_RandHF": {
        't': data_real['t'],
        'lf_filt': data_real['lf_filt'], 'lf_env': data_real['lf_env'], 'lf_ph': data_real['lf_ph'],
        'hf_filt': data_rand['hf_filt'], 'hf_env': data_rand['hf_env'], 'hf_ph': data_rand['hf_ph']
    }
}

# Real beat period for quarter beat plots
dphi_cycles_real = data_real['hf_ph'] - 2.0 * data_real['lf_ph']
slope_real, _ = np.polyfit(data_real['t'] / 1000.0, dphi_cycles_real, 1)
beat_freq_real = abs(slope_real)
beat_period_ms_real = (1.0 / beat_freq_real) * 1000.0
quarter_beat_ms = beat_period_ms_real * 0.25

with open(log_file, "a") as f:
    f.write(f"Real Ep1 Beat Period used for slices: {beat_period_ms_real:.4f} ms (1/4 = {quarter_beat_ms:.4f} ms)\n\n")

def find_contiguous(condition):
    idx = np.flatnonzero(condition)
    if len(idx) == 0: return []
    return np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)

# Task 1 & 3: Plotting for all datasets
for name, d in datasets.items():
    # A) Quarter beat period raw plot
    t_start = EP1_WIN[0]
    s_mask = (d['t'] >= t_start) & (d['t'] <= t_start + quarter_beat_ms)
    
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(d['lf_filt'][s_mask], d['hf_filt'][s_mask], 'b-')
    ax.set_title(f'1/4 Beat Raw: {name}')
    ax.set_xlabel('LF Amplitude')
    ax.set_ylabel('HF Amplitude')
    plt.savefig(out_dir / f"v5_quarterbeat_{name}.png", dpi=200, bbox_inches='tight')
    plt.close(fig)

    # B) Normalized Phase-Binned Overlay
    # Normalize:
    lf_norm = d['lf_filt'] / np.clip(d['lf_env'], 1e-12, None)
    hf_norm = d['hf_filt'] / np.clip(d['hf_env'], 1e-12, None)
    
    dphi_cycles = d['hf_ph'] - 2.0 * d['lf_ph']
    dphi_rad_wrap0 = (dphi_cycles * 2 * np.pi) % (2 * np.pi)
    
    num_bins = 8
    bin_edges = np.linspace(0, 2 * np.pi, num_bins + 1)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_bins))
    
    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(num_bins):
        c = (dphi_rad_wrap0 >= bin_edges[i]) & (dphi_rad_wrap0 < bin_edges[i+1])
        blocks = find_contiguous(c)
        label_added = False
        for blk in blocks:
            if len(blk) > 1:
                ax.plot(lf_norm[blk], hf_norm[blk], color=colors[i], alpha=0.5, linewidth=1.0, 
                        label=f"Bin {i+1}" if not label_added else "")
                label_added = True
    
    ax.set_title(f'Normalized Phase-Binned: {name}')
    ax.set_xlabel('LF(t) / A_LF(t)')
    ax.set_ylabel('HF(t) / A_HF(t)')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True)
    if name == "Real_Ep1":
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(out_dir / f"v5_norm_phasebinned_{name}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

# Task 2: Fix window-length problem. 
# 3 LF cycles = ~0.075 ms (assuming 40kHz -> 0.025ms per cycle)
win_len = 0.075
starts = [295.3, 295.5, 295.7, 295.9]

fig, ax = plt.subplots(figsize=(7, 7))
colors = ['red', 'green', 'blue', 'orange']

for i, start_ms in enumerate(starts):
    end_ms = start_ms + win_len
    mask = (data_real['t'] >= start_ms) & (data_real['t'] <= end_ms)
    
    lf = data_real['lf_filt'][mask]
    hf = data_real['hf_filt'][mask]
    
    ax.plot(lf, hf, color=colors[i], label=f'{start_ms:.1f}-{end_ms:.3f} ms', alpha=0.8, linewidth=1.5)

ax.set_title(f'Real Ep1: 4 Independent {win_len*1000:.0f} us windows (≥3 LF cycles)')
ax.set_xlabel('LF Amplitude (Narrow)')
ax.set_ylabel('HF Amplitude (Narrow)')
ax.legend()
ax.grid(True)
plt.savefig(out_dir / f"v5_independent_windows_Real_Ep1.png", dpi=300, bbox_inches='tight')
plt.close(fig)

with open(log_file, "a") as f:
    f.write("Completed V5 processing.\n")

print("Done phase1 v5 script.")
