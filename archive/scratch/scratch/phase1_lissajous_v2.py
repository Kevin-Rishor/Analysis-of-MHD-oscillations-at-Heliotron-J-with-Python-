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
WINDOWS = {
    "Global": (259.1, 275.0),
    "Episode_1": (295.3, 297.1),
    "Episode_2": (301.8, 304.6)
}

LF_BAND = (40e3, 80e3)
HF_BAND = (80e3, 120e3)
ORDER = 4

data_dir = Path(r"c:\TFG\data\hj88653")
out_dir = Path(r"c:\TFG\scratch")
log_file = out_dir / "lissajous_check_v2.log"

with open(log_file, "w") as f_log:
    f_log.write(f"--- Lissajous Check Phase 1 (V2) ---\n\n")

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

def weighted_mean(values, weights):
    return np.sum(values * weights) / np.sum(weights)

# 1 & 2: Instantaneous Freq and Phase Locking for MP1
for win_name, (t_start, t_end) in WINDOWS.items():
    pad_ms = 5.0
    mask = (t_ms_full >= t_start - pad_ms) & (t_ms_full <= t_end + pad_ms)
    t_win_pad = t_ms_full[mask]
    ys_win_pad = ys_full[mask]
    
    lf_env, lf_phase_cycles, lf_filt, lf_ifreq = extract_instantaneous_frequency(ys_win_pad, fs, LF_BAND[0], LF_BAND[1], order=ORDER)
    hf_env, hf_phase_cycles, hf_filt, hf_ifreq = extract_instantaneous_frequency(ys_win_pad, fs, HF_BAND[0], HF_BAND[1], order=ORDER)
    
    mask_exact = (t_win_pad >= t_start) & (t_win_pad <= t_end)
    t_exact = t_win_pad[mask_exact]
    
    lf_ifreq_exact = lf_ifreq[mask_exact]
    lf_env_exact = lf_env[mask_exact]
    lf_phase_exact = lf_phase_cycles[mask_exact]
    
    hf_ifreq_exact = hf_ifreq[mask_exact]
    hf_env_exact = hf_env[mask_exact]
    hf_phase_exact = hf_phase_cycles[mask_exact]
    
    # Simple means (for comparison)
    mean_lf_simple = np.mean(lf_ifreq_exact)
    mean_hf_simple = np.mean(hf_ifreq_exact)
    
    # Amplitude-weighted means
    mean_lf_weighted = weighted_mean(lf_ifreq_exact, lf_env_exact)
    mean_hf_weighted = weighted_mean(hf_ifreq_exact, hf_env_exact)
    
    # Thresholding for very low amplitude (e.g. < 10% of max envelope in window)
    env_threshold = 0.10 * np.max(lf_env_exact)
    low_amp_mask = lf_env_exact < env_threshold
    low_amp_frac = np.mean(low_amp_mask)
    
    valid_mask = ~low_amp_mask
    if np.sum(valid_mask) > 0:
        mean_lf_corrected = weighted_mean(lf_ifreq_exact[valid_mask], lf_env_exact[valid_mask])
        mean_hf_corrected = weighted_mean(hf_ifreq_exact[valid_mask], hf_env_exact[valid_mask])
        ratio_corrected = mean_hf_corrected / mean_lf_corrected
    else:
        mean_lf_corrected = mean_lf_weighted
        mean_hf_corrected = mean_hf_weighted
        ratio_corrected = mean_hf_weighted / mean_lf_weighted
        
    with open(log_file, "a") as f:
        f.write(f"--- Window: {win_name} ({t_start}-{t_end} ms) ---\n")
        f.write(f"Simple average LF freq: {mean_lf_simple/1000:.2f} kHz\n")
        f.write(f"Weighted average LF freq: {mean_lf_weighted/1000:.2f} kHz\n")
        f.write(f"Fraction of samples with LF envelope < 10% max: {low_amp_frac*100:.1f}%\n")
        f.write(f"Corrected (weighted & filtered) LF freq: {mean_lf_corrected/1000:.2f} kHz\n")
        f.write(f"Corrected HF freq: {mean_hf_corrected/1000:.2f} kHz\n")
        f.write(f"HF/LF ratio (corrected): {ratio_corrected:.3f}\n\n")

    # Phase difference Delta Phi = Phi_HF - 2 * Phi_LF
    # phase is in cycles. We convert to radians for standard phase analysis.
    dphi_cycles = hf_phase_exact - 2.0 * lf_phase_exact
    dphi_rad = (dphi_cycles * 2 * np.pi)
    # wrap to [-pi, pi]
    dphi_rad_wrapped = np.arctan2(np.sin(dphi_rad), np.cos(dphi_rad))
    
    plt.figure(figsize=(10, 4))
    plt.plot(t_exact, dphi_rad_wrapped, '.', markersize=1, alpha=0.5, color='purple')
    plt.title(f'Phase Difference $\\Delta\\phi = \\phi_{{HF}} - 2\\phi_{{LF}}$ (MP1, {win_name})')
    plt.xlabel('Time (ms)')
    plt.ylabel('$\\Delta\\phi$ (rad)')
    plt.ylim(-np.pi, np.pi)
    plt.grid(True)
    plt.savefig(out_dir / f"dphi_{win_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

# 3: Short sub-windows for Lissajous
short_win_len_ms = 0.5
short_episodes = ["Episode_1", "Episode_2"]
for ep_name in short_episodes:
    t_start, t_end = WINDOWS[ep_name]
    num_slices = int(np.floor((t_end - t_start) / short_win_len_ms))
    
    for i in range(num_slices):
        slice_start = t_start + i * short_win_len_ms
        slice_end = slice_start + short_win_len_ms
        
        pad_ms = 5.0
        mask = (t_ms_full >= slice_start - pad_ms) & (t_ms_full <= slice_end + pad_ms)
        t_win_pad = t_ms_full[mask]
        ys_win_pad = ys_full[mask]
        
        _, _, lf_filt, _ = extract_instantaneous_frequency(ys_win_pad, fs, LF_BAND[0], LF_BAND[1], order=ORDER)
        _, _, hf_filt, _ = extract_instantaneous_frequency(ys_win_pad, fs, HF_BAND[0], HF_BAND[1], order=ORDER)
        
        mask_exact = (t_win_pad >= slice_start) & (t_win_pad <= slice_end)
        t_exact = t_win_pad[mask_exact]
        lf_exact = lf_filt[mask_exact]
        hf_exact = hf_filt[mask_exact]
        
        fig, ax = plt.subplots(figsize=(6, 6))
        points = np.array([lf_exact, hf_exact]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(t_exact.min(), t_exact.max())
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(t_exact)
        lc.set_linewidth(1)
        lc.set_alpha(0.8)
        ax.add_collection(lc)
        ax.set_xlim(lf_exact.min(), lf_exact.max())
        ax.set_ylim(hf_exact.min(), hf_exact.max())
        ax.set_xlabel('LF Amplitude (40-80 kHz)')
        ax.set_ylabel('HF Amplitude (80-120 kHz)')
        ax.set_title(f'Lissajous: MP1, {ep_name} slice {i}\n({slice_start:.1f}-{slice_end:.1f} ms)')
        ax.grid(True)
        plt.savefig(out_dir / f"lissajous_short_{ep_name}_slice_{i}.png", dpi=200, bbox_inches='tight')
        plt.close(fig)

print("Done phase1 v2 script.")
