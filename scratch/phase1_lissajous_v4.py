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
    "Episode_2": (301.8, 304.6),
    "NBI_only": (293.0, 295.0)
}

LF_BAND = (36e3, 44e3)
HF_BAND = (84e3, 94e3)
ORDER = 4

data_dir = Path(r"c:\TFG\data\hj88653")
out_dir = Path(r"c:\TFG\scratch")
log_file = out_dir / "lissajous_check_v4.log"

with open(log_file, "w") as f_log:
    f_log.write(f"--- Lissajous Check Phase 1 (V4 - Beat Period & Phase Binning) ---\n\n")
    f_log.write("Filter Transient Note: The signal is padded by 5.0 ms prior to applying the 4th-order Bessel filter. Since the LF period is ~0.025 ms, 5.0 ms provides ~200 clean cycles for the filter to settle before the analysis window begins. Thus, no edge transients remain inside the analyzed slices, regardless of their short duration.\n\n")

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

def find_contiguous_regions(condition):
    idx = np.flatnonzero(condition)
    if len(idx) == 0:
        return []
    blocks = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    return blocks

for win_name, (t_start, t_end) in WINDOWS.items():
    pad_ms = 5.0
    mask = (t_ms_full >= t_start - pad_ms) & (t_ms_full <= t_end + pad_ms)
    t_win_pad = t_ms_full[mask]
    ys_win_pad = ys_full[mask]
    
    lf_env, lf_phase_cycles, lf_filt, _ = extract_instantaneous_frequency(ys_win_pad, fs, LF_BAND[0], LF_BAND[1], order=ORDER)
    hf_env, hf_phase_cycles, hf_filt, _ = extract_instantaneous_frequency(ys_win_pad, fs, HF_BAND[0], HF_BAND[1], order=ORDER)
    
    mask_exact = (t_win_pad >= t_start) & (t_win_pad <= t_end)
    t_exact = t_win_pad[mask_exact]
    t_sec_exact = t_exact / 1000.0
    
    lf_exact = lf_filt[mask_exact]
    hf_exact = hf_filt[mask_exact]
    lf_phase_exact = lf_phase_cycles[mask_exact]
    hf_phase_exact = hf_phase_cycles[mask_exact]
    
    # Compute unwrapped phase difference in cycles
    dphi_cycles = hf_phase_exact - 2.0 * lf_phase_exact
    
    # Fit linear slope to get exact beat frequency
    # dphi_cycles vs t_sec_exact
    slope, intercept = np.polyfit(t_sec_exact, dphi_cycles, 1)
    beat_freq_hz = abs(slope)
    beat_period_s = 1.0 / beat_freq_hz if beat_freq_hz > 0 else np.inf
    beat_period_ms = beat_period_s * 1000.0
    
    with open(log_file, "a") as f:
        f.write(f"--- Window: {win_name} ---\n")
        f.write(f"Fitted Beat Frequency: {beat_freq_hz/1000:.3f} kHz\n")
        f.write(f"Derived Beat Period: {beat_period_ms:.4f} ms\n\n")

    if win_name in ["Episode_1", "Episode_2", "NBI_only"]:
        # 1/4 and 1/2 beat period plots
        fractions = {"quarter": 0.25, "half": 0.5}
        for frac_name, frac_val in fractions.items():
            slice_dur = beat_period_ms * frac_val
            slice_end = t_start + slice_dur
            s_mask = (t_exact >= t_start) & (t_exact <= slice_end)
            
            t_s = t_exact[s_mask]
            lf_s = lf_exact[s_mask]
            hf_s = hf_exact[s_mask]
            
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(lf_s, hf_s, 'b-', linewidth=1.5, alpha=0.8)
            ax.set_xlabel('LF Amplitude')
            ax.set_ylabel('HF Amplitude')
            ax.set_title(f'Lissajous: {win_name} ({frac_name} beat period)\n{slice_dur:.4f} ms duration')
            ax.grid(True)
            plt.savefig(out_dir / f"lissajous_v4_{frac_name}beat_{win_name}.png", dpi=200, bbox_inches='tight')
            plt.close(fig)
            
        # Phase binning overlay
        # map dphi to [0, 2pi)
        dphi_rad_wrap0 = (dphi_cycles * 2 * np.pi) % (2 * np.pi)
        num_bins = 8
        bin_edges = np.linspace(0, 2 * np.pi, num_bins + 1)
        colors = plt.cm.rainbow(np.linspace(0, 1, num_bins))
        
        fig, ax = plt.subplots(figsize=(8, 8))
        for i in range(num_bins):
            # condition for current bin
            c = (dphi_rad_wrap0 >= bin_edges[i]) & (dphi_rad_wrap0 < bin_edges[i+1])
            blocks = find_contiguous_regions(c)
            
            label_added = False
            for blk in blocks:
                if len(blk) > 1:
                    ax.plot(lf_exact[blk], hf_exact[blk], color=colors[i], alpha=0.5, linewidth=1.0, 
                            label=f"Bin {i+1} ({bin_edges[i]:.1f}-{bin_edges[i+1]:.1f} rad)" if not label_added else "")
                    label_added = True
                    
        ax.set_xlabel('LF Amplitude')
        ax.set_ylabel('HF Amplitude')
        ax.set_title(f'Lissajous Phase-Binned Overlay: {win_name}')
        ax.grid(True)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(out_dir / f"lissajous_v4_phasebinned_{win_name}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

print("Done phase1 v4 script.")
