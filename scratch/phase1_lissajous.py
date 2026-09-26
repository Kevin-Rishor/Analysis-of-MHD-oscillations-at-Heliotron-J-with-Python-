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
PROBES = ["MP1", "MP3", "MP4"]
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
log_file = out_dir / "lissajous_check_lines.log"

with open(log_file, "w") as f_log:
    f_log.write(f"--- Lissajous Check Phase 1 (Lines) ---\n\n")

for probe in PROBES:
    # Load data
    fname = data_dir / f"{probe}@{SHOT}.edf"
    edf = TE.edf()
    dat = edf.load(str(fname))
    t_full = dat[:, 0]
    
    # Convert time to ms
    if edf.DimUnit[0] == 's':
        t_ms_full = t_full * 1000.0
    else:
        t_ms_full = t_full
    
    fs = 1.0 / (t_full[1] - t_full[0])
    if edf.DimUnit[0] == 'ms':
        fs = fs * 1000.0

    ys_full = dat[:, 1]
    
    for win_name, (t_start, t_end) in WINDOWS.items():
        # Crop to window with some padding for filtering
        pad_ms = 5.0
        mask = (t_ms_full >= t_start - pad_ms) & (t_ms_full <= t_end + pad_ms)
        t_win_pad = t_ms_full[mask]
        ys_win_pad = ys_full[mask]
        
        # Filter
        _, _, lf_filt, lf_ifreq = extract_instantaneous_frequency(
            ys_win_pad, fs, LF_BAND[0], LF_BAND[1], order=ORDER
        )
        _, _, hf_filt, hf_ifreq = extract_instantaneous_frequency(
            ys_win_pad, fs, HF_BAND[0], HF_BAND[1], order=ORDER
        )
        
        # Remove padding
        mask_exact = (t_win_pad >= t_start) & (t_win_pad <= t_end)
        t_exact = t_win_pad[mask_exact]
        lf_exact = lf_filt[mask_exact]
        hf_exact = hf_filt[mask_exact]
        lf_ifreq_exact = lf_ifreq[mask_exact]
        hf_ifreq_exact = hf_ifreq[mask_exact]
        
        # Compute ratio
        mean_lf_freq = np.mean(lf_ifreq_exact)
        mean_hf_freq = np.mean(hf_ifreq_exact)
        ratio = mean_hf_freq / mean_lf_freq
        
        # Log
        with open(log_file, "a") as f_log:
            f_log.write(f"Probe {probe}, Window {win_name} ({t_start}-{t_end} ms):\n")
            f_log.write(f"  Mean LF ifreq: {mean_lf_freq/1000:.2f} kHz\n")
            f_log.write(f"  Mean HF ifreq: {mean_hf_freq/1000:.2f} kHz\n")
            f_log.write(f"  Frequency ratio (HF/LF): {ratio:.3f}\n\n")
        
        # Plot with continuous lines
        fig, ax = plt.subplots(figsize=(8, 6))
        
        points = np.array([lf_exact, hf_exact]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        norm = plt.Normalize(t_exact.min(), t_exact.max())
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(t_exact)
        lc.set_linewidth(1)
        lc.set_alpha(0.6)
        
        line = ax.add_collection(lc)
        cbar = fig.colorbar(line, ax=ax, label='Time (ms)')
        
        ax.set_xlim(lf_exact.min(), lf_exact.max())
        ax.set_ylim(hf_exact.min(), hf_exact.max())
        
        ax.set_xlabel('LF Amplitude (40-80 kHz)')
        ax.set_ylabel('HF Amplitude (80-120 kHz)')
        ax.set_title(f'Lissajous: {probe}, {win_name} ({t_start}-{t_end} ms)')
        ax.grid(True)
        
        out_fig = out_dir / f"lissajous_lines_{probe}_{win_name}.png"
        plt.savefig(out_fig, dpi=300, bbox_inches='tight')
        plt.close(fig)

print(f"Done. Check {log_file}")
