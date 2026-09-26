import sys
import os
import numpy as np
from pathlib import Path

# Add the script directory to path
# Add the script directory to path
sys.path.append(os.path.abspath('c:/TFG/analysis/obj3_nonlinear_coupling'))
from mhd_analysis_obj3 import load_edf_signal

def get_on_off_times(fpath):
    if not os.path.exists(fpath):
        return None, None
    t, y, _, _ = load_edf_signal(fpath)
    
    # NBI/ECH signals are usually square pulses. Find where they are above 10% of their max.
    # Wp might not be a square pulse, so let's just find the first and last time it's above a certain value
    # But wait, Wp has a baseline that might be non-zero.
    baseline = np.mean(y[100:1000]) # First part
    peak = np.max(y)
    print(f"Debug {fpath[-15:]}: len={len(y)}, peak={peak:.4f}, base={baseline:.4f}")
    if peak - baseline < 1e-3:
        return None, None
        
    threshold = baseline + (peak - baseline) * 0.5
    
    # search after 50 ms to avoid initial spikes
    dt = t[1] - t[0]
    skip_samples = int(50.0 / dt) if len(t) > 1 and dt > 0 else 0
    if skip_samples >= len(y): skip_samples = 0
    
    on_mask = y[skip_samples:] > threshold
    if not np.any(on_mask):
        return None, None
        
    on_idx = np.where(on_mask)[0] + skip_samples
    return t[on_idx[0]], t[on_idx[-1]]

for shot in ["88653", "88654"]:
    print(f"SHOT {shot}:")
    dir_path = f"c:/TFG/data/hj{shot}"
    
    for nbi in ["NBIS10I", "NBIS3I", "NBIS4I", "NBIS9I"]:
        fpath = f"{dir_path}/{nbi}@{shot}.edf"
        t_on, t_off = get_on_off_times(fpath)
        if t_on is not None:
            print(f"  {nbi}: {t_on:.1f} - {t_off:.1f} ms")
            
    ech_path = f"{dir_path}/ECHRG500@{shot}.edf"
    t_on, t_off = get_on_off_times(ech_path)
    if t_on is not None:
        print(f"  ECHRG500: {t_on:.1f} - {t_off:.1f} ms")
    
    print()
