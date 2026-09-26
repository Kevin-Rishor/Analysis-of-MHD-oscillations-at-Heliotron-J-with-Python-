import sys
from pathlib import Path
import numpy as np
import scipy.signal as dsp

root_dir = Path("c:/TFG")
for p_add in [root_dir / "jpack", root_dir / "analysis", root_dir / "analysis" / "common"]:
    if str(p_add) not in sys.path:
        sys.path.append(str(p_add))

import turnelib as TE
from mhd_common import extract_instantaneous_frequency

shot = 88653
data_dir = root_dir / "data" / f"hj{shot}"
probes = ["MP1", "MP3", "MP4"]
t_start, t_end = 259.1, 275.0
edf = TE.edf()

results_by_factor = {}

for factor in [1, 2, 4]:
    nperseg = 2048 * factor
    noverlap = 1536 * factor
    nfft = 4096 * factor
    
    print(f"\n--- factor {factor}x ---")
    print(f"nperseg={nperseg}, noverlap={noverlap}, nfft={nfft}")
    
    probe_peaks = {1.22: [], 2.20: []}
    
    for p in probes:
        file_path = data_dir / f"{p}@{shot}.edf"
        dat = edf.load(str(file_path))
        t_raw = dat[:, 0]
        t_ms = t_raw if edf.DimUnit[0] == "ms" else t_raw * 1000.0
        fs = 1.0 / ((t_ms[1] - t_ms[0]) / 1000.0)
        ys = dat[:, 1]
        
        envelope, _, _, _ = extract_instantaneous_frequency(ys, fs, 80000, 120000, 4, 325)
        
        idx_win = np.where((t_ms >= t_start) & (t_ms <= t_end))[0]
        ys_win = ys[idx_win]
        env_win = envelope[idx_win]
        
        f, coh2 = dsp.coherence(ys_win, env_win, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, nfft=nfft)
        f_khz = f / 1000.0
        df = f_khz[1] - f_khz[0]
        
        n_seg = int(np.floor((len(ys_win) - nperseg) / (nperseg - noverlap))) + 1
        sig_floor = 1.0 - (0.05) ** (1.0 / max(1, n_seg - 1))
        
        # Find coherence at 1.22 and 2.20 kHz
        idx_122 = np.argmin(np.abs(f_khz - 1.22))
        idx_220 = np.argmin(np.abs(f_khz - 2.20))
        
        coh_122 = coh2[idx_122]
        coh_220 = coh2[idx_220]
        
        if coh_122 > sig_floor:
            probe_peaks[1.22].append(p)
        if coh_220 > sig_floor:
            probe_peaks[2.20].append(p)
            
        print(f"[{p}] df={df:.4f} kHz, floor={sig_floor:.3f} | coh@1.22={coh_122:.3f}, coh@2.20={coh_220:.3f}")

    print("Peaks > floor in probes:")
    print(f" 1.22 kHz in {len(probe_peaks[1.22])}/3 probes ({','.join(probe_peaks[1.22])})")
    print(f" 2.20 kHz in {len(probe_peaks[2.20])}/3 probes ({','.join(probe_peaks[2.20])})")
