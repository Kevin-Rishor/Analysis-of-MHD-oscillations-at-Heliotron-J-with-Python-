import sys
from pathlib import Path
import numpy as np
import scipy.signal as dsp

root_dir = Path("c:/TFG")
sys.path.append(str(root_dir / "jpack"))
sys.path.append(str(root_dir / "analysis" / "common"))
sys.path.append(str(root_dir / "analysis" / "heating_phase_comparison"))

import turnelib as TE
from comparison_common import load_channel

def run_step5():
    shot = 88653
    data_dir = root_dir / "data" / f"hj{shot}"
    
    # Load MP1 for coherence
    t_mp1, y_mp1, dt, fs = load_channel(shot, "MP1")
    
    if y_mp1 is None:
        print("Error: MP1 not found.")
        return
        
    # We want coherence with MP1 at ~89 kHz in the active window (e.g., ECH+NBI phase 259.1 - 291.3)
    # Let's use 259.1 to 291.3 for coherence
    t_start = 259.1
    t_end = 291.3
    idx_win = np.where((t_mp1 >= t_start) & (t_mp1 <= t_end))[0]
    y_mp1_win = y_mp1[idx_win]
    nperseg = 2048
    noverlap = int(nperseg * 0.75)
    nfft = 2048
    
    out_lines = ["\nStep 5: ECE Radial Profile"]
    out_lines.append("Channel | Frequency | RMS | Coherence @ 89 kHz | Notes")
    out_lines.append("-" * 75)
    
    results = []
    
    edf = TE.edf()
    for ece_file in data_dir.glob("ECE*FAST@88653.edf"):
        dat = edf.load(str(ece_file))
        if dat is None:
            continue
            
        ys = dat[:, 1]
        
        # Parse frequency
        freq_str = "Unknown"
        for comment in edf.comments:
            if "property =" in comment:
                # e.g., "property = '57.5 GHz'"
                try:
                    freq_str = comment.split("=")[1].strip().strip("'")
                except:
                    pass
        
        rms = np.sqrt(np.mean(ys**2))
        name = edf.Name
        
        coh_val = np.nan
        if rms > 1e-6:
            ys_win = ys[idx_win]
            f_coh, coh = dsp.coherence(y_mp1_win, ys_win, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=nfft)
            f_khz = f_coh / 1000.0
            idx_89 = np.argmin(np.abs(f_khz - 89.0))
            coh_val = coh[idx_89]
            
        results.append({
            "name": name,
            "freq": freq_str,
            "freq_val": float(freq_str.replace("GHz", "").strip()) if "GHz" in freq_str else 0.0,
            "rms": rms,
            "coh": coh_val
        })
        
    results.sort(key=lambda x: x["freq_val"])
    
    for r in results:
        notes = ""
        if r["rms"] < 1e-3:
            notes = "Dead/unconnected/different receiver"
        coh_str = f"{r['coh']:.4f}" if not np.isnan(r['coh']) else "N/A"
        line = f"{r['name']:<10} | {r['freq']:<10} | {r['rms']:<10.6f} | {coh_str:<18} | {notes}"
        out_lines.append(line)
        
    with open(root_dir / "scratch" / "round_final2.txt", "a") as f:
        f.write("\n".join(out_lines) + "\n")

if __name__ == "__main__":
    run_step5()
