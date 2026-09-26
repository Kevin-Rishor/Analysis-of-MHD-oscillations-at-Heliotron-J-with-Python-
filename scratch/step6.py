import sys
from pathlib import Path
import numpy as np

root_dir = Path("c:/TFG")
sys.path.append(str(root_dir / "jpack"))
sys.path.append(str(root_dir / "analysis"))
sys.path.append(str(root_dir / "analysis" / "common"))
sys.path.append(str(root_dir / "analysis" / "heating_phase_comparison"))

from comparison_common import load_channel
from mhd_plot_bicoherence_sliding_trace import custom_abicoh2_complex

def run_step6():
    shot = 88653
    t_ms, ys, dt, fs = load_channel(shot, "MP1")
    
    if ys is None:
        print("Error: MP1 not found.")
        return
        
    out_lines = ["\nStep 6: Avalanche Test"]
    
    # f1 = 101.6 kHz, f2 = -50.8 kHz -> f3 = 50.8 kHz
    target_f1 = 101.6
    target_f2 = -50.8
    
    def get_b2(t_start, t_end, num_segments, label):
        idx = np.where((t_ms >= t_start) & (t_ms <= t_end))[0]
        if len(idx) == 0:
            out_lines.append(f"{label}: No data in {t_start}-{t_end} ms.")
            return
            
        y_sub = ys[idx]
        
        # custom_abicoh2_complex nfft=512, nensemble
        # 512 points @ 1MHz = 0.512 ms. 
        # to get exactly num_segments, we pass nensemble=num_segments
        f1, f2, b2, b_comp, den = custom_abicoh2_complex(y_sub, dt=dt, nfft=512, nensemble=num_segments)
        
        # find closest frequencies
        ff1_k, ff2_k = np.meshgrid(f1 / 1000.0, f2 / 1000.0, indexing='ij')
        
        dist = (ff1_k - target_f1)**2 + (ff2_k - target_f2)**2
        best_c = np.unravel_index(np.argmin(dist), dist.shape)
        
        b2_val = b2[best_c]
        actual_f1 = f1[best_c[0]] / 1000.0
        actual_f2 = f2[best_c[1]] / 1000.0
        
        N = min(num_segments, len(y_sub) // 256 - 1)  # approx num segments
        N_eff = N / 2.0
        threshold = 1.0 / N_eff if N_eff > 0 else 1.0
        
        out_lines.append(f"{label} (actual f1={actual_f1:.1f}, f2={actual_f2:.1f}): b2 = {b2_val:.4f}, N = {N}, N_eff = {N_eff:.1f}, 1/N_eff = {threshold:.4f}")

    # (a) the first 30 segments of 291.3-300.0 ms
    get_b2(291.3, 300.0, 30, "(a) 291.3-300.0 ms, first 30 segs")
    
    # (b) 60 and 124 segments of 291.3-330 ms
    get_b2(291.3, 330.0, 60, "(b) 291.3-330.0 ms, first 60 segs")
    get_b2(291.3, 330.0, 124, "(b) 291.3-330.0 ms, first 124 segs")
    
    # (c) quiet 400-410 ms (first 124 segments for comparison? let's do 124)
    get_b2(400.0, 410.0, 124, "(c) 400.0-410.0 ms (quiet), 124 segs")

    with open(root_dir / "scratch" / "round_final2.txt", "a") as f:
        f.write("\n".join(out_lines) + "\n")

if __name__ == "__main__":
    run_step6()
