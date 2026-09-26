import sys
import os
import numpy as np
from pathlib import Path

# Add the script directory to path
sys.path.append(os.path.abspath('c:/TFG/analysis/obj3_nonlinear_coupling'))

import mhd_analysis_obj3
from mhd_analysis_obj3 import analyze_channel_bicoherence, triads_match, detect_hmode_burst_window

validation_shot = "88654"
validation_dir = Path("c:/TFG/data/hj88654")

val_wp_file = validation_dir / f"Wp@{validation_shot}.edf"
val_det = detect_hmode_burst_window(val_wp_file)
val_t_start_auto, val_t_end_auto = val_det['start_ms'], val_det['end_ms']

# Using the manual window corresponding to 88653
val_t_start_manual = 259.1
val_t_end_manual = 275.0

channels = ["MP1", "MP3", "MP4"]
nfft = 1024
noverlap_frac = 0.5
ensemble = 30
threshold = 0.5
f1max = 150.0
f2max = 150.0
topn = 10
peak_window_khz = 5.0
fmin_khz = 5.0
repro_tol_khz = 10.0

triads_to_test = {
    "PSD-confirmed triad": (89.8, 41.0, 0.0),
    "non-PSD-confirmed triad, for comparison only": (98.6, 48.8, 0.0)
}

output_path = "c:/TFG/shot_88654_phase2_results_v2.log"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("Phase 1 Clarification: Validation Shot 88654 Reproducibility Test\n")
    f.write(f"Wp auto-detected burst window: {val_t_start_auto:.1f}-{val_t_end_auto:.1f} ms\n")
    f.write(f"Manually applied matching window (same as 88653): {val_t_start_manual:.1f}-{val_t_end_manual:.1f} ms\n")
    f.write("We will use the manually applied window for this test.\n")
    
    for label, ref_triad in triads_to_test.items():
        f.write(f"\n{'='*80}\n")
        f.write(f"TESTING TRIAD: {label}\n")
        f.write(f"Reference Triad: f1={ref_triad[0]:.1f}, f2={ref_triad[1]:.1f} kHz\n")
        f.write(f"{'='*80}\n")
        
        for ch in channels:
            fpath = validation_dir / f"{ch}@{validation_shot}.edf"
            result_rep = analyze_channel_bicoherence(
                fpath, nfft, noverlap_frac, ensemble, threshold, f1max, f2max, topn,
                t_start_ms=val_t_start_manual, t_end_ms=val_t_end_manual,
                peak_window_khz=peak_window_khz, fmin_khz=fmin_khz
            )
            
            max_b2 = result_rep['diff_max_b2']
            best_freqs = result_rep['diff_best_freqs']
            
            # Find the bicoherence value specifically at the reference triad location
            bicoh2 = result_rep['bicoh2']
            f1_arr = result_rep['f1'] / 1000.0
            f2_arr = result_rep['f2'] / 1000.0
            
            f1_ref, f2_ref, _ = ref_triad
            
            f1_mask = np.abs(f1_arr - f1_ref) <= repro_tol_khz
            f2_mask = np.abs(f2_arr - (-f2_ref)) <= repro_tol_khz
            
            f1_idx = np.where(f1_mask)[0]
            f2_idx = np.where(f2_mask)[0]
            
            if len(f1_idx) > 0 and len(f2_idx) > 0:
                subgrid = bicoh2[np.ix_(f1_idx, f2_idx)]
                max_ref_b2 = np.max(subgrid)
                max_idx = np.unravel_index(np.argmax(subgrid), subgrid.shape)
                found_f1 = f1_arr[f1_idx[max_idx[0]]]
                found_f2 = -f2_arr[f2_idx[max_idx[1]]]
            else:
                max_ref_b2 = 0.0
                found_f1 = 0.0
                found_f2 = 0.0
            
            f.write(f"\nChannel {ch}:\n")
            f.write(f"  Global max peak anywhere: b² = {max_b2:.3f} at f1={best_freqs[0]:.1f}, f2={best_freqs[1]:.1f} kHz\n")
            f.write(f"  Max peak within ±{repro_tol_khz} kHz of ref triad: b² = {max_ref_b2:.3f} at f1={found_f1:.1f}, f2={found_f2:.1f} kHz\n")
            
            if max_b2 < threshold:
                f.write(f"  NOTE: The main script short-circuits here! Because global max {max_b2:.3f} < {threshold}, it FAILS immediately without ever checking triads_match.\n")
            else:
                f.write(f"  NOTE: The main script proceeds past the short-circuit.\n")

print("Done writing test_88654_v2.py")
