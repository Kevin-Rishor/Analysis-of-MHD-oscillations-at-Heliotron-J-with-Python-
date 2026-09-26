import sys
import os
import numpy as np
from pathlib import Path

# Add the script directory to path
sys.path.append(os.path.abspath('c:/TFG/analysis/obj3_nonlinear_coupling'))

import mhd_analysis_obj3
from mhd_analysis_obj3 import analyze_channel_bicoherence, triads_match, detect_hmode_burst_window, compute_flat_frequency_window

class Args:
    pass

args = Args()
args.flat_freq_low_khz = 5.0
args.flat_freq_high_khz = 150.0
args.flat_order = 4
args.flat_sg_win = 525
args.nfft = 1024
args.ensemble = 30
args.flat_min_duration_ms = 5.0
args.flat_scan_window_ms = 8.0
args.flat_growth_tolerance = 0.5
args.flat_slope_smooth_ms = 2.0
args.flat_window_start = None
args.flat_window_end = None
args.fmin = 5.0
args.f1max = 150.0
args.f2max = 150.0
args.topn = 10
args.threshold = 0.5
args.peak_window_khz = 5.0
args.repro_tol_khz = 10.0

validation_shot = "88654"
validation_dir = Path("c:/TFG/data/hj88654")

val_wp_file = validation_dir / f"Wp@{validation_shot}.edf"
val_det = detect_hmode_burst_window(val_wp_file)
val_t_start_auto, val_t_end_auto = val_det['start_ms'], val_det['end_ms']

print(f"Wp auto-detected burst window: {val_t_start_auto:.1f}-{val_t_end_auto:.1f} ms")

# Run the M9 flat-frequency sub-window detection algorithm
mp1_file = validation_dir / f"MP1@{validation_shot}.edf"
i0, i1, info = compute_flat_frequency_window(mp1_file, val_t_start_auto, val_t_end_auto, args)

if 'used_fallback' in info and info['used_fallback']:
    val_t_start_m9, val_t_end_m9 = val_t_start_auto, val_t_end_auto
    print(f"M9 algorithm failed or fell back: {info['reason']}")
else:
    val_t_start_m9, val_t_end_m9 = i0, i1
    print(f"M9 sub-window: {val_t_start_m9:.1f}-{val_t_end_m9:.1f} ms")

# Re-run the reproducibility test using val_t_start_m9, val_t_end_m9
channels = ["MP1", "MP3", "MP4"]
noverlap_frac = 0.5

triads_to_test = {
    "PSD-confirmed triad": (89.8, 41.0, 0.0),
    "non-PSD-confirmed triad, for comparison only": (98.6, 48.8, 0.0)
}

output_path = "c:/TFG/shot_88654_phase2_results_v3.log"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("Phase 1 Clarification: Validation Shot 88654 Reproducibility Test\n")
    f.write(f"Wp auto-detected burst window: {val_t_start_auto:.1f}-{val_t_end_auto:.1f} ms\n")
    f.write(f"M9 auto-detected flat-frequency sub-window: {val_t_start_m9:.1f}-{val_t_end_m9:.1f} ms\n")
    
    # Information about timestamps
    # Based on the auto-detected Wp burst windows:
    # 88653: 198.8 - 311.8 ms (from cleanup_inventory_v3.log)
    # 88654: 208.1 - 310.0 ms
    f.write("\n--- Timing Alignment ---\n")
    f.write("Shot 88653 Wp auto-detected burst window: 198.8 - 311.8 ms\n")
    f.write(f"Shot 88654 Wp auto-detected burst window: {val_t_start_auto:.1f} - {val_t_end_auto:.1f} ms\n")
    f.write("The two shots are offset by approximately +9.3 ms (88654 starts later). The absolute timestamps do not naturally align.\n")
    f.write("Therefore, rather than rigidly transposing the manual 259.1-275.0 ms window from 88653, we apply the identical M9 algorithm to 88654's own timeline.\n")
    
    for label, ref_triad in triads_to_test.items():
        f.write(f"\n{'='*80}\n")
        f.write(f"TESTING TRIAD: {label}\n")
        f.write(f"Reference Triad: f1={ref_triad[0]:.1f}, f2={ref_triad[1]:.1f} kHz\n")
        f.write(f"{'='*80}\n")
        
        for ch in channels:
            fpath = validation_dir / f"{ch}@{validation_shot}.edf"
            result_rep = analyze_channel_bicoherence(
                fpath, args.nfft, noverlap_frac, args.ensemble, args.threshold, args.f1max, args.f2max, args.topn,
                t_start_ms=val_t_start_m9, t_end_ms=val_t_end_m9,
                peak_window_khz=args.peak_window_khz, fmin_khz=args.fmin
            )
            
            max_b2 = result_rep['diff_max_b2']
            best_freqs = result_rep['diff_best_freqs']
            
            # Find the bicoherence value specifically at the reference triad location
            bicoh2 = result_rep['bicoh2']
            f1_arr = result_rep['f1'] / 1000.0
            f2_arr = result_rep['f2'] / 1000.0
            
            f1_ref, f2_ref, _ = ref_triad
            
            f1_mask = np.abs(f1_arr - f1_ref) <= args.repro_tol_khz
            f2_mask = np.abs(f2_arr - (-f2_ref)) <= args.repro_tol_khz
            
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
            f.write(f"  Max peak within ±{args.repro_tol_khz} kHz of ref triad: b² = {max_ref_b2:.3f} at f1={found_f1:.1f}, f2={found_f2:.1f} kHz\n")
            
            if max_b2 < args.threshold:
                f.write(f"  NOTE: The main script short-circuits here! Because global max {max_b2:.3f} < {args.threshold}, it FAILS immediately without ever checking triads_match.\n")
            else:
                if max_ref_b2 >= args.threshold:
                    f.write(f"  PASS: The script would find a match for this triad, because global max >= threshold AND local max >= threshold.\n")
                else:
                    f.write(f"  FAIL: The script would fail because the local max {max_ref_b2:.3f} is < threshold.\n")

print("Done writing test_88654_v3.py")
