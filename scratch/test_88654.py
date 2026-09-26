import sys
import os
from pathlib import Path

# Add the script directory to path
sys.path.append(os.path.abspath('c:/TFG/analysis/obj3_nonlinear_coupling'))

import mhd_analysis_obj3
from mhd_analysis_obj3 import analyze_channel_bicoherence, triads_match, detect_hmode_burst_window

validation_shot = "88654"
validation_dir = Path("c:/TFG/data/hj88654")

val_wp_file = validation_dir / f"Wp@{validation_shot}.edf"
val_det = detect_hmode_burst_window(val_wp_file)
val_t_start, val_t_end = val_det['start_ms'], val_det['end_ms']

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

output_path = "c:/TFG/shot_88654_phase2_results.log"
with open(output_path, "w") as f:
    f.write("Phase 2 Planning: Validation Shot 88654 Reproducibility Test\n")
    f.write(f"Validation shot burst window: {val_t_start:.1f}-{val_t_end:.1f} ms\n")
    
    for label, ref_triad in triads_to_test.items():
        f.write(f"\n{'='*80}\n")
        f.write(f"TESTING TRIAD: {label}\n")
        f.write(f"Reference Triad: f1={ref_triad[0]:.1f}, f2={ref_triad[1]:.1f} kHz\n")
        f.write(f"{'='*80}\n")
        
        for ch in channels:
            fpath = validation_dir / f"{ch}@{validation_shot}.edf"
            result_rep = analyze_channel_bicoherence(
                fpath, nfft, noverlap_frac, ensemble, threshold, f1max, f2max, topn,
                t_start_ms=val_t_start, t_end_ms=val_t_end,
                peak_window_khz=peak_window_khz, fmin_khz=fmin_khz
            )
            
            # The reference peaks are Difference coupling (f1 - f2 = f3)
            # 89.8 - 41.0 = 48.8 > 0
            max_b2 = result_rep['diff_max_b2']
            best_freqs = result_rep['diff_best_freqs']
            top_peaks = result_rep['diff_peaks']
            
            f.write(f"\nChannel {ch}:\n")
            if not top_peaks:
                f.write("  No coupling peaks found.\n")
                continue
                
            f.write(f"  Max peak b² = {max_b2:.3f} at f1={best_freqs[0]:.1f}, f2={best_freqs[1]:.1f}, f3={best_freqs[2]:.1f} kHz\n")
            
            if max_b2 < threshold:
                f.write(f"  FAIL: max peak (b²={max_b2:.3f}) does NOT exceed threshold ({threshold})\n")
                continue
                
            if triads_match(ref_triad, best_freqs, repro_tol_khz):
                f.write(f"  PASS: Triad matches within ±{repro_tol_khz} kHz!\n")
            else:
                matched_sec = False
                for p_idx, peak in enumerate(top_peaks):
                    _, p_f1, p_f2, p_f3 = peak
                    if triads_match(ref_triad, (p_f1, p_f2, p_f3), repro_tol_khz):
                        f.write(f"  PASS: Matches secondary peak #{p_idx+1} at f1={p_f1:.1f}, f2={p_f2:.1f}\n")
                        matched_sec = True
                        break
                
                if not matched_sec:
                    f.write(f"  FAIL: No peak found within ±{repro_tol_khz} kHz of {ref_triad[0]}/{ref_triad[1]} kHz.\n")
                    f.write(f"  Top peaks were:\n")
                    for p in top_peaks[:3]:
                        f.write(f"    b²={p[0]:.3f} at f1={p[1]:.1f}, f2={p[2]:.1f}\n")

print("Done writing test_88654.py")
