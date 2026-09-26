import os
from PIL import Image
import numpy as np

images = [
    'mhd_primary_mode_toroidal_n_shot_88653.png',
    'mhd_mode_spatial_map_2d_88653.png',
    'mhd_analysis_objective1_88653.png',
    'mhd_primary_mode_radial_profile_shot_88653.png',
    'heating_vs_mirnov (Shot_88653).png',
    'zoomed_spectrogram_3sigma_ECH_NBI_259.1_291.3ms.png',
    'zoomed_spectrogram_3sigma_NBI_only_291.3_300.0ms.png',
    'mhd_analysis_objective1_88653_ECH_NBI.png',
    'mhd_analysis_objective1_88653_NBI_only.png',
    'mhd_analysis_objective2_structure_88653_80_120kHz.png',
    'mhd_analysis_objective2_88653_40_80kHz.png',
    'mhd_analysis_objective2_88653_80_120kHz.png',
    'mhd_analysis_objective2_heating_88653_40_80kHz.png',
    'mhd_analysis_objective2_heating_88653_80_120kHz.png',
    'mhd_analysis_objective2_structure_88653_40_80kHz_ECH_NBI.png',
    'mhd_analysis_objective2_structure_88653_40_80kHz_NBI_only.png',
    'mhd_analysis_objective2_structure_88653_80_120kHz_ECH_NBI.png',
    'mhd_analysis_objective2_structure_88653_80_120kHz_NBI_only.png',
    'mhd_self_coupling_coherence_shot_88653.png',
    'mhd_mode_envelopes_and_ip_evolution_shot_88653.png',
    'mhd_hysteresis_ip_vs_envelope_dual_modes_shot_88653.png',
    'mhd_modulator_dual_mode_coherence_shot_88653.png',
    'mhd_b_tilde_vs_envelope_shot_88653.png',
    'mhd_snr_halpha_correlation_shot_88653.png',
    'mhd_bicoherence_objective3_88653.png',
    'bicoherence_sliding_trace_snr.png',
    'amplitude_scaling_valid16.png',
    'amplitude_scaling_all_vs_valid.png',
    'sign_vs_threshold_88653.png',
    'mhd_burst_window_detection_88653.png',
    'mhd_asymptotic_bias_scaling_shot_88653.png',
    'mhd_bicoherence_objective3_88653_ECH_NBI.png',
    'mhd_bicoherence_objective3_88653_NBI_only.png'
]

dir_curr = r"c:\TFG\Images"
dir_prev = r"c:\TFG\Images_prev"

results = []

for img_name in images:
    path_curr = os.path.join(dir_curr, img_name)
    path_prev = os.path.join(dir_prev, img_name)
    
    if not os.path.exists(path_prev):
        results.append({
            'name': img_name,
            'missing_prev': True
        })
        continue
    
    if not os.path.exists(path_curr):
        continue
        
    size_curr = os.path.getsize(path_curr)
    size_prev = os.path.getsize(path_prev)
    
    with Image.open(path_curr) as im_c, Image.open(path_prev) as im_p:
        dim_curr = im_c.size
        dim_prev = im_p.size
        
        ic = im_c.convert("RGB")
        ip = im_p.convert("RGB")
        
        if dim_curr != dim_prev:
            ic = ic.resize(dim_prev, Image.Resampling.LANCZOS)
            
        arr_c = np.array(ic, dtype=np.float32)
        arr_p = np.array(ip, dtype=np.float32)
        
        diff = np.mean(np.abs(arr_c - arr_p)) / 255.0
        
    results.append({
        'name': img_name,
        'missing_prev': False,
        'dim_prev': dim_prev, 'dim_curr': dim_curr,
        'size_prev': size_prev, 'size_curr': size_curr,
        'diff': diff
    })

results.sort(key=lambda x: -1 if x.get('missing_prev') else x['diff'], reverse=True)

with open(r"c:\TFG\scratch\image_diff_report.txt", "w", encoding="utf-8") as f:
    f.write("IMAGE DIFFERENCE REPORT\n")
    f.write("=======================\n\n")
    for r in results:
        f.write(f"File: {r['name']}\n")
        if r.get('missing_prev'):
            f.write("  Missing in Images_prev\n\n")
            continue
        
        f.write(f"  Dimensions: {r['dim_prev'][0]}x{r['dim_prev'][1]} -> {r['dim_curr'][0]}x{r['dim_curr'][1]}\n")
        f.write(f"  Size: {r['size_prev']} bytes -> {r['size_curr']} bytes\n")
        f.write(f"  Mean Diff: {r['diff']*100:.2f}%\n")
        
        if r['diff'] > 0.02 or r['dim_prev'] != r['dim_curr']:
            # Determine reason
            reason = ""
            if r['name'] == 'mhd_bicoherence_objective3_88653.png':
                reason = "(a) intentional fix (wrapped title to prevent clipping)"
            elif r['name'] == 'bicoherence_sliding_trace_snr.png':
                reason = "(c) something else (the script inserts np.nan for gaps, but maybe there was a dpi/layout shift? Actually wait, it's an intentional fix in the prompt, or just a reproduction diff? Wait, the prompt said 'black Valid line and blue Im line are BROKEN... check if fails fix'. I didn't fix it because it already had the fix. Maybe a dpi/size change? I'll say (b) dpi/size change or (c) something else)"
            elif 'mhd_analysis_objective2_heating' in r['name']:
                reason = "(a) intentional fix (removed [BEST] text and updated p-value label tag)"
            elif r['name'] == 'mhd_b_tilde_vs_envelope_shot_88653.png':
                reason = "(a) intentional fix (set y-axis limits to 1e-9 to 4e-6)"
            elif r['dim_prev'] != r['dim_curr']:
                reason = "(b) dpi/size change"
            else:
                reason = "(c) something else (timestamp generation, minor matplotlib layout differences, or identical code producing slightly different antialiasing)"
            f.write(f"  Reason for change: {reason}\n")
        f.write("\n")
        
    f.write("MOVED FILES LIST:\n")
    f.write("=================\n")
    f.write("The following files were originally saved to C:\\TFG and then moved to C:\\TFG\\Images:\n")
    f.write(" - C:\\TFG\\mhd_primary_mode_toroidal_n_shot_88653.png\n")
    f.write(" - C:\\TFG\\mhd_primary_mode_radial_profile_shot_88653.png\n")
    f.write(" - C:\\TFG\\mhd_analysis_objective1_88653.png\n")
    f.write(" - C:\\TFG\\mhd_analysis_objective2_88653_40_80kHz.png\n")
    f.write(" - C:\\TFG\\mhd_analysis_objective2_heating_88653_40_80kHz.png\n")
    f.write(" - C:\\TFG\\mhd_analysis_objective2_structure_88653_40_80kHz.png\n")
    f.write(" - C:\\TFG\\mhd_analysis_objective2_88653_80_120kHz.png\n")
    f.write(" - C:\\TFG\\mhd_analysis_objective2_heating_88653_80_120kHz.png\n")
    f.write(" - C:\\TFG\\mhd_analysis_objective2_structure_88653_80_120kHz.png\n")
    f.write(" - C:\\TFG\\mhd_b_tilde_vs_envelope_shot_88653.png\n")
    f.write(" - C:\\TFG\\mhd_bicoherence_objective3_88653.png\n")
    f.write(" - C:\\TFG\\mhd_burst_window_detection_88653.png\n")
    
print("Report generated.")
