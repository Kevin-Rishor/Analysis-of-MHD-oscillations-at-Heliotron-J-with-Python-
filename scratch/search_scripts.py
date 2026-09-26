import os, glob

files = [
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

dirs_to_search = ['c:/TFG/analysis', 'c:/TFG/scratch']
import re

print("filename | script | command | output folder")
for file_to_find in files:
    stem = os.path.splitext(file_to_find)[0]
    stem = stem.replace('(', '').replace(')', '').replace(' ', '_')
    stem = stem.replace('_shot_88653', '').replace('_88653', '')
    if 'heating_vs_mirnov' in stem: stem = 'heating_vs_mirnov'
    if 'amplitude_scaling' in stem: stem = 'amplitude_scaling'
    
    found = False
    for d in dirs_to_search:
        for root, _, fs in os.walk(d):
            for f in fs:
                if f.endswith('.py') or f.endswith('.sh') or f.endswith('.ps1'):
                    path = os.path.join(root, f)
                    try:
                        with open(path, 'r', encoding='utf-8') as fh:
                            content = fh.read()
                            if file_to_find in content or stem in content:
                                print(f"{file_to_find} | {path}")
                                found = True
                    except:
                        pass
