import os
import re
from pathlib import Path

repo_dir = Path('C:/TFG')

# Find all python files in the analysis objectives folders
obj_scripts = {
    'Obj1': list((repo_dir / 'analysis' / 'obj1_mode_identification').glob('*.py')),
    'Obj2': list((repo_dir / 'analysis' / 'obj2_dynamics_and_structure').glob('*.py')),
    'Obj3': list((repo_dir / 'analysis' / 'obj3_nonlinear_coupling').glob('*.py')),
}
# Check heating comparison too if relevant, though we can map it to Obj2?
# For now just use the 3 objectives.

image_patterns = {}
for obj, scripts in obj_scripts.items():
    for script in scripts:
        content = script.read_text(encoding='utf-8')
        # match savefig('...') or savefig(f'...') or output = '...' + safe_savefig(output)
        # It's tricky to parse all dynamically. Let's extract all strings ending with .png or .pdf
        # or things like mhd_analysis_objective2_{shot}_{args.lower}_{args.upper}kHz.png
        # Let's extract any string literal that looks like a filename prefix.
        strings = re.findall(r'["\']([^"\']+\.png)["\']', content)
        strings += re.findall(r'["\']([^"\']+)["\']', content)
        
        # also match typical patterns manually
        if obj == 'Obj1':
            image_patterns[obj] = image_patterns.get(obj, []) + ['mhd_analysis_objective1', 'mhd_primary_mode_radial_profile', 'mhd_primary_mode_toroidal_n']
        elif obj == 'Obj2':
            image_patterns[obj] = image_patterns.get(obj, []) + [
                'mhd_analysis_objective2', 'heating_vs_mirnov', 'mhd_b_tilde_vs_envelope',
                'mhd_mode_envelopes_and_ip_evolution', 'mhd_hysteresis_ip_vs_envelope',
                'mhd_modulator_dual_mode', 'mhd_self_coupling_coherence', 'mhd_snr_halpha_correlation',
                'mhd_mode_spatial_map_2d'
            ]
        elif obj == 'Obj3':
            image_patterns[obj] = image_patterns.get(obj, []) + [
                'mhd_bicoherence_objective3', 'mhd_burst_window_detection', 'mhd_asymptotic_bias_scaling',
                'bicoherence_sliding_trace_snr'
            ]

all_images = []
for p in repo_dir.glob('*.png'):
    all_images.append(p)
for p in (repo_dir / 'Images').glob('*.png'):
    all_images.append(p)
for p in repo_dir.glob('*.jpg'):
    all_images.append(p)
for p in (repo_dir / 'Images').glob('*.jpg'):
    all_images.append(p)

plan_md = []
plan_md.append("# Phase 2: Image Reorganization Plan\n")
plan_md.append("| Current Path | Proposed Destination | Producing Script / Prefix | Confidence |")
plan_md.append("|---|---|---|---|")

for img in all_images:
    rel_path = img.relative_to(repo_dir).as_posix()
    name = img.name
    
    proposed_obj = 'Unknown'
    confidence = 'Low'
    reason = 'No match'
    
    for obj, patterns in image_patterns.items():
        for pat in patterns:
            if pat in name:
                proposed_obj = f"Images/{obj.lower().replace('obj', 'obj')}_folder" # we will properly map it below
                if obj == 'Obj1': proposed_obj = 'Images/obj1_mode_identification/'
                if obj == 'Obj2': proposed_obj = 'Images/obj2_dynamics_and_structure/'
                if obj == 'Obj3': proposed_obj = 'Images/obj3_nonlinear_coupling/'
                confidence = 'High'
                reason = pat
                break
        if confidence == 'High':
            break
            
    # some fallbacks
    if confidence == 'Low':
        if 'obj1' in name.lower():
            proposed_obj = 'Images/obj1_mode_identification/'
            confidence = 'Medium'
            reason = 'Filename keyword'
        elif 'obj2' in name.lower() or 'lissajous' in name.lower():
            proposed_obj = 'Images/obj2_dynamics_and_structure/'
            confidence = 'Medium'
            reason = 'Filename keyword'
        elif 'obj3' in name.lower() or 'bicoherence' in name.lower():
            proposed_obj = 'Images/obj3_nonlinear_coupling/'
            confidence = 'Medium'
            reason = 'Filename keyword'
        else:
            proposed_obj = 'Keep in place'
            
    plan_md.append(f"| {rel_path} | {proposed_obj} | {reason} | {confidence} |")

(repo_dir / 'reorg_phase2_image_plan.md').write_text('\n'.join(plan_md), encoding='utf-8')
print("Image plan generated")
