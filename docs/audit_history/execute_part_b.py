import os
import shutil
import re
from pathlib import Path

repo_dir = Path('C:/TFG')
log_file = repo_dir / 'reorg_phase2_log.md'
tex_file = repo_dir / 'presentation_claude.tex'

def log_action(msg):
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f'- {msg}\n')
    print(msg)

def replace_tex(old_str, new_str):
    content = tex_file.read_text(encoding='utf-8')
    if old_str in content:
        content = content.replace(old_str, new_str)
        tex_file.write_text(content, encoding='utf-8')
        return True
    return False

# 1. Lissajous contradiction
archive_orphans = repo_dir / 'archive' / 'orphans'
archive_orphans.mkdir(parents=True, exist_ok=True)
lissajous = [
    'Images/mhd_lissajous_independent_windows_shot_88653.png',
    'Images/mhd_lissajous_phasebinned_real_shot_88653.png',
    'Images/mhd_lissajous_phasebinned_surrogateA_quietHF_shot_88653.png',
    'Images/mhd_lissajous_phasebinned_surrogateB_randHF_shot_88653.png'
]
for l in lissajous:
    src = repo_dir / l
    if src.exists():
        dst = archive_orphans / src.name
        shutil.move(src, dst)
        log_action(f'Moved {l} to {dst.relative_to(repo_dir)} (Lissajous unused in .tex)')

# 2. Heating phase comparison
heat_dir = repo_dir / 'Images' / 'heating_phase_comparison'
heat_dir.mkdir(parents=True, exist_ok=True)

# Find generated images by scanning heating_phase_comparison/*.py
heating_py = list((repo_dir / 'analysis' / 'heating_phase_comparison').glob('*.py'))
heat_images = set()
for py in heating_py:
    content = py.read_text(encoding='utf-8')
    # find explicit .png strings
    for m in re.findall(r'["\']([^"\']+\.png)["\']', content):
        heat_images.add(Path(m).name)
    # find dynamic prefixes if obvious
    if 'zoomed_spectrogram_3sigma' in content:
        for p in repo_dir.rglob('zoomed_spectrogram_3sigma*.png'):
            heat_images.add(p.name)
    if 'mhd_analysis_objective1_88653_' in content:
        heat_images.add('mhd_analysis_objective1_88653_ECH_NBI.png')
        heat_images.add('mhd_analysis_objective1_88653_NBI_only.png')
    if 'mhd_analysis_objective2_structure_' in content:
        heat_images.add('mhd_analysis_objective2_structure_88653_40_80kHz_ECH_NBI.png')
        heat_images.add('mhd_analysis_objective2_structure_88653_40_80kHz_NBI_only.png')
        heat_images.add('mhd_analysis_objective2_structure_88653_80_120kHz_ECH_NBI.png')
        heat_images.add('mhd_analysis_objective2_structure_88653_80_120kHz_NBI_only.png')
    if 'mhd_bicoherence_objective3_' in content:
        heat_images.add('mhd_bicoherence_objective3_88653_ECH_NBI.png')
        heat_images.add('mhd_bicoherence_objective3_88653_NBI_only.png')
    if 'mhd_mode_envelopes_and_ip_ECH_NBI_comparison.png' in content:
        heat_images.add('mhd_mode_envelopes_and_ip_ECH_NBI_comparison.png')

# Move them
for name in heat_images:
    for loc in [repo_dir / name, repo_dir / 'Images' / name]:
        if loc.exists():
            dst = heat_dir / name
            shutil.move(loc, dst)
            log_action(f'Moved {loc.relative_to(repo_dir)} to {dst.relative_to(repo_dir)} (Heating phase comparison script output)')
            old_tex = f'Images/{name}'
            new_tex = f'Images/heating_phase_comparison/{name}'
            if replace_tex(old_tex, new_tex):
                log_action(f'Updated presentation_claude.tex: {old_tex} -> {new_tex}')

# 3. Mode mixing
obj3_dir = repo_dir / 'analysis' / 'obj3_nonlinear_coupling'
obj3_img = repo_dir / 'Images' / 'obj3_nonlinear_coupling'
obj3_img.mkdir(parents=True, exist_ok=True)

src_py = repo_dir / 'analysis' / 'mhd_mode_mixing_analysis.py'
if src_py.exists():
    dst_py = obj3_dir / 'mhd_mode_mixing_analysis.py'
    shutil.move(src_py, dst_py)
    log_action(f'Moved {src_py.relative_to(repo_dir)} to {dst_py.relative_to(repo_dir)} (Mode mixing relocation)')

mix_imgs = ['mhd_mode_mixing_shot_88653_HF_90kHz.png', 'mhd_mode_mixing_shot_88653_LF_43kHz.png']
for m in mix_imgs:
    src = repo_dir / 'Images' / m
    if not src.exists(): src = repo_dir / m
    if src.exists():
        dst = obj3_img / m
        shutil.move(src, dst)
        log_action(f'Moved {src.relative_to(repo_dir)} to {dst.relative_to(repo_dir)} (Mode mixing relocation)')
        replace_tex(f'Images/{m}', f'Images/obj3_nonlinear_coupling/{m}')
        log_action(f'Updated presentation_claude.tex for {m}')

# 4. Refresh stale figures
stale = ['mhd_bicoherence_objective3_88653.png', 'mhd_burst_window_detection_88653.png']
for s in stale:
    root_copy = repo_dir / s
    img_copy = repo_dir / 'Images' / s
    dst = obj3_img / s
    if root_copy.exists():
        if img_copy.exists():
            img_copy.unlink() # Delete old
            log_action(f'Deleted stale {img_copy.relative_to(repo_dir)}')
        shutil.move(root_copy, dst)
        log_action(f'Moved {root_copy.relative_to(repo_dir)} to {dst.relative_to(repo_dir)} (Refresh stale figure)')
        replace_tex(f'Images/{s}', f'Images/obj3_nonlinear_coupling/{s}')
        log_action(f'Updated presentation_claude.tex for {s}')

# 5. Rest of image plan (including dedupe)
plan_lines = (repo_dir / 'reorg_phase2_image_plan.md').read_text(encoding='utf-8').split('\n')
for line in plan_lines:
    if line.startswith('| ') and not line.startswith('| Current'):
        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= 4:
            current_path = parts[1]
            dest_folder = parts[2]
            
            # Skip orphans since we handle them in step 6
            if dest_folder.startswith('archive/orphans') or dest_folder.startswith('docs/') or dest_folder.startswith('keep'):
                continue
                
            basename = Path(current_path).name
            if basename in lissajous or basename in heat_images or basename in mix_imgs or basename in stale:
                continue # Handled
            
            # 12 identical collisions deduplication
            root_c = repo_dir / basename
            img_c = repo_dir / 'Images' / basename
            if root_c.exists() and img_c.exists():
                # we confirmed them identical in the previous script (except the 2 stale which we handled)
                root_c.unlink()
                log_action(f'Deleted redundant root copy: {basename}')
                
            active_src = repo_dir / current_path
            if not active_src.exists():
                active_src = repo_dir / 'Images' / basename
            
            if active_src.exists() and dest_folder.startswith('Images/obj'):
                d_dir = repo_dir / dest_folder
                d_dir.mkdir(parents=True, exist_ok=True)
                dst = d_dir / basename
                shutil.move(active_src, dst)
                log_action(f'Moved {active_src.relative_to(repo_dir)} to {dst.relative_to(repo_dir)} (Image plan execution)')
                
                # Update tex if referenced
                old_tex = f'Images/{basename}'
                new_tex = f'{dest_folder}/{basename}'
                if replace_tex(old_tex, new_tex):
                    log_action(f'Updated presentation_claude.tex: {old_tex} -> {new_tex}')

# 6. Vetted orphan table execution
in_orphans = False
for line in plan_lines:
    if line.startswith('## Fate of Orphan Candidates'):
        in_orphans = True
        continue
    if in_orphans and line.startswith('| ') and not line.startswith('| Current'):
        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= 5:
            current_path = parts[1]
            dest_folder = parts[2]
            
            if 'archive/orphans' in dest_folder:
                # check lissajous exception, but they are already handled!
                if Path(current_path).name in [Path(l).name for l in lissajous]:
                    continue
                src = repo_dir / current_path
                if src.exists():
                    dst = archive_orphans / src.name
                    shutil.move(src, dst)
                    log_action(f'Moved {current_path} to {dst.relative_to(repo_dir)} (Orphan execution)')

# 7. Scratch / Images_prev
def is_genuine_ref(target):
    for r, d, fs in os.walk(repo_dir):
        rel_r = Path(r).relative_to(repo_dir)
        if target in rel_r.parts or 'archive' in rel_r.parts or '.git' in r:
            continue
        for f in fs:
            if f.endswith(('.py', '.tex', '.md', '.txt')):
                p = Path(r) / f
                if p.name in ['reorg_phase1_inventory.md', 'reorg_phase2_log.md', 'reorg_phase2_image_plan.md', 'do_part_a.py']:
                    continue
                if p.name.startswith('scratch_') or p.name.startswith('do_part_'):
                    continue
                try:
                    c = p.read_text(encoding='utf-8')
                    if f'{target}/' in c or f'{target}\\' in c:
                        return f'{target} referenced in {p.relative_to(repo_dir)}'
                except:
                    pass
    return None

scratch_ref = is_genuine_ref('scratch')
if not scratch_ref:
    src = repo_dir / 'scratch'
    if src.exists():
        dst = repo_dir / 'archive' / 'scratch'
        shutil.move(src, dst)
        log_action('Moved scratch/ to archive/scratch/ (No genuine references)')
else:
    log_action(f'HALTED scratch: {scratch_ref}')

imgprev_ref = is_genuine_ref('Images_prev')
if not imgprev_ref:
    src = repo_dir / 'Images_prev'
    if src.exists():
        dst = repo_dir / 'archive' / 'Images_prev'
        shutil.move(src, dst)
        log_action('Moved Images_prev/ to archive/Images_prev/ (No genuine references)')
else:
    log_action(f'HALTED Images_prev: {imgprev_ref}')

# 8. Sweep leftover reorg tooling
audit_dir = repo_dir / 'docs' / 'audit_history'
tooling = [
    'do_part_a.py', 'do_part_b1.py', 'do_part_b3.py', 
    'scratch_b4.py', 'scratch_b4_extra.py', 'scratch_b4_dynamic.py', 
    'scratch_diff.py', 'scratch_b5.py', 'execute_part_b.py'
]
for root, dirs, files in os.walk(repo_dir):
    if 'archive' in Path(root).parts or '.git' in root or 'docs' in Path(root).parts:
        continue
    for f in files:
        if f in tooling:
            src = Path(root) / f
            dst = audit_dir / f
            shutil.move(src, dst)
            log_action(f'Moved {src.relative_to(repo_dir)} to {dst.relative_to(repo_dir)} (Reorg tooling sweep)')

print("Part B completed.")
