import os
import hashlib
from pathlib import Path

repo_dir = Path('C:/TFG')
md_plan = repo_dir / 'reorg_phase2_image_plan.md'

# 1. Regenerate Orphan Table
inv_md = (repo_dir / 'reorg_phase1_inventory.md').read_text(encoding='utf-8')
lines = inv_md.split('\n')

orphans = []
in_orphans = False
for line in lines:
    if line.startswith('### Orphan Candidates List'):
        in_orphans = True
        continue
    if line.startswith('## 6.'):
        in_orphans = False
    
    if in_orphans and line.startswith('- '):
        # The line might have backticks. We should robustly strip spaces and backticks.
        o = line[2:].strip().replace('', '')
        orphans.append(o)

# Filter out generate_audit_v*.py
orphans = [o for o in orphans if not ('generate_audit_v' in o and o.endswith('.py'))]

valid_orphans = []
for o in orphans:
    p = repo_dir / o
    if p.exists():
        valid_orphans.append(o)

new_plan_lines = []
plan_content = md_plan.read_text(encoding='utf-8')
for line in plan_content.split('\n'):
    if line == '## Fate of Orphan Candidates':
        break
    new_plan_lines.append(line)

new_plan_lines.append("## Fate of Orphan Candidates")
new_plan_lines.append("| Current Path | Proposed Destination | Confidence | Reason |")
new_plan_lines.append("|---|---|---|---|")

for o in valid_orphans:
    p_lower = o.lower()
    if 'thesis.pdf' in p_lower or 'presentation.pdf' in p_lower or 'reorg_phase' in p_lower:
        dest = "keep — likely a final output"
        conf = "High"
        reason = "Root output document"
    elif 'recent_changes.diff' in p_lower or 'search_results.txt' in p_lower or 'temp_summary' in p_lower or p_lower.endswith('.log'):
        dest = "docs/audit_history/ or archive/"
        conf = "High"
        reason = "Log or temporary output"
    elif o.startswith('scratch/') or o.startswith('Images_prev/'):
        dest = "archive/"
        conf = "High"
        reason = "Bulk archive plan"
    elif o.startswith('.continue/'):
        dest = "archive/ or keep"
        conf = "Medium"
        reason = "IDE config"
    elif p_lower.endswith('.json'):
        dest = "keep — likely a final output"
        conf = "High"
        reason = "Data output"
    elif 'presentation_claude' in p_lower or 'run_all_objectives.py' in p_lower or 'generate_audit.py' in p_lower:
        dest = "keep"
        conf = "Medium"
        reason = "Orchestrator or source"
    else:
        dest = "archive/orphans/"
        conf = "Medium"
        reason = "No references"
    new_plan_lines.append(f"| {o} | {dest} | {conf} | {reason} |")

md_plan.write_text('\n'.join(new_plan_lines), encoding='utf-8')

print("--- 1. Orphan table regenerated ---")
print(f"Total valid orphans found and tabulated: {len(valid_orphans)}")

# 2. Image Collisions (root vs Images/)
print("\n--- 2. Image Collisions ---")
images_dir = repo_dir / 'Images'
root_images = list(repo_dir.glob('*.png')) + list(repo_dir.glob('*.jpg'))
collisions = []

for r_img in root_images:
    name = r_img.name
    i_img = images_dir / name
    if i_img.exists():
        r_size = r_img.stat().st_size
        i_size = i_img.stat().st_size
        
        if r_size == i_size:
            r_hash = hashlib.md5(r_img.read_bytes()).hexdigest()
            i_hash = hashlib.md5(i_img.read_bytes()).hexdigest()
            if r_hash == i_hash:
                collisions.append(f"Identical: {name} (propose deleting root copy, keeping Images/)")
            else:
                collisions.append(f"Different (Hash Mismatch): {name}\n  Root: {r_size} bytes, {r_img.stat().st_mtime}\n  Images/: {i_size} bytes, {i_img.stat().st_mtime}")
        else:
            from datetime import datetime
            r_dt = datetime.fromtimestamp(r_img.stat().st_mtime)
            i_dt = datetime.fromtimestamp(i_img.stat().st_mtime)
            collisions.append(f"Different: {name}\n  Root: {r_size} bytes, {r_dt}\n  Images/: {i_size} bytes, {i_dt}")

for c in collisions:
    print(c)
if not collisions:
    print("No collisions found.")

# 3. Resolve unmatched images
print("\n--- 3. Resolve Unmatched Images via Grep ---")
targets = [
    'amplitude_scaling_all_vs_valid.png',
    'amplitude_scaling_valid16.png',
    'mhd_mode_mixing_shot_88653_HF_90kHz.png',
    'mhd_mode_mixing_shot_88653_LF_43kHz.png',
    'sign_vs_threshold_88653.png',
    'zoomed_spectrogram_3sigma_'
]

py_files = list((repo_dir / 'analysis').rglob('*.py'))
found = {}
for py_f in py_files:
    content = py_f.read_text(encoding='utf-8')
    for t in targets:
        # Search for exact string or something very close to it
        if t in content:
            found.setdefault(t, set()).add(py_f.relative_to(repo_dir).as_posix())
        elif 'zoomed_spectrogram_3sigma_' in t and 'zoomed_spectrogram_3sigma' in content:
             found.setdefault(t, set()).add(py_f.relative_to(repo_dir).as_posix())

for t in targets:
    sources = found.get(t, [])
    if sources:
        print(f"{t} -> Produced by: {', '.join(sources)}")
    else:
        # try regex for dynamic f-strings
        dynamic_found = []
        import re
        if 'zoomed_spectrogram_3sigma' in t:
             for py_f in py_files:
                  c = py_f.read_text(encoding='utf-8')
                  if re.search(r'zoomed_spectrogram_3sigma.*\.png', c):
                       dynamic_found.append(py_f.relative_to(repo_dir).as_posix())
        if dynamic_found:
             print(f"{t} -> Produced dynamically by: {', '.join(set(dynamic_found))}")
        else:
             print(f"{t} -> NOT FOUND in any analysis/*.py script")

# 4. Check tex for mhd_lissajous_*.png
print("\n--- 4. Tex references for mhd_lissajous_*.png ---")
lissajous_images = [
    'mhd_lissajous_independent_windows_shot_88653.png',
    'mhd_lissajous_phasebinned_real_shot_88653.png',
    'mhd_lissajous_phasebinned_surrogateA_quietHF_shot_88653.png',
    'mhd_lissajous_phasebinned_surrogateB_randHF_shot_88653.png'
]

tex_files = list(repo_dir.rglob('*.tex'))
for img in lissajous_images:
    referenced = False
    for tex in tex_files:
        if 'archive' in tex.parts: continue
        try:
            content = tex.read_text(encoding='utf-8')
            if img in content:
                referenced = True
                print(f"Yes: {img} is referenced in {tex.name}")
                break
        except:
            pass
    if not referenced:
        print(f"No: {img} is NOT referenced in any .tex file")

print("\nAll findings computed.")
