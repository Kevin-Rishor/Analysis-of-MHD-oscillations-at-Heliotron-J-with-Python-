import re
from pathlib import Path

repo_dir = Path('C:/TFG')
tex_files = ['Theoretical framework.tex', 'Methodology.tex', 'presentation_claude.tex']

# 1. Parse \includegraphics
graphics = {} # map image_filename -> list of (tex_file, exact_path)
for tf in tex_files:
    p = repo_dir / tf
    if p.exists():
        content = p.read_text(encoding='utf-8')
        # match \includegraphics[...]{path} or \includegraphics{path}
        # \includegraphics[width=0.8\linewidth]{Images/mhd_...png}
        matches = re.findall(r'\\includegraphics(?:\[.*?\])?\{([^}]+)\}', content)
        for m in matches:
            basename = Path(m).name
            graphics.setdefault(basename, []).append((tf, m))

print("### 1. \\includegraphics calls in specified .tex files")
print("| Image Filename | Included? | .tex File(s) | Exact Relative Path |")
print("|---|---|---|---|")
for basename, usages in sorted(graphics.items()):
    files = ", ".join(sorted(set(u[0] for u in usages)))
    paths = ", ".join(sorted(set(u[1] for u in usages)))
    print(f"| {basename} | Yes | {files} | {paths} |")
if not graphics:
    print("No \\includegraphics calls found.")

print("\n### 2. Collisions (mhd_bicoherence & mhd_burst_window_detection)")
targets_2 = ['mhd_bicoherence_objective3_88653.png', 'mhd_burst_window_detection_88653.png']
for t in targets_2:
    if t in graphics:
        paths = set(u[1] for u in graphics[t])
        is_images_copy = any(p.startswith('Images/') or p.startswith('Images\\') for p in paths)
        is_root_copy = any(not (p.startswith('Images/') or p.startswith('Images\\')) for p in paths)
        
        msg = f"{t}: "
        if is_images_copy and not is_root_copy:
            msg += "The older Images/ copy is actually cited (thesis PDF is using a stale figure)."
        elif is_root_copy and not is_images_copy:
            msg += "The newer root copy is cited."
        else:
            msg += f"Mixed citations: {paths}"
        print(msg)
    else:
        print(f"{t}: Not cited in these .tex files.")

print("\n### 3. Missing Source Script Images")
targets_3 = ['amplitude_scaling_all_vs_valid.png', 'amplitude_scaling_valid16.png', 'sign_vs_threshold_88653.png']
for t in targets_3:
    if t in graphics:
        print(f"{t}: FIGURE IN USE, SOURCE SCRIPT MISSING. (Do not archive, keeps its place).")
    else:
        print(f"{t}: Not included. Genuine archive candidate.")

print("\n### 4. Mode Mixing Images")
targets_4 = ['mhd_mode_mixing_shot_88653_HF_90kHz.png', 'mhd_mode_mixing_shot_88653_LF_43kHz.png']
for t in targets_4:
    if t in graphics:
        print(f"{t}: FIGURE IN USE.")
    else:
        print(f"{t}: Not included.")

print("\n### 5. Preview Diffs for .tex Updates")
# Read image plan to see what is slated to move
plan_lines = (repo_dir / 'reorg_phase2_image_plan.md').read_text(encoding='utf-8').split('\n')
move_map = {} # basename -> new_folder (e.g. Images/obj1_mode_identification)
for line in plan_lines:
    if line.startswith('| ') and not line.startswith('| Current'):
        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= 4:
            current_path = parts[1]
            dest = parts[2]
            if dest.startswith('Images/obj'):
                basename = Path(current_path).name
                move_map[basename] = dest

# Prepare diffs
import difflib
for tf in tex_files:
    p = repo_dir / tf
    if not p.exists(): continue
    original_lines = p.read_text(encoding='utf-8').splitlines(keepends=True)
    new_lines = []
    changed = False
    
    for line in original_lines:
        new_line = line
        matches = re.finditer(r'(\\includegraphics(?:\[.*?\])?\{)([^}]+)(\})', line)
        
        # We need to replace from the end to not mess up indices, or just string replace if unique
        # String replace is safer for simple paths
        for m in matches:
            full_match = m.group(0)
            prefix = m.group(1)
            old_path = m.group(2)
            suffix = m.group(3)
            
            basename = Path(old_path).name
            if basename in move_map and basename in graphics:
                # Slated to move AND currently included
                # New path should be: proposed_dest/basename
                dest_folder = move_map[basename]
                if not dest_folder.endswith('/'): dest_folder += '/'
                new_path = dest_folder + basename
                new_line = new_line.replace(old_path, new_path)
                changed = True
                
        new_lines.append(new_line)
        
    if changed:
        print(f"\n--- Proposed Edits for {tf} ---")
        diff = difflib.unified_diff(original_lines, new_lines, fromfile=tf, tofile=tf)
        for d_line in diff:
            print(d_line.strip('\r\n'))

