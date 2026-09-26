import os
from pathlib import Path

repo_dir = Path('C:/TFG')
targets = [
    'amplitude_scaling_all_vs_valid.png',
    'amplitude_scaling_valid16.png',
    'mhd_mode_mixing_shot_88653_HF_90kHz.png',
    'mhd_mode_mixing_shot_88653_LF_43kHz.png',
    'sign_vs_threshold_88653.png'
]

py_files = list(repo_dir.rglob('*.py'))
found = {}
for py_f in py_files:
    if 'archive' in py_f.parts: continue
    try:
        content = py_f.read_text(encoding='utf-8')
        for t in targets:
            if t in content or t.replace('.png', '') in content:
                found.setdefault(t, set()).add(py_f.relative_to(repo_dir).as_posix())
    except:
        pass

for t in targets:
    if t in found:
        print(f"{t} -> Found in: {', '.join(found[t])}")
    else:
        print(f"{t} -> Not found anywhere")
