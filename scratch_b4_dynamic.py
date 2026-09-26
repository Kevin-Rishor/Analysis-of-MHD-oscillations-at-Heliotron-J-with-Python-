import os
import re
from pathlib import Path

repo_dir = Path('C:/TFG')
for py_f in (repo_dir / 'analysis').rglob('*.py'):
    c = py_f.read_text(encoding='utf-8')
    if 'mode_mixing' in c:
        print(f"Found 'mode_mixing' in {py_f.relative_to(repo_dir)}")
        for line in c.split('\n'):
            if 'mode_mixing' in line and ('savefig' in line or 'png' in line or '=' in line):
                print(f"  {line.strip()}")
    if 'amplitude_scaling' in c:
        print(f"Found 'amplitude_scaling' in {py_f.relative_to(repo_dir)}")
        for line in c.split('\n'):
            if 'amplitude_scaling' in line and ('savefig' in line or 'png' in line or '=' in line):
                print(f"  {line.strip()}")
