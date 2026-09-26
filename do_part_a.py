import os
import shutil
from pathlib import Path

repo_dir = Path('C:/TFG')
log_file = repo_dir / 'reorg_phase2_log.md'

with open(log_file, 'w', encoding='utf-8') as f:
    f.write('# Reorganization Phase 2 Log\n\n')

def log_action(msg):
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f'- {msg}\n')
    print(msg)

# 1 & 2: Superseded duplicates
archive_dup_dir = repo_dir / 'archive' / 'superseded_duplicates'
archive_dup_dir.mkdir(parents=True, exist_ok=True)

root_dups = [
    'analysis/mhd_analysis_obj1.py',
    'analysis/mhd_analysis_obj2.py',
    'analysis/mhd_analysis_obj3.py',
    'analysis/mhd_analysis_obj3_v1_backup.py',
    'analysis/mhd_common.py'
]

for p in root_dups:
    src = repo_dir / p
    if src.exists():
        dst = archive_dup_dir / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src, dst)
        log_action(f'Moved {p} to {dst.relative_to(repo_dir)} (Verified dead root-level duplicate)')

# Find all *_v1_backup.* and *_backup.* files
for root, dirs, files in os.walk(repo_dir):
    if 'archive' in Path(root).parts or '.git' in root or '.venv' in root:
        continue
    for f in files:
        if ('_v1_backup' in f or f.endswith('_backup.png') or f.endswith('_backup.py') or '_backup' in f) and f != 'reorg_phase2_log.md':
            src = Path(root) / f
            rel_src = src.relative_to(repo_dir)
            dst = archive_dup_dir / rel_src
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(src, dst)
            log_action(f'Moved {rel_src} to {dst.relative_to(repo_dir)} (Backup file archiving)')

# 3: Audit history clutter
audit_dir = repo_dir / 'docs' / 'audit_history'
audit_dir.mkdir(parents=True, exist_ok=True)

leave_alone = {'generate_audit_v6.py', 'thesis_audit_v6.log', 'reorg_phase1_inventory.md', 'reorg_phase2_log.md'}

clutter_patterns = [
    'thesis_audit', 'log_audit', 'general_summary', 'cleanup_inventory',
    'shot_88654_', 'objective3_within_episode_scaling', 'ece03_validity_check',
    'phase1_verification.log', 'slide11_self_coupling_check.log',
    'generate_audit', 'recent_changes.diff', 'search_results.txt', 'temp_summary_check.txt'
]

for f in os.listdir(repo_dir):
    p = repo_dir / f
    if p.is_file() and f not in leave_alone:
        if any(f.startswith(pat) or pat in f for pat in clutter_patterns):
            dst = audit_dir / f
            shutil.move(p, dst)
            log_action(f'Moved {f} to {dst.relative_to(repo_dir)} (Audit history consolidation)')

# 4: Grep for scratch/ and Images_prev/
def check_references(target_dir_name):
    # search for target_dir_name in files outside of it
    found_refs = []
    for r, d, fs in os.walk(repo_dir):
        rel_r = Path(r).relative_to(repo_dir)
        if target_dir_name in rel_r.parts or 'archive' in rel_r.parts or '.git' in r:
            continue
        for f in fs:
            if f.endswith(('.py', '.tex', '.md', '.txt')):
                p = Path(r) / f
                try:
                    content = p.read_text(encoding='utf-8')
                    if f'{target_dir_name}/' in content or f'{target_dir_name}\\' in content:
                        found_refs.append(f'{target_dir_name} referenced in {p.relative_to(repo_dir)}')
                except:
                    pass
    return found_refs

scratch_refs = check_references('scratch')
if scratch_refs:
    for ref in scratch_refs:
        log_action(f'HALTED archiving scratch: {ref}')
else:
    src = repo_dir / 'scratch'
    if src.exists():
        dst = repo_dir / 'archive' / 'scratch'
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src, dst)
        log_action(f'Moved scratch/ to rchive/scratch/ (No external references found)')

images_prev_refs = check_references('Images_prev')
if images_prev_refs:
    for ref in images_prev_refs:
        log_action(f'HALTED archiving Images_prev: {ref}')
else:
    src = repo_dir / 'Images_prev'
    if src.exists():
        dst = repo_dir / 'archive' / 'Images_prev'
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src, dst)
        log_action(f'Moved Images_prev/ to rchive/Images_prev/ (No external references found)')

print('Part A Script Complete')
