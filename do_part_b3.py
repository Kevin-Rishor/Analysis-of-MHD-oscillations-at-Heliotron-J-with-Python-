import json
from pathlib import Path

repo_dir = Path('C:/TFG')
out_dir = Path('C:/Users/kevin/.gemini/antigravity/brain/8c48972e-b334-4ce7-a039-3891fa50fa1d/scratch')

# Load the phase 1 inventory
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
        o = line.strip()[3:-1]
        orphans.append(o)

# Filter out generate_audit_v*.py
orphans = [o for o in orphans if not ('generate_audit_v' in o and o.endswith('.py'))]

md = []
md.append("\n\n## Fate of Orphan Candidates\n")
md.append("| Current Path | Proposed Destination | Confidence | Reason |")
md.append("|---|---|---|---|")

for o in orphans:
    p = o.lower()
    o_path = repo_dir / o
    
    if 'thesis.pdf' in p or 'presentation.pdf' in p or 'reorg_phase' in p:
        dest = "keep — likely a final output"
        conf = "High"
        reason = "Root output document"
    elif 'recent_changes.diff' in p or 'search_results.txt' in p or 'temp_summary' in p or p.endswith('.log'):
        dest = "docs/audit_history/ or archive/"
        conf = "High"
        reason = "Log or temporary output"
    elif o.startswith('scratch/') or o.startswith('Images_prev/'):
        dest = "archive/"
        conf = "High"
        reason = "Already part of bulk archive plan"
    elif o.startswith('.continue/'):
        dest = "archive/ or keep"
        conf = "Medium"
        reason = "IDE configuration"
    elif o.endswith('.json') and ('discrete_modes' in p or 'primary_mode' in p or 'mode_mixing' in p):
        dest = "keep — likely a final output"
        conf = "High"
        reason = "Data output from scripts"
    elif 'v1_backup' in p:
        dest = "archive/superseded_duplicates/"
        conf = "High"
        reason = "Backup file"
    elif 'presentation_claude' in p or 'run_all_objectives.py' in p or 'generate_audit.py' in p:
        dest = "keep"
        conf = "Medium"
        reason = "Orchestrator or presentation source"
    else:
        dest = "archive/orphans/"
        conf = "Medium"
        reason = "No references found in codebase"
        
    md.append(f"| {o} | {dest} | {conf} | {reason} |")

# Append to plan
with open(repo_dir / 'reorg_phase2_image_plan.md', 'a', encoding='utf-8') as f:
    f.write('\n'.join(md))

print("Appended orphans to image plan")
