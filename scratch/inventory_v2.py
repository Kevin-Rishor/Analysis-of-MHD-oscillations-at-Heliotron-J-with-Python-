import os
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import hashlib
import difflib

repo_dir = Path(r"c:\TFG")
target_exts = {".txt", ".tex", ".log", ".md"}
search_exts = {".txt", ".tex", ".log", ".md", ".py"}

base_core_names = {
    "presentation_claude.tex",
    "general_summary.log",
    "bispectral_analysis_conclusion.md",
    "predator_prey_sloshing_theory.md"
}

target_files = []
search_files = []

for p in repo_dir.rglob("*"):
    if ".git" in p.parts or ".venv" in p.parts or "archive" in p.parts:
        continue
    if p.is_dir():
        continue
    if p.suffix.lower() in target_exts:
        target_files.append(p)
    if p.suffix.lower() in search_exts:
        search_files.append(p)

file_contents = {}
for p in search_files:
    try:
        with open(p, "r", encoding="utf-8") as f:
            file_contents[p] = f.read()
    except Exception:
        try:
            with open(p, "r", encoding="latin-1") as f:
                file_contents[p] = f.read()
        except Exception:
            file_contents[p] = ""

# --- FIX 1: TOPOLOGICAL CORE RESOLUTION ---
roots = [f for f in search_files if f.name in base_core_names]
reachable = set(roots)
changed = True
while changed:
    changed = False
    for f in search_files:
        if f in reachable: continue
        # A file is added to reachable if any ALREADY REACHABLE file cites it
        t_name = f.name
        for r in reachable:
            if t_name in file_contents[r]:
                reachable.add(f)
                changed = True
                break

core_files = {f for f in target_files if f in reachable}

# Track exactly WHO cites each file for the output
cited_by = defaultdict(list)
for f in core_files:
    if f.name in base_core_names:
        continue
    for r in reachable:
        if r != f and f.name in file_contents[r]:
            cited_by[f].append(r)


# --- FIX 2: DUPLICATES IDENTIFICATION ---
target_by_name = defaultdict(list)
for f in target_files:
    target_by_name[f.name].append(f)

def get_hash(fpath):
    with open(fpath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

duplicates_report = []
for name, paths in target_by_name.items():
    if len(paths) > 1:
        hashes = {p: get_hash(p) for p in paths}
        unique_hashes = set(hashes.values())
        if len(unique_hashes) == 1:
            duplicates_report.append(f"Filename: {name}\n  - All {len(paths)} copies are BYTE-IDENTICAL.\n  - Paths: {', '.join([str(p) for p in paths])}")
        else:
            duplicates_report.append(f"Filename: {name}\n  - Contains DIFFERENT content across copies. Needs human decision.")
            for p in paths:
                h = hashes[p]
                extra = ""
                if name == "mhd_obj2_88653.log":
                    if "obj2_dynamics_and_structure" in p.parts:
                        extra = " [Corresponds to ACTIVE script: obj2_dynamics_and_structure\\mhd_analysis_obj2.py]"
                    elif p.parent.name == "analysis":
                        extra = " [Corresponds to ORPHANED script: analysis\\mhd_analysis_obj2.py]"
                duplicates_report.append(f"  - {p} (Hash: {h[:8]}){extra}")

# --- LINEAGES AND SCRATCH ---
def get_base_name(f):
    stem = f.stem
    base = stem
    base = re.sub(r'(_[vV]\d+)?(_backup)?(_copy)?(_DRAFT)?(_fresh)?(_LF)?$', '', base)
    return base

all_by_base = defaultdict(list)
for f in target_files:
    all_by_base[get_base_name(f)].append(f)

superseded_groups = {}
scratch_list = []

for b, group in all_by_base.items():
    if len(group) > 1:
        core_in_group = [f for f in group if f in core_files]
        if core_in_group:
            core_in_group.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            kept = core_in_group[0]
            removed = [f for f in group if f not in core_files]
            if removed:
                superseded_groups[b] = {'kept': kept, 'removed': removed, 'reason': 'CORE'}
        else:
            group.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            kept = group[0]
            removed = group[1:]
            superseded_groups[b] = {'kept': kept, 'removed': removed, 'reason': 'MOD_TIME'}
    else:
        f = group[0]
        if f not in core_files:
            scratch_list.append(f)

# --- FIX 3: SUPERSEDED DIFFS ---
diff_reports = []
for b, d in superseded_groups.items():
    kept = d['kept']
    with open(kept, 'r', encoding='utf-8', errors='replace') as kf:
        kept_lines = kf.readlines()
        
    for rm in d['removed']:
        with open(rm, 'r', encoding='utf-8', errors='replace') as rf:
            rm_lines = rf.readlines()
            
        diff = list(difflib.unified_diff(rm_lines, kept_lines, fromfile=str(rm), tofile=str(kept), n=3))
        diff_hdr = f"Lineage '{b}': [REMOVE] {rm.name} vs [KEEP] {kept.name}"
        
        if not diff:
            diff_reports.append(f"{diff_hdr}\n  - BYTE-IDENTICAL. (Safe to remove older).")
        else:
            adds = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
            dels = sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))
            diff_reports.append(f"{diff_hdr}")
            diff_reports.append(f"  - Older file has {dels} unique lines, Newer file has {adds} unique lines.")
            if dels > 0 and adds == 0:
                diff_reports.append("  - MEANINGFUL DIFFERENCE: The older file has content that was DROPPED in the newer file!")
            elif dels > 0 and adds > 0:
                diff_reports.append("  - MEANINGFUL DIFFERENCE: Both files have unique content. The newer file may have dropped or rewritten old content.")
            else:
                diff_reports.append("  - The newer file strictly adds or modifies content without dropping unique lines from the older (dels=0).")
            
            snippet = "".join(diff[:15])
            diff_reports.append("  --- Diff Snippet ---")
            diff_reports.append(snippet.rstrip())
            if len(diff) > 15:
                diff_reports.append(f"  ... (+ {len(diff)-15} more lines)")
            diff_reports.append("  --------------------")


def format_file(f):
    st = f.stat()
    sz = st.st_size
    dt = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return f"{f} (Size: {sz} bytes, Modified: {dt})"

out_lines = []

out_lines.append("=== DUPLICATES ANALYSIS ===")
for d in duplicates_report:
    out_lines.append(d)

out_lines.append("\n=== CATEGORY: CORE ===")
out_lines.append("These files are foundational or actively cited topologically tracing back to the core files.")
for f in sorted(list(core_files)):
    if f.name in base_core_names:
        out_lines.append(f"- {format_file(f)}\n  Reason: Base CORE file")
    else:
        refs = cited_by[f]
        out_lines.append(f"- {format_file(f)}\n  Reason: Topologically cited by -> {', '.join([r.name for r in refs])}")

out_lines.append("\n=== CATEGORY: SUPERSEDED-BY-A-LATER-VERSION ===")
out_lines.append("Files with an obvious version lineage where a later or CORE version exists.")
for b in sorted(superseded_groups.keys()):
    d = superseded_groups[b]
    out_lines.append(f"\nLineage Base: '{b}'")
    kept = d['kept']
    reason_str = "Already in CORE" if d['reason'] == 'CORE' else "Picked by newest modification time"
    out_lines.append(f"  [KEEP] {format_file(kept)} - {reason_str}")
    for r in d['removed']:
        out_lines.append(f"  [REMOVE] {format_file(r)}")

out_lines.append("\n=== SUPERSEDED DIFFS ===")
for dr in diff_reports:
    out_lines.append(dr)

out_lines.append("\n=== CATEGORY: SCRATCH / ONE-OFF ===")
out_lines.append("Unreferenced singletons in scratch/ or log files serving temporary purposes.")
for f in sorted(scratch_list):
    out_lines.append(f"- {format_file(f)}")

out_lines.append("\n=== PROPOSED MOVE LIST (c:\\TFG\\archive\\...) ===")
total_files = 0
total_size = 0
moves = []

def add_move(f):
    global total_files, total_size
    rel_path = f.relative_to(repo_dir)
    dest = repo_dir / "archive" / rel_path
    moves.append(f"{f} -> {dest}")
    total_files += 1
    total_size += f.stat().st_size

for d in superseded_groups.values():
    for r in d['removed']:
        add_move(r)

for f in scratch_list:
    add_move(f)

for m in sorted(moves):
    out_lines.append(m)

out_lines.append(f"\nTotal files to move: {total_files}")
out_lines.append(f"Total size to move: {total_size} bytes")

with open(repo_dir / "cleanup_inventory_v2.log", "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))
