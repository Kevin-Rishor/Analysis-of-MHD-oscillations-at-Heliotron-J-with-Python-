import os
import re
from pathlib import Path
from datetime import datetime

repo_dir = Path(r"c:\TFG")
target_exts = {".txt", ".tex", ".log", ".md"}
search_exts = {".txt", ".tex", ".log", ".md", ".py"}

# The base CORE files explicitly requested by the user
base_core_names = {
    "presentation_claude.tex",
    "general_summary.log",
    "bispectral_analysis_conclusion.md",
    "predator_prey_sloshing_theory.md"
}

target_files = []
search_files = []

# Gather files (ignoring .git, .venv, and archive if they exist)
for p in repo_dir.rglob("*"):
    if ".git" in p.parts or ".venv" in p.parts or "archive" in p.parts:
        continue
    if p.is_dir():
        continue
    if p.suffix.lower() in target_exts:
        target_files.append(p)
    if p.suffix.lower() in search_exts:
        search_files.append(p)

# Pre-load text contents for searching citations
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
            pass

# Determine cross-references: if a target file's name appears anywhere in another file
referenced_by = {f: [] for f in target_files}
for target in target_files:
    t_name = target.name
    for s_path, content in file_contents.items():
        if s_path != target and t_name in content:
            referenced_by[target].append(s_path)

core_files = set()
non_core_files = set()

for f in target_files:
    if f.name in base_core_names or len(referenced_by[f]) > 0:
        core_files.add(f)
    else:
        non_core_files.add(f)

# Group remaining non-core files by base lineage name
def get_base_name(f):
    stem = f.stem
    # Strip versions like _v1, _v2, _v1_backup, _backup, _copy, _DRAFT
    # Examples: general_summary_v2_DRAFT -> general_summary
    # objective3_within_episode_scaling_fresh -> objective3_within_episode_scaling
    # Try multiple sub passes for safety
    base = stem
    base = re.sub(r'(_[vV]\d+)?(_backup)?(_copy)?(_DRAFT)?(_fresh)?$', '', base)
    # also handle e.g. ece03_validity_check_v2 -> ece03_validity_check
    # if it just ended in _v2, the above takes care of it
    return base

all_by_base = {}
for f in target_files:
    b = get_base_name(f)
    if b not in all_by_base:
        all_by_base[b] = []
    all_by_base[b].append(f)

superseded_groups = {} # base -> {'kept': file, 'removed': [files]}
scratch_list = []

for b, group in all_by_base.items():
    if len(group) > 1:
        # Check if any in this group are CORE
        core_in_group = [f for f in group if f in core_files]
        if core_in_group:
            # We have at least one CORE file for this base.
            # All other files in this group that are NOT core are superseded.
            removed = [f for f in group if f not in core_files]
            if removed:
                # Keep the latest modified core file as the designated "kept" representation
                core_in_group.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                superseded_groups[b] = {'kept': core_in_group[0], 'removed': removed}
        else:
            # None are core. Sort by modification time to keep the newest.
            group.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            superseded_groups[b] = {'kept': group[0], 'removed': group[1:]}
    else:
        # Single file
        f = group[0]
        if f not in core_files:
            scratch_list.append(f)

# Let's verify that any file in "scratch_list" isn't actually a version of something in CORE?
# We already did a global base matching above, so if it shared a base, it would be in a group > 1.
# What if it's named log_audit_v1.log, but there is no other log_audit file? Then len(group) == 1,
# it ends up in scratch_list. That is correct per "round-specific investigation logs...".

def format_file(f):
    st = f.stat()
    sz = st.st_size
    dt = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return f"{f} (Size: {sz} bytes, Modified: {dt})"

out_lines = []
out_lines.append("=== CATEGORY: CORE ===")
out_lines.append("These files are foundational or actively cited by other files in the repository.")
for f in sorted(list(core_files)):
    refs = referenced_by[f]
    ref_str = f"Cited by: {', '.join([r.name for r in refs])}" if refs else "Base CORE file"
    out_lines.append(f"- {format_file(f)}")
    out_lines.append(f"  Reason: {ref_str}")

out_lines.append("\n=== CATEGORY: SUPERSEDED-BY-A-LATER-VERSION ===")
out_lines.append("Files with an obvious version lineage where a later or CORE version exists.")
for b in sorted(superseded_groups.keys()):
    d = superseded_groups[b]
    out_lines.append(f"\nLineage Base: '{b}'")
    kept = d['kept']
    kept_reason = "Already in CORE (Actively cited or base file)" if kept in core_files else "Latest modified version in lineage, keeping as representative"
    out_lines.append(f"  [KEEP] {format_file(kept)} - {kept_reason}")
    for r in d['removed']:
        out_lines.append(f"  [REMOVE] {format_file(r)}")

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

with open(repo_dir / "cleanup_inventory.log", "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))

print(f"Inventory completed. Wrote {len(moves)} proposed moves to cleanup_inventory.log")
