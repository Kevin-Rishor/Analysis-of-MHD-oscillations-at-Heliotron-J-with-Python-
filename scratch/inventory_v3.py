import os
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
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
lines_cache = {}
for p in search_files:
    try:
        with open(p, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        try:
            with open(p, "r", encoding="latin-1") as f:
                text = f.read()
        except Exception:
            text = ""
    file_contents[p] = text
    lines_cache[p] = text.splitlines(keepends=True)

# Topological CORE
roots = [f for f in search_files if f.name in base_core_names]
reachable = set(roots)
changed = True
while changed:
    changed = False
    for f in search_files:
        if f in reachable: continue
        t_name = f.name
        for r in reachable:
            if t_name in file_contents[r]:
                reachable.add(f)
                changed = True
                break

core_files = {f for f in target_files if f in reachable}

def get_base_name(f):
    stem = f.stem
    return re.sub(r'(_[vV]\d+)?(_backup)?(_copy)?(_DRAFT)?(_fresh)?(_LF)?$', '', stem)

all_by_base = defaultdict(list)
for f in target_files:
    all_by_base[get_base_name(f)].append(f)

# Strict Subset Check
def subsumes(big_f, small_f):
    big_lines = lines_cache[big_f]
    small_lines = lines_cache[small_f]
    sm = difflib.SequenceMatcher(None, small_lines, big_lines)
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        if op in ('delete', 'replace'):
            for line in small_lines[a0:a1]:
                if line.strip():  # non-whitespace deletion means small is NOT a subset
                    return False
    return True

# Scope Extraction
scope_annotations = {}
for f in target_files:
    b = get_base_name(f)
    if "mhd_obj2" in b or "mhd_obj3" in b:
        lines = lines_cache[f][:25]
        scope = "UNKNOWN SCOPE"
        for l in lines:
            if "FREQUENCY BAND" in l.upper() or "Band:" in l or "Burst window" in l or "Objective 2 Multi-Shot" in l:
                scope = l.strip()
                break
        scope_annotations[f] = scope

def score_path(p):
    # Higher score = better candidate to KEEP if identical
    score = 0
    if "scratch" in p.parts: score -= 10
    if "fresh_run" in p.parts: score -= 5
    score += p.stat().st_mtime / 1e9
    return score

auto_resolved = []
ambiguous_groups = {}
scratch_list = []

for b, group in all_by_base.items():
    if len(group) > 1:
        removes = set()
        for i in range(len(group)):
            for j in range(len(group)):
                if i == j: continue
                f_i = group[i]
                f_j = group[j]
                if f_i in removes: continue
                
                if subsumes(f_j, f_i):
                    # Check tie-breaker if identical
                    identical = subsumes(f_i, f_j)
                    if identical:
                        i_is_core = f_i in core_files
                        j_is_core = f_j in core_files
                        if i_is_core and not j_is_core:
                            pass # i wins
                        elif not i_is_core and j_is_core:
                            removes.add(f_i) # j wins
                        else:
                            # Break tie with score
                            if score_path(f_j) > score_path(f_i):
                                if not i_is_core: removes.add(f_i)
                            elif score_path(f_j) == score_path(f_i):
                                # Ultimate fallback: shortest path string
                                if len(str(f_j)) < len(str(f_i)) and not i_is_core:
                                    removes.add(f_i)
                    else:
                        if not (f_i in core_files):
                            removes.add(f_i)
                            
        survivors = [f for f in group if f not in removes]
        
        for r in removes:
            subsumer = None
            for s in survivors:
                if subsumes(s, r):
                    subsumer = s
                    break
            auto_resolved.append({'remove': r, 'keep': subsumer, 'base': b})
            
        if len(survivors) > 1:
            ambiguous_groups[b] = survivors
    else:
        f = group[0]
        if f not in core_files:
            scratch_list.append(f)

# Format Output
out_lines = []

def format_file(f):
    st = f.stat()
    sz = st.st_size
    dt = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    scope = f"\n    -> Scope Context: {scope_annotations[f]}" if f in scope_annotations else ""
    return f"{f} (Size: {sz}, Modified: {dt}){scope}"

out_lines.append("=== SCOPE IDENTIFICATION (mhd_obj2 & mhd_obj3) ===")
out_lines.append("Checking top lines for scope identifiers before any grouping decisions:")
for f, scope in sorted(scope_annotations.items(), key=lambda x: str(x[0])):
    out_lines.append(f"- {f}\n    -> {scope}")

out_lines.append("\n=== CATEGORY: AMBIGUOUS (DO NOT AUTO-RESOLVE) ===")
out_lines.append("These files share a base name but have DIVERGING content (neither is a strict superset of the other).")
out_lines.append("They are left exactly where they are for manual human resolution. No moves proposed for these.")
for b, survivors in sorted(ambiguous_groups.items()):
    out_lines.append(f"\nLineage Base: '{b}' ({len(survivors)} diverging versions)")
    for s in survivors:
        out_lines.append(f"  [SURVIVOR] {format_file(s)}")
    
    survs_sorted = sorted(survivors, key=lambda x: x.stat().st_mtime)
    for i in range(len(survs_sorted) - 1):
        f1 = survs_sorted[i]
        f2 = survs_sorted[i+1]
        diff = list(difflib.unified_diff(lines_cache[f1], lines_cache[f2], fromfile=str(f1), tofile=str(f2), n=3))
        out_lines.append(f"\n  --- Full Diff Snippet ({f1.name} -> {f2.name}) ---")
        snippet = "".join(diff[:25])
        out_lines.append(snippet.rstrip())
        if len(diff) > 25:
            out_lines.append(f"  ... (+ {len(diff)-25} more lines)")
        out_lines.append("  --------------------------------")

out_lines.append("\n=== CATEGORY: TRUE VERSION PAIRS (AUTO-RESOLVED) ===")
out_lines.append("These files are strict subsets of another file (or byte-identical older versions). Safe to archive.")
ar_by_base = defaultdict(list)
for ar in auto_resolved:
    ar_by_base[ar['base']].append(ar)
    
for b in sorted(ar_by_base.keys()):
    out_lines.append(f"\nLineage Base: '{b}'")
    for ar in ar_by_base[b]:
        out_lines.append(f"  [KEEP / SUPERSET] {format_file(ar['keep'])}")
        out_lines.append(f"  [REMOVE / SUBSET] {format_file(ar['remove'])}")

out_lines.append("\n=== CATEGORY: CORE ===")
out_lines.append("Topologically traced from baseline files.")
for f in sorted(list(core_files)):
    out_lines.append(f"- {format_file(f)}")

out_lines.append("\n=== CATEGORY: SCRATCH / ONE-OFF ===")
out_lines.append("Singletons not matching any version pairs or core references.")
for f in sorted(scratch_list):
    out_lines.append(f"- {format_file(f)}")

out_lines.append("\n=== PROPOSED MOVE LIST (c:\\TFG\\archive\\...) ===")
total_files = 0
moves = []

for ar in auto_resolved:
    r = ar['remove']
    dest = repo_dir / "archive" / r.relative_to(repo_dir)
    moves.append(f"{r} -> {dest}")
    total_files += 1
    
for f in scratch_list:
    dest = repo_dir / "archive" / f.relative_to(repo_dir)
    moves.append(f"{f} -> {dest}")
    total_files += 1

for m in sorted(moves):
    out_lines.append(m)
    
out_lines.append(f"\nTotal files to move: {total_files}")

with open(repo_dir / "cleanup_inventory_v3.log", "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))
