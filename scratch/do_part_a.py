import os
import shutil
from pathlib import Path

repo_dir = Path(r"c:\TFG")
log_file = repo_dir / "cleanup_inventory_v3.log"
archive_dir = repo_dir / "archive"

with open(log_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

in_section = False
files_to_move = []

for line in lines:
    if line.startswith("=== CATEGORY: TRUE VERSION PAIRS (AUTO-RESOLVED) ==="):
        in_section = True
        continue
    elif line.startswith("=== CATEGORY:"):
        if in_section:
            break # Exited the section
            
    if in_section and "[REMOVE / SUBSET]" in line:
        # Line format:   [REMOVE / SUBSET] c:\TFG\general_summary_v4_DRAFT.log (Size: 18493, Modified: ...)
        path_str = line.split("[REMOVE / SUBSET] ")[1].split(" (Size:")[0].strip()
        files_to_move.append(Path(path_str))

moved_count = 0
for src in files_to_move:
    if src.exists():
        rel_path = src.relative_to(repo_dir)
        dest = archive_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        moved_count += 1
        print(f"MOVED: {src} -> {dest}")
    else:
        print(f"NOT FOUND: {src}")

print(f"\nTotal files successfully moved: {moved_count} out of {len(files_to_move)} expected from TRUE VERSION PAIRS list.")
