import os
from pathlib import Path

def extract_summaries(lines):
    summaries = []
    in_summary_block = False
    current_summary = []
    
    for i, line in enumerate(lines):
        # Heuristics for start of a summary block
        is_summary_header = "SUMMARY" in line.upper() or "COMPARISON" in line.upper()
        if is_summary_header:
            if not in_summary_block:
                start_idx = i
                # Backtrack to include any border if present
                if i > 0 and len(lines[i-1].strip()) > 10 and set(lines[i-1].strip()) <= {'=', '-', '#'}:
                    start_idx = i - 1
                in_summary_block = True
                current_summary = [lines[start_idx]]
                if start_idx != i:
                    current_summary.append(line)
                continue
                
        if in_summary_block:
            current_summary.append(line)
            # End of block heuristic
            is_border = len(line.strip()) > 10 and set(line.strip()) <= {'=', '-', '#'}
            if is_border and len(current_summary) > 2:
                # If we hit a closing border, end the block
                summaries.append("".join(current_summary))
                in_summary_block = False
                current_summary = []
            elif line.strip() == "" and i+1 < len(lines) and (len(lines[i+1].strip()) > 10 and set(lines[i+1].strip()) <= {'=', '-', '#'}):
                # If we hit an empty line before a new section
                summaries.append("".join(current_summary))
                in_summary_block = False
                current_summary = []
                
    if in_summary_block:
        summaries.append("".join(current_summary))
        
    return summaries

def main():
    base_dir = Path(r"c:\TFG")
    output_file = base_dir / "general_summary.log"
    
    log_files = list(base_dir.rglob("*.log"))
    json_files = list(base_dir.rglob("*.json"))
    md_files = list(base_dir.rglob("*.md"))
    
    all_files = log_files + json_files + md_files
    
    if output_file in all_files:
        all_files.remove(output_file)
        
    # Ignore hidden folders like .git or .venv, and exclude 'jpack'
    all_files = [f for f in all_files if not any(part.startswith('.') for part in f.parts) and 'jpack' not in f.parts]
        
    master_summary = []
    full_contents = []
    
    for lf in all_files:
        try:
            with open(lf, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(lf, 'r', encoding='latin-1') as f:
                content = f.read()
                
        lines = content.splitlines(keepends=True)
        file_summaries = extract_summaries(lines)
        
        rel_path = lf.relative_to(base_dir)
        
        if file_summaries:
            master_summary.append(f"[{rel_path}] Summaries Extracted:\n")
            master_summary.append("-" * 80 + "\n")
            for s in file_summaries:
                master_summary.append(s)
                if not s.endswith("\n"):
                    master_summary.append("\n")
            master_summary.append("-" * 80 + "\n\n")
            
        full_contents.append(f"{'='*80}\n")
        full_contents.append(f"FULL LOG: {rel_path}\n")
        full_contents.append(f"{'='*80}\n")
        full_contents.append(content)
        full_contents.append("\n\n")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("################################################################################\n")
        f.write("                          MASTER SUMMARY OF ALL LOGS                            \n")
        f.write("################################################################################\n\n")
        if master_summary:
            for line in master_summary:
                f.write(line)
        else:
            f.write("No summary sections found.\n\n")
            
        f.write("\n")
        f.write("################################################################################\n")
        f.write("                          FULL CONTENTS OF ALL LOGS                             \n")
        f.write("################################################################################\n\n")
        for line in full_contents:
            f.write(line)
            
    print(f"Aggregated {len(log_files)} log files into {output_file}")

if __name__ == '__main__':
    main()
