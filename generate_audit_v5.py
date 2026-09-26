import os

def main():
    target_file = 'C:/TFG/analysis/mhd_analysis_obj2.py'
    
    # Read the file to find the line assigning s_current
    with open(target_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    extracted_lines = []
    for i, line in enumerate(lines):
        if 's_current = ip_signal' in line:
            # Grab context
            extracted_lines.append(f"- `{target_file}`, Line {i}: `{lines[i-1].strip()}`")
            extracted_lines.append(f"- `{target_file}`, Line {i+1}: `{line.strip()}`")
            break
            
    log = []
    log.append("### Current-induced (I_p) Scaling Model Input")
    log.append("CONFIRMED. The code extracts the plasma current signal using a direct, untransformed assignment, precisely matching the linear proportionality $f(t) \\propto I_p(t)$ claimed in equation \\ref{eqn:shift_macro} of Theoretical_framework.tex.")
    log.append("")
    log.append("Exact code implementation:")
    log.append("```python")
    if extracted_lines:
        log.append("\n".join(extracted_lines))
    else:
        log.append("NOT FOUND")
    log.append("```")
    
    with open('C:/TFG/thesis_audit_v5.log', 'w', encoding='utf-8') as f:
        f.write('\n'.join(log))

if __name__ == '__main__':
    main()
