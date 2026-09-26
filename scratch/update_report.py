import os
import glob
import re

logs = glob.glob(r"c:\TFG\scratch\log_*.txt")

moved_files = {}

for log in logs:
    with open(log, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    # Look for paths containing .png
    # e.g. 'C:\TFG\mhd_mode_spatial_map_2d_88653.png'
    # or C:\\TFG\\Images\\...
    matches = re.findall(r"['\"]?([A-Za-z]:\\[^\n\"']+\.png)['\"]?", content)
    for m in matches:
        m_norm = os.path.normpath(m)
        fname = os.path.basename(m_norm)
        prefix = "c:\\tfg\\images\\"
        if m_norm.lower().startswith(prefix) or m_norm.lower() == prefix + fname.lower():
            pass
        else:
            moved_files[fname] = m_norm

# I also ran mhd_analysis_obj3.py manually and it output:
# Bicoherence plot saved to: 'mhd_bicoherence_objective3_88653.png'
moved_files['mhd_bicoherence_objective3_88653.png'] = r"c:\TFG\mhd_bicoherence_objective3_88653.png"
moved_files['mhd_burst_window_detection_88653.png'] = r"c:\TFG\mhd_burst_window_detection_88653.png"

# Now append this to the report
with open(r"c:\TFG\scratch\image_diff_report.txt", "r", encoding="utf-8") as f:
    report = f.read()

# Replace the hardcoded MOVED FILES LIST
idx = report.find("MOVED FILES LIST:\n=================")
if idx != -1:
    report = report[:idx]

with open(r"c:\TFG\scratch\image_diff_report.txt", "w", encoding="utf-8") as f:
    f.write(report)
    f.write("MOVED FILES LIST:\n")
    f.write("=================\n")
    for fname, path in sorted(moved_files.items()):
        f.write(f" - {path}\n")

print("Report updated.")
