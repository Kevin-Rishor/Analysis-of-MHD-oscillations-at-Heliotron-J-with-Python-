#!/usr/bin/env python3
"""
analysis/obj_ech/run_mode_structure.py

Task 24 Subtask B:
Calls run_mode_structure_comparison from analysis/heating_phase_comparison/mhd_compare_mode_structure.py
for shot 88653 into analysis/obj_ech/mode_structure_run2/,
captures stdout to mode_structure_run2/stdout.txt,
and writes mode_structure_run2/results.json from parsed stdout (no recomputation).
Also explains the sign convention of PMP angles and k_grid.
"""

import io
import os
import re
import sys
import json
from pathlib import Path

# Setup paths
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "analysis" / "heating_phase_comparison"))
sys.path.insert(0, str(REPO_ROOT / "analysis" / "common"))
sys.path.insert(0, str(REPO_ROOT / "jpack"))

from mhd_compare_mode_structure import run_mode_structure_comparison
from comparison_common import PMP_RAW_ANGLE_LABELS_DEG, PMP_ANGLES_DEG

OUT_DIR = REPO_ROOT / "analysis" / "obj_ech" / "mode_structure_run2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=================================================================")
print("RUNNING run_mode_structure_comparison FOR SHOT 88653")
print("=================================================================")

# Capture stdout
old_stdout = sys.stdout
captured_io = io.StringIO()
sys.stdout = captured_io

try:
    run_mode_structure_comparison(shot=88653, out_dir=OUT_DIR, verbose=True)
finally:
    sys.stdout = old_stdout

stdout_content = captured_io.getvalue()

# Save captured stdout to mode_structure_run2/stdout.txt
stdout_path = OUT_DIR / "stdout.txt"
with open(stdout_path, "w", encoding="utf-8") as f:
    f.write(stdout_content)

print(f"Captured stdout written to: {stdout_path}")
print("--- stdout.txt content ---")
print(stdout_content)

# Parse stdout to build results.json without recomputation
# Expected blocks:
#   Processing Secondary Band (40-80 kHz) - ECH+NBI...
#     Dominant m = +6 at 48.8 kHz | r_circ = 0.216, err = 92.3 deg -> UNCONFIRMED
#   Processing Secondary Band (40-80 kHz) - Just NBI...
#     Dominant m = +2 at 41.0 kHz | r_circ = 0.632, err = 81.1 deg -> UNCONFIRMED
#   Processing Primary Band (80-120 kHz) - ECH+NBI...
#     Dominant m = +3 at 89.8 kHz | r_circ = 0.882, err = 26.7 deg -> CONFIRMED
#   Processing Primary Band (80-120 kHz) - Just NBI...
#     Dominant m = +3 at 85.9 kHz | r_circ = 0.902, err = 22.2 deg -> CONFIRMED

runs_meta = [
    {
        "key": "secondary_ech_nbi",
        "band": "Secondary Band (40-80 kHz)",
        "phase": "ECH+NBI",
        "time_window_ms": [259.1, 291.3],
        "pattern": r"Processing Secondary Band \(40-80 kHz\) - ECH\+NBI\.\.\.\s+Dominant m = ([+-]?\d+) at ([\d\.]+) kHz \| r_circ = ([\d\.]+), err = ([\d\.]+) deg -> (CONFIRMED|UNCONFIRMED)"
    },
    {
        "key": "secondary_nbi_only",
        "band": "Secondary Band (40-80 kHz)",
        "phase": "NBI-only",
        "time_window_ms": [291.3, 300.0],
        "pattern": r"Processing Secondary Band \(40-80 kHz\) - Just NBI\.\.\.\s+Dominant m = ([+-]?\d+) at ([\d\.]+) kHz \| r_circ = ([\d\.]+), err = ([\d\.]+) deg -> (CONFIRMED|UNCONFIRMED)"
    },
    {
        "key": "primary_ech_nbi",
        "band": "Primary Band (80-120 kHz)",
        "phase": "ECH+NBI",
        "time_window_ms": [259.1, 291.3],
        "pattern": r"Processing Primary Band \(80-120 kHz\) - ECH\+NBI\.\.\.\s+Dominant m = ([+-]?\d+) at ([\d\.]+) kHz \| r_circ = ([\d\.]+), err = ([\d\.]+) deg -> (CONFIRMED|UNCONFIRMED)"
    },
    {
        "key": "primary_nbi_only",
        "band": "Primary Band (80-120 kHz)",
        "phase": "NBI-only",
        "time_window_ms": [291.3, 300.0],
        "pattern": r"Processing Primary Band \(80-120 kHz\) - Just NBI\.\.\.\s+Dominant m = ([+-]?\d+) at ([\d\.]+) kHz \| r_circ = ([\d\.]+), err = ([\d\.]+) deg -> (CONFIRMED|UNCONFIRMED)"
    }
]

results = {}
for meta in runs_meta:
    match = re.search(meta["pattern"], stdout_content)
    if not match:
        raise ValueError(f"Could not parse stdout for {meta['key']}")
    m_val = int(match.group(1))
    f_val = float(match.group(2))
    rc_val = float(match.group(3))
    err_val = float(match.group(4))
    is_conf = (match.group(5) == "CONFIRMED")
    
    results[meta["key"]] = {
        "band": meta["band"],
        "phase": meta["phase"],
        "time_window_ms": meta["time_window_ms"],
        "frequency_khz": f_val,
        "m": m_val,
        "r_circ": rc_val,
        "err_deg": err_val,
        "is_confirmed": is_conf
    }

results_path = OUT_DIR / "results.json"
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"Parsed results written to: {results_path}")
print(json.dumps(results, indent=2))

print("\n=================================================================")
print("SIGN CONVENTION EXPLANATION (PMP ANGLES AND k_grid)")
print("=================================================================")
print(f"PMP_RAW_ANGLE_LABELS_DEG: {PMP_RAW_ANGLE_LABELS_DEG}")
print(f"PMP_ANGLES_DEG:           {PMP_ANGLES_DEG}")
print("Formula: PMP_ANGLES_DEG = [(360.0 - a) % 360.0 for a in PMP_RAW_ANGLE_LABELS_DEG]")
print("k_grid:  np.arange(-6, 7)")
print("Kernel:  E = np.exp(-1j * k_grid[:, None] * angles_rad[None, :]) / len(angles_rad)")
print("\nExplanation:")
print("1. Poloidal angle coordinate inversion:")
print("   The raw coil angles 'a' are inverted via theta = (360 - a) % 360. This inverts the")
print("   angular coordinate direction so that positive poloidal rotation aligns with the")
print("   plasma drift / co-current direction.")
print("2. Fourier NUDFT Kernel:")
print("   The basis is exp(-1j * k * theta). For a spatial mode exp(1j * m * theta), the peak")
print("   projection occurs at k = +m.")
print("3. Primary vs Secondary Modes:")
print("   - For the Primary mode (80-120 kHz), the poloidal projection sharply peaks at k = +3")
print("     (m = +3) with high circular concentration (r_circ = 0.882 and 0.902, err = 26.7 deg and 22.2 deg).")
print("   - For the Secondary modes (40-80 kHz), the raw power argmax yields k = +6 (ECH+NBI) and")
print("     k = +2 (NBI-only). However, their circular concentrations are very low (r_circ = 0.216 and 0.632)")
print("     and mean angular phase errors are large (err = 92.3 deg and 81.1 deg), failing the")
print("     confirmation threshold (r_circ >= 0.70, err <= 45 deg). Consequently, secondary modes")
print("     are classified UNRESOLVED.")
print("   - If an opposing sign convention were used (omitting theta -> -theta or conjugating the kernel),")
print("     the primary peak would appear at k = -3. Any attribution of m = -3 to the secondary mode")
print("     arises either from an inverted sign convention or an assumed three-wave coupling selection rule")
print("     (m_daughter = -m_parent), but is NOT empirically supported by the poloidal array.")
