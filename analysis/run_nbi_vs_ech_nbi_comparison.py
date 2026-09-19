"""Comparative Analysis Pipeline: ECH+NBI Phase vs. Pure NBI Phase
Shot 88653 - Heliotron J MHD Analysis

Measured ECH Turn-Off: t_ECH_off = 291.3 ms (from ECHRG500@88653.edf)
Output directory: c:/TFG/NBI vs NBI and ECH/
"""

import sys
from pathlib import Path

current = Path(__file__).resolve().parent
root_dir = None
for p in [current] + list(current.parents):
    if (p / "jpack").exists():
        root_dir = p
        break
if root_dir is None:
    root_dir = Path("c:/TFG")

comp_dir = root_dir / "analysis" / "heating_phase_comparison"
if str(comp_dir) not in sys.path:
    sys.path.append(str(comp_dir))

from run_heating_comparison import main

if __name__ == "__main__":
    main()
