"""Runner script to reproduce all comparative heating phase analysis plots and JSON data."""
import sys
from pathlib import Path

current = Path(__file__).resolve().parent
root_dir = current.parent
comp_dir = root_dir / "analysis" / "heating_phase_comparison"
if str(comp_dir) not in sys.path:
    sys.path.append(str(comp_dir))

from run_heating_comparison import main

if __name__ == "__main__":
    main()
