import sys
from pathlib import Path

current = Path(__file__).resolve().parent
comp_dir = current / "heating_phase_comparison"
if str(comp_dir) not in sys.path:
    sys.path.append(str(comp_dir))

from mhd_plot_bicoherence_sliding_trace import run_sliding_bicoherence_trace

if __name__ == "__main__":
    run_sliding_bicoherence_trace()
