"""
run_obj3.py
Master runner script for Specific Objective 3:
Non-Linear Three-Wave Coupling, Bicoherence Geometry & Asymptotic Bias Validation
Shot #88653 (Heliotron J)
"""

import sys
import subprocess
from pathlib import Path

# Dynamic repository root finder
current = Path(__file__).resolve().parent
ROOT_DIR = None
for p in [current] + list(current.parents):
    if (p / "jpack").exists():
        ROOT_DIR = p
        break
if ROOT_DIR is None:
    ROOT_DIR = Path("c:/TFG")

def main():
    print("=" * 80)
    print("RUNNING PIPELINE: SPECIFIC OBJECTIVE 3 (NON-LINEAR BICOHERENCE)")
    print(f"Working Directory: {ROOT_DIR}")
    print("=" * 80)

    scripts = [
        ("1. Bicoherence Matrix & Top 10 Couplings (Stationary Window: 259.1 - 275.0 ms)",
         [sys.executable, str(current / "mhd_analysis_obj3.py"), "--shots", "88653", "--burst-start", "259.1", "--burst-end", "275.0"]),
        ("2. Asymptotic Noise Convergence Test (1/N -> 0 for Bicoherence & PSD)",
         [sys.executable, str(current / "mhd_plot_asymptotic_bias_scaling.py")])
    ]

    for desc, cmd in scripts:
        print(f"\n>>> Executing: {desc}...")
        res = subprocess.run(cmd, cwd=str(ROOT_DIR))
        if res.returncode != 0:
            print(f"ERROR: Command failed with code {res.returncode}")
            sys.exit(res.returncode)

    print("\n" + "=" * 80)
    print("OBJECTIVE 3 PIPELINE COMPLETED SUCCESSFULLY!")
    print("Generated Figures:")
    print("  - mhd_bicoherence_objective3_88653.png")
    print("  - mhd_asymptotic_bias_scaling_shot_88653.png")
    print("  - mhd_burst_window_detection_88653.png")
    print("=" * 80)

if __name__ == "__main__":
    main()
