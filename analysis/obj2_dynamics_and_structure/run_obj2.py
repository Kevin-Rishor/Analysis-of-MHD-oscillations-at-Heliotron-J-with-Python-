"""
run_obj2.py
Master runner script for Specific Objective 2:
MHD Mode Dynamics, Heating Synchronization, Spatial Structure & Self-Coupling
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
    print("RUNNING PIPELINE: SPECIFIC OBJECTIVE 2 (DYNAMICS & MODAL STRUCTURE)")
    print(f"Working Directory: {ROOT_DIR}")
    print("=" * 80)

    scripts = [
        ("1. Macroscopic Heating & Confinement vs. Mirnov Fluctuation",
         [sys.executable, str(current / "heating_vs_mirnov.py")]),
        ("2. Secondary Mode Band (40 - 80 kHz): Heating Sync & Structure",
         [sys.executable, str(current / "mhd_analysis_obj2.py"), "--shots", "88653", "-l", "40", "-u", "80"]),
        ("3. Primary Mode Band (80 - 120 kHz): Heating Sync & Structure",
         [sys.executable, str(current / "mhd_analysis_obj2.py"), "--shots", "88653", "-l", "80", "-u", "120"]),
        ("4. 2D Unfolded Magnetic Surface Map ((m,n) Structure)",
         [sys.executable, str(current / "mhd_plot_2d_torus_maps.py")]),
        ("5. B_tilde vs. Envelope Self-Coupling Cross-Spectral Coherence",
         [sys.executable, str(current / "mhd_plot_self_coupling_coherence.py")])
    ]

    for desc, cmd in scripts:
        print(f"\n>>> Executing: {desc}...")
        res = subprocess.run(cmd, cwd=str(ROOT_DIR))
        if res.returncode != 0:
            print(f"ERROR: Command failed with code {res.returncode}")
            sys.exit(res.returncode)

    print("\n" + "=" * 80)
    print("OBJECTIVE 2 PIPELINE COMPLETED SUCCESSFULLY!")
    print("Generated Figures:")
    print("  - heating_vs_mirnov (Shot_88653).png")
    print("  - mhd_analysis_objective2_88653_40_80kHz.png")
    print("  - mhd_analysis_objective2_88653_80_120kHz.png")
    print("  - mhd_analysis_objective2_heating_88653_40_80kHz.png")
    print("  - mhd_analysis_objective2_heating_88653_80_120kHz.png")
    print("  - mhd_analysis_objective2_structure_88653_40_80kHz.png")
    print("  - mhd_analysis_objective2_structure_88653_80_120kHz.png")
    print("  - mhd_analysis_objective2_zhong_88653_40_80kHz.png")
    print("  - mhd_analysis_objective2_zhong_88653_80_120kHz.png")
    print("  - mhd_mode_spatial_map_2d_88653.png")
    print("  - mhd_self_coupling_coherence_shot_88653.png")
    print("=" * 80)

if __name__ == "__main__":
    main()
