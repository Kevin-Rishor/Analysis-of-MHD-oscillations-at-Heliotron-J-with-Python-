import sys
import subprocess
from pathlib import Path

current = Path(__file__).resolve().parent
root_dir = None
for p in [current] + list(current.parents):
    if (p / "jpack").exists():
        root_dir = p
        break
if root_dir is None:
    root_dir = current.parents[1]


def main():
    print(f"Running Objective 2 pipeline in: {root_dir}")

    is_verbose = "--verbose" in sys.argv or "-v" in sys.argv

    scripts = [
        ("Heating & Confinement vs. Mirnov", [sys.executable, str(current / "heating_vs_mirnov.py")]),
        ("Dual-Band Dynamics & Structure (40-80, 80-120 kHz)", [sys.executable, str(current / "mhd_analysis_obj2.py"), "--shots", "88653", "88654", "--bands", "40:80", "80:120"]),
        ("Two-Mode Poloidal Combination Search", [sys.executable, str(current / "mhd_two_mode_combination_search.py"), "--shot", "88653"]),
        ("ECE vs. LPF Te Verification (Multi-panel)", [sys.executable, str(current / "verify_ece_lpf_temperature.py")]),
        ("ECE vs. LPF Te Verification (Final 2-panel Figure)", [sys.executable, str(current / "plot_ece_lpf_te_single_figure.py")]),
        ("2D Magnetic Surface Map", [sys.executable, str(current / "mhd_plot_2d_torus_maps.py")]),
        ("Self-Coupling Coherence", [sys.executable, str(current / "mhd_plot_self_coupling_coherence.py")]),
    ]

    for desc, cmd in scripts:
        if is_verbose and "mhd_analysis_obj2.py" in cmd[1]:
            cmd.append("--verbose")
        print(f"Executing {desc}...")
        res = subprocess.run(cmd, cwd=str(root_dir))
        if res.returncode != 0:
            print(f"Error: Command failed with code {res.returncode}")
            sys.exit(res.returncode)

    print("Objective 2 pipeline completed successfully.")


if __name__ == "__main__":
    main()

