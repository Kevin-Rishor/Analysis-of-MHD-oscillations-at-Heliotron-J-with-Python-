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
    root_dir = Path("c:/TFG")


def main():
    print(f"Running Objective 2 pipeline in: {root_dir}")

    scripts = [
        ("Heating & Confinement vs. Mirnov", [sys.executable, str(current / "heating_vs_mirnov.py")]),
        ("Secondary Band (40-80 kHz)", [sys.executable, str(current / "mhd_analysis_obj2.py"), "--shots", "88653", "-l", "40", "-u", "80"]),
        ("Primary Band (80-120 kHz)", [sys.executable, str(current / "mhd_analysis_obj2.py"), "--shots", "88653", "-l", "80", "-u", "120"]),
        ("2D Magnetic Surface Map", [sys.executable, str(current / "mhd_plot_2d_torus_maps.py")]),
        ("Self-Coupling Coherence", [sys.executable, str(current / "mhd_plot_self_coupling_coherence.py")])
    ]

    for desc, cmd in scripts:
        print(f"Executing {desc}...")
        res = subprocess.run(cmd, cwd=str(root_dir))
        if res.returncode != 0:
            print(f"Error: Command failed with code {res.returncode}")
            sys.exit(res.returncode)

    print("Objective 2 pipeline completed successfully.")


if __name__ == "__main__":
    main()

