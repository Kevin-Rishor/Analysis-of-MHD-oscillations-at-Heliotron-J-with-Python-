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
    print(f"Running Objective 3 pipeline in: {root_dir}")

    scripts = [
        ("Bicoherence Matrix & Top 10 Couplings", [sys.executable, str(current / "mhd_analysis_obj3.py"), "--shots", "88653", "--burst-start", "259.1", "--burst-end", "275.0"]),
        ("Asymptotic Noise Convergence Test", [sys.executable, str(current / "mhd_plot_asymptotic_bias_scaling.py")])
    ]

    for desc, cmd in scripts:
        print(f"Executing {desc}...")
        res = subprocess.run(cmd, cwd=str(root_dir))
        if res.returncode != 0:
            print(f"Error: Command failed with code {res.returncode}")
            sys.exit(res.returncode)

    print("Objective 3 pipeline completed successfully.")


if __name__ == "__main__":
    main()

