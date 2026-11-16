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
    print(f"Running Objective 1 pipeline in: {root_dir}")

    scripts = [
        ("Multi-panel Overview", [sys.executable, str(current / "mhd_analysis_obj1.py"), "--shots", "88653"], True),
        ("Toroidal & Radial Localization", [sys.executable, str(current / "mhd_primary_mode_id.py"), "--shots", "88653"], True),
        ("16-ch ECE x 18-coil Radial Localization", [sys.executable, str(current / "analyze_ece_all_probes_radial.py")], False),
        ("ECE12-ECE13 Radial Phase Delay & Flow", [sys.executable, str(current / "analyze_ece12_13_phase_delay.py")], False),
    ]

    is_verbose = "--verbose" in sys.argv or "-v" in sys.argv

    for desc, cmd, supports_verbose in scripts:
        if is_verbose and supports_verbose:
            cmd.append("--verbose")
        print(f"Executing {desc}...")
        res = subprocess.run(cmd, cwd=str(root_dir))
        if res.returncode != 0:
            print(f"Error: Command failed with code {res.returncode}")
            sys.exit(res.returncode)

    print("Objective 1 pipeline completed successfully.")


if __name__ == "__main__":
    main()

