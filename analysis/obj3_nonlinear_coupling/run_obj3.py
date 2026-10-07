import json
import os
import subprocess
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


def get_objective1_window(shot=88653):
    json_path = root_dir / f"primary_mode_identification_shot_{shot}.json"
    fallback_window = (259.1, 275.0)
    if not json_path.exists():
        print(f"Warning: {json_path.name} not found; falling back to {fallback_window[0]}/{fallback_window[1]} ms.")
        return fallback_window

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        win = data.get("window") or data.get("analyzed_subwindow_ms")
        if win and len(win) == 2:
            return float(win[0]), float(win[1])
        else:
            print(f"Warning: window field missing or invalid in {json_path.name}; falling back to {fallback_window[0]}/{fallback_window[1]} ms.")
            return fallback_window
    except Exception as e:
        print(f"Warning: Error reading {json_path.name}: {e}; falling back to {fallback_window[0]}/{fallback_window[1]} ms.")
        return fallback_window


def main(target_dir=None):
    if os.environ.get("ALLOW_LEGACY_LIBRARY_ESTIMATOR") != "1":
        print("Objective 3 legacy pipeline disabled: it uses the library abicoh2 routine, which has an index error (see analysis/obj3_nonlinear_coupling/SUPERSEDED.md). Set ALLOW_LEGACY_LIBRARY_ESTIMATOR=1 to run it anyway.")
        sys.exit(2)
    run_cwd = Path(target_dir) if target_dir else root_dir
    print(f"Running Objective 3 pipeline in: {run_cwd}")

    w_start, w_end = get_objective1_window(88653)
    print(f"Using window: {w_start:.1f} - {w_end:.1f} ms")

    scripts = [
        ("Bicoherence Matrix & Top 10 Couplings", [sys.executable, str(current / "mhd_analysis_obj3.py"), "--shots", "88653", "--burst-start", f"{w_start:.1f}", "--burst-end", f"{w_end:.1f}"]),
        ("Asymptotic Noise Convergence Test", [sys.executable, str(current / "mhd_plot_asymptotic_bias_scaling.py")])
    ]

    for desc, cmd in scripts:
        print(f"Executing {desc}...")
        res = subprocess.run(cmd, cwd=str(run_cwd))
        if res.returncode != 0:
            print(f"Error: Command failed with code {res.returncode}")
            sys.exit(res.returncode)

    print("Objective 3 pipeline completed successfully.")


if __name__ == "__main__":
    out_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(target_dir=out_arg)

