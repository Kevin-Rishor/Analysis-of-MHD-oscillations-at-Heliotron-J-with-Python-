"""
run_all_objectives.py
Master entrypoint for the complete Heliotron J MHD Analysis TFG codebase.
Allows running individual objectives or all 3 objectives sequentially.
Usage:
    python run_all_objectives.py --obj 1
    python run_all_objectives.py --obj 2
    python run_all_objectives.py --obj 3
    python run_all_objectives.py --all
"""

import sys
import argparse
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

def run_objective(obj_num):
    obj_dirs = {
        1: ("Specific Objective 1: Mode Identification", ROOT_DIR / "analysis" / "obj1_mode_identification" / "run_obj1.py"),
        2: ("Specific Objective 2: Dynamics & Modal Structure", ROOT_DIR / "analysis" / "obj2_dynamics_and_structure" / "run_obj2.py"),
        3: ("Specific Objective 3: Non-Linear Coupling & Bicoherence", ROOT_DIR / "analysis" / "obj3_nonlinear_coupling" / "run_obj3.py"),
    }

    if obj_num not in obj_dirs:
        print(f"Unknown objective number: {obj_num}")
        return False

    name, script = obj_dirs[obj_num]
    print("\n" + "#" * 80)
    print(f"# LAUNCHING {name.upper()}")
    print("#" * 80)

    res = subprocess.run([sys.executable, str(script)], cwd=str(ROOT_DIR))
    return res.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="Master Runner for Heliotron J MHD Analysis TFG")
    parser.add_argument("--obj", type=int, choices=[1, 2, 3], help="Run a specific objective (1, 2, or 3)")
    parser.add_argument("--all", action="store_true", help="Run all 3 objectives sequentially")

    args = parser.parse_args()

    if not args.obj and not args.all:
        parser.print_help()
        print("\nTip: Run 'python run_all_objectives.py --obj 3' to execute Objective 3.")
        return

    if args.all:
        for o in [1, 2, 3]:
            ok = run_objective(o)
            if not ok:
                print(f"Objective {o} failed. Halting pipeline.")
                sys.exit(1)
    elif args.obj:
        ok = run_objective(args.obj)
        if not ok:
            sys.exit(1)

if __name__ == "__main__":
    main()
