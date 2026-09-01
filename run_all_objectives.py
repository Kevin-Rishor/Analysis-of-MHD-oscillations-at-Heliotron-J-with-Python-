import sys
import argparse
import subprocess
from pathlib import Path

root_dir = Path(__file__).resolve().parent


def run_objective(obj_num):
    obj_dirs = {
        1: ("Objective 1: Mode Identification", root_dir / "analysis" / "obj1_mode_identification" / "run_obj1.py"),
        2: ("Objective 2: Dynamics & Structure", root_dir / "analysis" / "obj2_dynamics_and_structure" / "run_obj2.py"),
        3: ("Objective 3: Non-Linear Coupling", root_dir / "analysis" / "obj3_nonlinear_coupling" / "run_obj3.py"),
    }

    if obj_num not in obj_dirs:
        print(f"Unknown objective number: {obj_num}")
        return False

    name, script = obj_dirs[obj_num]
    print(f"\nLaunching {name}...")
    res = subprocess.run([sys.executable, str(script)], cwd=str(root_dir))
    return res.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Master Runner for Heliotron J MHD Analysis")
    parser.add_argument("--obj", type=int, choices=[1, 2, 3], help="Run a specific objective (1, 2, or 3)")
    parser.add_argument("--all", action="store_true", help="Run all 3 objectives sequentially")

    args = parser.parse_args()

    if not args.obj and not args.all:
        parser.print_help()
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

