import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Configure standard streams for UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

root_dir = Path(__file__).resolve().parent


class LoggerTee:
    """Tees messages to both console and a log file."""
    def __init__(self, log_path=None):
        self.log_file = None
        self.log_path = log_path
        if log_path:
            self.log_path = Path(log_path)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_file = open(self.log_path, "w", encoding="utf-8", errors="replace")

    def write(self, message):
        sys.stdout.write(message)
        sys.stdout.flush()
        if self.log_file:
            self.log_file.write(message)
            self.log_file.flush()

    def log_line(self, line=""):
        self.write(line + "\n")

    def close(self):
        if self.log_file:
            self.log_file.close()
            self.log_file = None


def run_objective(obj_num, logger, verbose=False):
    obj_dirs = {
        1: ("Objective 1: Mode Identification", root_dir / "analysis" / "obj1_mode_identification" / "run_obj1.py"),
        2: ("Objective 2: Dynamics & Structure", root_dir / "analysis" / "obj2_dynamics_and_structure" / "run_obj2.py"),
        3: ("Objective 3: Non-Linear Coupling", root_dir / "analysis" / "obj3_nonlinear_coupling" / "run_obj3.py"),
    }

    if obj_num not in obj_dirs:
        logger.log_line(f"Unknown objective number: {obj_num}")
        return False, 0.0

    name, script = obj_dirs[obj_num]
    separator = "=" * 80
    logger.log_line(f"\n{separator}")
    logger.log_line(f"[{datetime.now().strftime('%H:%M:%S')}] STARTING {name.upper()}")
    logger.log_line(f"{separator}")

    t_start = time.time()
    cmd = [sys.executable, str(script)]
    if verbose:
        cmd.append("--verbose")

    proc = subprocess.Popen(
        cmd,
        cwd=str(root_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    for line in iter(proc.stdout.readline, ""):
        logger.write(line)
    proc.stdout.close()
    proc.wait()

    elapsed = time.time() - t_start
    status_str = "SUCCESS" if proc.returncode == 0 else f"FAILED (code {proc.returncode})"

    logger.log_line(f"{separator}")
    logger.log_line(f"[{datetime.now().strftime('%H:%M:%S')}] COMPLETED {name.upper()}: {status_str} (Elapsed: {elapsed:.1f}s)")
    logger.log_line(f"{separator}\n")

    return proc.returncode == 0, elapsed


def main():
    parser = argparse.ArgumentParser(description="Master Runner for Heliotron J MHD Analysis")
    parser.add_argument("--obj", type=int, choices=[1, 2, 3], help="Run a specific objective (1, 2, or 3)")
    parser.add_argument("--all", action="store_true", help="Run all 3 objectives sequentially")
    parser.add_argument("--log-file", type=str, default=str(root_dir / "run_all_objectives.log"),
                        help="Path to complete execution log file (default: run_all_objectives.log)")
    parser.add_argument("--no-log", action="store_true", help="Disable writing log file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Pass verbose flag to sub-analyses")

    args = parser.parse_args()

    if not args.obj and not args.all:
        parser.print_help()
        return

    log_path = None if args.no_log else args.log_file
    logger = LoggerTee(log_path)

    header_bar = "#" * 80
    logger.log_line(f"\n{header_bar}")
    logger.log_line("  HELIOTRON J MHD ANALYSIS - MASTER PIPELINE RUNNER")
    logger.log_line(f"  Start time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if log_path:
        logger.log_line(f"  Log file   : {log_path}")
    logger.log_line(f"{header_bar}")

    pipeline_start = time.time()
    results = {}

    to_run = [1, 2, 3] if args.all else [args.obj]
    for o in to_run:
        ok, elapsed = run_objective(o, logger, verbose=args.verbose)
        results[o] = (ok, elapsed)
        if not ok and args.all:
            logger.log_line(f"Pipeline halted: Objective {o} failed.")
            break

    total_time = time.time() - pipeline_start

    logger.log_line(f"\n{header_bar}")
    logger.log_line("  PIPELINE EXECUTION SUMMARY")
    logger.log_line(f"{header_bar}")
    all_ok = True
    for o, (ok, elapsed) in results.items():
        res_text = "PASSED" if ok else "FAILED"
        if not ok:
            all_ok = False
        logger.log_line(f"  Objective {o}: {res_text:<7} (Duration: {elapsed:5.1f} s)")
    logger.log_line(f"  Total Pipeline Time: {total_time:.1f} s")
    if log_path:
        logger.log_line(f"  Full log saved to  : {log_path}")
    logger.log_line(f"{header_bar}\n")

    logger.close()
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()


