import argparse
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
    root_dir = current.parents[1]


def get_objective1_window(shot=88653):
    json_path = root_dir / f"primary_mode_identification_shot_{shot}.json"
    fallback_window = (259.1, 275.0)
    if not json_path.exists():
        return fallback_window

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        win = data.get("window") or data.get("analyzed_subwindow_ms")
        if win and len(win) == 2:
            return float(win[0]), float(win[1])
        return fallback_window
    except Exception:
        return fallback_window


def _run_step(desc, cmd, cwd):
    print(f"Executing {desc}...")
    res = subprocess.run(cmd, cwd=str(cwd))
    if res.returncode != 0:
        print(f"Error: Command failed with code {res.returncode}")
        sys.exit(res.returncode)


def run_legacy_pipeline(run_cwd):
    if os.environ.get("ALLOW_LEGACY_LIBRARY_ESTIMATOR") != "1":
        print(
            "Warning: Running deprecated LAS.abicoh2 Objective 3 scripts (--legacy-library-estimator). "
            "See analysis/obj3_nonlinear_coupling/SUPERSEDED.md for index-convention details."
        )
    w_start, w_end = get_objective1_window(88653)
    print(f"Using window: {w_start:.1f} - {w_end:.1f} ms")
    scripts = [
        ("Legacy Bicoherence Matrix & Top 10 Couplings", [sys.executable, str(current / "mhd_analysis_obj3.py"), "--shots", "88653", "--burst-start", f"{w_start:.1f}", "--burst-end", f"{w_end:.1f}"]),
        ("Legacy Asymptotic Noise Convergence Test", [sys.executable, str(current / "mhd_plot_asymptotic_bias_scaling.py")]),
    ]
    for desc, cmd in scripts:
        _run_step(desc, cmd, run_cwd)


def main(target_dir=None, recompute_surrogates=False, legacy_library_estimator=False):
    run_cwd = Path(target_dir) if target_dir else root_dir
    print(f"Running Objective 3 pipeline in: {run_cwd}")

    if legacy_library_estimator:
        run_legacy_pipeline(run_cwd)
        print("Objective 3 legacy pipeline completed.")
        return

    cache_53 = root_dir / "cache" / "shot_88653_full_record_cache.npz"
    cache_54 = root_dir / "cache" / "shot_88654_full_record_cache.npz"
    caches_present = cache_53.exists() and cache_54.exists()

    steps = [
        ("Corrected Bicoherence Unit Tests", [sys.executable, str(root_dir / "analysis" / "common" / "test_abicoh2_corrected.py")]),
        ("Control Windows Comparison Figure", [sys.executable, str(current / "plot_control_windows.py")]),
    ]

    if caches_present:
        steps.append(("Converged Bicoherence Maps (#88653 & #88654)", [sys.executable, str(current / "update_bicoherence_maps_converged.py")]))
    else:
        print(
            "[Skip] update_bicoherence_maps_converged.py skipped: cache/shot_{88653,88654}_full_record_cache.npz not found. "
            "Build caches from raw .edf files first with `python analysis/obj_ech/build_full_record_cache.py`."
        )

    if recompute_surrogates:
        if not caches_present:
            print("Error: --recompute-surrogates requires cache/shot_{88653,88654}_full_record_cache.npz.")
            sys.exit(2)
        steps.append(("500-Surrogate Asymptotic 1/K Convergence", [sys.executable, str(current / "run_convergence_500_surrogates.py")]))
    else:
        steps.append(("Two-Panel Asymptotic 1/K Convergence Figure", [sys.executable, str(current / "generate_two_panel_convergence_fig.py")]))

    for desc, cmd in steps:
        _run_step(desc, cmd, run_cwd)

    print("Objective 3 rectified pipeline completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Objective 3: Nonlinear Bispectral & Wave-Wave Coupling Pipeline")
    parser.add_argument("target_dir", nargs="?", default=None, help="Optional working directory override")
    parser.add_argument("--recompute-surrogates", action="store_true",
                        help="Re-run the 500-subset / 500-surrogate Monte Carlo convergence sweep (requires .npz caches)")
    parser.add_argument("--legacy-library-estimator", action="store_true",
                        help="Run superseded LAS.abicoh2 scripts instead of the rectified pipeline")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose flag (accepted for unified CLI compatibility)")
    args = parser.parse_args()
    main(
        target_dir=args.target_dir,
        recompute_surrogates=args.recompute_surrogates,
        legacy_library_estimator=args.legacy_library_estimator,
    )
