import sys
import argparse
from pathlib import Path

current = Path(__file__).resolve().parent
root_dir = None
for p in [current] + list(current.parents):
    if (p / "jpack").exists():
        root_dir = p
        break
if root_dir is None:
    root_dir = Path("c:/TFG")

for p_add in [
    root_dir / "jpack", root_dir / "analysis", root_dir / "analysis" / "common",
    root_dir / "analysis" / "obj2_dynamics_and_structure", current
]:
    if str(p_add) not in sys.path:
        sys.path.append(str(p_add))

from comparison_common import DEFAULT_SHOT, DEFAULT_OUT_DIR
from mhd_compare_spectrograms import run_spectrogram_comparison
from mhd_compare_mode_identification import run_mode_identification_comparison
from mhd_compare_mode_structure import run_mode_structure_comparison
from mhd_compare_bicoherence import run_bicoherence_comparison
from mhd_compare_toroidal_and_radial import run_toroidal_and_radial_comparison
from mhd_compare_self_coupling import run_self_coupling_comparison
from mhd_plot_bicoherence_sliding_trace import run_sliding_bicoherence_trace
from mhd_plot_nbi_vs_ech_envelope_ip import generate_nbi_comparison_envelope_ip_plot


class LoggerTee:
    """Tees standard output to both console and a log file."""
    def __init__(self, log_path=None):
        self.terminal = sys.stdout
        self.log_file = None
        if log_path:
            p = Path(log_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self.log_file = open(p, "w", encoding="utf-8", errors="replace")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        if self.log_file:
            self.log_file.write(message)
            self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        if self.log_file:
            self.log_file.flush()

    def close(self):
        if self.log_file:
            self.log_file.close()
            self.log_file = None


def main():
    parser = argparse.ArgumentParser(
        description="Comparative Heating Phase Analysis Pipeline (ECH+NBI vs Pure NBI)"
    )
    parser.add_argument("-s", "--shots", type=int, default=DEFAULT_SHOT, help=f"Shot number (default: {DEFAULT_SHOT})")
    parser.add_argument("-o", "--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "nbi_vs_ech_nbi_comparison.log"

    tee = LoggerTee(log_path)
    old_stdout = sys.stdout
    sys.stdout = tee

    try:
        print(f"================================================================================")
        print(f"Heliotron J MHD Comparative Analysis: ECH+NBI vs. Just NBI Phase (Shot {args.shots})")
        print(f"Output directory: {out_dir}")
        print(f"================================================================================")

        # 1. Zoomed Spectrograms with 3-Sigma Dominance
        run_spectrogram_comparison(shot=args.shots, out_dir=out_dir, verbose=args.verbose)

        # 2. Objective 1 Mode Identification Overview
        run_mode_identification_comparison(shot=args.shots, out_dir=out_dir, verbose=args.verbose)

        # 3. Objective 2 Spatial Modal Structure (4 figures)
        run_mode_structure_comparison(shot=args.shots, out_dir=out_dir, verbose=args.verbose)

        # 4. Objective 3 Non-Linear Bicoherence Matrices
        run_bicoherence_comparison(shot=args.shots, out_dir=out_dir, verbose=args.verbose)

        # 5. Toroidal Mode Number n & Radial Localization
        run_toroidal_and_radial_comparison(shot=args.shots, out_dir=out_dir, verbose=args.verbose)

        # 6. Self-Coupling Coherence
        run_self_coupling_comparison(shot=args.shots, out_dir=out_dir, verbose=args.verbose)

        # 7. Dual-Mode Envelope & Plasma Current Evolution Comparison
        print(f"--- Mode Envelopes & Ip Evolution Comparison (Shot {args.shots}) ---")
        generate_nbi_comparison_envelope_ip_plot(shot=args.shots, out_dir=out_dir)
        print(f"    Saved figure: mhd_mode_envelopes_and_ip_ECH_NBI_comparison.png")

        # 8. Sliding Bicoherence Trace with SNR Cutoff
        run_sliding_bicoherence_trace(shot=args.shots, out_dir=out_dir, verbose=args.verbose)

        print(f"\nComparative heating phase analysis completed successfully.")
        print(f"All figures and summary JSON files saved to: '{out_dir}'\n")

    finally:
        sys.stdout = old_stdout
        tee.close()


if __name__ == "__main__":
    main()
