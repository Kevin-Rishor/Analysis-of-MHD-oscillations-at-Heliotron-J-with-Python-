import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

SHOT = 88653

current = Path(__file__).resolve().parent
root_dir = None
for p in [current] + list(current.parents):
    if (p / "jpack").exists():
        root_dir = p
        break
if root_dir is None:
    root_dir = current.parents[1]

for p_add in [root_dir / "jpack", root_dir / "analysis", root_dir / "analysis" / "common"]:
    if str(p_add) not in sys.path:
        sys.path.append(str(p_add))

import turnelib as TE


def plot_heating_vs_mirnov(shot=SHOT, out_dir=None, ne_mask_until_ms=None):
    if out_dir is None:
        out_dir = root_dir / "Images" / "obj2_dynamics_and_structure"
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_dir = root_dir / "data" / f"hj{shot}"
    wp_file = base_dir / f"Wp@{shot}.edf"
    mp1_file = base_dir / f"MP1@{shot}.edf"
    ech_file = base_dir / f"ECHRG500@{shot}.edf"
    nbi_files = sorted(list(base_dir.glob(f"NBIS*@{shot}.edf")))
    nir_file = base_dir / f"NIRMON@{shot}.edf"

    edf_wp = TE.edf()
    dat_wp = edf_wp.load(str(wp_file))
    t_wp = dat_wp[:, 0]
    y_wp = dat_wp[:, 1]

    edf_mp1 = TE.edf()
    dat_mp1 = edf_mp1.load(str(mp1_file))
    t_mp1 = dat_mp1[:, 0]
    y_mp1 = dat_mp1[:, 1]

    edf_ech = TE.edf()
    dat_ech = edf_ech.load(str(ech_file))
    t_ech = dat_ech[:, 0]
    y_ech = dat_ech[:, 1]

    total_nbi = None
    t_nbi = None
    for fpath in nbi_files:
        edf_nbi = TE.edf()
        dat_nbi = edf_nbi.load(str(fpath))
        if t_nbi is None:
            t_nbi = dat_nbi[:, 0]
            total_nbi = np.zeros_like(dat_nbi[:, 1])
        total_nbi += dat_nbi[:, 1]

    edf_nir = TE.edf()
    dat_nir = edf_nir.load(str(nir_file))
    t_nir = dat_nir[:, 0]
    y_nir = dat_nir[:, 1]

    nave_file = base_dir / f"nave@{shot}.edf"
    if nave_file.exists():
        edf_nave = TE.edf()
        dat_nave = edf_nave.load(str(nave_file))
        t_nave = dat_nave[:, 0]
        y_nave = dat_nave[:, 1]
    else:
        t_nave, y_nave = None, None

    y_pellet = np.where(y_nir >= 0.15, y_nir, 0.0)
    max_pellet = y_pellet.max()
    if max_pellet > 0:
        y_pellet = y_pellet / max_pellet

    if t_nave is not None:
        dt_nave = t_nave[1] - t_nave[0]
        boxcar_size = max(1, int(round(1.0 / dt_nave)))
        y_nave_smooth = uniform_filter1d(y_nave, size=boxcar_size)

        win_mask = (t_nave >= 150.0) & (t_nave <= 350.0)
        idx_max = win_mask.nonzero()[0][np.argmax(y_nave_smooth[win_mask])]
        t_max = t_nave[idx_max]
        val_max_raw = y_nave[idx_max]
        val_max_smooth = y_nave_smooth[idx_max]

        before_max_mask = (t_nave >= 150.0) & (t_nave <= t_max)
        ne_range = np.ptp(y_nave_smooth[win_mask])
        peaks, _ = find_peaks(-y_nave_smooth[before_max_mask], prominence=0.015 * ne_range)
        if len(peaks) > 0:
            last_min_idx = before_max_mask.nonzero()[0][peaks[-1]]
            t_cut_auto = t_nave[last_min_idx]
            val_cut_raw = y_nave[last_min_idx]
            val_cut_smooth = y_nave_smooth[last_min_idx]
        else:
            last_min_idx = before_max_mask.nonzero()[0][np.argmin(y_nave_smooth[before_max_mask])]
            t_cut_auto = t_nave[last_min_idx]
            val_cut_raw = y_nave[last_min_idx]
            val_cut_smooth = y_nave_smooth[last_min_idx]

        dt_wp = t_wp[1] - t_wp[0]
        size_wp = max(1, int(round(1.0 / dt_wp)))
        y_wp_smooth = uniform_filter1d(y_wp, size=size_wp)
        pellet_indices = np.where(y_pellet > 0)[0]
        if len(pellet_indices) > 0:
            t_pellet = t_nir[pellet_indices[0]]
            mask_wp = (t_wp >= t_pellet) & (t_wp <= 350.0)
            peaks_wp, _ = find_peaks(-y_wp_smooth[mask_wp], prominence=0.02 * np.ptp(y_wp_smooth[mask_wp]))
            if len(peaks_wp) > 0:
                t_wp_min = t_wp[mask_wp][peaks_wp[0]]
            else:
                t_wp_min = t_wp[mask_wp][np.argmin(y_wp_smooth[mask_wp])]
        else:
            t_wp_min = None

        if ne_mask_until_ms is not None:
            t_cut = ne_mask_until_ms
            print(f"[Override] Using user-specified t_cut = {t_cut:.3f} ms (auto: {t_cut_auto:.3f} ms)")
        else:
            t_cut = t_cut_auto

        diff = abs(t_cut - t_wp_min) if t_wp_min is not None else None
        print(f"t_cut: {t_cut:.3f} ms (n_e raw: {val_cut_raw:.4f}, smooth: {val_cut_smooth:.4f})")
        print(f"t_max: {t_max:.3f} ms (n_e raw: {val_max_raw:.4f}, smooth: {val_max_smooth:.4f})")
        if t_wp_min is not None:
            print(f"W_p local minimum time: {t_wp_min:.3f} ms (difference: {diff:.3f} ms)")
            if diff > 5.0:
                print(f"WARNING: Difference between t_cut and W_p minimum ({diff:.3f} ms) exceeds 5.0 ms!")
            else:
                print(f"Cross-check passed: Difference ({diff:.3f} ms) <= 5.0 ms.")

        y_nave_plot = y_nave.copy()
        mask_cut = t_nave < t_cut
        y_nave_plot[mask_cut] = np.nan
        print(f"Masked samples: {int(np.sum(mask_cut))} / {len(t_nave)}")
    else:
        y_nave_plot = None

    plt.rcdefaults()
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6.5))

    ax1.plot(t_wp, y_wp, color='blue', linewidth=2.0, label='Wp (kJ)')
    ax1.plot(t_nir, y_pellet * 0.5, color='orange', linewidth=1.5, label='Pellet Injection')
    ax1.plot(t_ech, y_ech, color='green', linewidth=1.8, label='ECH (V)')
    if total_nbi is not None:
        ax1.plot(t_nbi, total_nbi, color='red', linewidth=1.8, label='NBI (V)')

    ax1.set_title(f"Plasma Heating & Confinement / Mirnov Coil Comparison (#{shot})", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Heating & Plasma", fontsize=10)
    ax1.set_ylim(-0.1, 5.0)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, edgecolor='gray')

    if t_nave is not None:
        ax3 = ax1.twinx()
        ax3.plot(t_nave, y_nave_plot, color='purple', linewidth=1.5, label='n_e (10^19 m^-3)')
        ax3.set_ylabel("n_e (10^19 m^-3)", fontsize=10, color='purple')
        ax3.tick_params(axis='y', labelcolor='purple')
        ax3.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, edgecolor='gray')
        ax3.text(0.31, 0.88, f"n_e masked for t < {t_cut:.1f} ms", transform=ax3.transAxes,
                 fontsize=8.5, color='purple', ha='center', va='top', style='italic',
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='purple', alpha=0.85, linewidth=0.5))

    ax2.plot(t_mp1, y_mp1, color='blue', linewidth=0.5, alpha=0.8, label='MP1')
    ax2.set_ylabel("Mirnov (V)", fontsize=10)
    ax2.set_xlabel("Time (ms)", fontsize=10)
    ax2.set_ylim(-1.5, 1.5)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.xlim(150, 350)
    plt.tight_layout()

    output_path = out_dir / f"heating_vs_mirnov (Shot_{shot}).png"
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot plasma heating & confinement vs Mirnov coil.")
    parser.add_argument("--shot", type=int, default=SHOT, help=f"Discharge shot number (default: {SHOT})")
    parser.add_argument("--ne-mask-until-ms", type=float, default=None,
                        help="Override time (ms) up to which n_e is masked (default: automatic)")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    out_path = Path(args.out_dir) if args.out_dir else root_dir / "Images" / "obj2_dynamics_and_structure"
    out_path.mkdir(parents=True, exist_ok=True)
    plot_heating_vs_mirnov(shot=args.shot, out_dir=out_path, ne_mask_until_ms=args.ne_mask_until_ms)

