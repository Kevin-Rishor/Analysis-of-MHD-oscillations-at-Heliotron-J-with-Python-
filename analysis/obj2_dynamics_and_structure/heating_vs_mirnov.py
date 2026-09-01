import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

SHOT = 88653

current = Path(__file__).resolve().parent
root_dir = None
for p in [current] + list(current.parents):
    if (p / "jpack").exists():
        root_dir = p
        break
if root_dir is None:
    root_dir = Path("c:/TFG")

for p_add in [root_dir / "jpack", root_dir / "analysis", root_dir / "analysis" / "common"]:
    if str(p_add) not in sys.path:
        sys.path.append(str(p_add))

import turnelib as TE


def plot_heating_vs_mirnov(shot=SHOT, out_dir=None):
    if out_dir is None:
        out_dir = root_dir
    else:
        out_dir = Path(out_dir)

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

    y_pellet = np.where(y_nir >= 0.15, y_nir, 0.0)
    max_pellet = y_pellet.max()
    if max_pellet > 0:
        y_pellet = y_pellet / max_pellet

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
    ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, edgecolor='gray')

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
    plot_heating_vs_mirnov()

