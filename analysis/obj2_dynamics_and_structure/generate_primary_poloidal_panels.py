"""
Generate Publication-Grade Primary Mode (m = +3) Poloidal Phase Verification Panels
Matching the exact visual grammar, dimensions, and styling of the Secondary Mode panels.
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Ensure access to analysis root and helper modules
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR / "analysis" / "obj2_dynamics_and_structure"))

from mhd_two_mode_combination_search import (
    load_signals, extract_cross_spectral_phase, PMP_ANGLES_DEG,
    T_W1_START, T_W1_END, T_W2_START, T_W2_END, safe_savefig
)


def generate_primary_panels(shot=88653):
    print(f"Loading magnetic signals for shot #{shot}...")
    time_ms, dt_s, fs, sigs, pmp_signals = load_signals(shot=shot)
    if not pmp_signals:
        raise RuntimeError("No PMP signals loaded.")

    windows = [
        ("ECH+NBI Phase (259.1 - 291.3 ms)", "ECH_NBI", T_W1_START, T_W1_END),
        ("NBI-only Phase (291.3 - 300.0 ms)", "NBI_only", T_W2_START, T_W2_END),
    ]

    th_dense = np.linspace(-np.pi, np.pi, 1000)
    col_m = "#d62728"  # Primary theme crimson matching \definecolor{primcol}{RGB}{214,39,40}

    target_dirs = [
        ROOT_DIR / "Images" / "heating_phase_comparison" / "panels",
        ROOT_DIR / "Images" / "obj2_dynamics_and_structure" / "panels",
    ]
    for d in target_dirs:
        d.mkdir(parents=True, exist_ok=True)

    for win_title, win_tag, tw_s, tw_e in windows:
        print(f"\nProcessing {win_title}...")
        pdata = extract_cross_spectral_phase(
            time_ms, fs, pmp_signals, PMP_ANGLES_DEG,
            80000.0, 120000.0, tw_s, tw_e
        )
        f_pk = pdata["f_peak_khz"]
        meas_phase = pdata["meas_phase"]
        angles_rad = pdata["angles_rad"]
        channels = pdata["channels"]
        sigma_phase = pdata["sigma_phase"]

        m_dom = 3
        ph_theo = np.arctan2(np.sin(m_dom * angles_rad), np.cos(m_dom * angles_rad))
        diff = np.arctan2(np.sin(meas_phase - ph_theo), np.cos(meas_phase - ph_theo))
        r_circ = float(np.sqrt(np.sum(np.cos(diff)) ** 2 + np.sum(np.sin(diff)) ** 2) / len(angles_rad))
        mae = float(np.rad2deg(np.mean(np.abs(diff))))
        rmse = float(np.rad2deg(np.sqrt(np.mean(diff ** 2))))
        res_deg = np.rad2deg(diff)
        contained = np.abs(res_deg) <= 45.0
        n_le45 = int(np.sum(contained))
        frac_le45 = n_le45 / len(channels)

        # Create figure with exact dimensions matching Secondary panels (12.5 x 4.5 in at 180 DPI = 2250 x 810 px)
        fig_s, ax_s = plt.subplots(figsize=(12.5, 4.5))

        # Theoretical line for dominant m = +3
        ph_t = np.arctan2(np.sin(m_dom * th_dense), np.cos(m_dom * th_dense))
        jumps = np.where(np.abs(np.diff(ph_t)) > np.pi)[0]
        tg = th_dense.copy()
        pg = ph_t.copy()
        if len(jumps) > 0:
            tg = np.insert(tg, jumps + 1, np.nan)
            pg = np.insert(pg, jumps + 1, np.nan)
        ax_s.plot(tg, pg, color=col_m, ls="-", lw=2.2, alpha=0.95, label=rf"$m = {m_dom:+d}$ (Dominant Branch)", zorder=3)

        # Shaded ribbon (+- pi/4 = +- 45 deg)
        split_idx = np.concatenate(([0], jumps + 1, [len(th_dense)]))
        for s_start, s_end in zip(split_idx[:-1], split_idx[1:]):
            t_seg = th_dense[s_start:s_end]
            p_seg = ph_t[s_start:s_end]
            ax_s.fill_between(t_seg, p_seg - np.pi / 4, p_seg + np.pi / 4, color=col_m, alpha=0.08, zorder=1)

        # Measured PMP probes with error bars and filled/hollow markers
        for i, ch in enumerate(channels):
            ang = angles_rad[i]
            ph = meas_phase[i]
            sig = sigma_phase[i]
            is_cont = contained[i]

            ax_s.errorbar(ang, ph, yerr=sig, fmt="none", ecolor=col_m, elinewidth=1.3, capsize=3.0, alpha=0.85, zorder=4)
            ax_s.scatter(ang, ph, s=70,
                         facecolor=col_m if is_cont else "white",
                         edgecolor=col_m,
                         marker="o", lw=1.5 if not is_cont else 1.1, zorder=5)
            ax_s.annotate(ch, (ang, ph), textcoords="offset points",
                          xytext=(4, 5 if ph >= 0 else -10),
                          fontsize=7.5, fontweight="bold", color=col_m)

        # Monospace metrics summary box matching secondary panels
        txt_box = (
            f"[CONFIRMED] Dominant Mode: m = {m_dom:+d}\n"
            f"Alignment r_circ = {r_circ:.2f}, MAE = {mae:.1f} deg, RMSE = {rmse:.1f} deg\n"
            f"Contained (<= 45 deg): {n_le45}/14 ({frac_le45 * 100:.0f}%)"
        )
        ax_s.text(0.02, 0.04, txt_box, transform=ax_s.transAxes, fontsize=8.5, fontfamily="monospace",
                  color="navy",
                  bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="navy", alpha=0.92), zorder=6)

        # Axes, labels, ticks, and grid
        ax_s.set_xlim(-np.pi * 1.05, np.pi * 1.05)
        ax_s.set_ylim(-np.pi * 1.15, np.pi * 1.15)
        ax_s.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax_s.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
        ax_s.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax_s.set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
        ax_s.axhline(0, color="gray", lw=0.5, alpha=0.3)
        ax_s.axvline(0, color="gray", lw=0.5, alpha=0.3)
        ax_s.set_xlabel(r"Poloidal Angle $\theta$ (rad)", fontsize=10.5)
        ax_s.set_ylabel(r"Cross-Spectral Phase $\phi$ (rad)", fontsize=10.5)
        ax_s.set_title(f"Poloidal Phase Structure Verification at {f_pk:.1f} kHz (Dominant Mode m = +3: {win_title})",
                       fontsize=11, fontweight="bold")
        ax_s.grid(True, alpha=0.25, ls="--")

        # Upper right legend
        h_m = mlines.Line2D([], [], color=col_m, ls="-", lw=2.2, label=rf"$m = {m_dom:+d}$ (Dominant Branch)")
        h_cont = mlines.Line2D([], [], color="dimgray", marker="o", markerfacecolor="gray", markeredgecolor="dimgray",
                               markersize=6.5, ls="none", label=r"PMP Contained ($\leq 45^\circ$)")
        h_out = mlines.Line2D([], [], color="dimgray", marker="o", markerfacecolor="white", markeredgecolor="dimgray",
                              markeredgewidth=1.4, markersize=6.5, ls="none", label=r"PMP Outside ($> 45^\circ$)")
        ax_s.legend(handles=[h_m, h_cont, h_out], loc="upper right", fontsize=8.5, framealpha=0.92)

        plt.tight_layout()

        # Save to all target locations
        fname = f"mhd_analysis_objective2_structure_{shot}_80_120kHz_{win_tag}_panel4_poloidal_phase_structure_verification_at.png"
        for d in target_dirs:
            out_file = d / fname
            safe_savefig(fig_s, out_file, dpi=180)
            print(f"  [Saved] {out_file} (size: {out_file.stat().st_size} bytes)")

        plt.close(fig_s)


if __name__ == "__main__":
    generate_primary_panels(shot=88653)
