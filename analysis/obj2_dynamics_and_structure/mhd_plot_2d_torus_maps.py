"""
mhd_plot_2d_torus_maps.py
Publication-style 2D spatial mapping of the primary MHD mode (m=3, n=-1) on Heliotron J:
(a) Equatorial plane top-down view (toroidal structure n=-1)
(b) Poloidal cross-section (modal structure m=3, localized at rho ~ 0.6)
Typography matched identically to the rest of the analysis codebase (default sans-serif, bold headers).
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def generate_2d_torus_maps(shot=88653, out_dir="."):
    out_dir = Path(out_dir)

    # Reset rcParams to match the rest of the codebase (standard sans-serif)
    plt.rcdefaults()

    # Machine & Mode parameters
    R0 = 1.20        # Major radius (m)
    a = 0.17         # Minor radius (m)
    a_wall = 0.20    # Vessel radius (m)
    M_periods = 4    # Field periodicity
    m = 3            # Poloidal mode number
    n = -1           # Toroidal mode number
    f_khz = 89.0     # Mode frequency (kHz)
    rho_peak = 0.62  # Radial peak location
    rho_width = 0.16 # Radial width

    # Mirnov probes
    tor_probes = {
        "MP1": (33.3, 0.0),
        "MP3": (213.3, 0.0),
        "MP4": (303.3, 0.0),
        "MP2": (131.3, 90.0),
    }

    pmp_raw_deg = [0., 10., 20., 30., 40., 50., 60., 80., 90., 100., 110., 120., 150., 180.]
    pmp_angles_deg = [(360.0 - ang) % 360.0 for ang in pmp_raw_deg]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), facecolor="white")
    fig.suptitle(f"Heliotron J #{shot} — Primary Mode 2D Spatial Structure (f ≈ {f_khz:.1f} kHz, m = 3, n = -1)", fontsize=13, fontweight="bold", y=0.97)

    # =========================================================================
    # (a) TOP-DOWN EQUATORIAL VIEW (X-Y)
    # =========================================================================
    r_grid = np.linspace(R0 - a, R0 + a, 140)
    phi_grid = np.linspace(0, 2 * np.pi, 400)
    R_mesh, Phi_mesh = np.meshgrid(r_grid, phi_grid)

    rho_tor = np.abs(R_mesh - R0) / a
    rad_env_tor = np.exp(-((rho_tor - rho_peak) / rho_width) ** 2)
    delta_B_tor = rad_env_tor * np.cos(n * Phi_mesh)

    X_mesh = R_mesh * np.cos(Phi_mesh)
    Y_mesh = R_mesh * np.sin(Phi_mesh)

    levels = np.linspace(-1.0, 1.0, 51)
    cmap = plt.cm.RdBu_r

    cf1 = ax1.contourf(X_mesh, Y_mesh, delta_B_tor, levels=levels, cmap=cmap, extend="both")

    # Boundaries
    phi_dense = np.linspace(0, 2 * np.pi, 500)
    ax1.plot((R0 - a) * np.cos(phi_dense), (R0 - a) * np.sin(phi_dense), "k--", lw=1.1, label=r"LCFS ($a = 0.17$ m)")
    ax1.plot((R0 + a) * np.cos(phi_dense), (R0 + a) * np.sin(phi_dense), "k--", lw=1.1)
    ax1.plot((R0 - a_wall) * np.cos(phi_dense), (R0 - a_wall) * np.sin(phi_dense), "k-", lw=1.6, label=r"Vessel ($a_\mathrm{wall} = 0.20$ m)")
    ax1.plot((R0 + a_wall) * np.cos(phi_dense), (R0 + a_wall) * np.sin(phi_dense), "k-", lw=1.6)

    # Helical axis
    R_axis = R0 + 0.045 * np.cos(M_periods * phi_dense)
    ax1.plot(R_axis * np.cos(phi_dense), R_axis * np.sin(phi_dense), color="darkgoldenrod", ls="-.", lw=1.5, label="Magnetic axis ($M = 4$)")

    # Toroidal Probes
    for p_name, (phi_deg, theta_deg) in tor_probes.items():
        phi_rad = np.radians(phi_deg)
        xp = (R0 + a_wall) * np.cos(phi_rad)
        yp = (R0 + a_wall) * np.sin(phi_rad)
        is_coplanar = (theta_deg == 0.0)
        marker = "o" if is_coplanar else "^"
        col = "black" if is_coplanar else "dimgray"
        ax1.scatter(xp, yp, marker=marker, color=col, s=55, zorder=6, edgecolors="white", lw=0.8)

        dx = 0.12 * np.cos(phi_rad)
        dy = 0.12 * np.sin(phi_rad)
        ha = "left" if dx >= 0 else "right"
        va = "bottom" if dy >= 0 else "top"
        lbl = f"{p_name}" if is_coplanar else rf"{p_name} ($\theta \neq 0^\circ$)"
        ax1.text(xp + dx, yp + dy, lbl, fontsize=9, ha=ha, va=va, fontweight="bold")

    # NBI injection vectors: BL1 (counter-injection) and BL2 (co-injection)
    ax1.annotate(
        "NBI (BL1, Ctr-Inj)", xy=(0.85, 0.90), xytext=(0.35, 1.35),
        arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.8),
        fontsize=9, fontweight="bold", color="tab:red"
    )
    ax1.annotate(
        "NBI (BL2, Co-Inj)", xy=(-0.90, -0.85), xytext=(-1.40, -0.40),
        arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.8),
        fontsize=9, fontweight="bold", color="tab:red"
    )

    ax1.set_aspect("equal")
    ax1.set_xlim(-1.65, 1.65)
    ax1.set_ylim(-1.65, 1.65)
    ax1.set_xlabel("X (m)", fontsize=10)
    ax1.set_ylabel("Y (m)", fontsize=10)
    ax1.set_title(r"(a) Top-down Equatorial View ($n = -1$)", fontsize=11, fontweight="bold", pad=8)
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="lower left", fontsize=8.5, framealpha=0.9)

    cbar1 = fig.colorbar(cf1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label(r"$\delta B_\theta$ (arb. units)", fontsize=9.5)

    # =========================================================================
    # (b) POLOIDAL CROSS-SECTION VIEW (R - R0 vs Z)
    # =========================================================================
    r_arr = np.linspace(0, a_wall, 160)
    theta_arr = np.linspace(-np.pi, np.pi, 400)
    R_pol, Theta_pol = np.meshgrid(r_arr, theta_arr)

    kappa = 1.22
    delta_tri = 0.08
    X_cs = R_pol * np.cos(Theta_pol) - delta_tri * (R_pol ** 2 / a) * np.cos(2 * Theta_pol)
    Z_cs = kappa * R_pol * np.sin(Theta_pol)

    rho_pol = R_pol / a
    rad_env_pol = np.exp(-((rho_pol - rho_peak) / rho_width) ** 2)
    rad_env_pol[rho_pol > 1.0] *= np.exp(-((rho_pol[rho_pol > 1.0] - 1.0) / 0.05) ** 2)

    delta_Phi_pol = rad_env_pol * np.cos(m * Theta_pol)

    cf2 = ax2.contourf(X_cs, Z_cs, delta_Phi_pol, levels=levels, cmap=cmap, extend="both")

    # Magnetic surfaces
    for rho_val in [0.2, 0.4, 0.6, 0.8, 1.0]:
        rf = rho_val * a
        xf = rf * np.cos(theta_arr) - delta_tri * (rf ** 2 / a) * np.cos(2 * theta_arr)
        zf = kappa * rf * np.sin(theta_arr)
        if abs(rho_val - 1.0) < 1e-4:
            ax2.plot(xf, zf, "k--", lw=1.2, label=r"LCFS ($\rho = 1.0$)")
        elif abs(rho_val - rho_peak) < 0.05:
            ax2.plot(xf, zf, color="purple", ls="--", lw=1.3, label=r"Resonant surface ($\rho \approx 0.62$)")
        else:
            ax2.plot(xf, zf, color="gray", ls=":", lw=0.7)

    # Vacuum vessel wall
    ax2.plot(a_wall * np.cos(theta_arr), kappa * a_wall * np.sin(theta_arr), "k-", lw=1.6, label="Vessel wall")

    # PMP coils
    for i, ang in enumerate(pmp_angles_deg):
        th = np.radians(ang)
        xp = (a_wall + 0.012) * np.cos(th)
        zp = kappa * (a_wall + 0.012) * np.sin(th)
        ax2.scatter(xp, zp, color="black", s=30, zorder=6, edgecolors="white", lw=0.6)
        if i in [0, 4, 8, 13]:
            label = f"PMP{i+1}"
            dx = 0.03 * np.cos(th)
            dz = 0.03 * np.sin(th)
            ha = "left" if dx >= 0 else "right"
            va = "bottom" if dz >= 0 else "top"
            if i == 8: # PMP9 at bottom, shift slightly to avoid border clipping
                va = "top"
                dz = -0.015
            ax2.text(xp + dx, zp + dz, label, fontsize=8.5, ha=ha, va=va)

    # ECE diagnostic chord
    ax2.axhline(0, color="darkgreen", ls="-.", lw=1.0, alpha=0.7)
    ece_peak_x = rho_peak * a
    ax2.scatter([ece_peak_x], [0], marker="x", color="darkgreen", s=80, lw=2.0, zorder=7, label=r"ECE12/13 peak ($\rho \approx 0.62$)")

    ax2.set_aspect("equal")
    ax2.set_xlim(-0.27, 0.27)
    ax2.set_ylim(-0.31, 0.31)
    ax2.set_xlabel(r"$R - R_0$ (m)", fontsize=10)
    ax2.set_ylabel("Z (m)", fontsize=10)
    ax2.set_title(r"(b) Poloidal Cross-Section ($m = 3$)", fontsize=11, fontweight="bold", pad=8)
    ax2.grid(True, ls=":", alpha=0.4)
    ax2.legend(loc="lower left", fontsize=8.5, framealpha=0.9)

    cbar2 = fig.colorbar(cf2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label(r"$\tilde{\phi}$ (arb. units)", fontsize=9.5)

    plt.tight_layout()
    out_png = out_dir / f"mhd_mode_spatial_map_2d_{shot}.png"
    plt.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Clean 2D torus map saved to: '{out_png}'")
    return str(out_png)

if __name__ == "__main__":
    generate_2d_torus_maps(88653, out_dir=r"c:\TFG")
