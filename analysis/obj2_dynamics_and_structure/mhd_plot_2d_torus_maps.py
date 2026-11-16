import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

current = Path(__file__).resolve().parent
root_dir = None
for p in [current] + list(current.parents):
    if (p / "jpack").exists():
        root_dir = p
        break
if root_dir is None:
    root_dir = current.parents[1]


def generate_2d_torus_maps(shot=88653, out_dir=None):
    if out_dir is None:
        out_dir = root_dir / "Images" / "obj2_dynamics_and_structure"
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = out_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    plt.rcdefaults()
    plt.rcParams.update({
        "mathtext.fontset": "dejavusans",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "axes.edgecolor": "#333333",
        "font.size": 12.0,
        "axes.labelsize": 13.0,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
    })

    R0 = 1.20
    a = 0.17
    a_wall = 0.20
    M_periods = 4
    n = -1
    rho_peak = 0.38
    rho_width = 0.15

    tor_probes = {
        "MP1": (33.3, 0.0),
        "MP3": (213.3, 0.0),
        "MP4": (303.3, 0.0),
        "MP2": (131.3, None),  # Non-coplanar (theta != 0 deg)
    }

    pmp_raw_deg = [0., 10., 20., 30., 40., 50., 60., 80., 90., 100., 110., 120., 150., 180.]
    pmp_angles_deg = [(360.0 - ang) % 360.0 for ang in pmp_raw_deg]

    levels = np.linspace(-1.0, 1.0, 51)
    cmap = plt.cm.RdBu_r

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.6, 4.9), facecolor="white")
    fig.suptitle(f"Heliotron J #{shot}", fontsize=15.0, fontweight="bold", y=0.97)

    # =========================================================================
    # (a) Top-down equatorial view (Shared n = -1 Toroidal Structure)
    # =========================================================================
    r_grid = np.linspace(R0 - a, R0 + a, 140)
    phi_grid = np.linspace(0, 2 * np.pi, 400)
    R_mesh, Phi_mesh = np.meshgrid(r_grid, phi_grid)

    rho_tor = np.abs(R_mesh - R0) / a
    rad_env_tor = np.exp(-((rho_tor - rho_peak) / rho_width) ** 2)
    delta_B_tor = rad_env_tor * np.cos(n * Phi_mesh)

    X_mesh = R_mesh * np.cos(Phi_mesh)
    Y_mesh = R_mesh * np.sin(Phi_mesh)

    cf1 = ax1.contourf(X_mesh, Y_mesh, delta_B_tor, levels=levels, cmap=cmap, extend="both")

    phi_dense = np.linspace(0, 2 * np.pi, 500)
    ax1.plot((R0 - a) * np.cos(phi_dense), (R0 - a) * np.sin(phi_dense), "k--", lw=1.1, label=r"LCFS ($a = 0.17$ m)")
    ax1.plot((R0 + a) * np.cos(phi_dense), (R0 + a) * np.sin(phi_dense), "k--", lw=1.1)
    ax1.plot((R0 - a_wall) * np.cos(phi_dense), (R0 - a_wall) * np.sin(phi_dense), "k-", lw=1.6, label=r"Vessel ($a_\mathrm{wall} = 0.20$ m)")
    ax1.plot((R0 + a_wall) * np.cos(phi_dense), (R0 + a_wall) * np.sin(phi_dense), "k-", lw=1.6)

    # Resonant flux surface circle
    ax1.plot((R0 + rho_peak * a) * np.cos(phi_dense), (R0 + rho_peak * a) * np.sin(phi_dense), color="purple", ls=":", lw=1.2, label=r"$\rho \sim 0.29–0.47$ (ECE peak)")
    ax1.plot((R0 - rho_peak * a) * np.cos(phi_dense), (R0 - rho_peak * a) * np.sin(phi_dense), color="purple", ls=":", lw=1.2)

    R_axis = R0 + 0.045 * np.cos(M_periods * phi_dense)
    ax1.plot(R_axis * np.cos(phi_dense), R_axis * np.sin(phi_dense), color="darkgoldenrod", ls="-.", lw=1.5, label="Magnetic axis ($M = 4$)")

    for p_name, (phi_deg, theta_deg) in tor_probes.items():
        phi_rad = np.radians(phi_deg)
        xp = (R0 + a_wall) * np.cos(phi_rad)
        yp = (R0 + a_wall) * np.sin(phi_rad)
        is_coplanar = (theta_deg == 0.0)
        marker = "o" if is_coplanar else "^"
        col = "black" if is_coplanar else "firebrick"
        s = 60 if is_coplanar else 80
        ax1.scatter(xp, yp, marker=marker, color=col, s=s, zorder=6, edgecolors="white", lw=0.9)

        dx = 0.16 * np.cos(phi_rad)
        dy = 0.16 * np.sin(phi_rad)
        ha = "left" if dx >= 0 else "right"
        va = "bottom" if dy >= 0 else "top"
        if p_name == "MP1":
            ha, va, dx, dy = "left", "center", 0.16, 0.04
        elif p_name == "MP2":
            ha, va, dx, dy = "right", "bottom", -0.15, 0.14
        elif p_name == "MP3":
            ha, va, dx, dy = "right", "top", -0.16, -0.16
        elif p_name == "MP4":
            ha, va, dx, dy = "left", "top", 0.14, -0.15
        lbl = f"{p_name}" if is_coplanar else rf"{p_name} ($\theta \neq 0^\circ$)"
        ax1.text(xp + dx, yp + dy, lbl, fontsize=10.5, ha=ha, va=va, fontweight="bold",
                 color="black" if is_coplanar else "firebrick",
                 bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.82),
                 zorder=8)

    ax1.annotate(
        "NBI (BL1, Ctr)", xy=(0.78, 0.92), xytext=(0.05, 1.66),
        arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.8),
        fontsize=10.0, fontweight="bold", color="tab:red", ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="tab:red", lw=0.6, alpha=0.90),
        zorder=9
    )
    ax1.annotate(
        "NBI (BL2, Co)", xy=(-1.12, -0.35), xytext=(-2.04, 0.18),
        arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.8),
        fontsize=10.0, fontweight="bold", color="tab:red", ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="tab:red", lw=0.6, alpha=0.90),
        zorder=9
    )

    ax1.set_aspect("equal")
    ax1.set_xlim(-2.12, 2.02)
    ax1.set_ylim(-2.02, 2.02)
    ax1.set_xlabel(r"Equatorial Coordinate $X$ [m]", fontsize=13.0)
    ax1.set_ylabel(r"Equatorial Coordinate $Y$ [m]", fontsize=13.0)
    ax1.set_title(r"(a) Equatorial View ($n = -1$)", fontsize=13.5, fontweight="bold", pad=8)
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9.0, framealpha=0.92)

    cbar1 = fig.colorbar(cf1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label(r"Normalized Perturbation $\delta B_\theta$ [a.u.]", fontsize=11.0)
    cbar1.ax.tick_params(labelsize=11.0)

    # =========================================================================
    # (b) Poloidal Cross-Section: Secondary Mode (m = +1 and +12 coexisting)
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

    # Secondary coexisting m = +1 fundamental and m = +12 boundary ripple
    delta_Phi_sec = rad_env_pol * (0.65 * np.cos(Theta_pol) + 0.35 * np.cos(12 * Theta_pol))
    cf2 = ax2.contourf(X_cs, Z_cs, delta_Phi_sec, levels=levels, cmap=cmap, extend="both")

    for rho_val in [0.2, 0.4, 0.6, 0.8, 1.0]:
        rf = rho_val * a
        xf = rf * np.cos(theta_arr) - delta_tri * (rf ** 2 / a) * np.cos(2 * theta_arr)
        zf = kappa * rf * np.sin(theta_arr)
        if abs(rho_val - 1.0) < 1e-4:
            ax2.plot(xf, zf, "k--", lw=1.2, label=r"LCFS ($\rho = 1.0$)")
        elif abs(rho_val - 0.4) < 1e-4:
            ax2.plot(xf, zf, color="purple", ls="--", lw=1.5, label=r"Resonant ($\rho \sim 0.29–0.47$)")
        else:
            ax2.plot(xf, zf, color="gray", ls=":", lw=0.7)

    ax2.plot(a_wall * np.cos(theta_arr), kappa * a_wall * np.sin(theta_arr), "k-", lw=1.6, label="Vessel wall")

    for i, ang in enumerate(pmp_angles_deg):
        th = np.radians(ang)
        xp = (a_wall + 0.012) * np.cos(th)
        zp = kappa * (a_wall + 0.012) * np.sin(th)
        ax2.scatter(xp, zp, color="black", s=28, zorder=6, edgecolors="white", lw=0.6)
        if i in [0, 4, 8, 13]:
            label = f"PMP{i+1}"
            dx = 0.024 * np.cos(th)
            dz = 0.024 * np.sin(th)
            ha = "left" if dx >= 0 else "right"
            va = "bottom" if dz >= 0 else "top"
            if i == 0:
                dx, dz, ha, va = 0.016, 0.024, "left", "bottom"
            elif i == 8:
                dx, dz, ha, va = 0.0, -0.022, "center", "top"
            elif i == 13:
                dx, dz, ha, va = -0.016, 0.022, "right", "bottom"
            ax2.text(xp + dx, zp + dz, label, fontsize=10.2, ha=ha, va=va, fontweight="bold")

    ax2.axhline(0, color="darkgreen", ls="-.", lw=1.0, alpha=0.7)
    ece_peak_x = rho_peak * a
    ax2.scatter([ece_peak_x], [0], marker="X", color="darkgreen", s=95, lw=1.2, edgecolors="black", zorder=7,
                label=r"ECE12/13 ($\rho \sim 0.29–0.47$)")

    ax2.set_aspect("equal")
    ax2.set_xlim(-0.34, 0.34)
    ax2.set_ylim(-0.34, 0.34)
    ax2.set_xlabel(r"Major Radius Offset $R - R_0$ [m]", fontsize=13.0)
    ax2.set_ylabel(r"Vertical Coordinate $Z$ [m]", fontsize=13.0)
    ax2.set_title(r"(b) Secondary ($m = +1 \wedge +12$)", fontsize=13.5, fontweight="bold", pad=8)
    ax2.grid(True, ls=":", alpha=0.4)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9.0, framealpha=0.92)

    cbar2 = fig.colorbar(cf2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label(r"Normalized Potential $\tilde{\phi}_{m=1\wedge 12}$ [a.u.]", fontsize=10.8)
    cbar2.ax.tick_params(labelsize=11.0)

    # =========================================================================
    # (c) Poloidal Cross-Section: Primary Mode (m = +3)
    # =========================================================================
    delta_Phi_prim = rad_env_pol * np.cos(3 * Theta_pol)
    cf3 = ax3.contourf(X_cs, Z_cs, delta_Phi_prim, levels=levels, cmap=cmap, extend="both")

    for rho_val in [0.2, 0.4, 0.6, 0.8, 1.0]:
        rf = rho_val * a
        xf = rf * np.cos(theta_arr) - delta_tri * (rf ** 2 / a) * np.cos(2 * theta_arr)
        zf = kappa * rf * np.sin(theta_arr)
        if abs(rho_val - 1.0) < 1e-4:
            ax3.plot(xf, zf, "k--", lw=1.2, label=r"LCFS ($\rho = 1.0$)")
        elif abs(rho_val - 0.4) < 1e-4:
            ax3.plot(xf, zf, color="purple", ls="--", lw=1.5, label=r"Resonant ($\rho \sim 0.29–0.47$)")
        else:
            ax3.plot(xf, zf, color="gray", ls=":", lw=0.7)

    ax3.plot(a_wall * np.cos(theta_arr), kappa * a_wall * np.sin(theta_arr), "k-", lw=1.6, label="Vessel wall")

    for i, ang in enumerate(pmp_angles_deg):
        th = np.radians(ang)
        xp = (a_wall + 0.012) * np.cos(th)
        zp = kappa * (a_wall + 0.012) * np.sin(th)
        ax3.scatter(xp, zp, color="black", s=28, zorder=6, edgecolors="white", lw=0.6)
        if i in [0, 4, 8, 13]:
            label = f"PMP{i+1}"
            dx = 0.024 * np.cos(th)
            dz = 0.024 * np.sin(th)
            ha = "left" if dx >= 0 else "right"
            va = "bottom" if dz >= 0 else "top"
            if i == 0:
                dx, dz, ha, va = 0.016, 0.024, "left", "bottom"
            elif i == 8:
                dx, dz, ha, va = 0.0, -0.022, "center", "top"
            elif i == 13:
                dx, dz, ha, va = -0.016, 0.022, "right", "bottom"
            ax3.text(xp + dx, zp + dz, label, fontsize=10.2, ha=ha, va=va, fontweight="bold")

    ax3.axhline(0, color="darkgreen", ls="-.", lw=1.0, alpha=0.7)
    ax3.scatter([ece_peak_x], [0], marker="X", color="darkgreen", s=95, lw=1.2, edgecolors="black", zorder=7,
                label=r"ECE12/13 ($\rho \sim 0.29–0.47$)")

    ax3.set_aspect("equal")
    ax3.set_xlim(-0.34, 0.34)
    ax3.set_ylim(-0.34, 0.34)
    ax3.set_xlabel(r"Major Radius Offset $R - R_0$ [m]", fontsize=13.0)
    ax3.set_ylabel(r"Vertical Coordinate $Z$ [m]", fontsize=13.0)
    ax3.set_title(r"(c) Primary ($m = +3$)", fontsize=13.5, fontweight="bold", pad=8)
    ax3.grid(True, ls=":", alpha=0.4)
    ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9.0, framealpha=0.92)

    cbar3 = fig.colorbar(cf3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label(r"Normalized Potential $\tilde{\phi}_{m=3}$ [a.u.]", fontsize=10.8)
    cbar3.ax.tick_params(labelsize=11.0)

    plt.tight_layout(rect=[0, 0.04, 1, 0.91], w_pad=1.6)
    out_png = out_dir / f"mhd_mode_spatial_map_2d_{shot}.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"2D torus map saved to: '{out_png}'")

    # Save standalone panels for high-res slide inclusion
    # Panel 1: Equatorial View
    fig1, ax_p1 = plt.subplots(figsize=(6.2, 5.5), facecolor="white")
    cf_p1 = ax_p1.contourf(X_mesh, Y_mesh, delta_B_tor, levels=levels, cmap=cmap, extend="both")
    ax_p1.plot((R0 - a) * np.cos(phi_dense), (R0 - a) * np.sin(phi_dense), "k--", lw=1.1, label=r"LCFS ($a = 0.17$\,m)")
    ax_p1.plot((R0 + a) * np.cos(phi_dense), (R0 + a) * np.sin(phi_dense), "k--", lw=1.1)
    ax_p1.plot((R0 - a_wall) * np.cos(phi_dense), (R0 - a_wall) * np.sin(phi_dense), "k-", lw=1.6, label=r"Vessel ($a_\mathrm{wall} = 0.20$\,m)")
    ax_p1.plot((R0 + a_wall) * np.cos(phi_dense), (R0 + a_wall) * np.sin(phi_dense), "k-", lw=1.6)
    ax_p1.plot((R0 + rho_peak * a) * np.cos(phi_dense), (R0 + rho_peak * a) * np.sin(phi_dense), color="purple", ls=":", lw=1.2, label=r"$\rho \sim 0.29–0.47$ (ECE peak)")
    ax_p1.plot((R0 - rho_peak * a) * np.cos(phi_dense), (R0 - rho_peak * a) * np.sin(phi_dense), color="purple", ls=":", lw=1.2)
    ax_p1.plot(R_axis * np.cos(phi_dense), R_axis * np.sin(phi_dense), color="darkgoldenrod", ls="-.", lw=1.5, label="Magnetic axis ($M = 4$)")
    for p_name, (phi_deg, theta_deg) in tor_probes.items():
        phi_rad = np.radians(phi_deg)
        xp = (R0 + a_wall) * np.cos(phi_rad)
        yp = (R0 + a_wall) * np.sin(phi_rad)
        is_coplanar = (theta_deg == 0.0)
        marker = "o" if is_coplanar else "^"
        col = "black" if is_coplanar else "firebrick"
        s = 60 if is_coplanar else 80
        ax_p1.scatter(xp, yp, marker=marker, color=col, s=s, zorder=6, edgecolors="white", lw=0.9)
        dx = 0.14 * np.cos(phi_rad)
        dy = 0.14 * np.sin(phi_rad)
        ha = "left" if dx >= 0 else "right"
        va = "bottom" if dy >= 0 else "top"
        lbl = f"{p_name}" if is_coplanar else rf"{p_name} ($\theta \neq 0^\circ$)"
        ax_p1.text(xp + dx, yp + dy, lbl, fontsize=8.5, ha=ha, va=va, fontweight="bold",
                   color="black" if is_coplanar else "firebrick")
    ax_p1.annotate("NBI (BL1, Ctr)", xy=(0.85, 0.90), xytext=(0.30, 1.40),
                   arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.6),
                   fontsize=8.5, fontweight="bold", color="tab:red")
    ax_p1.annotate("NBI (BL2, Co)", xy=(-0.90, -0.85), xytext=(-1.45, -0.35),
                   arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.6),
                   fontsize=8.5, fontweight="bold", color="tab:red")
    ax_p1.set_aspect("equal")
    ax_p1.set_xlim(-1.75, 1.75)
    ax_p1.set_ylim(-1.75, 1.75)
    ax_p1.set_xlabel(r"Equatorial Coordinate $X$ [m]", fontsize=9.5)
    ax_p1.set_ylabel(r"Equatorial Coordinate $Y$ [m]", fontsize=9.5)
    ax_p1.set_title(r"(a) Equatorial Torus View ($n = -1$)", fontsize=11, fontweight="bold", pad=8)
    ax_p1.grid(True, ls=":", alpha=0.4)
    ax_p1.legend(loc="lower left", fontsize=8, framealpha=0.92)
    cb_p1 = fig1.colorbar(cf_p1, ax=ax_p1, fraction=0.046, pad=0.04)
    cb_p1.set_label(r"Normalized Perturbation $\delta B_\theta$ [a.u.]", fontsize=9)
    plt.tight_layout()
    fig1.savefig(panels_dir / f"mhd_mode_spatial_map_2d_{shot}_panel1_a_top_down_equatorial_view.png", dpi=200)
    plt.close(fig1)

    # Panel 2: Secondary Mode (m = +1 and +12 coexisting)
    fig2, ax_p2 = plt.subplots(figsize=(5.5, 5.5), facecolor="white")
    cf_p2 = ax_p2.contourf(X_cs, Z_cs, delta_Phi_sec, levels=levels, cmap=cmap, extend="both")
    for rho_val in [0.2, 0.4, 0.6, 0.8, 1.0]:
        rf = rho_val * a
        xf = rf * np.cos(theta_arr) - delta_tri * (rf ** 2 / a) * np.cos(2 * theta_arr)
        zf = kappa * rf * np.sin(theta_arr)
        if abs(rho_val - 1.0) < 1e-4:
            ax_p2.plot(xf, zf, "k--", lw=1.2, label=r"LCFS ($\rho = 1.0$)")
        elif abs(rho_val - 0.4) < 1e-4:
            ax_p2.plot(xf, zf, color="purple", ls="--", lw=1.5, label=r"Resonant ($\rho \sim 0.29–0.47$)")
        else:
            ax_p2.plot(xf, zf, color="gray", ls=":", lw=0.7)
    ax_p2.plot(a_wall * np.cos(theta_arr), kappa * a_wall * np.sin(theta_arr), "k-", lw=1.6, label="Vessel wall")
    for i, ang in enumerate(pmp_angles_deg):
        th = np.radians(ang)
        xp = (a_wall + 0.012) * np.cos(th)
        zp = kappa * (a_wall + 0.012) * np.sin(th)
        ax_p2.scatter(xp, zp, color="black", s=28, zorder=6, edgecolors="white", lw=0.6)
        if i in [0, 4, 8, 13]:
            label = f"PMP{i+1}"
            dx = 0.032 * np.cos(th)
            dz = 0.032 * np.sin(th)
            ha = "left" if dx >= 0 else "right"
            va = "bottom" if dz >= 0 else "top"
            if i == 8:
                va = "top"
                dz = -0.015
            ax_p2.text(xp + dx, zp + dz, label, fontsize=8, ha=ha, va=va)
    ax_p2.axhline(0, color="darkgreen", ls="-.", lw=1.0, alpha=0.7)
    ax_p2.scatter([ece_peak_x], [0], marker="X", color="darkgreen", s=90, lw=1.2, edgecolors="black", zorder=7,
                 label=r"ECE12/13 ($\rho \sim 0.29–0.47$)")
    ax_p2.set_aspect("equal")
    ax_p2.set_xlim(-0.27, 0.27)
    ax_p2.set_ylim(-0.31, 0.31)
    ax_p2.set_xlabel(r"Major Radius Offset $R - R_0$ [m]", fontsize=9.5)
    ax_p2.set_ylabel(r"Vertical Coordinate $Z$ [m]", fontsize=9.5)
    ax_p2.set_title(r"(b) Secondary Mode Poloidal ($m = +1 \wedge +12$)", fontsize=11, fontweight="bold", pad=8)
    ax_p2.grid(True, ls=":", alpha=0.4)
    ax_p2.legend(loc="lower left", fontsize=8, framealpha=0.92)
    cb_p2 = fig2.colorbar(cf_p2, ax=ax_p2, fraction=0.046, pad=0.04)
    cb_p2.set_label(r"Normalized Potential $\tilde{\phi}_{m=1\wedge 12}$ [a.u.]", fontsize=9)
    plt.tight_layout()
    fig2.savefig(panels_dir / f"mhd_mode_spatial_map_2d_{shot}_panel2_b_poloidal_cross_section_secondary_m1_m12.png", dpi=200)
    fig2.savefig(panels_dir / f"mhd_mode_spatial_map_2d_{shot}_panel2_b_poloidal_cross_section_secondary_m2.png", dpi=200)
    plt.close(fig2)

    # Panel 3: Primary Mode (m = +3)
    fig3, ax_p3 = plt.subplots(figsize=(5.5, 5.5), facecolor="white")
    cf_p3 = ax_p3.contourf(X_cs, Z_cs, delta_Phi_prim, levels=levels, cmap=cmap, extend="both")
    for rho_val in [0.2, 0.4, 0.6, 0.8, 1.0]:
        rf = rho_val * a
        xf = rf * np.cos(theta_arr) - delta_tri * (rf ** 2 / a) * np.cos(2 * theta_arr)
        zf = kappa * rf * np.sin(theta_arr)
        if abs(rho_val - 1.0) < 1e-4:
            ax_p3.plot(xf, zf, "k--", lw=1.2, label=r"LCFS ($\rho = 1.0$)")
        elif abs(rho_val - 0.4) < 1e-4:
            ax_p3.plot(xf, zf, color="purple", ls="--", lw=1.5, label=r"Resonant ($\rho \sim 0.29–0.47$)")
        else:
            ax_p3.plot(xf, zf, color="gray", ls=":", lw=0.7)
    ax_p3.plot(a_wall * np.cos(theta_arr), kappa * a_wall * np.sin(theta_arr), "k-", lw=1.6, label="Vessel wall")
    for i, ang in enumerate(pmp_angles_deg):
        th = np.radians(ang)
        xp = (a_wall + 0.012) * np.cos(th)
        zp = kappa * (a_wall + 0.012) * np.sin(th)
        ax_p3.scatter(xp, zp, color="black", s=28, zorder=6, edgecolors="white", lw=0.6)
        if i in [0, 4, 8, 13]:
            label = f"PMP{i+1}"
            dx = 0.032 * np.cos(th)
            dz = 0.032 * np.sin(th)
            ha = "left" if dx >= 0 else "right"
            va = "bottom" if dz >= 0 else "top"
            if i == 8:
                va = "top"
                dz = -0.015
            ax_p3.text(xp + dx, zp + dz, label, fontsize=8, ha=ha, va=va)
    ax_p3.axhline(0, color="darkgreen", ls="-.", lw=1.0, alpha=0.7)
    ax_p3.scatter([ece_peak_x], [0], marker="X", color="darkgreen", s=90, lw=1.2, edgecolors="black", zorder=7,
                 label=r"ECE12/13 ($\rho \sim 0.29–0.47$)")
    ax_p3.set_aspect("equal")
    ax_p3.set_xlim(-0.27, 0.27)
    ax_p3.set_ylim(-0.31, 0.31)
    ax_p3.set_xlabel(r"Major Radius Offset $R - R_0$ [m]", fontsize=9.5)
    ax_p3.set_ylabel(r"Vertical Coordinate $Z$ [m]", fontsize=9.5)
    ax_p3.set_title(r"(c) Primary Mode Poloidal ($m = +3$)", fontsize=11, fontweight="bold", pad=8)
    ax_p3.grid(True, ls=":", alpha=0.4)
    ax_p3.legend(loc="lower left", fontsize=8, framealpha=0.92)
    cb_p3 = fig3.colorbar(cf_p3, ax=ax_p3, fraction=0.046, pad=0.04)
    cb_p3.set_label(r"Normalized Potential $\tilde{\phi}_{m=3}$ [a.u.]", fontsize=9)
    plt.tight_layout()
    fig3.savefig(panels_dir / f"mhd_mode_spatial_map_2d_{shot}_panel3_c_poloidal_cross_section_primary_m3.png", dpi=200)
    plt.close(fig3)

    return str(out_png)


if __name__ == "__main__":
    generate_2d_torus_maps()
