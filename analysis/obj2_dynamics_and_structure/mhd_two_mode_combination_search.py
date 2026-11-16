#!/usr/bin/env python3
"""
mhd_two_mode_combination_search.py

Exhaustive search and evaluation of two-mode poloidal combinations (m1, m2)
coexisting near the secondary mode frequencies in Heliotron J discharge #88653.

Tests the hypothesis that the secondary mode is composed of two coexisting modes,
evaluating:
1. Combinations involving m = +9 and m = -9 (suggested by professor)
2. Exhaustive grid search over all pairs (m1, m2) in [-20, +20] (820 pairs)
3. Both Model A (Dual-Branch Closest-Assignment) and Model B (Complex Superposition)
4. Evaluation across:
   - Full 14-probe array (PMP1-PMP14)
   - Active 12-probe array (excluding PMP8, PMP14)
   - Across ECH+NBI (48.8 kHz), NBI-only (41.0 kHz), and Flat (42.97 kHz) windows.
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import scipy.signal as dsp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# Setup paths
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = None
for p in [CURRENT_DIR] + list(CURRENT_DIR.parents):
    if (p / "jpack").exists():
        ROOT_DIR = p
        break
if ROOT_DIR is None:
    ROOT_DIR = CURRENT_DIR.parents[1]

for p_add in [ROOT_DIR / "jpack", ROOT_DIR / "analysis", ROOT_DIR / "analysis" / "common", ROOT_DIR / "analysis" / "heating_phase_comparison"]:
    if str(p_add) not in sys.path:
        sys.path.append(str(p_add))

from comparison_common import (
    load_signals, PMP_ANGLES_DEG,
    T_W1_START, T_W1_END, T_W2_START, T_W2_END,
    safe_savefig
)

T_FLAT_START = 259.1
T_FLAT_END = 275.0


def extract_cross_spectral_phase(time_ms, fs, pmp_signals, pmp_angles_deg, fl_hz, fu_hz, tw_start, tw_end,
                                 nfft_max=512):
    """
    Computes cross-spectral phase of each PMP probe relative to PMP1 at peak frequency.
    """
    pmp_channels = sorted(pmp_signals.keys(), key=lambda c: int(c[3:]))
    raw_angles_deg = np.array([pmp_angles_deg[int(ch[3:]) - 1] for ch in pmp_channels])
    angles_rad = np.deg2rad(raw_angles_deg)
    # Map to [-pi, pi]
    angles_rad_in_pi = np.arctan2(np.sin(angles_rad), np.cos(angles_rad))

    idx_win = np.where((time_ms >= tw_start) & (time_ms <= tw_end))[0]
    n_samples = len(idx_win)
    sig_matrix = np.array([pmp_signals[ch][idx_win] for ch in pmp_channels])
    nfft = min(nfft_max, n_samples)

    # 1. Peak frequency via 2D NUDFT power
    f_pmp, _, S_pmp = dsp.spectrogram(
        sig_matrix, fs=fs, window='hann', nperseg=nfft, noverlap=nfft // 2,
        nfft=nfft, detrend='constant', return_onesided=False, scaling='density', axis=-1, mode='complex'
    )
    band_mask = (f_pmp >= fl_hz) & (f_pmp <= fu_hz)
    k_grid = np.arange(-6, 7)
    S_band = S_pmp[:, band_mask, :]
    E = np.exp(-1j * k_grid[:, None] * angles_rad_in_pi[None, :]) / len(angles_rad_in_pi)
    Sk = np.tensordot(E, S_band, axes=(1, 0))
    P2d = np.mean((Sk * Sk.conj()).real, axis=2)
    f_band_khz = f_pmp[band_mask] / 1000.0
    f_peak_khz = float(f_band_khz[int(np.argmax(np.sum(P2d, axis=0)))])

    # 2. Cross-spectral phase at peak frequency relative to PMP1
    f_spec, _, S_cross = dsp.spectrogram(
        sig_matrix, fs=fs, window='hann', nperseg=nfft, noverlap=nfft // 2,
        nfft=nfft, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='complex'
    )
    idx_f_peak = int(np.argmin(np.abs(f_spec - f_peak_khz * 1000.0)))
    S_pk = S_cross[:, idx_f_peak, :]
    ref_pk = S_pk[0, :]
    cross_pk = S_pk * np.conj(ref_pk)[None, :]
    avg_cross = np.mean(cross_pk, axis=1)
    n_seg = S_pk.shape[1]

    p_ref = np.mean(np.abs(ref_pk) ** 2)
    p_ch = np.mean(np.abs(S_pk) ** 2, axis=1)
    coherence_sq = np.clip(np.abs(avg_cross) ** 2 / (p_ref * p_ch + 1e-30), 0.0, 1.0)
    sigma_phase = np.sqrt(np.maximum(0.0, 1.0 - coherence_sq) / (2.0 * np.maximum(coherence_sq, 1e-4) * max(1, n_seg)))
    sigma_phase = np.clip(sigma_phase, 0.0, np.pi)

    # Measured phase relative to PMP1 anchored at 0
    raw_phase = np.angle(avg_cross)
    meas_phase = np.arctan2(np.sin(raw_phase - raw_phase[0]),
                            np.cos(raw_phase - raw_phase[0]))

    return {
        "f_peak_khz": f_peak_khz,
        "meas_phase": meas_phase,
        "avg_cross": avg_cross,
        "coherence_sq": coherence_sq,
        "sigma_phase": sigma_phase,
        "n_seg": n_seg,
        "angles_rad": angles_rad_in_pi,
        "angles_deg": np.rad2deg(angles_rad_in_pi),
        "channels": pmp_channels,
    }


def evaluate_single_mode(angles_rad, meas_phase, m_val):
    """Evaluates single-mode theoretical trajectory phi = m * theta (mod 2pi)."""
    th_theo = np.arctan2(np.sin(m_val * angles_rad), np.cos(m_val * angles_rad))
    diff = np.arctan2(np.sin(meas_phase - th_theo), np.cos(meas_phase - th_theo))
    rc = float(np.sqrt(np.sum(np.cos(diff)) ** 2 + np.sum(np.sin(diff)) ** 2) / len(angles_rad))
    mae_deg = float(np.rad2deg(np.mean(np.abs(diff))))
    rmse_deg = float(np.rad2deg(np.sqrt(np.mean(diff ** 2))))
    max_err_deg = float(np.rad2deg(np.max(np.abs(diff))))
    residuals_deg = np.rad2deg(diff)
    n_le45 = int(np.sum(np.abs(residuals_deg) <= 45.0))
    n_le30 = int(np.sum(np.abs(residuals_deg) <= 30.0))
    return {
        "m": int(m_val),
        "r_circ": rc,
        "mae_deg": mae_deg,
        "rmse_deg": rmse_deg,
        "max_err_deg": max_err_deg,
        "n_le45": n_le45,
        "frac_le45": n_le45 / len(angles_rad),
        "n_le30": n_le30,
        "frac_le30": n_le30 / len(angles_rad),
        "residuals_deg": residuals_deg.tolist(),
    }


def evaluate_two_mode_model_a(angles_rad, meas_phase, m1, m2):
    """
    Model A: Dual-Branch Closest-Assignment Model (m1 ^ m2).
    Each probe is assigned to the closer of the two theoretical branches.
    """
    th_theo1 = np.arctan2(np.sin(m1 * angles_rad), np.cos(m1 * angles_rad))
    th_theo2 = np.arctan2(np.sin(m2 * angles_rad), np.cos(m2 * angles_rad))

    diff1 = np.arctan2(np.sin(meas_phase - th_theo1), np.cos(meas_phase - th_theo1))
    diff2 = np.arctan2(np.sin(meas_phase - th_theo2), np.cos(meas_phase - th_theo2))

    abs_diff1 = np.abs(diff1)
    abs_diff2 = np.abs(diff2)

    # Choice of branch
    choose_branch2 = abs_diff2 < abs_diff1
    best_diff = np.where(choose_branch2, diff2, diff1)
    assigned_branch = np.where(choose_branch2, m2, m1)

    n_probes = len(angles_rad)
    rc = float(np.sqrt(np.sum(np.cos(best_diff)) ** 2 + np.sum(np.sin(best_diff)) ** 2) / n_probes)
    mean_cos = float(np.mean(np.cos(best_diff)))
    mae_deg = float(np.rad2deg(np.mean(np.abs(best_diff))))
    rmse_deg = float(np.rad2deg(np.sqrt(np.mean(best_diff ** 2))))
    max_err_deg = float(np.rad2deg(np.max(np.abs(best_diff))))

    best_diff_deg = np.rad2deg(best_diff)
    n_le45 = int(np.sum(np.abs(best_diff_deg) <= 45.0))
    n_le30 = int(np.sum(np.abs(best_diff_deg) <= 30.0))

    n_m1 = int(np.sum(~choose_branch2))
    n_m2 = int(np.sum(choose_branch2))

    return {
        "m1": int(m1),
        "m2": int(m2),
        "r_circ": rc,
        "mean_cos": mean_cos,
        "mae_deg": mae_deg,
        "rmse_deg": rmse_deg,
        "max_err_deg": max_err_deg,
        "n_le45": n_le45,
        "frac_le45": n_le45 / n_probes,
        "n_le30": n_le30,
        "frac_le30": n_le30 / n_probes,
        "n_m1": n_m1,
        "n_m2": n_m2,
        "assigned_branch": assigned_branch.tolist(),
        "residuals_deg": best_diff_deg.tolist(),
    }


def evaluate_two_mode_model_b(angles_rad, meas_phase, avg_cross, m1, m2):
    """
    Model B: Complex 2-Mode Wavefield Superposition.
    B_hat(theta) = c1 * exp(i*m1*theta) + c2 * exp(i*m2*theta)
    Fits complex amplitudes (c1, c2) via linear least squares.
    """
    n_probes = len(angles_rad)
    z_meas = np.exp(1j * meas_phase)
    # Basis matrix (n_probes, 2)
    A = np.column_stack([np.exp(1j * m1 * angles_rad), np.exp(1j * m2 * angles_rad)])
    
    # Solve complex least squares
    c, residuals, rank, s = np.linalg.lstsq(A, z_meas, rcond=None)
    c1, c2 = c[0], c[1]

    # Reconstructed phasor
    z_fit = A @ c
    # Reconstructed phase relative to probe 0
    phase_fit = np.angle(z_fit)
    phase_fit = np.arctan2(np.sin(phase_fit - phase_fit[0]), np.cos(phase_fit - phase_fit[0]))

    diff = np.arctan2(np.sin(meas_phase - phase_fit), np.cos(meas_phase - phase_fit))
    rc = float(np.sqrt(np.sum(np.cos(diff)) ** 2 + np.sum(np.sin(diff)) ** 2) / n_probes)
    mae_deg = float(np.rad2deg(np.mean(np.abs(diff))))
    rmse_deg = float(np.rad2deg(np.sqrt(np.mean(diff ** 2))))
    max_err_deg = float(np.rad2deg(np.max(np.abs(diff))))

    amp_ratio = float(np.abs(c2) / (np.abs(c1) + 1e-12))
    phase_offset_deg = float(np.rad2deg(np.angle(c2 / (c1 + 1e-12))))

    return {
        "m1": int(m1),
        "m2": int(m2),
        "r_circ_sup": rc,
        "mae_deg_sup": mae_deg,
        "rmse_deg_sup": rmse_deg,
        "max_err_deg_sup": max_err_deg,
        "amp_ratio_c2_c1": amp_ratio,
        "phase_offset_deg": phase_offset_deg,
        "c1": [float(c1.real), float(c1.imag)],
        "c2": [float(c2.real), float(c2.imag)],
    }


def run_full_grid_search(phase_data, active_only=False, max_m=20):
    """
    Runs exhaustive grid search over all pairs (m1, m2) in [-max_m, +max_m].
    """
    angles_rad = phase_data["angles_rad"]
    meas_phase = phase_data["meas_phase"]
    avg_cross = phase_data["avg_cross"]
    channels = phase_data["channels"]

    if active_only:
        active_mask = np.array([ch not in ["PMP8", "PMP14"] for ch in channels])
        angles_rad = angles_rad[active_mask]
        meas_phase = meas_phase[active_mask]
        avg_cross = avg_cross[active_mask]
        channels = [ch for ch, m in zip(channels, active_mask) if m]

    m_candidates = np.arange(-max_m, max_m + 1)
    
    # Single mode evaluations
    single_res = {}
    for m in m_candidates:
        single_res[int(m)] = evaluate_single_mode(angles_rad, meas_phase, m)

    # Pair evaluations
    pair_results = []
    m9_results = []

    for i, m1 in enumerate(m_candidates):
        for m2 in m_candidates[i + 1:]:
            res_a = evaluate_two_mode_model_a(angles_rad, meas_phase, m1, m2)
            res_b = evaluate_two_mode_model_b(angles_rad, meas_phase, avg_cross, m1, m2)
            combined = {**res_a, **res_b}
            pair_results.append(combined)

            if m1 in (9, -9) or m2 in (9, -9):
                m9_results.append(combined)

    return {
        "channels": channels,
        "single_mode": single_res,
        "all_pairs": pair_results,
        "m9_pairs": m9_results,
    }


def generate_heatmaps_figure(pair_results, m_range=12, out_path=None, title_suffix="", key_pairs=None):
    """
    Generates 2D heatmaps over (m1, m2) space for MAE, r_circ, and Containment Count.
    """
    m_vals = np.arange(-m_range, m_range + 1)
    n_m = len(m_vals)
    m_to_idx = {m: i for i, m in enumerate(m_vals)}

    mae_mat = np.full((n_m, n_m), np.nan)
    rc_mat = np.full((n_m, n_m), np.nan)
    n45_mat = np.full((n_m, n_m), np.nan)
    sup_rc_mat = np.full((n_m, n_m), np.nan)

    for r in pair_results:
        m1, m2 = r["m1"], r["m2"]
        if abs(m1) <= m_range and abs(m2) <= m_range:
            i, j = m_to_idx[m1], m_to_idx[m2]
            mae_mat[i, j] = r["mae_deg"]
            mae_mat[j, i] = r["mae_deg"]
            rc_mat[i, j] = r["r_circ"]
            rc_mat[j, i] = r["r_circ"]
            n45_mat[i, j] = r["n_le45"]
            n45_mat[j, i] = r["n_le45"]
            sup_rc_mat[i, j] = r["r_circ_sup"]
            sup_rc_mat[j, i] = r["r_circ_sup"]

    # Fill diagonal with single-mode values
    # (they are already recorded or can be evaluated)

    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Two-Mode Combination (m1, m2) Grid Search Space {title_suffix}", fontsize=13, fontweight='bold')

    extent = [-m_range - 0.5, m_range + 0.5, -m_range - 0.5, m_range + 0.5]

    # Panel 1: MAE (lower is better)
    im1 = axs[0, 0].imshow(mae_mat, origin='lower', extent=extent, cmap='viridis_r', vmin=15, vmax=80)
    plt.colorbar(im1, ax=axs[0, 0], label="Mean Absolute Phase Error (deg)")
    axs[0, 0].set_title("(a) Model A: Mean Absolute Phase Error (deg)")
    axs[0, 0].set_xlabel("Poloidal Mode Number m2")
    axs[0, 0].set_ylabel("Poloidal Mode Number m1")

    # Panel 2: r_circ (higher is better)
    im2 = axs[0, 1].imshow(rc_mat, origin='lower', extent=extent, cmap='plasma', vmin=0.5, vmax=0.98)
    plt.colorbar(im2, ax=axs[0, 1], label="Dual-Branch Circular Concentration r_circ")
    axs[0, 1].set_title("(b) Model A: Circular Concentration r_circ(m1, m2)")
    axs[0, 1].set_xlabel("Poloidal Mode Number m2")
    axs[0, 1].set_ylabel("Poloidal Mode Number m1")

    # Panel 3: Number of probes contained <= 45 deg
    im3 = axs[1, 0].imshow(n45_mat, origin='lower', extent=extent, cmap='RdYlGn', vmin=6, vmax=14)
    plt.colorbar(im3, ax=axs[1, 0], label="Probes Contained (|err| <= 45 deg)")
    axs[1, 0].set_title("(c) Model A: PMP Containment Count (<= 45 deg, max 14)")
    axs[1, 0].set_xlabel("Poloidal Mode Number m2")
    axs[1, 0].set_ylabel("Poloidal Mode Number m1")

    # Panel 4: Superposition r_circ
    im4 = axs[1, 1].imshow(sup_rc_mat, origin='lower', extent=extent, cmap='magma', vmin=0.5, vmax=0.98)
    plt.colorbar(im4, ax=axs[1, 1], label="Superposition r_circ (Model B)")
    axs[1, 1].set_title("(d) Model B: Complex 2-Mode Superposition r_circ")
    axs[1, 1].set_xlabel("Poloidal Mode Number m2")
    axs[1, 1].set_ylabel("Poloidal Mode Number m1")

    # Mark top 2 global pairs on all panels
    if key_pairs is None:
        key_pairs = []
    for ax in axs.flat:
        ax.set_xticks(np.arange(-m_range, m_range + 1, 2))
        ax.set_yticks(np.arange(-m_range, m_range + 1, 2))
        ax.grid(True, alpha=0.3, ls=':')
        for k1, k2 in key_pairs:
            if abs(k1) <= m_range and abs(k2) <= m_range:
                ax.plot([k2, k1], [k1, k2], 'ro', markersize=7, fillstyle='none', markeredgewidth=1.8)

    plt.tight_layout()
    if out_path:
        safe_savefig(fig, out_path, dpi=180)
        print(f"  [Figure] Saved heatmap: {out_path.name}")
    plt.close(fig)


def plot_phase_containment_comparison(windows_data, out_path=None, standalone_dir=None, shot=88653, m1=1, m2=12):
    """
    Plots poloidal phase phi(theta) for coexisting modes (m1 = +1, m2 = +12)
    across the two primary heating windows (ECH+NBI and NBI-only).
    Generates:
      1. Two distinct standalone figures (one per window) without any background lines or non-relevant lines.
      2. Side-by-side 1x2 comparative figure for both windows.
    """
    th_dense = np.linspace(-np.pi, np.pi, 1200)

    # 1. Generate Side-by-Side (1 row x 2 columns) Combined Figure
    fig, axs = plt.subplots(1, 2, figsize=(15.5, 5.5), sharey=True)

    for idx_w, (win_name, pdata) in enumerate(windows_data.items()):
        ax = axs[idx_w]
        meas_phase = pdata["meas_phase"]
        angles_rad = pdata["angles_rad"]
        channels = pdata["channels"]
        sigma_phase = pdata["sigma_phase"]
        f_pk = pdata["f_peak_khz"]

        # Evaluate (m1=+1, m2=+12)
        res = evaluate_two_mode_model_a(angles_rad, meas_phase, m1, m2)
        assigned_branches = res["assigned_branch"]

        # Theoretical lines for m1=+1 and m2=+12
        for m_val, col, ls, lw, lbl in [
            (m1, "#1f77b4", "-", 2.2, rf"$m_1 = {m1:+d}$ (Low-$m$ Branch)"),
            (m2, "#9467bd", "--", 2.2, rf"$m_2 = {m2:+d}$ (Steep Branch)"),
        ]:
            ph_t = np.arctan2(np.sin(m_val * th_dense), np.cos(m_val * th_dense))
            jumps = np.where(np.abs(np.diff(ph_t)) > np.pi)[0]
            tg = th_dense.copy()
            pg = ph_t.copy()
            if len(jumps) > 0:
                tg = np.insert(tg, jumps + 1, np.nan)
                pg = np.insert(pg, jumps + 1, np.nan)
            ax.plot(tg, pg, color=col, ls=ls, lw=lw, alpha=0.95, label=lbl, zorder=3)

        # Shaded +/- 45 deg containment ribbons
        for m_val, col in [(m1, "#1f77b4"), (m2, "#9467bd")]:
            ph_t = np.arctan2(np.sin(m_val * th_dense), np.cos(m_val * th_dense))
            jumps = np.where(np.abs(np.diff(ph_t)) > np.pi)[0]
            split_idx = np.concatenate(([0], jumps + 1, [len(th_dense)]))
            for s_start, s_end in zip(split_idx[:-1], split_idx[1:]):
                t_seg = th_dense[s_start:s_end]
                p_seg = ph_t[s_start:s_end]
                ax.fill_between(t_seg, p_seg - np.pi/4, p_seg + np.pi/4, color=col, alpha=0.08, zorder=1)

        # Measured PMP probes
        for i, ch in enumerate(channels):
            ang = angles_rad[i]
            ph = meas_phase[i]
            sig = sigma_phase[i]
            br = assigned_branches[i]
            col_probe = "#1f77b4" if br == m1 else "#9467bd"
            res_err = abs(res["residuals_deg"][i])
            is_contained = res_err <= 45.0

            ax.errorbar(ang, ph, yerr=sig, fmt='none', ecolor=col_probe, elinewidth=1.3, capsize=3.0, alpha=0.85, zorder=4)
            ax.scatter(ang, ph, s=70,
                       facecolor=col_probe if is_contained else "white",
                       edgecolor=col_probe,
                       marker="o", lw=1.5 if not is_contained else 1.1, zorder=5)
            ax.annotate(ch, (ang, ph), textcoords="offset points",
                        xytext=(4, 5 if ph >= 0 else -10),
                        fontsize=7.5, fontweight='bold', color=col_probe)

        # Status text box
        panel_letter = "(a)" if idx_w == 0 else "(b)"
        txt_box = (
            f"[SUPPORTED] Coexisting Modes: m1 = {m1:+d}, m2 = {m2:+d}\n"
            f"  MAE = {res['mae_deg']:.1f} deg, RMSE = {res['rmse_deg']:.1f} deg, r_circ = {res['r_circ']:.3f}\n"
            f"  Contained (<=45 deg): {res['n_le45']}/14 ({res['frac_le45']*100:.0f}%) [{res['n_m1']} on m1, {res['n_m2']} on m2]"
        )
        ax.text(0.02, 0.04, txt_box, transform=ax.transAxes, fontsize=8.5, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="black", alpha=0.92), zorder=6)

        ax.set_xlim(-np.pi * 1.05, np.pi * 1.05)
        ax.set_ylim(-np.pi * 1.15, np.pi * 1.15)
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
        if idx_w == 0:
            ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
            ax.set_ylabel(r"Cross-Spectral Phase $\phi$ (rad)", fontsize=10.5)
        ax.axhline(0, color="gray", lw=0.5, alpha=0.3)
        ax.axvline(0, color="gray", lw=0.5, alpha=0.3)
        ax.set_xlabel(r"Poloidal Angle $\theta$ (rad)", fontsize=10.5)
        ax.set_title(f"{panel_letter} {win_name} (f_peak = {f_pk:.2f} kHz)", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.25, ls="--")
        
        # Legend with self-explanatory proxy handles
        h_m1 = mlines.Line2D([], [], color="#1f77b4", ls="-", lw=2.2, label=rf"$m_1 = {m1:+d}$ (Low-$m$ Branch)")
        h_m2 = mlines.Line2D([], [], color="#9467bd", ls="--", lw=2.2, label=rf"$m_2 = {m2:+d}$ (Steep Branch)")
        h_cont = mlines.Line2D([], [], color="dimgray", marker="o", markerfacecolor="gray", markeredgecolor="dimgray",
                               markersize=6.5, ls="none", label=r"PMP Contained ($\leq 45^\circ$)")
        h_out = mlines.Line2D([], [], color="dimgray", marker="o", markerfacecolor="white", markeredgecolor="dimgray",
                              markeredgewidth=1.4, markersize=6.5, ls="none", label=r"PMP Outside ($> 45^\circ$)")
        ax.legend(handles=[h_m1, h_m2, h_cont, h_out], loc="upper right", fontsize=8.0, framealpha=0.92)

    plt.tight_layout()
    if out_path:
        safe_savefig(fig, out_path, dpi=180)
        print(f"  [Figure] Saved 2-window side-by-side plot: {out_path.name}")
        # Also copy to heating_phase_comparison
        h_dir = ROOT_DIR / "Images" / "heating_phase_comparison"
        h_dir.mkdir(parents=True, exist_ok=True)
        safe_savefig(fig, h_dir / out_path.name, dpi=180)
    plt.close(fig)

    # 2. Generate TWO DISTINCT Standalone Figure Files (one per window)
    heating_panels_dir = ROOT_DIR / "Images" / "heating_phase_comparison" / "panels"
    heating_panels_dir.mkdir(parents=True, exist_ok=True)
    obj2_panels_dir = ROOT_DIR / "Images" / "obj2_dynamics_and_structure" / "panels"
    obj2_panels_dir.mkdir(parents=True, exist_ok=True)

    for idx_w, (win_name, pdata) in enumerate(windows_data.items()):
        fig_s, ax_s = plt.subplots(figsize=(12.5, 4.5))
        meas_phase = pdata["meas_phase"]
        angles_rad = pdata["angles_rad"]
        channels = pdata["channels"]
        sigma_phase = pdata["sigma_phase"]
        f_pk = pdata["f_peak_khz"]

        res = evaluate_two_mode_model_a(angles_rad, meas_phase, m1, m2)
        assigned_branches = res["assigned_branch"]

        # Theoretical lines for m1=+1 and m2=+12
        for m_val, col, ls, lw, lbl in [
            (m1, "#1f77b4", "-", 2.2, rf"$m_1 = {m1:+d}$ (Low-$m$ Branch)"),
            (m2, "#9467bd", "--", 2.2, rf"$m_2 = {m2:+d}$ (Steep Branch)"),
        ]:
            ph_t = np.arctan2(np.sin(m_val * th_dense), np.cos(m_val * th_dense))
            jumps = np.where(np.abs(np.diff(ph_t)) > np.pi)[0]
            tg = th_dense.copy()
            pg = ph_t.copy()
            if len(jumps) > 0:
                tg = np.insert(tg, jumps + 1, np.nan)
                pg = np.insert(pg, jumps + 1, np.nan)
            ax_s.plot(tg, pg, color=col, ls=ls, lw=lw, alpha=0.95, label=lbl, zorder=3)

        # Shaded ribbons
        for m_val, col in [(m1, "#1f77b4"), (m2, "#9467bd")]:
            ph_t = np.arctan2(np.sin(m_val * th_dense), np.cos(m_val * th_dense))
            jumps = np.where(np.abs(np.diff(ph_t)) > np.pi)[0]
            split_idx = np.concatenate(([0], jumps + 1, [len(th_dense)]))
            for s_start, s_end in zip(split_idx[:-1], split_idx[1:]):
                t_seg = th_dense[s_start:s_end]
                p_seg = ph_t[s_start:s_end]
                ax_s.fill_between(t_seg, p_seg - np.pi/4, p_seg + np.pi/4, color=col, alpha=0.08, zorder=1)

        # Measured PMP probes
        for i, ch in enumerate(channels):
            ang = angles_rad[i]
            ph = meas_phase[i]
            sig = sigma_phase[i]
            br = assigned_branches[i]
            col_probe = "#1f77b4" if br == m1 else "#9467bd"
            res_err = abs(res["residuals_deg"][i])
            is_contained = res_err <= 45.0

            ax_s.errorbar(ang, ph, yerr=sig, fmt='none', ecolor=col_probe, elinewidth=1.3, capsize=3.0, alpha=0.85, zorder=4)
            ax_s.scatter(ang, ph, s=70,
                         facecolor=col_probe if is_contained else "white",
                         edgecolor=col_probe,
                         marker="o", lw=1.5 if not is_contained else 1.1, zorder=5)
            ax_s.annotate(ch, (ang, ph), textcoords="offset points",
                          xytext=(4, 5 if ph >= 0 else -10),
                          fontsize=7.5, fontweight='bold', color=col_probe)

        txt_box = (
            f"[SUPPORTED] Coexisting Modes: m1 = {m1:+d}, m2 = {m2:+d}\n"
            f"Alignment r_circ = {res['r_circ']:.2f}, MAE = {res['mae_deg']:.1f} deg, RMSE = {res['rmse_deg']:.1f} deg\n"
            f"Contained (<= 45 deg): {res['n_le45']}/14 ({res['frac_le45']*100:.0f}%) [{res['n_m1']} on m1, {res['n_m2']} on m2]"
        )
        ax_s.text(0.02, 0.04, txt_box, transform=ax_s.transAxes, fontsize=8.5, fontfamily="monospace",
                  color="navy",
                  bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="navy", alpha=0.92), zorder=6)

        ax_s.set_xlim(-np.pi * 1.05, np.pi * 1.05)
        ax_s.set_ylim(-np.pi * 1.15, np.pi * 1.15)
        ax_s.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax_s.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
        ax_s.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax_s.set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
        ax_s.axhline(0, color="gray", lw=0.5, alpha=0.3)
        ax_s.axvline(0, color="gray", lw=0.5, alpha=0.3)
        ax_s.set_xlabel(r"Poloidal Angle $\theta$ (rad)", fontsize=10.5)
        ax_s.set_ylabel(r"Cross-Spectral Phase $\phi$ (rad)", fontsize=10.5)
        ax_s.set_title(f"Poloidal Phase Structure Verification at {f_pk:.1f} kHz (Coexisting Modes m = +1 and +12: {win_name})", fontsize=11, fontweight='bold')
        ax_s.grid(True, alpha=0.25, ls="--")
        
        # Legend with self-explanatory proxy handles
        h_m1 = mlines.Line2D([], [], color="#1f77b4", ls="-", lw=2.2, label=rf"$m_1 = {m1:+d}$ (Low-$m$ Branch)")
        h_m2 = mlines.Line2D([], [], color="#9467bd", ls="--", lw=2.2, label=rf"$m_2 = {m2:+d}$ (Steep Branch)")
        h_cont = mlines.Line2D([], [], color="dimgray", marker="o", markerfacecolor="gray", markeredgecolor="dimgray",
                               markersize=6.5, ls="none", label=r"PMP Contained ($\leq 45^\circ$)")
        h_out = mlines.Line2D([], [], color="dimgray", marker="o", markerfacecolor="white", markeredgecolor="dimgray",
                              markeredgewidth=1.4, markersize=6.5, ls="none", label=r"PMP Outside ($> 45^\circ$)")
        ax_s.legend(handles=[h_m1, h_m2, h_cont, h_out], loc="upper right", fontsize=8.5, framealpha=0.92)

        plt.tight_layout()
        tag = "ECH_NBI" if "ECH" in win_name else "NBI_only"
        
        # Save to heating_phase_comparison/panels/
        p1 = heating_panels_dir / f"mhd_analysis_objective2_structure_{shot}_40_80kHz_{tag}_panel4_poloidal_phase_structure_verification_at.png"
        safe_savefig(fig_s, p1, dpi=180)
        print(f"  [Figure] Saved distinct figure 1: {p1.name}")

        # Save to obj2_dynamics_and_structure/panels/
        p2 = obj2_panels_dir / f"mhd_secondary_two_mode_phase_containment_{shot}_{tag}.png"
        safe_savefig(fig_s, p2, dpi=180)
        print(f"  [Figure] Saved distinct figure 2: {p2.name}")

        plt.close(fig_s)


def main():
    parser = argparse.ArgumentParser(description="Exhaustive Two-Mode Poloidal Combination Search")
    parser.add_argument("--shot", type=int, default=88653, help="Shot number (default: 88653)")
    parser.add_argument("--max-m", type=int, default=20, help="Maximum m candidate to evaluate (default: 20)")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for figures and tables")
    args = parser.parse_args()

    shot = args.shot
    out_dir = Path(args.out_dir) if args.out_dir else ROOT_DIR / "Images" / "obj2_dynamics_and_structure"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_out_dir = ROOT_DIR / "analysis" / "obj2_dynamics_and_structure"
    data_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f"TWO-MODE POLOIDAL COMBINATION (m1, m2) SEARCH - SHOT {shot}")
    print(f"Testing all 820 pairs in [-20, +20] tabula rasa across all PMPs")
    print(f"{'=' * 80}\n")

    time_ms, dt_s, fs, sigs, pmp_signals = load_signals(shot=shot)
    if not pmp_signals:
        print("[Error] No PMP signals found.")
        sys.exit(1)

    windows = {
        "ECH+NBI Phase (259.1 - 291.3 ms)": (T_W1_START, T_W1_END, 40000.0, 80000.0),
        "NBI-only Phase (291.3 - 300.0 ms)": (T_W2_START, T_W2_END, 40000.0, 80000.0),
    }

    all_window_results = {}
    windows_phase_data = {}

    for win_name, (tw_s, tw_e, fl, fu) in windows.items():
        print(f"--- Processing {win_name} ---")
        pdata = extract_cross_spectral_phase(time_ms, fs, pmp_signals, PMP_ANGLES_DEG, fl, fu, tw_s, tw_e)
        windows_phase_data[win_name] = pdata
        print(f"  Detected Secondary Peak Frequency: {pdata['f_peak_khz']:.2f} kHz")

        # Full 14 probes
        res_full = run_full_grid_search(pdata, active_only=False, max_m=args.max_m)
        # Active 12 probes (exclude PMP8, PMP14)
        res_act = run_full_grid_search(pdata, active_only=True, max_m=args.max_m)

        all_window_results[win_name] = {
            "full_14": res_full,
            "active_12": res_act,
            "phase_data": {
                "f_peak_khz": pdata["f_peak_khz"],
                "meas_phase_deg": np.rad2deg(pdata["meas_phase"]).tolist(),
                "angles_deg": pdata["angles_deg"].tolist(),
                "coherence_sq": pdata["coherence_sq"].tolist(),
                "channels": pdata["channels"],
            }
        }

    # Extract top 2 combinations per window (Rank 1 and Rank 2)
    top2_per_window = {}
    for win_name in windows:
        res_full = all_window_results[win_name]["full_14"]
        pairs_sorted = sorted(res_full["all_pairs"], key=lambda r: (r["mae_deg"], -r["n_le45"]))
        top2_per_window[win_name] = {
            "rank1": pairs_sorted[0],
            "rank2": pairs_sorted[1],
        }

        print(f"\n=========================================================================================")
        print(f"UNBIASED TOP COMBINATIONS FOR {win_name.upper()} (All 14 PMP Probes, All {len(pairs_sorted)} Pairs)")
        print(f"=========================================================================================")
        p1 = pairs_sorted[0]
        p2 = pairs_sorted[1]
        print(f"1. Rank 1 Combination (m1 = {p1['m1']:+d}, m2 = {p1['m2']:+d}):")
        print(f"   MAE = {p1['mae_deg']:5.1f}°, RMSE = {p1['rmse_deg']:5.1f}°, MaxErr = {p1['max_err_deg']:5.1f}°, r_circ = {p1['r_circ']:.3f}, Contained (<=45°): {p1['n_le45']}/14 ({p1['frac_le45']*100:.0f}%), Split: {p1['n_m1']} on m1, {p1['n_m2']} on m2")
        print(f"2. Rank 2 Combination (m1 = {p2['m1']:+d}, m2 = {p2['m2']:+d}):")
        print(f"   MAE = {p2['mae_deg']:5.1f}°, RMSE = {p2['rmse_deg']:5.1f}°, MaxErr = {p2['max_err_deg']:5.1f}°, r_circ = {p2['r_circ']:.3f}, Contained (<=45°): {p2['n_le45']}/14 ({p2['frac_le45']*100:.0f}%), Split: {p2['n_m1']} on m1, {p2['n_m2']} on m2")

        print(f"\nTop 10 Combinations for {win_name}:")
        for rank, p_k in enumerate(pairs_sorted[:10], 1):
            print(f"   Rank {rank:2d}: ({p_k['m1']:+2d}, {p_k['m2']:+2d}) | MAE = {p_k['mae_deg']:5.1f}°, RMSE = {p_k['rmse_deg']:5.1f}°, MaxErr = {p_k['max_err_deg']:5.1f}°, r_circ = {p_k['r_circ']:.3f}, Contained: {p_k['n_le45']}/14 ({p_k['frac_le45']*100:.0f}%), Split: {p_k['n_m1']}/{p_k['n_m2']}")

    # Joint Cross-Phase Ranking (ECH+NBI and NBI-only combined)
    win_ech = list(windows.keys())[0]
    win_nbi = list(windows.keys())[1]
    p_ech = { (r["m1"], r["m2"]): r for r in all_window_results[win_ech]["full_14"]["all_pairs"] }
    p_nbi = { (r["m1"], r["m2"]): r for r in all_window_results[win_nbi]["full_14"]["all_pairs"] }

    joint_pairs = []
    for pair_key, r_e in p_ech.items():
        if pair_key in p_nbi:
            r_n = p_nbi[pair_key]
            combined_mae = 0.5 * (r_e["mae_deg"] + r_n["mae_deg"])
            combined_n45 = r_e["n_le45"] + r_n["n_le45"]
            combined_rc = 0.5 * (r_e["r_circ"] + r_n["r_circ"])
            combined_max_err = max(r_e["max_err_deg"], r_n["max_err_deg"])
            joint_pairs.append({
                "m1": pair_key[0],
                "m2": pair_key[1],
                "joint_mae_deg": combined_mae,
                "joint_n_le45": combined_n45,
                "joint_r_circ": combined_rc,
                "joint_max_err": combined_max_err,
                "ech_mae": r_e["mae_deg"],
                "nbi_mae": r_n["mae_deg"],
                "ech_n45": r_e["n_le45"],
                "nbi_n45": r_n["n_le45"],
            })

    joint_sorted = sorted(joint_pairs, key=lambda r: (r["joint_mae_deg"], -r["joint_n_le45"]))

    print(f"\n=========================================================================================")
    print(f"JOINT DISCHARGE RANKING (ECH+NBI + NBI-only Combined Score, All {len(joint_sorted)} Pairs)")
    print(f"=========================================================================================")
    print(f"Top 2 Globally Optimal Combinations:")
    for rank in [1, 2]:
        p = joint_sorted[rank - 1]
        print(f"  Rank {rank}: ({p['m1']:+2d}, {p['m2']:+2d}) | Joint MAE = {p['joint_mae_deg']:5.1f}° (ECH: {p['ech_mae']:.1f}°, NBI: {p['nbi_mae']:.1f}°), "
              f"Contained: {p['joint_n_le45']}/28, Joint r_circ = {p['joint_r_circ']:.3f}, MaxErr = {p['joint_max_err']:.1f}°")

    print(f"\nTop 15 Overall Combinations Across Both Heating Windows:")
    for rank, p in enumerate(joint_sorted[:15], 1):
        print(f"  Rank {rank:2d}: ({p['m1']:+2d}, {p['m2']:+2d}) | Joint MAE = {p['joint_mae_deg']:5.1f}° (ECH: {p['ech_mae']:.1f}°, NBI: {p['nbi_mae']:.1f}°), "
              f"Contained: {p['joint_n_le45']}/28, Joint r_circ = {p['joint_r_circ']:.3f}, MaxErr = {p['joint_max_err']:.1f}°")

    # Save CSV files
    import csv
    csv_rankings_path = data_out_dir / f"two_mode_m_combinations_{shot}_ranking.csv"
    with open(csv_rankings_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=[
            "rank", "m1", "m2", "joint_mae_deg", "joint_n_le45", "joint_r_circ", "joint_max_err",
            "ech_mae", "nbi_mae", "ech_n45", "nbi_n45"
        ])
        writer.writeheader()
        for rank, row in enumerate(joint_sorted, 1):
            writer.writerow({"rank": rank, **row})
    print(f"\nSaved CSV ranking to: {csv_rankings_path}")

    # Build and save detailed probe breakdown CSV for Rank 1 and Rank 2 in both windows
    csv_top2_path = data_out_dir / f"two_mode_top2_combinations_{shot}.csv"
    top2_rows = []
    for win_name in windows:
        pdata = windows_phase_data[win_name]
        for rank_label, r_dict in [("Rank 1", top2_per_window[win_name]["rank1"]), ("Rank 2", top2_per_window[win_name]["rank2"])]:
            m1, m2 = r_dict["m1"], r_dict["m2"]
            eval_res = evaluate_two_mode_model_a(pdata["angles_rad"], pdata["meas_phase"], m1, m2)
            for i, ch in enumerate(pdata["channels"]):
                res_deg = float(eval_res["residuals_deg"][i])
                top2_rows.append({
                    "window": win_name,
                    "rank": rank_label,
                    "m1": m1,
                    "m2": m2,
                    "channel": ch,
                    "theta_deg": pdata["angles_deg"][i],
                    "meas_phase_deg": np.rad2deg(pdata["meas_phase"][i]),
                    "assigned_mode": eval_res["assigned_branch"][i],
                    "residual_deg": res_deg,
                    "contained_le45": abs(res_deg) <= 45.0
                })
    with open(csv_top2_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=[
            "window", "rank", "m1", "m2", "channel", "theta_deg", "meas_phase_deg",
            "assigned_mode", "residual_deg", "contained_le45"
        ])
        writer.writeheader()
        writer.writerows(top2_rows)
    print(f"Saved Top 2 Probe Breakdown CSV to: {csv_top2_path}")

    def to_serializable(val):
        if isinstance(val, np.ndarray):
            return val.tolist()
        if isinstance(val, (np.integer, np.int64, np.int32)):
            return int(val)
        if isinstance(val, (np.floating, np.float64, np.float32)):
            return float(val)
        if isinstance(val, (np.bool_, bool)):
            return bool(val)
        return str(val)

    # Save JSON summary
    json_summary = {
        "shot": shot,
        "top2_per_window": top2_per_window,
        "top_joint_combinations": joint_sorted[:25],
    }
    json_out_path = data_out_dir / f"two_mode_m_combinations_{shot}.json"
    with open(json_out_path, "w", encoding="utf-8") as f_json:
        json.dump(json_summary, f_json, indent=2, default=to_serializable)
    print(f"Saved full JSON summary to: {json_out_path}")

    # Generate Figures
    print("\n--- Generating Publication-Grade Figures ---")
    key_pairs = [
        (joint_sorted[0]["m1"], joint_sorted[0]["m2"]),
        (joint_sorted[1]["m1"], joint_sorted[1]["m2"])
    ]
    heatmap_out = out_dir / f"mhd_secondary_two_mode_heatmaps_{shot}.png"
    generate_heatmaps_figure(
        all_window_results[win_ech]["full_14"]["all_pairs"],
        m_range=15,
        out_path=heatmap_out,
        title_suffix=f"(Shot #{shot}, ECH+NBI Phase, All 14 PMP Probes)",
        key_pairs=key_pairs
    )

    containment_out = out_dir / f"mhd_secondary_two_mode_phase_containment_{shot}.png"
    standalone_dir = out_dir / "panels"
    plot_phase_containment_comparison(
        windows_phase_data,
        out_path=containment_out,
        standalone_dir=standalone_dir,
        shot=shot,
        m1=1,
        m2=12
    )

    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
