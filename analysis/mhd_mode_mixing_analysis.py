"""Poloidal Mode Mixing Analysis across the PMP1-PMP14 Array
Heliotron J MHD Analysis (Shot #88653 and multi-shot capable)

Physics Context:
In an ideal circular plasma, an MHD mode is a single Fourier harmonic:
phi(theta) = m * theta (modulo 2pi).
In Heliotron J (L=1, M=4 helical-axis heliotron), non-axisymmetry, toroidicity,
and non-circular flux-surface shaping couple multiple poloidal harmonics at the
exact same frequency (mode mixing).

This causes:
1. Spatial phase ripple: measured phi_j(theta_j) oscillates around the dominant
   linear fit due to sideband interference (m +/- 1, m +/- 2).
2. Spatial spectral broadening: NUDFT power spectrum P(m) at mode peak f_0 distributes
   power into adjacent sidebands rather than an isolated delta peak at m_dom.

Features:
- Cross-spectral phase phi(theta) and metrological uncertainty (Bendat & Piersol 1986)
- Theoretical harmonic trajectory phi = m_dom * theta (modulo 2pi) in radians [-pi, pi]
- Spatial phase residual delta_phi(theta) visualizing sideband ripple across all 14 PMP probes
- NUDFT spatial power spectrum P(m) at mode peak f_0 across PMP1-PMP14 array
- Harmonic Purity Index: Purity = P(m_dom) / sum_m P(m)
- Sideband Mixing Ratios: R_{m +/- 1} = P(m +/- 1) / P(m_dom), R_{m +/- 2} = P(m +/- 2) / P(m_dom)
- Academic publication-style visualization (Nucl. Fusion / PPCF standard) and JSON export.
"""

import sys
import os
import argparse
import json
import logging
from pathlib import Path
import numpy as np
import scipy.signal as dsp
import matplotlib.pyplot as plt

# Encoding setup
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

# Path setup
current_dir = Path(__file__).resolve().parent
ROOT_DIR = None
for p in [current_dir] + list(current_dir.parents):
    if (p / "jpack").exists():
        ROOT_DIR = p
        break
if ROOT_DIR is None:
    ROOT_DIR = Path("c:/TFG")

for p_add in [ROOT_DIR / "jpack", ROOT_DIR / "analysis", ROOT_DIR / "analysis" / "common"]:
    if str(p_add) not in sys.path:
        sys.path.append(str(p_add))

import turnelib as TE

# PMP Poloidal Array Geometry & Hardware Calibration
PMP_COIL_CONST = 4.5e-3   # coil parameter (m^2 * turns)
PMP_GAIN = 2.0            # amplifier gain
PMP_CHANNELS = [f"PMP{i}" for i in range(1, 15)]  # PMP1..PMP14
PMP_RAW_ANGLE_LABELS_DEG = [0., 10., 20., 30., 40., 50., 60., 80., 90., 100., 110., 120., 150., 180.]
# Physical poloidal angle in degrees: plab = (360.0 - raw_label) % 360.0
PMP_ANGLES_DEG = [(360.0 - a) % 360.0 for a in PMP_RAW_ANGLE_LABELS_DEG]
PMP_INVERT_CHANNELS_DEFAULT = ("PMP1", "PMP2", "PMP3", "PMP4")

# Hardware known dead/disconnected channels on Heliotron J for these discharges:
# PMP8 (at -80 deg) and PMP14 (at +180 deg) exhibit near-zero coherence (< 0.05)
DEFAULT_DEAD_CHANNELS = ["PMP8", "PMP14"]

# Logger setup
logger = logging.getLogger("mhd_mode_mixing")
logger.setLevel(logging.INFO)
logger.handlers.clear()
c_handler = logging.StreamHandler(sys.stdout)
c_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(c_handler)

def log(msg=""):
    logger.info(msg)


def load_pmp_array(shot, data_dir, invert_channels=PMP_INVERT_CHANNELS_DEFAULT):
    """Loads and calibrates all available PMP channels for the specified shot."""
    signals = {}
    plab_rad = {}
    angles_deg_dict = {}
    missing = []
    t_ms = None
    dt = None

    for ch, angle_deg in zip(PMP_CHANNELS, PMP_ANGLES_DEG):
        pmp_file = data_dir / f"{ch}@{shot}.edf"
        if not pmp_file.exists():
            missing.append(ch)
            continue
        edf_pmp = TE.edf()
        dat_pmp = edf_pmp.load(str(pmp_file))
        t_raw = dat_pmp[:, 0]
        t_ch_ms = t_raw if edf_pmp.DimUnit[0] == "ms" else t_raw * 1000.0
        ys_pmp = dat_pmp[:, 1] / (PMP_GAIN * PMP_COIL_CONST)

        if ch in invert_channels:
            ys_pmp = -ys_pmp

        if t_ms is None:
            t_ms = t_ch_ms
            dt = (t_ms[1] - t_ms[0]) / 1000.0  # seconds
            signals[ch] = ys_pmp
        else:
            if len(t_ch_ms) != len(t_ms) or np.max(np.abs(t_ch_ms - t_ms)) > 1e-6:
                ys_pmp = np.interp(t_ms, t_ch_ms, ys_pmp)
            signals[ch] = ys_pmp

        # Map angle to [-pi, pi]
        rad_val = np.deg2rad(angle_deg)
        rad_in_pi = np.arctan2(np.sin(rad_val), np.cos(rad_val))
        plab_rad[ch] = rad_in_pi
        angles_deg_dict[ch] = np.rad2deg(rad_in_pi)

    if missing:
        log(f"  [Warning] Missing poloidal channels: {missing} ({len(signals)}/14 loaded)")
    else:
        log(f"  Loaded all {len(signals)} poloidal PMP channels (polarity inverted for {list(invert_channels)}).")

    return signals, plab_rad, angles_deg_dict, t_ms, dt


def compute_cross_spectral_phases(sig_matrix, channels, plab_rad, fs, f_target_hz, nfft=256):
    """Computes cross-spectral phase, coherence, and uncertainty relative to reference channel (PMP1)."""
    n_channels, n_samples = sig_matrix.shape
    nfft_use = min(nfft, n_samples)

    # Complex spectrogram
    f_spec, t_spec, S = dsp.spectrogram(
        sig_matrix, fs=fs, window="hann", nperseg=nfft_use, noverlap=nfft_use // 2,
        nfft=nfft_use, detrend="constant", return_onesided=True, scaling="density",
        axis=-1, mode="complex"
    )

    # Find closest frequency bin
    f_idx = int(np.argmin(np.abs(f_spec - f_target_hz)))
    actual_f_hz = float(f_spec[f_idx])

    S_peak = S[:, f_idx, :]  # shape: (n_channels, n_segments)
    n_seg = S_peak.shape[1]

    ref_spectrum = S_peak[0, :]  # Reference is channels[0] = PMP1
    cross_spectra = S_peak * np.conj(ref_spectrum)[None, :]
    avg_cross = np.mean(cross_spectra, axis=1)

    p_ref = np.mean(np.abs(ref_spectrum)**2)
    p_ch = np.mean(np.abs(S_peak)**2, axis=1)
    coherence_sq = np.clip(np.abs(avg_cross)**2 / (p_ref * p_ch + 1e-30), 0.0, 1.0)

    # Bendat & Piersol (1986) metrological phase standard deviation:
    sigma_phase = np.sqrt(np.maximum(0.0, 1.0 - coherence_sq) / (2.0 * np.maximum(coherence_sq, 1e-4) * n_seg))
    sigma_phase = np.clip(sigma_phase, 0.0, np.pi)

    # Measured phase anchored to reference probe:
    raw_phase = np.angle(avg_cross)
    measured_phase = raw_phase - raw_phase[0]
    measured_phase = np.arctan2(np.sin(measured_phase), np.cos(measured_phase))

    return {
        "actual_f_hz": actual_f_hz,
        "measured_phase": measured_phase,
        "coherence_sq": coherence_sq,
        "sigma_phase": sigma_phase,
        "n_seg": n_seg,
        "f_spec": f_spec,
        "S": S,
        "f_idx": f_idx,
    }


def compute_nudft_spectrum_at_freq(S_at_f, angles_rad, max_m=10):
    """Computes spatial NUDFT power spectrum P(m) across the poloidal array at frequency f_0.
    
    S_at_f: complex spectrum across probes at f_0, shape (n_channels, n_segments)
    angles_rad: probe angles in radians, shape (n_channels,)
    """
    k_grid = np.arange(-max_m, max_m + 1)
    # Projection operator E: shape (n_modes, n_channels)
    E = np.exp(-1j * k_grid[:, None] * angles_rad[None, :]) / len(angles_rad)
    Sk = np.dot(E, S_at_f)  # shape: (n_modes, n_segments)
    P_m = np.mean((Sk * np.conj(Sk)).real, axis=1)  # shape: (n_modes,)
    return k_grid, P_m


def evaluate_harmonic_alignment(angles_rad, measured_phase, m_val):
    """Evaluates the circular alignment and residuals against phi = m * theta (modulo 2pi).
    
    Matches the exact M10-PHASE presentation formulation (anchored at theta=0, phi=0).
    """
    th_theory = np.arctan2(np.sin(m_val * angles_rad), np.cos(m_val * angles_rad))
    diff = np.arctan2(np.sin(measured_phase - th_theory), np.cos(measured_phase - th_theory))
    r_circ = float(np.sqrt(np.sum(np.cos(diff))**2 + np.sum(np.sin(diff))**2) / len(angles_rad))
    rmse_deg = float(np.rad2deg(np.sqrt(np.mean(diff**2))))
    mean_err_deg = float(np.rad2deg(np.mean(np.abs(diff))))
    return {
        "m": m_val,
        "r_circ": r_circ,
        "rmse_deg": rmse_deg,
        "mean_error_deg": mean_err_deg,
        "residuals_rad": diff,
        "residuals_deg": np.rad2deg(diff),
    }


def analyze_mode_mixing(shot, data_dir, t_start_ms, t_end_ms, mode_configs, out_dir=None,
                         nfft=256, max_m=6, dead_channels=DEFAULT_DEAD_CHANNELS):
    """Performs comprehensive poloidal mode mixing analysis across all available PMP channels."""
    if out_dir is None:
        out_dir = ROOT_DIR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if dead_channels is None:
        dead_channels = []

    log(f"\n{'=' * 85}")
    log(f"POLOIDAL MODE MIXING (phi vs. theta) ANALYSIS - SHOT {shot}")
    log(f"Analysis Window: {t_start_ms:.1f} - {t_end_ms:.1f} ms (Duration: {t_end_ms - t_start_ms:.1f} ms)")
    if dead_channels:
        log(f"Excluded Channels: {dead_channels}")
    else:
        log("Array Configuration: Full 14-Channel PMP Array Included (PMP1–PMP14)")
    log(f"{'=' * 85}")

    pmp_signals, plab_rad, angles_deg_dict, t_ms, dt = load_pmp_array(shot, data_dir)
    if len(pmp_signals) < 4:
        log("  [Error] Insufficient PMP channels loaded (< 4). Mode mixing analysis cannot proceed.")
        return None

    fs = 1.0 / dt
    idx_win = np.where((t_ms >= t_start_ms) & (t_ms <= t_end_ms))[0]
    if len(idx_win) < nfft:
        log(f"  [Error] Window sample count ({len(idx_win)}) is smaller than nfft ({nfft}).")
        return None

    all_channels = sorted(pmp_signals.keys(), key=lambda c: int(c[3:]))
    active_channels = [ch for ch in all_channels if ch not in dead_channels]

    sig_matrix_all = np.array([pmp_signals[ch][idx_win] for ch in all_channels])
    sig_matrix_act = np.array([pmp_signals[ch][idx_win] for ch in active_channels])
    angles_rad_all = np.array([plab_rad[ch] for ch in all_channels])
    angles_rad_act = np.array([plab_rad[ch] for ch in active_channels])
    angles_deg_all = np.array([angles_deg_dict[ch] for ch in all_channels])
    angles_deg_act = np.array([angles_deg_dict[ch] for ch in active_channels])

    log(f"  Active Array: {len(active_channels)} channels ({active_channels})")

    overall_results = {
        "shot": shot,
        "time_window_ms": [float(t_start_ms), float(t_end_ms)],
        "all_channels": all_channels,
        "active_channels": active_channels,
        "excluded_channels": dead_channels,
        "modes": {},
    }

    # Set up publication-grade styling
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.titlesize": 12,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
    })

    for mode_cfg in mode_configs:
        label = mode_cfg["label"]
        f_target_khz = mode_cfg["f_target_khz"]
        f_target_hz = f_target_khz * 1000.0
        expected_m = mode_cfg.get("expected_m", None)

        log(f"\n--- Analyzing Mode: {label} (Target: {f_target_khz:.2f} kHz) ---")

        # 1. Cross-spectral phase on active probes
        spec_res_act = compute_cross_spectral_phases(sig_matrix_act, active_channels, plab_rad, fs, f_target_hz, nfft=nfft)
        spec_res_all = compute_cross_spectral_phases(sig_matrix_all, all_channels, plab_rad, fs, f_target_hz, nfft=nfft)

        actual_f_khz = spec_res_act["actual_f_hz"] / 1000.0
        mean_coh_act = float(np.mean(spec_res_act["coherence_sq"]))
        log(f"  STFT Bin Center: {actual_f_khz:.2f} kHz (Resolution: {fs/(nfft*1000.0):.2f} kHz, Segments: {spec_res_act['n_seg']})")
        log(f"  Mean Coherence gamma^2 across {len(active_channels)} channels: {mean_coh_act:.3f}")

        # 2. NUDFT Spatial Power Spectrum P(m) at mode peak
        S_at_f_act = spec_res_act["S"][:, spec_res_act["f_idx"], :]
        k_grid, P_m = compute_nudft_spectrum_at_freq(S_at_f_act, angles_rad_act, max_m=max_m)

        m_dom_nudft = int(k_grid[np.argmax(P_m)])
        log(f"  Dominant Poloidal Harmonic from NUDFT: m = {m_dom_nudft:+d}")

        m_dom = expected_m if expected_m is not None else m_dom_nudft
        log(f"  Target Poloidal Harmonic: m = {m_dom:+d}")

        # 3. Harmonic Alignment and Residuals (M10-PHASE presentation formulation: phi = m*theta modulo 2pi)
        fit_dom = evaluate_harmonic_alignment(angles_rad_act, spec_res_act["measured_phase"], m_dom)
        r_circ = fit_dom["r_circ"]
        rmse_deg = fit_dom["rmse_deg"]
        mean_err_deg = fit_dom["mean_error_deg"]
        residuals_rad = fit_dom["residuals_rad"]
        residuals_deg = fit_dom["residuals_deg"]

        log(f"  Circular Alignment against phi = {m_dom:+d}*theta: r_circ = {r_circ:.3f}, RMSE = {rmse_deg:.1f} deg, Mean Error = {mean_err_deg:.1f} deg")

        # 4. Harmonic Purity Index & Sideband Mixing Ratios
        p_total = float(np.sum(P_m))
        p_dom = float(P_m[k_grid == m_dom][0])
        purity_index = p_dom / (p_total + 1e-30)

        def get_mode_power(m_val):
            mask = k_grid == m_val
            return float(P_m[mask][0]) if np.any(mask) else 0.0

        p_m_minus_1 = get_mode_power(m_dom - 1)
        p_m_plus_1 = get_mode_power(m_dom + 1)
        p_m_minus_2 = get_mode_power(m_dom - 2)
        p_m_plus_2 = get_mode_power(m_dom + 2)

        r_m_minus_1 = p_m_minus_1 / (p_dom + 1e-30)
        r_m_plus_1 = p_m_plus_1 / (p_dom + 1e-30)
        r_m_minus_2 = p_m_minus_2 / (p_dom + 1e-30)
        r_m_plus_2 = p_m_plus_2 / (p_dom + 1e-30)

        sideband_ratio_first = (p_m_minus_1 + p_m_plus_1) / (p_dom + 1e-30)
        sideband_ratio_second = (p_m_minus_2 + p_m_plus_2) / (p_dom + 1e-30)

        log(f"  Mode Mixing Metrics ({len(active_channels)} Probes):")
        log(f"    - Harmonic Purity Index: {purity_index * 100.0:.1f}%")
        log(f"    - Sideband Ratio R(m-1 = {m_dom-1:+d}): {r_m_minus_1:.3f} ({r_m_minus_1*100:.1f}%)")
        log(f"    - Sideband Ratio R(m+1 = {m_dom+1:+d}): {r_m_plus_1:.3f} ({r_m_plus_1*100:.1f}%)")
        log(f"    - Total 1st-Order Sideband Ratio (m +/- 1): {sideband_ratio_first:.3f} ({sideband_ratio_first*100:.1f}%)")
        log(f"    - Total 2nd-Order Sideband Ratio (m +/- 2): {sideband_ratio_second:.3f} ({sideband_ratio_second*100:.1f}%)")

        mode_data = {
            "label": label,
            "f_target_khz": f_target_khz,
            "actual_f_khz": actual_f_khz,
            "m_dominant": int(m_dom),
            "r_circ": float(r_circ),
            "rmse_deg": float(rmse_deg),
            "mean_error_deg": float(mean_err_deg),
            "harmonic_purity_index": float(purity_index),
            "sideband_ratios": {
                f"m_{m_dom-1:+d}": float(r_m_minus_1),
                f"m_{m_dom+1:+d}": float(r_m_plus_1),
                f"m_{m_dom-2:+d}": float(r_m_minus_2),
                f"m_{m_dom+2:+d}": float(r_m_plus_2),
                "total_first_order_m_pm_1": float(sideband_ratio_first),
                "total_second_order_m_pm_2": float(sideband_ratio_second),
            },
            "per_probe_data": {},
        }

        for i, ch in enumerate(all_channels):
            is_active = ch in active_channels
            idx_act = active_channels.index(ch) if is_active else None
            mode_data["per_probe_data"][ch] = {
                "theta_deg": float(angles_deg_all[i]),
                "theta_rad": float(angles_rad_all[i]),
                "is_active": is_active,
                "measured_phase_rad": float(spec_res_all["measured_phase"][i]),
                "measured_phase_deg": float(np.rad2deg(spec_res_all["measured_phase"][i])),
                "coherence_sq": float(spec_res_all["coherence_sq"][i]),
                "sigma_phase_deg": float(np.rad2deg(spec_res_all["sigma_phase"][i])),
                "phase_residual_deg": float(residuals_deg[idx_act]) if is_active else None,
                "phase_residual_rad": float(residuals_rad[idx_act]) if is_active else None,
            }

        overall_results["modes"][label] = mode_data

        # ---------------------------------------------------------------------------------
        # 4-PANEL PUBLICATION-GRADE FIGURE (Nucl. Fusion / PPCF styling)
        # ---------------------------------------------------------------------------------
        fig, axs = plt.subplots(4, 1, figsize=(8.5, 12.5), gridspec_kw={"height_ratios": [1.15, 0.85, 0.95, 0.75]})
        fig.patch.set_facecolor("white")
        title_str = (
            f"Poloidal Mode Structure & Sideband Mixing — Shot #{shot}\n"
            f"[{label}: f = {actual_f_khz:.2f} kHz, Window: {t_start_ms:.1f}–{t_end_ms:.1f} ms, Target m = {m_dom:+d}]"
        )
        fig.suptitle(title_str, fontweight="bold", y=0.985)

        # ---------------- Panel (a): Measured Phase phi(theta) in Radians ----------------
        th_grid = np.linspace(-np.pi, np.pi, 1200)

        # Theoretical sideband lines (m +/- 1, m +/- 2) modulo 2pi
        candidate_ms = sorted(set([-3, -2, -1, 1, 2, 3, 4, m_dom]))
        for m_cand in candidate_ms:
            if m_cand == m_dom:
                continue
            ph_cand = np.arctan2(np.sin(m_cand * th_grid), np.cos(m_cand * th_grid))
            jumps = np.where(np.abs(np.diff(ph_cand)) > np.pi)[0]
            th_p = th_grid.copy()
            ph_p = ph_cand.copy()
            if len(jumps) > 0:
                th_p = np.insert(th_p, jumps + 1, np.nan)
                ph_p = np.insert(ph_p, jumps + 1, np.nan)
            is_sideband = abs(m_cand - m_dom) == 1
            axs[0].plot(th_p, ph_p, color="#888888" if is_sideband else "#cccccc",
                        ls="--" if is_sideband else ":", lw=1.0 if is_sideband else 0.7,
                        label=f"m = {m_cand:+d}" if is_sideband else None, zorder=1)

        # Dominant m theoretical trajectory
        ph_dom = np.arctan2(np.sin(m_dom * th_grid), np.cos(m_dom * th_grid))
        jumps = np.where(np.abs(np.diff(ph_dom)) > np.pi)[0]
        th_p = th_grid.copy()
        ph_p = ph_dom.copy()
        if len(jumps) > 0:
            th_p = np.insert(th_p, jumps + 1, np.nan)
            ph_p = np.insert(ph_p, jumps + 1, np.nan)
        axs[0].plot(th_p, ph_p, color="#d62728", lw=2.0, zorder=2,
                    label=rf"Dominant $m = {m_dom:+d}$ ($\phi = {m_dom}\theta$)")

        # Active probe points with Bendat-Piersol error bars
        phi_act = spec_res_act["measured_phase"]
        sig_act = spec_res_act["sigma_phase"]
        axs[0].errorbar(angles_rad_act, phi_act, yerr=sig_act, fmt="none",
                        ecolor="#1f77b4", elinewidth=1.2, capsize=3.0, capthick=0.8, alpha=0.8, zorder=3)
        axs[0].scatter(angles_rad_act, phi_act, s=55, facecolor="#1f77b4", edgecolor="#0b3c61",
                       lw=1.0, zorder=4, label=r"Measured phase $\phi_j$ (ref: PMP1)")

        for i, ch in enumerate(active_channels):
            axs[0].annotate(ch, (angles_rad_act[i], phi_act[i]), textcoords="offset points",
                            xytext=(4, 5), fontsize=7.5, color="#0b3c61", fontweight="bold")

        # Flag any dead channels if specified
        if dead_channels:
            dead_indices = [all_channels.index(ch) for ch in dead_channels]
            th_dead = angles_rad_all[dead_indices]
            ph_dead = spec_res_all["measured_phase"][dead_indices]
            axs[0].scatter(th_dead, ph_dead, s=45, facecolor="white", edgecolor="#888888",
                           marker="s", lw=1.2, zorder=4, label="Excluded channel")
            for i, ch in enumerate(dead_channels):
                axs[0].annotate(f"{ch}*", (th_dead[i], ph_dead[i]), textcoords="offset points",
                                xytext=(4, -10), fontsize=7.5, color="#777777", fontstyle="italic")

        axs[0].set_xlim(-np.pi * 1.05, np.pi * 1.05)
        axs[0].set_ylim(-np.pi * 1.12, np.pi * 1.12)
        axs[0].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        axs[0].set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
        axs[0].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        axs[0].set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
        axs[0].set_xlabel(r"Poloidal Angle $\theta$ (rad)")
        axs[0].set_ylabel(r"Cross-Spectral Phase $\phi$ (rad)")
        axs[0].set_title(r"(a) Measured Cross-Spectral Phase $\phi(\theta)$ vs. Theoretical Harmonics", loc="left", fontweight="bold")
        axs[0].axhline(0, color="#888888", lw=0.5, alpha=0.5)
        axs[0].axvline(0, color="#888888", lw=0.5, alpha=0.5)
        axs[0].grid(True, alpha=0.25, ls="--")
        axs[0].legend(loc="lower right" if m_dom > 0 else "upper right", framealpha=0.9)

        # ---------------- Panel (b): Phase Residual delta_phi(theta) in Radians ----------------
        order_act = np.argsort(angles_rad_act)
        th_act_sorted = angles_rad_act[order_act]
        res_rad_sorted = residuals_rad[order_act]

        axs[1].axhline(0.0, color="black", ls="-", lw=0.8, alpha=0.7)
        axs[1].axhline(+np.pi/4, color="#888888", ls="--", lw=0.8, label=r"Tolerance Limit ($\pm\pi/4 = \pm 45^\circ$)")
        axs[1].axhline(-np.pi/4, color="#888888", ls="--", lw=0.8)

        # Connecting curve along active probes (split on wrap jumps > pi)
        diff_res = np.abs(np.diff(res_rad_sorted))
        jump_idx = np.where(diff_res > np.pi)[0]
        th_line = th_act_sorted.copy()
        res_line = res_rad_sorted.copy()
        if len(jump_idx) > 0:
            th_line = np.insert(th_line, jump_idx + 1, np.nan)
            res_line = np.insert(res_line, jump_idx + 1, np.nan)
        axs[1].plot(th_line, res_line, color="#1f77b4", ls="-", lw=1.3, alpha=0.75, zorder=2)
        axs[1].scatter(angles_rad_act, residuals_rad, s=50, facecolor="#1f77b4", edgecolor="#0b3c61", lw=0.9, zorder=3)

        for i, ch in enumerate(active_channels):
            y_pos = residuals_rad[i]
            axs[1].annotate(ch, (angles_rad_act[i], y_pos), textcoords="offset points",
                            xytext=(3, 4 if y_pos >= 0 else -10),
                            fontsize=7.5, color="#990000" if abs(y_pos) > np.pi/4 else "#0b3c61")

        if dead_channels:
            axs[1].text(0.98, 0.08, f"*{', '.join(dead_channels)} excluded (coherence near zero / disconnected)",
                        transform=axs[1].transAxes, fontsize=8, ha="right", color="#666666", fontstyle="italic")

        axs[1].set_xlim(-np.pi * 1.05, np.pi * 1.05)
        max_res_rad = max(np.pi/2, float(np.max(np.abs(residuals_rad))) * 1.15)
        axs[1].set_ylim(-max_res_rad, max_res_rad)
        axs[1].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        axs[1].set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
        axs[1].set_yticks([-np.pi, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, np.pi])
        axs[1].set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$-\pi/4$", r"$0$", r"$\pi/4$", r"$\pi/2$", r"$\pi$"])
        axs[1].set_xlabel(r"Poloidal Angle $\theta$ (rad)")
        axs[1].set_ylabel(r"Phase Residual $\delta\phi$ (rad)")
        axs[1].set_title(r"(b) Phase Residual $\delta\phi(\theta) = \phi_j - m_{\mathrm{dom}}\theta_j \ (\mathrm{mod}\ 2\pi)$ (Sideband Ripple)", loc="left", fontweight="bold")
        axs[1].grid(True, alpha=0.25, ls="--")
        axs[1].legend(loc="upper right", framealpha=0.9)

        # ---------------- Panel (c): NUDFT Spatial Spectrum P(m) ----------------
        bar_cols = []
        for m_val in k_grid:
            if m_val == m_dom:
                bar_cols.append("#2b5c8f")  # Deep slate navy
            elif abs(m_val - m_dom) == 1:
                bar_cols.append("#e67e22")  # Muted amber/orange
            elif abs(m_val - m_dom) == 2:
                bar_cols.append("#f39c12")  # Warm gold
            else:
                bar_cols.append("#bdc3c7")  # Clean muted gray

        norm_power = P_m / np.max(P_m)
        bars = axs[2].bar(k_grid, norm_power, color=bar_cols, width=0.62, edgecolor="#2c3e50", lw=0.7, zorder=2)
        axs[2].set_xlim(-max_m - 0.6, max_m + 0.6)
        axs[2].set_xticks(k_grid)
        axs[2].set_ylabel(r"Normalized Power $P(m) / P_{\max}$")
        axs[2].set_xlabel(r"Poloidal Mode Number $m$")
        axs[2].set_title(f"(c) NUDFT Spatial Power Spectrum $P(m)$ ({len(active_channels)} Probes)", loc="left", fontweight="bold")
        axs[2].grid(True, alpha=0.25, ls="--", axis="y")

        stat_box = (
            f"Mode peak: f = {actual_f_khz:.2f} kHz\n"
            f"Dominant: m = {m_dom:+d}\n"
            f"Harmonic Purity: {purity_index*100.0:.1f}%\n"
            f"Sideband R(m-1 = {m_dom-1:+d}): {r_m_minus_1*100.0:.1f}%\n"
            f"Sideband R(m+1 = {m_dom+1:+d}): {r_m_plus_1*100.0:.1f}%\n"
            f"Circular Alignment: r_circ = {r_circ:.2f} (RMSE = {rmse_deg:.1f}°)"
        )
        axs[2].text(0.02, 0.94, stat_box, transform=axs[2].transAxes, fontsize=8.2,
                    verticalalignment="top", fontfamily="monospace",
                    bbox=dict(boxstyle="square,pad=0.5", facecolor="white", edgecolor="#aaaaaa", alpha=0.95))

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#2b5c8f", edgecolor="#2c3e50", label=f"Dominant ($m = {m_dom:+d}$)"),
            Patch(facecolor="#e67e22", edgecolor="#2c3e50", label=r"1st Sideband ($m \pm 1$)"),
            Patch(facecolor="#f39c12", edgecolor="#2c3e50", label=r"2nd Sideband ($m \pm 2$)"),
            Patch(facecolor="#bdc3c7", edgecolor="#2c3e50", label="Other Harmonics"),
        ]
        axs[2].legend(handles=legend_elements, loc="upper right", framealpha=0.9)

        # ---------------- Panel (d): Diagnostic Coherence Across Array ----------------
        ch_indices = np.arange(len(all_channels))
        coh_all = spec_res_all["coherence_sq"]
        bar_colors_coh = []
        for ch, c in zip(all_channels, coh_all):
            if ch in dead_channels:
                bar_colors_coh.append("#e0e0e0")
            elif c >= 0.5:
                bar_colors_coh.append("#27ae60")  # Confirmed green
            elif c >= 0.4:
                bar_colors_coh.append("#f39c12")  # Acceptable amber
            else:
                bar_colors_coh.append("#e74c3c")  # Low red

        bars_coh = axs[3].bar(ch_indices, coh_all, color=bar_colors_coh, width=0.58, edgecolor="#2c3e50", lw=0.7, zorder=2)
        if dead_channels:
            dead_indices = [all_channels.index(ch) for ch in dead_channels]
            for idx in dead_indices:
                bars_coh[idx].set_hatch("//")
                bars_coh[idx].set_edgecolor("#888888")

        axs[3].axhline(0.5, color="#555555", ls="--", lw=0.8, label=r"Significance Threshold ($\gamma^2 = 0.5$)")
        axs[3].axhline(0.4, color="#888888", ls=":", lw=0.8, label=r"Acceptance Floor ($\gamma^2 = 0.4$)")
        axs[3].set_xticks(ch_indices)
        labels_ch = []
        for ch in all_channels:
            lbl = f"{ch}\n[N/C]" if ch in dead_channels else f"{ch}\n({angles_deg_dict[ch]:+.0f}°)"
            labels_ch.append(lbl)
        axs[3].set_xticklabels(labels_ch, fontsize=7.5)
        axs[3].set_ylim(0.0, 1.05)
        axs[3].set_ylabel(r"Coherence $\gamma^2$ vs. PMP1")
        axs[3].set_xlabel("Poloidal Mirnov Probe Channel")
        axs[3].set_title(r"(d) Measurement Quality Across PMP1–PMP14 Array at Mode Peak", loc="left", fontweight="bold")
        axs[3].grid(True, alpha=0.25, ls="--", axis="y")
        axs[3].legend(loc="upper right", framealpha=0.9)

        plt.tight_layout()
        out_png_name = f"mhd_mode_mixing_shot_{shot}_{label}.png"
        out_png_path = out_dir / out_png_name
        # Remove existing file if present to avoid Windows lock issues
        if out_png_path.exists():
            try:
                out_png_path.unlink()
            except Exception:
                pass
        plt.savefig(out_png_path, dpi=180)
        plt.close(fig)
        log(f"  Saved figure: {out_png_name}")

    # Export summary JSON
    out_json_name = f"mhd_mode_mixing_shot_{shot}.json"
    out_json_path = out_dir / out_json_name
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(overall_results, f, indent=2)
    log(f"\nSaved mode mixing metrics JSON: {out_json_name}")

    return overall_results


def main():
    parser = argparse.ArgumentParser(description="Poloidal Mode Mixing Analysis across PMP1-PMP14 Array (phi vs. theta)")
    parser.add_argument("--shot", type=int, default=88653, help="Shot number (default: 88653)")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory (default: data/hj{shot})")
    parser.add_argument("--window-start", type=float, default=259.1,
                        help="Start time of analysis window in ms (default: 259.1 ms, Objective-3 mode-active window)")
    parser.add_argument("--window-end", type=float, default=275.0,
                        help="End time of analysis window in ms (default: 275.0 ms)")
    parser.add_argument("--nfft", type=int, default=256, help="FFT size for STFT (default: 256)")
    parser.add_argument("--max-m", type=int, default=6, help="Maximum poloidal mode number evaluated in spectrum (default: 6)")
    parser.add_argument("--exclude-channels", type=str, nargs="*", default=DEFAULT_DEAD_CHANNELS,
                        help=f"Channels to exclude from fit and NUDFT (default: {DEFAULT_DEAD_CHANNELS})")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (default: project root)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else ROOT_DIR / "data" / f"hj{args.shot}"

    mode_configs = [
        {
            "label": "LF_43kHz",
            "f_target_khz": 42.97,
            "expected_m": 2,
        },
        {
            "label": "HF_90kHz",
            "f_target_khz": 89.84,
            "expected_m": 3,
        }
    ]

    analyze_mode_mixing(
        shot=args.shot,
        data_dir=data_dir,
        t_start_ms=args.window_start,
        t_end_ms=args.window_end,
        mode_configs=mode_configs,
        out_dir=args.out_dir,
        nfft=args.nfft,
        max_m=args.max_m,
        dead_channels=args.exclude_channels,
    )


if __name__ == "__main__":
    main()
