import sys
import argparse
from pathlib import Path
import numpy as np
import scipy.signal as dsp
import matplotlib.pyplot as plt

current = Path(__file__).resolve().parent
root_dir = None
for p in [current] + list(current.parents):
    if (p / "jpack").exists():
        root_dir = p
        break
if root_dir is None:
    root_dir = Path("c:/TFG")

for p_add in [root_dir / "jpack", root_dir / "analysis", root_dir / "analysis" / "common", current]:
    if str(p_add) not in sys.path:
        sys.path.append(str(p_add))

from mhd_common import extract_instantaneous_frequency
from comparison_common import (
    DEFAULT_SHOT, DEFAULT_OUT_DIR,
    T_W1_START, T_W1_END, T_W2_START, T_W2_END,
    PMP_ANGLES_DEG, load_signals, safe_savefig
)


def run_mode_structure_comparison(shot=DEFAULT_SHOT, out_dir=None, verbose=False):
    if out_dir is None:
        out_dir = DEFAULT_OUT_DIR
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Spatial Modal Structure (m) Comparison (Shot {shot}) ---")

    time_ms, dt_s, fs, signals_dict, pmp_signals = load_signals(shot=shot)
    if "MP1" not in signals_dict or len(pmp_signals) == 0:
        print("Error: Required probe signals not available.")
        return

    runs = [
        ("Secondary Band (40-80 kHz) - ECH+NBI", 40000.0, 80000.0, T_W1_START, T_W1_END, "mhd_analysis_objective2_structure_88653_40_80kHz_ECH_NBI.png", "ECH + NBI (259.1 - 291.3 ms)"),
        ("Secondary Band (40-80 kHz) - Just NBI", 40000.0, 80000.0, T_W2_START, T_W2_END, "mhd_analysis_objective2_structure_88653_40_80kHz_NBI_only.png", "Just NBI (291.3 - 300.0 ms)"),
        ("Primary Band (80-120 kHz) - ECH+NBI", 80000.0, 120000.0, T_W1_START, T_W1_END, "mhd_analysis_objective2_structure_88653_80_120kHz_ECH_NBI.png", "ECH + NBI (259.1 - 291.3 ms)"),
        ("Primary Band (80-120 kHz) - Just NBI", 80000.0, 120000.0, T_W2_START, T_W2_END, "mhd_analysis_objective2_structure_88653_80_120kHz_NBI_only.png", "Just NBI (291.3 - 300.0 ms)"),
    ]

    pmp_channels = sorted(pmp_signals.keys(), key=lambda c: int(c[3:]))
    angles_rad = np.array([np.deg2rad(PMP_ANGLES_DEG[int(ch[3:]) - 1]) for ch in pmp_channels])

    for desc, fl_hz, fu_hz, tw_start, tw_end, out_png, label_win in runs:
        print(f"  Processing {desc}...")
        idx_win = np.where((time_ms >= tw_start) & (time_ms <= tw_end))[0]
        n_samples = len(idx_win)

        # Carrier & Envelope extraction
        envelopes = {}
        for p in ["MP1", "MP3", "MP4"]:
            env, _, _, _ = extract_instantaneous_frequency(signals_dict[p], fs, fl_hz, fu_hz, order=4, sg_win=325)
            envelopes[p] = env[idx_win]

        pairs = [("MP1", "MP3"), ("MP1", "MP4"), ("MP3", "MP4")]
        styles = {("MP1", "MP3"): ("tab:blue", "-"), ("MP1", "MP4"): ("tab:green", "--"), ("MP3", "MP4"): ("tab:purple", "-.")}

        carrier_coh = {}
        envelope_coh = {}
        nperseg = min(1024, n_samples // 2)
        noverlap = nperseg // 2

        for p1, p2 in pairs:
            fc, coh_c = dsp.coherence(signals_dict[p1][idx_win], signals_dict[p2][idx_win], fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap)
            carrier_coh[(p1, p2)] = (fc, coh_c)
            fe, coh_e = dsp.coherence(envelopes[p1], envelopes[p2], fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap)
            envelope_coh[(p1, p2)] = (fe, coh_e)

        # Poloidal NUDFT Power Map
        sig_pmp_win = np.array([pmp_signals[ch][idx_win] for ch in pmp_channels])
        nfft_pmp = min(512, n_samples)
        f_pmp, _, S_pmp = dsp.spectrogram(
            sig_pmp_win, fs=fs, window='hann', nperseg=nfft_pmp, noverlap=nfft_pmp // 2,
            nfft=nfft_pmp, detrend='constant', return_onesided=False, scaling='density', axis=-1, mode='complex'
        )
        band_mask = (f_pmp >= fl_hz) & (f_pmp <= fu_hz)
        k_grid = np.arange(-6, 7)
        S_band = S_pmp[:, band_mask, :]
        E = np.exp(-1j * k_grid[:, None] * angles_rad[None, :]) / len(angles_rad)
        Sk = np.tensordot(E, S_band, axes=(1, 0))
        P2d = np.mean((Sk * Sk.conj()).real, axis=2)
        f_band_khz = f_pmp[band_mask] / 1000.0

        m_per_f = k_grid[np.argmax(P2d, axis=0)]
        f_peak_idx = int(np.argmax(np.sum(P2d, axis=0)))
        m_dominant = int(m_per_f[f_peak_idx])
        f_peak_khz = float(f_band_khz[f_peak_idx])

        # Poloidal Phase Structure Verification at f_peak
        f_spec, _, S_cross = dsp.spectrogram(
            sig_pmp_win, fs=fs, window='hann', nperseg=nfft_pmp, noverlap=nfft_pmp // 2,
            nfft=nfft_pmp, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='complex'
        )
        idx_f_peak = int(np.argmin(np.abs(f_spec - f_peak_khz * 1000.0)))
        S_pk = S_cross[:, idx_f_peak, :]
        ref_pk = S_pk[0, :]
        cross_pk = S_pk * np.conj(ref_pk)[None, :]
        avg_cross = np.mean(cross_pk, axis=1)
        n_seg = S_pk.shape[1]

        p_ref = np.mean(np.abs(ref_pk) ** 2)
        p_ch = np.mean(np.abs(S_pk) ** 2, axis=1)
        coh_sq = np.clip(np.abs(avg_cross) ** 2 / (p_ref * p_ch + 1e-30), 0.0, 1.0)
        sigma_phase = np.sqrt(np.maximum(0.0, 1.0 - coh_sq) / (2.0 * np.maximum(coh_sq, 1e-4) * max(1, n_seg)))
        sigma_phase = np.clip(sigma_phase, 0.0, np.pi)
        meas_phase = np.arctan2(np.sin(np.angle(avg_cross) - np.angle(avg_cross)[0]),
                                np.cos(np.angle(avg_cross) - np.angle(avg_cross)[0]))

        # Phase alignment with theoretical m
        th_in_pi = np.arctan2(np.sin(angles_rad), np.cos(angles_rad))
        th_theo_dom = np.arctan2(np.sin(m_dominant * th_in_pi), np.cos(m_dominant * th_in_pi))
        diff_dom = np.arctan2(np.sin(meas_phase - th_theo_dom), np.cos(meas_phase - th_theo_dom))
        r_circ_dom = float(np.sqrt(np.sum(np.cos(diff_dom)) ** 2 + np.sum(np.sin(diff_dom)) ** 2) / len(th_in_pi))
        mean_err_dom = float(np.rad2deg(np.mean(np.abs(diff_dom))))
        is_phase_confirmed = (r_circ_dom >= 0.70) and (mean_err_dom <= 45.0)

        if verbose:
            print(f"    Dominant m = {m_dominant:+d} at {f_peak_khz:.1f} kHz | r_circ = {r_circ_dom:.3f}, err = {mean_err_dom:.1f} deg -> {'CONFIRMED' if is_phase_confirmed else 'UNCONFIRMED'}")

        # 4-Panel Figure
        plt.rcdefaults()
        fig, axs = plt.subplots(4, 1, figsize=(12.5, 17.0))
        fig.suptitle(f"Objective 2 Spatial Modal Structure: {desc}\n({label_win})", fontsize=13, fontweight='bold')

        # Panel 0: Carrier Coherence
        for (p1, p2), (col, ls) in styles.items():
            fc, cc = carrier_coh[(p1, p2)]
            m_fc = (fc >= fl_hz) & (fc <= fu_hz)
            axs[0].plot(fc[m_fc] / 1000.0, cc[m_fc], color=col, ls=ls, lw=2.0, label=f"{p1}-{p2} Carrier Coherence")
        axs[0].axhline(0.5, color='black', ls=':', label='Significance Floor (0.5)')
        axs[0].set_title(f"Cross-Spectral Coherence Between Mirnov Probes: Carrier Oscillations ({fl_hz/1e3:.0f}-{fu_hz/1e3:.0f} kHz)", fontsize=10.5)
        axs[0].set_xlabel("Frequency (kHz)")
        axs[0].set_ylabel(r"Coherence $\gamma^2$")
        axs[0].set_ylim(0, 1.05)
        axs[0].set_xlim(fl_hz / 1000.0, fu_hz / 1000.0)
        axs[0].grid(True, alpha=0.3, ls=':')
        axs[0].legend(loc='upper right')

        # Panel 1: Envelope Coherence
        for (p1, p2), (col, ls) in styles.items():
            fe, ce = envelope_coh[(p1, p2)]
            m_fe = (fe >= 0) & (fe <= 10000.0)
            axs[1].plot(fe[m_fe] / 1000.0, ce[m_fe], color=col, ls=ls, lw=2.0, label=f"{p1}-{p2} Envelope Coherence")
        axs[1].axhline(0.5, color='black', ls=':', label='Significance Floor (0.5)')
        axs[1].set_title("Cross-Spectral Coherence Between Probe Envelopes: Modulation Structure", fontsize=10.5)
        axs[1].set_xlabel("Modulation Frequency (kHz)")
        axs[1].set_ylabel(r"Coherence $\gamma^2$")
        axs[1].set_ylim(0, 1.05)
        axs[1].set_xlim(0, 10.0)
        axs[1].grid(True, alpha=0.3, ls=':')
        axs[1].legend(loc='upper right')

        # Panel 2: Poloidal Mode Decomposition
        pcm = axs[2].pcolormesh(f_band_khz, k_grid, np.log10(P2d + 1e-30), cmap='jet', shading='auto')
        plt.colorbar(pcm, ax=axs[2], label=r"$\log_{10}$ Poloidal Power")
        axs[2].axhline(m_dominant, color='white', ls='--', lw=1.8, label=f"Dominant m = {m_dominant:+d}")
        axs[2].set_title(f"Poloidal Mode-Number Decomposition (PMP1-PMP14 Array, NUDFT)", fontsize=10.5)
        axs[2].set_xlabel("Frequency (kHz)")
        axs[2].set_ylabel("Poloidal Mode Number m")
        axs[2].set_ylim(-6, 6)
        axs[2].set_xlim(fl_hz / 1000.0, fu_hz / 1000.0)
        axs[2].legend(loc='upper right')

        # Panel 3: Poloidal Phase Structure
        th_theory_grid = np.linspace(-np.pi, np.pi, 1000)
        candidate_ms_to_show = sorted(set([-4, -3, -2, -1, 1, 2, 3, 4, m_dominant]))
        for m_candidate in candidate_ms_to_show:
            ph_theo = np.arctan2(np.sin(m_candidate * th_theory_grid), np.cos(m_candidate * th_theory_grid))
            jumps = np.where(np.abs(np.diff(ph_theo)) > np.pi)[0]
            tg = th_theory_grid.copy()
            pg = ph_theo.copy()
            if len(jumps) > 0:
                tg = np.insert(tg, jumps + 1, np.nan)
                pg = np.insert(pg, jumps + 1, np.nan)
            is_dom = (m_candidate == m_dominant)
            axs[3].plot(tg, pg, color='red' if is_dom else 'gray', lw=2.2 if is_dom else 0.8,
                        alpha=0.95 if is_dom else 0.25,
                        label=f"m = {m_candidate:+d} (Dominant)" if is_dom else (f"m = {m_candidate:+d}" if abs(m_candidate) <= 2 else None),
                        zorder=4 if is_dom else 1)

        axs[3].errorbar(th_in_pi, meas_phase, yerr=sigma_phase, fmt='none', ecolor='tab:blue', elinewidth=1.4, capsize=3.5, alpha=0.8)
        axs[3].scatter(th_in_pi, meas_phase, s=75, c='blue', edgecolors='black', lw=1.2, zorder=5, label='Measured PMP Phase')
        for i, ch in enumerate(pmp_channels):
            axs[3].annotate(ch, (th_in_pi[i], meas_phase[i]), textcoords="offset points", xytext=(4, 5), fontsize=7, fontweight='bold', color='darkblue')

        status_box = (f"[CONFIRMED] m = {m_dominant:+d} Fit\nAlignment r_circ = {r_circ_dom:.2f}, Mean Err = {mean_err_dom:.1f} deg"
                      if is_phase_confirmed else
                      f"[UNCONFIRMED] m = {m_dominant:+d} Fit\nAlignment r_circ = {r_circ_dom:.2f}, Mean Err = {mean_err_dom:.1f} deg")
        axs[3].text(0.02, 0.05, status_box, transform=axs[3].transAxes, fontsize=8.5, fontweight='bold',
                    color='darkgreen' if is_phase_confirmed else 'darkred',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='honeydew' if is_phase_confirmed else 'linen', alpha=0.9))

        axs[3].set_xlim(-np.pi * 1.05, np.pi * 1.05)
        axs[3].set_ylim(-np.pi * 1.15, np.pi * 1.15)
        axs[3].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        axs[3].set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
        axs[3].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        axs[3].set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
        axs[3].set_title(f"Poloidal Phase Structure Verification at {f_peak_khz:.1f} kHz", fontsize=10.5)
        axs[3].set_xlabel(r"Poloidal Angle $\theta$ (rad)")
        axs[3].set_ylabel(r"Cross-Spectral Phase $\phi$ (rad)")
        axs[3].grid(True, alpha=0.3)
        axs[3].legend(loc='upper right', fontsize=8, ncol=2)

        plt.tight_layout()
        out_path = out_dir / out_png
        safe_savefig(fig, out_path, dpi=150)
        plt.close(fig)
        print(f"    Saved figure: {out_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spatial Modal Structure Comparison (ECH+NBI vs Just NBI)")
    parser.add_argument("-s", "--shots", type=int, default=DEFAULT_SHOT, help="Shot number to analyze")
    parser.add_argument("-o", "--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode diagnostics")
    args = parser.parse_args()

    run_mode_structure_comparison(shot=args.shots, out_dir=args.out_dir, verbose=args.verbose)
