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

from comparison_common import (
    DEFAULT_SHOT, DEFAULT_OUT_DIR, T_NOISE_START, T_NOISE_END,
    T_W1_START, T_W1_END, T_W2_START, T_W2_END,
    load_signals, safe_savefig
)


def run_spectrogram_comparison(shot=DEFAULT_SHOT, out_dir=None, verbose=False):
    if out_dir is None:
        out_dir = DEFAULT_OUT_DIR
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Zoomed Spectrogram & 3-Sigma Dominance Analysis (Shot {shot}) ---")

    time_ms, dt_s, fs, signals_dict, _ = load_signals(shot=shot)
    if "MP1" not in signals_dict or time_ms is None:
        print("Error: MP1 signal not available.")
        return

    ys_mp1 = signals_dict["MP1"]

    nperseg = 1024
    noverlap = 960  # fine temporal resolution: step = 64 us
    f_stft, t_s, Zxx = dsp.stft(ys_mp1, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann')
    t_stft_ms = time_ms[0] + t_s * 1000.0
    psd = np.abs(Zxx) ** 2
    f_khz = f_stft / 1000.0

    mask_noise = (t_stft_ms >= T_NOISE_START) & (t_stft_ms <= T_NOISE_END)
    mean_noise = np.mean(psd[:, mask_noise], axis=1)
    std_noise = np.std(psd[:, mask_noise], axis=1)
    thresh_3sigma = mean_noise + 3.0 * std_noise

    windows = [
        ("ECH + NBI Phase", T_W1_START, T_W1_END, "zoomed_spectrogram_3sigma_ECH_NBI_259.1_291.3ms.png", "ECH + NBI"),
        ("Just NBI Phase", T_W2_START, T_W2_END, "zoomed_spectrogram_3sigma_NBI_only_291.3_300.0ms.png", "Just NBI")
    ]

    for title_win, tw_start, tw_end, out_name, label_phase in windows:
        mask_win = (t_stft_ms >= tw_start) & (t_stft_ms <= tw_end)
        t_sub = t_stft_ms[mask_win]
        psd_sub = psd[:, mask_win]

        dom_sub = psd_sub > thresh_3sigma[:, None]

        m_pri = (f_khz >= 70.0) & (f_khz <= 110.0)
        m_sec = (f_khz >= 30.0) & (f_khz <= 65.0)

        f_pri_track = []
        p_pri_track = []
        dom_pri_track = []

        f_sec_track = []
        p_sec_track = []
        dom_sec_track = []

        for col in psd_sub.T:
            idx_p = np.argmax(col[m_pri])
            fp = f_khz[m_pri][idx_p]
            pp = col[m_pri][idx_p]
            f_pri_track.append(fp)
            p_pri_track.append(pp)
            dom_pri_track.append(pp > thresh_3sigma[m_pri][idx_p])

            idx_s = np.argmax(col[m_sec])
            fs_val = f_khz[m_sec][idx_s]
            ps_val = col[m_sec][idx_s]
            f_sec_track.append(fs_val)
            p_sec_track.append(ps_val)
            dom_sec_track.append(ps_val > thresh_3sigma[m_sec][idx_s])

        f_pri_track = np.array(f_pri_track)
        p_pri_track = np.array(p_pri_track)
        dom_pri_track = np.array(dom_pri_track)

        f_sec_track = np.array(f_sec_track)
        p_sec_track = np.array(p_sec_track)
        dom_sec_track = np.array(dom_sec_track)

        jump_pri = np.abs(np.diff(f_pri_track))
        n_jumps_pri = int(np.sum(jump_pri > 2.5))
        jump_rate_pri = n_jumps_pri / (tw_end - tw_start)

        jump_sec = np.abs(np.diff(f_sec_track))
        n_jumps_sec = int(np.sum(jump_sec > 2.5))
        jump_rate_sec = n_jumps_sec / (tw_end - tw_start)

        counts_pri, bin_edges_pri = np.histogram(f_pri_track, bins=25)
        pks_hist_pri, _ = dsp.find_peaks(counts_pri, height=len(f_pri_track) * 0.15)
        branch_freqs_pri = [(bin_edges_pri[i] + bin_edges_pri[i + 1]) / 2.0 for i in pks_hist_pri]

        counts_sec, bin_edges_sec = np.histogram(f_sec_track, bins=25)
        pks_hist_sec, _ = dsp.find_peaks(counts_sec, height=len(f_sec_track) * 0.15)
        branch_freqs_sec = [(bin_edges_sec[i] + bin_edges_sec[i + 1]) / 2.0 for i in pks_hist_sec]

        print(f"  [{label_phase}] ({tw_start:.1f} - {tw_end:.1f} ms)")
        print(f"    Primary Mode:   {np.mean(f_pri_track):.2f} +/- {np.std(f_pri_track):.2f} kHz, 3-sigma occupancy: {np.mean(dom_pri_track)*100:.1f}%, jumps: {jump_rate_pri:.2f}/ms")
        print(f"    Secondary Mode: {np.mean(f_sec_track):.2f} +/- {np.std(f_sec_track):.2f} kHz, 3-sigma occupancy: {np.mean(dom_sec_track)*100:.1f}%, jumps: {jump_rate_sec:.2f}/ms")

        # Plot 2-Panel Figure
        plt.rcdefaults()
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11.5, 9.0), gridspec_kw={'height_ratios': [2.2, 1.2]}, sharex=True)
        tt, ff = np.meshgrid(t_sub, f_khz)
        mask_fzoom = (f_khz >= 20.0) & (f_khz <= 120.0)

        c0 = ax_top.pcolormesh(tt[mask_fzoom, :], ff[mask_fzoom, :], np.log10(psd_sub[mask_fzoom, :] + 1e-30),
                               cmap='jet', shading='auto', vmin=-11, vmax=-4)
        fig.colorbar(c0, ax=ax_top, label=r"$\log_{10}\mathrm{PSD}\ (\mathrm{V}^2/\mathrm{Hz})$", pad=0.02)

        dmap_plot = dom_sub[mask_fzoom, :].astype(float)
        ax_top.contourf(tt[mask_fzoom, :], ff[mask_fzoom, :], dmap_plot, levels=[0.5, 1.5], colors=['black'], alpha=0.15)
        ax_top.contour(tt[mask_fzoom, :], ff[mask_fzoom, :], dmap_plot, levels=[0.5], colors='black', linewidths=1.2, linestyles=':')

        ax_top.scatter(t_sub, f_pri_track, c='white', edgecolors='black', s=12, alpha=0.7, label='Primary Ridge Track', zorder=4)
        ax_top.scatter(t_sub, f_sec_track, c='yellow', edgecolors='black', s=12, alpha=0.7, label='Secondary Ridge Track', zorder=4)

        ax_top.set_title(f"Heliotron J #{shot} — Zoomed MP1 Spectrogram with 3-$\\sigma$ Dominance\n"
                         f"[{label_phase.upper()} PHASE: {tw_start:.1f} – {tw_end:.1f} ms]",
                         fontsize=12, fontweight='bold')
        ax_top.set_ylabel("Frequency (kHz)", fontsize=11)
        ax_top.set_ylim(20.0, 120.0)
        ax_top.grid(True, linestyle=':', alpha=0.4)
        ax_top.legend(loc='upper right', fontsize=9, framealpha=0.9)

        # Bottom panel: Instantaneous frequency trajectory
        ax_bot.plot(t_sub, f_pri_track, color='crimson', lw=1.8,
                    label=f'Primary: $\\mu={np.mean(f_pri_track):.1f}$ kHz, $\\sigma={np.std(f_pri_track):.1f}$ kHz')
        for b in branch_freqs_pri:
            ax_bot.axhline(b, color='darkred', ls='--', lw=1.0, alpha=0.6)
            ax_bot.text(tw_start + 0.2, b + 0.6, f"Branch {b:.1f} kHz", color='darkred', fontsize=8, fontweight='bold')

        ax_bot.plot(t_sub, f_sec_track, color='darkorange', lw=1.8,
                    label=f'Secondary: $\\mu={np.mean(f_sec_track):.1f}$ kHz, $\\sigma={np.std(f_sec_track):.1f}$ kHz')
        for b in branch_freqs_sec:
            ax_bot.axhline(b, color='chocolate', ls='--', lw=1.0, alpha=0.6)
            ax_bot.text(tw_start + 0.2, b + 0.6, f"Branch {b:.1f} kHz", color='chocolate', fontsize=8, fontweight='bold')

        ax_bot.set_xlabel("Time (ms)", fontsize=11)
        ax_bot.set_ylabel("Peak Freq (kHz)", fontsize=11)
        ax_bot.set_ylim(25.0, 105.0)
        ax_bot.set_title(r"Tracked Mode Trajectories & Frequency Branch Switching", fontsize=10.5, fontweight='bold')
        ax_bot.grid(True, linestyle=':', alpha=0.4)
        ax_bot.legend(loc='lower right', fontsize=8.5, framealpha=0.9)

        plt.tight_layout()
        out_fig = out_dir / out_name
        safe_savefig(fig, out_fig, dpi=150)
        plt.close(fig)
        print(f"    Saved figure: {out_fig.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zoomed Spectrogram Comparison (ECH+NBI vs Just NBI)")
    parser.add_argument("-s", "--shots", type=int, default=DEFAULT_SHOT, help="Shot number to analyze")
    parser.add_argument("-o", "--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    run_spectrogram_comparison(shot=args.shots, out_dir=args.out_dir, verbose=args.verbose)
