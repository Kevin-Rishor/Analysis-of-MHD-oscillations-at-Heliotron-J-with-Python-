import sys
import json
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
    DEFAULT_SHOT, DEFAULT_OUT_DIR,
    T_W1_START, T_W1_END, T_W2_START, T_W2_END,
    PROBE_ANGLES_TOR, load_channel, load_signals, safe_savefig
)


def run_toroidal_and_radial_comparison(shot=DEFAULT_SHOT, out_dir=None, verbose=False):
    if out_dir is None:
        out_dir = DEFAULT_OUT_DIR
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Toroidal Mode (n) & Radial Profile Comparison (Shot {shot}) ---")

    time_ms, dt_s, fs, signals_dict, _ = load_signals(shot=shot)
    if "MP1" not in signals_dict or "MP3" not in signals_dict or "MP4" not in signals_dict:
        print("Error: Required toroidal Mirnov probes (MP1, MP3, MP4) not available.")
        return

    windows = [
        ("ECH + NBI Phase", T_W1_START, T_W1_END, 89.0, "mhd_primary_mode_toroidal_n_shot_88653_ECH_NBI.png",
         "mhd_primary_mode_radial_profile_shot_88653_ECH_NBI.png", "primary_mode_identification_ECH_NBI.json"),
        ("Just NBI Phase", T_W2_START, T_W2_END, 86.0, "mhd_primary_mode_toroidal_n_shot_88653_NBI_only.png",
         "mhd_primary_mode_radial_profile_shot_88653_NBI_only.png", "primary_mode_identification_NBI_only.json"),
    ]

    pairs = [("MP1", "MP3"), ("MP1", "MP4"), ("MP3", "MP4")]

    for title_win, tw_start, tw_end, mode_freq, out_png_tor, out_png_rad, out_json in windows:
        print(f"  Processing {title_win} (Target: {mode_freq:.1f} kHz)...")
        idx_win = np.where((time_ms >= tw_start) & (time_ms <= tw_end))[0]
        n_samples = len(idx_win)

        nperseg = min(2048, n_samples // 2 if n_samples > 1000 else n_samples)
        noverlap = nperseg // 2

        measured_pairs = {}
        for p1, p2 in pairs:
            x1 = signals_dict[p1][idx_win]
            x2 = signals_dict[p2][idx_win]
            f_c, Pxy = dsp.csd(x1, x2, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
            _, coh = dsp.coherence(x1, x2, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
            f_khz = f_c / 1000.0
            idx_f = np.argmin(np.abs(f_khz - mode_freq))
            actual_f = f_khz[idx_f]
            gamma2 = float(coh[idx_f])
            phase_rad = float(np.angle(Pxy[idx_f]))
            phase_deg = float(np.degrees(phase_rad))
            d_tor_deg = PROBE_ANGLES_TOR[p2] - PROBE_ANGLES_TOR[p1]
            measured_pairs[(p1, p2)] = {
                "d_tor_deg": d_tor_deg,
                "actual_f_khz": actual_f,
                "gamma2": gamma2,
                "phase_rad": phase_rad,
                "phase_deg": phase_deg,
                "f_khz": f_khz,
                "coh_curve": coh,
            }

        n_candidates = list(range(-6, 7))
        n_fit_results = {}
        for n in n_candidates:
            sq_err = 0.0
            pair_diffs = []
            for (p1, p2), pdict in measured_pairs.items():
                d_tor_rad = np.radians(pdict["d_tor_deg"])
                theo_rad = (n * d_tor_rad + np.pi) % (2 * np.pi) - np.pi
                diff_rad = (pdict["phase_rad"] - theo_rad + np.pi) % (2 * np.pi) - np.pi
                pair_diffs.append(float(np.degrees(diff_rad)))
                sq_err += diff_rad ** 2
            rmse_deg = float(np.degrees(np.sqrt(sq_err / len(pairs))))
            n_fit_results[n] = {"rmse_deg": rmse_deg, "pair_diffs_deg": pair_diffs}

        best_n_list = sorted(n_fit_results.keys(), key=lambda n: n_fit_results[n]["rmse_deg"])
        best_n = best_n_list[0]
        second_best_n = best_n_list[1]

        # 3-Panel Toroidal Figure
        plt.rcdefaults()
        fig_tor, axs_tor = plt.subplots(3, 1, figsize=(10, 11))
        fig_tor.suptitle(f"Primary Mode Toroidal Identification — Shot {shot}\n({title_win}: f ~ {mode_freq:.1f} kHz, {tw_start:.1f}–{tw_end:.1f} ms)", fontsize=12, fontweight="bold")

        for (p1, p2), pdict in measured_pairs.items():
            axs_tor[0].plot(pdict["f_khz"], pdict["coh_curve"], label=f"{p1}-{p2} (delta_phi={pdict['d_tor_deg']:+.0f} deg)", lw=1.8)
        axs_tor[0].axvline(mode_freq, color="red", ls="--", lw=1.2, label=f"Mode Target ({mode_freq:.1f} kHz)")
        axs_tor[0].axhline(0.5, color="gray", ls=":", label="Significance Floor (0.5)")
        axs_tor[0].set_xlim(60.0, 120.0)
        axs_tor[0].set_ylim(0.0, 1.05)
        axs_tor[0].set_ylabel(r"Cross-Coherence $\gamma^2$")
        axs_tor[0].set_title("Inter-Probe Cross-Spectral Coherence", fontsize=10)
        axs_tor[0].legend(loc="upper right", fontsize=8.5)
        axs_tor[0].grid(True, alpha=0.3)

        d_angles_plot = np.linspace(-180, 270, 300)
        measured_dphi = [measured_pairs[("MP1", "MP3")]["d_tor_deg"], measured_pairs[("MP1", "MP4")]["d_tor_deg"], measured_pairs[("MP3", "MP4")]["d_tor_deg"]]
        measured_phases_deg = [measured_pairs[("MP1", "MP3")]["phase_deg"], measured_pairs[("MP1", "MP4")]["phase_deg"], measured_pairs[("MP3", "MP4")]["phase_deg"]]
        pair_names = ["MP1-MP3 (+180 deg)", "MP1-MP4 (+270 deg)", "MP3-MP4 (+90 deg)"]

        for n_test, col, ls in [(-1, "crimson", "-"), (+3, "navy", "--"), (+1, "gray", ":"), (+2, "darkorange", "-.")]:
            theo_line = [(n_test * np.radians(a) + np.pi) % (2 * np.pi) - np.pi for a in d_angles_plot]
            axs_tor[1].plot(d_angles_plot, np.degrees(theo_line), color=col, ls=ls, lw=1.5,
                            label=f"Theory n = {n_test:+d} (RMSE={n_fit_results[n_test]['rmse_deg']:.1f} deg)")

        for a, p, lbl in zip(measured_dphi, measured_phases_deg, pair_names):
            axs_tor[1].scatter([a], [p], color="blue", s=90, zorder=5, edgecolors="black", lw=1.5)
            axs_tor[1].annotate(f"{lbl}\n({p:+.1f} deg)", (a, p), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8.5, fontweight="bold")

        axs_tor[1].set_xlim(-10, 290)
        axs_tor[1].set_ylim(-195, 195)
        axs_tor[1].set_xlabel(r"Toroidal Probe Separation $\Delta\varphi$ (deg)")
        axs_tor[1].set_ylabel(r"Cross-Spectral Phase $\Delta\phi$ (deg)")
        axs_tor[1].set_title(r"Measured Phase vs. Theoretical Toroidal Phase $\Delta\phi = n \Delta\varphi$", fontsize=10)
        axs_tor[1].legend(loc="lower right", fontsize=8)
        axs_tor[1].grid(True, alpha=0.3)

        rmse_plot = [n_fit_results[n]["rmse_deg"] for n in n_candidates]
        bar_cols = ["forestgreen" if r < 15.0 else "lightcoral" for r in rmse_plot]
        axs_tor[2].bar(n_candidates, rmse_plot, color=bar_cols, width=0.65, edgecolor="black", alpha=0.85)
        axs_tor[2].axhline(45.0, color="gray", ls="--", label="Phase Tolerance Limit (45 deg)")
        axs_tor[2].set_xlabel("Toroidal Mode Number (n)")
        axs_tor[2].set_ylabel("Phase RMSE (deg)")
        axs_tor[2].set_title(f"Toroidal Mode Phase Error (Best: n = {best_n:+d} / {second_best_n:+d}, RMSE = {n_fit_results[best_n]['rmse_deg']:.1f} deg)", fontsize=10)
        axs_tor[2].set_xticks(n_candidates)
        axs_tor[2].legend(loc="upper right", fontsize=8.5)
        axs_tor[2].grid(True, alpha=0.3)

        plt.tight_layout()
        out_path_tor = out_dir / out_png_tor
        safe_savefig(fig_tor, out_path_tor, dpi=150)
        plt.close(fig_tor)
        print(f"    Saved figure: {out_path_tor.name}")

        # ECE & HAFAST Radial Profiles
        b_band, a_band = dsp.bessel(4, [(mode_freq - 10.0) * 1e3 / (fs / 2.0), (mode_freq + 10.0) * 1e3 / (fs / 2.0)], btype="bandpass")
        ece_results = []
        for i in range(1, 17):
            t_ece, y_ece, _, _ = load_channel(shot, f"ECE{i}FAST")
            if y_ece is None:
                continue
            y_win = y_ece[idx_win]
            _, coh_e = dsp.coherence(signals_dict["MP1"][idx_win], y_win, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
            f_e, Pxy_e = dsp.csd(signals_dict["MP1"][idx_win], y_win, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
            idx_e = np.argmin(np.abs(f_e / 1000.0 - mode_freq))
            ghz = 57.5 + (i - 1) * 1.0
            y_fluc = dsp.filtfilt(b_band, a_band, y_win)
            rms = float(np.std(y_fluc))
            gam = float(coh_e[idx_e])
            ece_results.append({"channel": f"ECE{i:02d}FAST", "freq_ghz": ghz, "gamma2": gam, "rms_v": rms})

        ece_results.sort(key=lambda d: d["freq_ghz"])

        ha_results = []
        for ha in ["3.5", "7.5", "11.5", "15.5"]:
            t_ha, y_ha, _, _ = load_channel(shot, f"HAFAST{ha}")
            if y_ha is None:
                continue
            y_win = y_ha[idx_win]
            _, coh_h = dsp.coherence(signals_dict["MP1"][idx_win], y_win, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
            f_h, _ = dsp.csd(signals_dict["MP1"][idx_win], y_win, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
            idx_h = np.argmin(np.abs(f_h / 1000.0 - mode_freq))
            y_fluc = dsp.filtfilt(b_band, a_band, y_win)
            ha_results.append({"name": f"HAFAST{ha}", "gamma2": float(coh_h[idx_h]), "rms_v": float(np.std(y_fluc))})

        # 3-Panel Radial Profile Figure
        fig_rad, axs_rad = plt.subplots(3, 1, figsize=(10, 11))
        fig_rad.suptitle(f"Primary Mode Radial Fluctuation & Edge Coupling — Shot {shot}\n({title_win}: f ~ {mode_freq:.1f} kHz, {tw_start:.1f}–{tw_end:.1f} ms)", fontsize=12, fontweight="bold")

        ghz_arr = [d["freq_ghz"] for d in ece_results]
        rms_arr = [d["rms_v"] for d in ece_results]
        gam_arr = [d["gamma2"] for d in ece_results]

        axs_rad[0].plot(ghz_arr, rms_arr, marker="o", color="crimson", lw=2.0, ms=7, label=r"RMS Fluctuation $\tilde{T}_e$ (V)")
        axs_rad[0].axvspan(64.0, 67.0, color="gold", alpha=0.2, label="Peak Fluctuation Zone (64.5 - 66.5 GHz)")
        axs_rad[0].set_xlabel("ECE Frequency (GHz)")
        axs_rad[0].set_ylabel(r"Fluctuation RMS (V)")
        axs_rad[0].set_title(r"Radial Electron Temperature Fluctuation Profile $\tilde{T}_e(r)$", fontsize=10)
        axs_rad[0].grid(True, alpha=0.3)
        axs_rad[0].legend(loc="upper left", fontsize=8.5)

        axs_rad[1].plot(ghz_arr, gam_arr, marker="s", color="tab:blue", lw=1.8, ms=6, label=rf"Coherence $\gamma^2({mode_freq:.0f}\mathrm{{kHz}})$ with MP1")
        axs_rad[1].set_xlabel("ECE Frequency (GHz)")
        axs_rad[1].set_ylabel(r"Cross-Coherence $\gamma^2$")
        axs_rad[1].set_title("Radial Coherence with Mirnov Probe MP1", fontsize=10)
        axs_rad[1].grid(True, alpha=0.3)
        axs_rad[1].legend(loc="upper right", fontsize=8.5)

        ha_names = [d["name"] for d in ha_results]
        ha_cohs = [d["gamma2"] for d in ha_results]
        ha_rms = [d["rms_v"] for d in ha_results]

        ax_ha_twin = axs_rad[2].twinx()
        axs_rad[2].bar(np.arange(len(ha_names)) - 0.15, ha_cohs, width=0.3, color="teal", alpha=0.85, label=r"Coherence $\gamma^2$ with MP1")
        ax_ha_twin.plot(np.arange(len(ha_names)) + 0.15, ha_rms, marker="D", color="darkorange", lw=2.0, label="RMS Fluctuation (V)")
        axs_rad[2].set_xticks(np.arange(len(ha_names)))
        axs_rad[2].set_xticklabels(ha_names)
        axs_rad[2].set_ylabel(r"Cross-Coherence $\gamma^2$", color="teal")
        ax_ha_twin.set_ylabel("RMS Fluctuation (V)", color="darkorange")
        axs_rad[2].set_title(r"Edge Fluctuation Coupling: Fast $H_\alpha$ Array (HAFAST)", fontsize=10)
        axs_rad[2].grid(True, alpha=0.3)
        h1, l1 = axs_rad[2].get_legend_handles_labels()
        h2, l2 = ax_ha_twin.get_legend_handles_labels()
        axs_rad[2].legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8.5)

        plt.tight_layout()
        out_path_rad = out_dir / out_png_rad
        safe_savefig(fig_rad, out_path_rad, dpi=150)
        plt.close(fig_rad)
        print(f"    Saved figure: {out_path_rad.name}")

        out_json_path = out_dir / out_json
        with open(out_json_path, "w", encoding="utf-8") as f_out:
            json.dump({
                "shot": shot, "window_ms": [tw_start, tw_end], "mode_freq_khz": mode_freq,
                "best_n": best_n, "second_best_n": second_best_n, "best_rmse_deg": n_fit_results[best_n]["rmse_deg"],
                "ece_peak_channel": max(ece_results, key=lambda d: d["rms_v"])["channel"],
                "ha_peak_channel": max(ha_results, key=lambda d: d["gamma2"])["name"]
            }, f_out, indent=2)
        print(f"    Saved JSON: {out_json_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Toroidal Mode n & Radial Profiles Comparison")
    parser.add_argument("-s", "--shots", type=int, default=DEFAULT_SHOT, help="Shot number to analyze")
    parser.add_argument("-o", "--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    run_toroidal_and_radial_comparison(shot=args.shots, out_dir=args.out_dir, verbose=args.verbose)
