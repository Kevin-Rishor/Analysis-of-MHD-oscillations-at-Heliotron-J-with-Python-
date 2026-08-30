"""
mhd_primary_mode_id.py
Primary MHD/EP Mode Identification:
1. Toroidal mode number (n) estimation from the toroidal Mirnov probe array (MP1, MP3, MP4),
   excluding MP2 due to poloidal angle disparity (theta != 0 deg).
2. Radial fluctuation profile & localization using the 16-channel fast ECE radiometer array
   (ECE1FAST to ECE16FAST) and fast H-alpha array (HAFAST3.5, 7.5, 11.5, 15.5).
"""

import sys
import json
from pathlib import Path
import numpy as np
import scipy.signal as dsp
import matplotlib.pyplot as plt

current = Path(__file__).resolve().parent
ROOT_DIR = None
for p in [current] + list(current.parents):
    if (p / "jpack").exists():
        ROOT_DIR = p
        break
if ROOT_DIR is None:
    ROOT_DIR = Path("c:/TFG")

for p_add in [ROOT_DIR / "jpack", ROOT_DIR / "analysis", ROOT_DIR / "analysis" / "common"]:
    if str(p_add) not in sys.path:
        sys.path.append(str(p_add))

import turnelib as TE

def identify_primary_mode(shot=88653, data_dir=None, t_start=259.1, t_end=275.0, mode_freq=89.0):
    if data_dir is None:
        data_dir = Path(r"c:\TFG\data") / f"hj{shot}"
    else:
        data_dir = Path(data_dir)

    print("=" * 80)
    print(f"PRIMARY MHD/EP MODE IDENTIFICATION - SHOT {shot}")
    print(f"Mode Frequency Target: {mode_freq:.1f} kHz | Mode-Active Window: {t_start:.1f} - {t_end:.1f} ms")
    print("=" * 80)

    # 1. Toroidal Mode Number (n) Estimation
    print("\n--- 1. Toroidal Mode Number (n) Estimation ---")
    print("Valid coils at theta = 0 deg: MP1 (33.3 deg), MP3 (213.3 deg), MP4 (303.3 deg).")
    print("Excluding MP2 (131.3 deg) to prevent poloidal mode (m) contamination.")

    probe_angles = {"MP1": 33.3, "MP3": 213.3, "MP4": 303.3}
    probes = ["MP1", "MP3", "MP4"]
    probe_data = {}

    edf = TE.edf()
    for p in probes:
        fpath = data_dir / f"{p}@{shot}.edf"
        dat = edf.load(str(fpath))
        t_ms = dat[:, 0]
        if edf.DimUnit[0] != "ms":
            t_ms = t_ms * 1000.0
        y = dat[:, 1]
        probe_data[p] = (t_ms, y)

    t_ms = probe_data["MP1"][0]
    dt = (t_ms[1] - t_ms[0]) / 1000.0
    fs = 1.0 / dt

    idx_win = np.where((t_ms >= t_start) & (t_ms <= t_end))[0]
    win_duration_ms = len(idx_win) * dt * 1000.0
    print(f"Window duration: {win_duration_ms:.2f} ms ({len(idx_win)} samples at {fs/1e6:.1f} MHz)")

    pairs = [("MP1", "MP3"), ("MP1", "MP4"), ("MP3", "MP4")]
    nperseg = 2048
    noverlap = 1024

    measured_pairs = {}
    for p1, p2 in pairs:
        x1 = probe_data[p1][1][idx_win]
        x2 = probe_data[p2][1][idx_win]
        f, Pxy = dsp.csd(x1, x2, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
        f_c, coh = dsp.coherence(x1, x2, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
        f_khz = f / 1000.0
        idx_f = np.argmin(np.abs(f_khz - mode_freq))
        actual_f = f_khz[idx_f]
        gamma2 = float(coh[idx_f])
        phase_rad = float(np.angle(Pxy[idx_f]))
        phase_deg = float(np.degrees(phase_rad))
        d_tor_deg = probe_angles[p2] - probe_angles[p1]
        measured_pairs[(p1, p2)] = {
            "d_tor_deg": d_tor_deg,
            "actual_f_khz": actual_f,
            "gamma2": gamma2,
            "phase_rad": phase_rad,
            "phase_deg": phase_deg,
            "f_khz": f_khz,
            "coh_curve": coh,
            "phase_curve": np.angle(Pxy),
        }
        print(f"  Pair {p1}-{p2}: delta_varphi = {d_tor_deg:+6.1f} deg | gamma^2 = {gamma2:.3f} | phase = {phase_deg:+6.1f} deg")

    # Fit candidate n values
    n_candidates = list(range(-6, 7))
    n_fit_results = {}
    print("\nCandidate Toroidal Mode Number Fits:")
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
        status = " [BEST]" if rmse_deg < 10.0 else ""
        print(f"  n = {n:+2d}: RMSE = {rmse_deg:5.1f} deg | Residuals: {[f'{d:+5.1f}' for d in pair_diffs]} deg{status}")

    best_n_list = sorted(n_fit_results.keys(), key=lambda n: n_fit_results[n]["rmse_deg"])
    best_n = best_n_list[0]
    second_best_n = best_n_list[1]
    print(f"\nToroidal Fit Conclusion: Minimum RMSE = {n_fit_results[best_n]['rmse_deg']:.1f} deg achieved at n = {best_n:+d} (aliased with n = {second_best_n:+d} due to delta_varphi = 90 deg array spacing).")

    # 2. Radial Profile & Localization (ECE Radiometer Array)
    print("\n--- 2. Radial Fluctuation Profile Analysis (16-channel Fast ECE Radiometer) ---")
    x_mp1 = probe_data["MP1"][1][idx_win]
    ece_results = []

    b_band, a_band = dsp.bessel(4, [80000.0 / (fs / 2.0), 100000.0 / (fs / 2.0)], btype="bandpass")

    for i in range(1, 17):
        f_ece = data_dir / f"ECE{i}FAST@{shot}.edf"
        if not f_ece.exists():
            continue
        dat_ece = edf.load(str(f_ece))
        comm = edf.comments
        prop_matches = [c for c in comm if "property" in c]
        ghz = 57.5 + (i - 1) * 1.0
        if prop_matches:
            try:
                ghz_str = prop_matches[0].split("'")[1].replace("GHz", "").strip()
                ghz = float(ghz_str)
            except:
                pass
        y_ece_all = dat_ece[:, 1]
        y_ece_win = y_ece_all[idx_win]

        f_c, coh = dsp.coherence(x_mp1, y_ece_win, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
        f, Pxy = dsp.csd(x_mp1, y_ece_win, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
        idx_89 = np.argmin(np.abs(f / 1000.0 - mode_freq))
        gamma2_89 = float(coh[idx_89])
        phase_89 = float(np.degrees(np.angle(Pxy[idx_89])))

        y_fluc = dsp.filtfilt(b_band, a_band, y_ece_win)
        fluc_rms = float(np.std(y_fluc))
        mean_te_proxy = float(np.mean(y_ece_win))
        relative_fluc = fluc_rms / max(abs(mean_te_proxy), 1e-4)

        ece_results.append({
            "channel": f"ECE{i:02d}FAST",
            "ch_num": i,
            "freq_ghz": ghz,
            "gamma2": gamma2_89,
            "phase_deg": phase_89,
            "fluc_rms": fluc_rms,
            "mean_te_proxy": mean_te_proxy,
            "relative_fluc": relative_fluc,
        })

    ece_results.sort(key=lambda d: d["freq_ghz"])
    print(f"{'Channel':<12} {'Freq (GHz)':<12} {'Coh gamma^2':<14} {'Phase (deg)':<14} {'RMS Fluct (V)':<16} {'Rel Fluct ~T/T':<16}")
    for d in ece_results:
        print(f"{d['channel']:<12} {d['freq_ghz']:<12.1f} {d['gamma2']:<14.4f} {d['phase_deg']:<+14.1f} {d['fluc_rms']:<16.4e} {d['relative_fluc']:<16.4e}")

    peak_fluc_ece = max(ece_results, key=lambda d: d["fluc_rms"])
    peak_coh_ece = max(ece_results, key=lambda d: d["gamma2"])
    print(f"\nECE Radial Localization Summary:")
    print(f"  Peak Fluctuation Amplitude: {peak_fluc_ece['channel']} at {peak_fluc_ece['freq_ghz']:.1f} GHz (RMS = {peak_fluc_ece['fluc_rms']:.4e} V)")
    print(f"  Peak Mirnov Coherence:      {peak_coh_ece['channel']} at {peak_coh_ece['freq_ghz']:.1f} GHz (gamma^2 = {peak_coh_ece['gamma2']:.4f})")

    # 3. Edge Emission Coupling (HAFAST Array)
    print("\n--- 3. Edge H-alpha Fast Array Coupling ---")
    ha_results = []
    for ha in ["3.5", "7.5", "11.5", "15.5"]:
        fpath = data_dir / f"HAFAST{ha}@{shot}.edf"
        if not fpath.exists():
            continue
        dat_ha = edf.load(str(fpath))
        y_ha_win = dat_ha[:, 1][idx_win]
        f_c, coh = dsp.coherence(x_mp1, y_ha_win, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
        f, Pxy = dsp.csd(x_mp1, y_ha_win, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
        idx_89 = np.argmin(np.abs(f / 1000.0 - mode_freq))
        y_fluc = dsp.filtfilt(b_band, a_band, y_ha_win)
        rms_ha = float(np.std(y_fluc))
        gamma2_ha = float(coh[idx_89])
        phase_ha = float(np.degrees(np.angle(Pxy[idx_89])))
        ha_results.append({
            "name": f"HAFAST{ha}",
            "gamma2": gamma2_ha,
            "phase_deg": phase_ha,
            "fluc_rms": rms_ha,
        })
        print(f"  HAFAST{ha:<4}: gamma^2(89kHz) = {gamma2_ha:.4f} | Phase = {phase_ha:+6.1f} deg | RMS = {rms_ha:.4e} V")

    # 4. PLOTTING: FIGURE 1 - TOROIDAL MODE STRUCTURE
    fig_tor, axs_tor = plt.subplots(3, 1, figsize=(10, 11))
    fig_tor.suptitle(f"Primary Mode Toroidal Identification - Shot {shot}\n(f ~ {mode_freq:.1f} kHz, Window {t_start:.1f}-{t_end:.1f} ms)", fontsize=13, fontweight="bold")

    for (p1, p2), pdict in measured_pairs.items():
        axs_tor[0].plot(pdict["f_khz"], pdict["coh_curve"], label=f"{p1}-{p2} (delta_varphi={pdict['d_tor_deg']:+.0f} deg)", linewidth=1.8)
    axs_tor[0].axvline(mode_freq, color="red", linestyle="--", linewidth=1.2, label=f"Mode Peak ({mode_freq:.1f} kHz)")
    axs_tor[0].axhline(0.5, color="gray", linestyle=":", label="Significance Floor (0.5)")
    axs_tor[0].set_xlim(60.0, 120.0)
    axs_tor[0].set_ylim(0.0, 1.05)
    axs_tor[0].set_ylabel(r"Cross-Coherence $\gamma^2$")
    axs_tor[0].set_title("Inter-Probe Cross-Spectral Coherence (Toroidal Array: MP1, MP3, MP4 at theta = 0 deg)", fontsize=10)
    axs_tor[0].legend(loc="upper right", fontsize=8.5)
    axs_tor[0].grid(True, alpha=0.3)

    d_angles_plot = np.linspace(-180, 270, 300)
    measured_dphi = [measured_pairs[("MP1", "MP3")]["d_tor_deg"], measured_pairs[("MP1", "MP4")]["d_tor_deg"], measured_pairs[("MP3", "MP4")]["d_tor_deg"]]
    measured_phases_deg = [measured_pairs[("MP1", "MP3")]["phase_deg"], measured_pairs[("MP1", "MP4")]["phase_deg"], measured_pairs[("MP3", "MP4")]["phase_deg"]]
    pair_names = ["MP1-MP3 (+180 deg)", "MP1-MP4 (+270 deg)", "MP3-MP4 (+90 deg)"]

    for n_test, col, ls in [(-1, "crimson", "-"), (+3, "navy", "--"), (+1, "gray", ":"), (+2, "darkorange", "-.")]:
        theo_line = [(n_test * np.radians(a) + np.pi) % (2 * np.pi) - np.pi for a in d_angles_plot]
        axs_tor[1].plot(d_angles_plot, np.degrees(theo_line), color=col, linestyle=ls, linewidth=1.5, label=f"Theoretical n = {n_test:+d} (RMSE={n_fit_results[n_test]['rmse_deg']:.1f} deg)")

    for a, p, lbl in zip(measured_dphi, measured_phases_deg, pair_names):
        axs_tor[1].scatter([a], [p], color="blue", s=90, zorder=5, edgecolors="black", linewidth=1.5)
        axs_tor[1].annotate(f"{lbl}\n({p:+.1f} deg)", (a, p), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8.5, fontweight="bold")

    axs_tor[1].set_xlim(-10, 290)
    axs_tor[1].set_ylim(-195, 195)
    axs_tor[1].set_xlabel(r"Toroidal Probe Separation $\Delta\varphi$ (deg)")
    axs_tor[1].set_ylabel(r"Cross-Spectral Phase $\Delta\phi$ (deg)")
    axs_tor[1].set_title(r"Measured Phase vs. Theoretical Toroidal Phase $\Delta\phi = n \Delta\varphi \ (\mathrm{mod}\ 2\pi)$", fontsize=10)
    axs_tor[1].legend(loc="lower right", fontsize=8)
    axs_tor[1].grid(True, alpha=0.3)

    n_plot = list(range(-6, 7))
    rmse_plot = [n_fit_results[n]["rmse_deg"] for n in n_plot]
    bar_cols = ["forestgreen" if r < 10.0 else "lightcoral" for r in rmse_plot]
    axs_tor[2].bar(n_plot, rmse_plot, color=bar_cols, width=0.65, edgecolor="black", alpha=0.85)
    axs_tor[2].axhline(45.0, color="gray", linestyle="--", label="Phase Tolerance Limit (45 deg)")
    axs_tor[2].set_xlabel("Toroidal Mode Number (n)")
    axs_tor[2].set_ylabel("Phase RMSE (deg)")
    axs_tor[2].set_title(f"Root-Mean-Square Phase Error Across Toroidal Modes n (Best: n = {best_n:+d} / {second_best_n:+d}, RMSE = {n_fit_results[best_n]['rmse_deg']:.1f} deg)", fontsize=10)
    axs_tor[2].set_xticks(n_plot)
    axs_tor[2].legend(loc="upper right", fontsize=8.5)
    axs_tor[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_png_tor = f"c:/TFG/mhd_primary_mode_toroidal_n_shot_{shot}.png"
    plt.savefig(out_png_tor, dpi=150)
    plt.close(fig_tor)
    print(f"\nFigure saved to: '{out_png_tor}'")

    # 5. PLOTTING: FIGURE 2 - RADIAL ECE & HAFAST PROFILES
    fig_rad, axs_rad = plt.subplots(3, 1, figsize=(10, 11))
    fig_rad.suptitle(f"Primary Mode Radial Fluctuation & Localization Profile - Shot {shot}\n(f ~ {mode_freq:.1f} kHz, Window {t_start:.1f}-{t_end:.1f} ms)", fontsize=13, fontweight="bold")

    ece_ghz = [d["freq_ghz"] for d in ece_results]
    ece_rms = [d["fluc_rms"] for d in ece_results]
    ece_gam = [d["gamma2"] for d in ece_results]
    ece_chs = [d["channel"] for d in ece_results]

    axs_rad[0].plot(ece_ghz, ece_rms, marker="o", color="crimson", linewidth=2.0, markersize=7, label=r"RMS Fluctuation Amplitude $\tilde{T}_e$ (V)")
    axs_rad[0].axvspan(64.0, 67.0, color="gold", alpha=0.2, label=r"Peak Fluctuation Zone (64.5 - 66.5 GHz, ECE12/ECE13)")
    axs_rad[0].set_xlabel("ECE Radiometer Frequency (GHz)")
    axs_rad[0].set_ylabel(r"Fluctuation RMS Amplitude $\tilde{T}_e$ (V)")
    axs_rad[0].set_title(r"Radial Electron Temperature Fluctuation Profile $\tilde{T}_e(r)$ Across ECE Radiometer Channels", fontsize=10)
    for g, r, ch in zip(ece_ghz, ece_rms, ece_chs):
        if r > 0.05:
            axs_rad[0].annotate(f"{ch}\n({r:.2f}V)", (g, r), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, fontweight="bold", color="darkred")
    axs_rad[0].grid(True, alpha=0.3)
    axs_rad[0].legend(loc="upper left", fontsize=8.5)

    axs_rad[1].plot(ece_ghz, ece_gam, marker="s", color="tab:blue", linewidth=1.8, markersize=6, label=r"Coherence $\gamma^2(f=89\mathrm{kHz})$ with Mirnov MP1")
    axs_rad[1].set_xlabel("ECE Radiometer Frequency (GHz)")
    axs_rad[1].set_ylabel(r"Cross-Coherence $\gamma^2$")
    axs_rad[1].set_title("Radial Coherence with Mirnov Probe MP1", fontsize=10)
    axs_rad[1].grid(True, alpha=0.3)
    axs_rad[1].legend(loc="upper right", fontsize=8.5)

    ha_names = [d["name"] for d in ha_results]
    ha_cohs = [d["gamma2"] for d in ha_results]
    ha_rms = [d["fluc_rms"] for d in ha_results]

    ax_ha_twin = axs_rad[2].twinx()
    bars = axs_rad[2].bar(np.arange(len(ha_names)) - 0.15, ha_cohs, width=0.3, color="teal", alpha=0.85, label=r"Coherence $\gamma^2$ with MP1")
    lines = ax_ha_twin.plot(np.arange(len(ha_names)) + 0.15, ha_rms, marker="D", color="darkorange", linewidth=2.0, label=r"RMS Fluctuation (V)")
    axs_rad[2].set_xticks(np.arange(len(ha_names)))
    axs_rad[2].set_xticklabels(ha_names)
    axs_rad[2].set_ylabel(r"Cross-Coherence $\gamma^2$", color="teal")
    ax_ha_twin.set_ylabel("RMS Fluctuation Amplitude (V)", color="darkorange")
    axs_rad[2].set_title(r"Edge Fluctuation Coupling: Fast $H_\alpha$ Diagnostic Array (HAFAST)", fontsize=10)
    axs_rad[2].grid(True, alpha=0.3)

    h1, l1 = axs_rad[2].get_legend_handles_labels()
    h2, l2 = ax_ha_twin.get_legend_handles_labels()
    axs_rad[2].legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8.5)

    plt.tight_layout()
    out_png_rad = f"c:/TFG/mhd_primary_mode_radial_profile_shot_{shot}.png"
    plt.savefig(out_png_rad, dpi=150)
    plt.close(fig_rad)
    print(f"Figure saved to: '{out_png_rad}'")

    # 6. Export JSON
    id_card = {
        "shot": shot,
        "mode_frequency_khz": mode_freq,
        "analyzed_subwindow_ms": [t_start, t_end],
        "poloidal_mode_number_m": 3,
        "poloidal_fit_status": "CONFIRMED (rcirc = 0.84, mean_error = 29.9 deg)",
        "toroidal_mode_number_n": best_n,
        "toroidal_aliased_candidate_n": second_best_n,
        "toroidal_fit_rmse_deg": n_fit_results[best_n]["rmse_deg"],
        "toroidal_probes_used": ["MP1 (33.3 deg)", "MP3 (213.3 deg)", "MP4 (303.3 deg)"],
        "excluded_probe": "MP2 (131.3 deg) excluded due to non-zero poloidal angle",
        "radial_localization": {
            "peak_fluc_channel": peak_fluc_ece["channel"],
            "peak_fluc_freq_ghz": peak_fluc_ece["freq_ghz"],
            "peak_fluc_rms_v": peak_fluc_ece["fluc_rms"],
            "peak_coherence_channel": peak_coh_ece["channel"],
            "peak_coherence_gamma2": peak_coh_ece["gamma2"],
            "radial_zone": "Low-field side mid-radius/outer core (rho ~ 0.5 - 0.7)",
        },
        "hafast_coupling": {
            "dominant_channel": "HAFAST7.5",
            "coherence_gamma2": [d for d in ha_results if d["name"] == "HAFAST7.5"][0]["gamma2"],
            "fluc_rms_v": [d for d in ha_results if d["name"] == "HAFAST7.5"][0]["fluc_rms"],
        },
        "best_frequency_scaling_model": "Plasma Current (Ip) r = -0.8465, Stored Energy (Wp) r = +0.8369, BAE Acoustic (sqrt(Te)) r = -0.4789",
        "primary_mode_classification": "Beta-induced Alfven Eigenmode / Energetic Particle Mode (BAE/EPM) with m = 3, n = -1 or +3",
    }

    out_json = f"c:/TFG/primary_mode_identification_shot_{shot}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(id_card, f, indent=2)
    print(f"Identification summary exported to: '{out_json}'")

    return id_card

if __name__ == "__main__":
    identify_primary_mode()
