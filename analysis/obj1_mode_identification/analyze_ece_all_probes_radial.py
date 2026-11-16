import sys
import json
from pathlib import Path
import numpy as np
import scipy.signal as dsp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

root_dir = Path(__file__).resolve().parents[2]
sys.path.extend([
    str(root_dir / "jpack"),
    str(root_dir / "analysis"),
    str(root_dir / "analysis" / "common"),
    str(root_dir / "analysis" / "heating_phase_comparison"),
    str(root_dir / "analysis" / "obj1_mode_identification"),
])

from comparison_common import load_channel, PMP_INVERT, safe_savefig, extract_panel_axes
from utils.heliotron_j_ece import freq_to_rho, rho_to_freq, freq_to_r_cm, ECE_CHANNELS

shot = 88653
data_dir = root_dir / "data" / f"hj{shot}"
out_dir = root_dir / "Images" / "obj1_mode_identification"
out_dir.mkdir(parents=True, exist_ok=True)
panels_dir = out_dir / "panels"
panels_dir.mkdir(parents=True, exist_ok=True)

# Exact 16-channel Heliotron J radiometer frequencies
ECE_FREQS_GHZ = {
    1: 57.5, 9: 58.5, 2: 59.5, 10: 60.5, 3: 61.5, 11: 62.5, 4: 63.5, 12: 64.5,
    5: 65.5, 13: 66.5, 6: 67.5, 14: 68.5, 7: 69.5, 15: 70.5, 8: 71.5, 16: 72.5
}
ece_order = sorted(range(1, 17), key=lambda i: ECE_FREQS_GHZ[i])

mag_probes = ["MP1", "MP2", "MP3", "MP4"] + [f"PMP{i}" for i in range(1, 15)]

def run_analysis():
    print(f"=== Complete ECE vs All Magnetic Coils Analysis (Shot {shot}) ===")
    
    # 1. Load magnetic data
    mag_data = {}
    t_ms = None
    fs = None
    for p in mag_probes:
        t, y, dt, f_s = load_channel(shot, p, data_dir=data_dir)
        if y is not None:
            if p in PMP_INVERT:
                y = -y
            mag_data[p] = y
            if t_ms is None:
                t_ms = t
                fs = f_s
    print(f"Loaded {len(mag_data)} magnetic coils at fs={fs:.0f} Hz")
    
    # 2. Load ECE data
    ece_data = {}
    for i in ece_order:
        ch = f"ECE{i}FAST"
        _, y, _, _ = load_channel(shot, ch, data_dir=data_dir)
        ece_data[ch] = y
    print(f"Loaded {len(ece_data)} ECE channels")
    
    # Define analysis windows
    idx_quiet = np.where((t_ms >= 150.0) & (t_ms <= 175.0))[0]
    idx_w1 = np.where((t_ms >= 259.1) & (t_ms <= 291.3))[0]    # ECH+NBI
    idx_w2 = np.where((t_ms >= 291.3) & (t_ms <= 300.0))[0]    # NBI-only
    idx_w0 = np.where((t_ms >= 259.1) & (t_ms <= 275.0))[0]    # Orig subwindow
    idx_w2_steady = np.where((t_ms >= 294.2) & (t_ms <= 300.0))[0] # Clean steady NBI
    
    windows_dict = {
        "ECH_NBI": {"idx": idx_w1, "label": "ECH + NBI (259.1 - 291.3 ms)", "f_pri": 89.0, "f_sec": 48.8, "nperseg": 2048},
        "NBI_only": {"idx": idx_w2, "label": "NBI only (291.3 - 300.0 ms)", "f_pri": 86.0, "f_sec": 41.0, "nperseg": 1024},
        "orig_259_275": {"idx": idx_w0, "label": "ECH+NBI Subwindow (259.1 - 275.0 ms)", "f_pri": 89.0, "f_sec": 49.0, "nperseg": 2048},
        "NBI_steady": {"idx": idx_w2_steady, "label": "NBI Steady Tail (294.2 - 300.0 ms)", "f_pri": 86.0, "f_sec": 41.0, "nperseg": 512},
    }
    
    bands = [
        {"name": "Secondary", "label": "Secondary Mode (40-80 kHz)", "flo": 40000.0, "fhi": 80000.0, "color": "darkorange"},
        {"name": "Primary", "label": "Primary Mode (80-100 kHz)", "flo": 80000.0, "fhi": 100000.0, "color": "crimson"}
    ]
    
    # Store complete numerical results for JSON export
    results_export = {
        "shot": shot,
        "ece_frequencies_ghz": {f"ECE{i:02d}FAST": ECE_FREQS_GHZ[i] for i in ece_order},
        "magnetic_probes": mag_probes,
        "windows": {}
    }
    
    # Structure for plotting
    plot_data = {}
    
    for wkey, win_info in windows_dict.items():
        idx_w = win_info["idx"]
        n_samples = len(idx_w)
        nperseg = min(win_info["nperseg"], n_samples // 2)
        noverlap = nperseg // 2
        k_segs = 1 + (n_samples - nperseg) // (nperseg - noverlap)
        floor95 = 1.0 - 0.05 ** (1.0 / max(1, k_segs - 1))
        
        win_results = {
            "label": win_info["label"],
            "n_samples": n_samples,
            "k_segments": k_segs,
            "coherence_floor_95": floor95,
            "bands": {}
        }
        
        plot_data[wkey] = {
            "win_info": win_info,
            "k_segs": k_segs,
            "floor95": floor95,
            "bands": {}
        }
        
        for band in bands:
            bname = band["name"]
            flo = band["flo"]
            fhi = band["fhi"]
            f_target = win_info["f_pri"] if bname == "Primary" else win_info["f_sec"]
            
            b_bp, a_bp = dsp.bessel(4, [flo / (fs / 2.0), fhi / (fs / 2.0)], btype="bandpass")
            
            # Filter magnetic probes
            mag_f = {p: dsp.filtfilt(b_bp, a_bp, mag_data[p][idx_w]) for p in mag_probes}
            mag_env = {p: np.abs(dsp.hilbert(mag_f[p])) for p in mag_probes}
            
            # Mutual coherence between ECE12 and ECE13
            fc_12_13, coh_12_13 = dsp.coherence(ece_data["ECE12FAST"][idx_w], ece_data["ECE13FAST"][idx_w], fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
            m_band_12_13 = (fc_12_13 >= flo) & (fc_12_13 <= fhi)
            ece12_13_mean_coh = float(np.mean(coh_12_13[m_band_12_13]))
            ece12_13_max_coh = float(np.max(coh_12_13[m_band_12_13]))
            
            # 16 ECE x 18 Coils matrices
            matrix_xcorr_car = np.zeros((16, len(mag_probes)))
            matrix_xcorr_env = np.zeros((16, len(mag_probes)))
            matrix_coh_fpk = np.zeros((16, len(mag_probes)))
            matrix_coh_max = np.zeros((16, len(mag_probes)))
            matrix_coh_mean = np.zeros((16, len(mag_probes)))
            
            ece_rms_active = []
            ece_rms_quiet = []
            ece_ratio = []
            ece_ghz_list = []
            ece_ch_names = []
            
            max_lag_car = int(0.025e-3 * fs)  # +/- 25 us
            max_lag_env = int(0.100e-3 * fs)  # +/- 100 us
            
            for row_idx, i in enumerate(ece_order):
                ch = f"ECE{i}FAST"
                ghz = ECE_FREQS_GHZ[i]
                ece_ch_names.append(ch)
                ece_ghz_list.append(ghz)
                
                yw = ece_data[ch][idx_w]
                yq = ece_data[ch][idx_quiet]
                yf = dsp.filtfilt(b_bp, a_bp, yw)
                yfq = dsp.filtfilt(b_bp, a_bp, yq)
                
                rms_act = float(np.std(yf))
                rms_q = float(np.std(yfq))
                ratio = float(rms_act / (rms_q + 1e-30))
                
                ece_rms_active.append(rms_act)
                ece_rms_quiet.append(rms_q)
                ece_ratio.append(ratio)
                
                yf_norm = (yf - np.mean(yf)) / (np.std(yf) + 1e-30)
                yenv = np.abs(dsp.hilbert(yf))
                yenv_norm = (yenv - np.mean(yenv)) / (np.std(yenv) + 1e-30)
                
                for col_idx, p in enumerate(mag_probes):
                    mf = mag_f[p]
                    mf_norm = (mf - np.mean(mf)) / (np.std(mf) + 1e-30)
                    xc_c = dsp.correlate(yf_norm, mf_norm, mode="full") / len(yf_norm)
                    mid = len(yf_norm) - 1
                    r_car = float(np.max(np.abs(xc_c[mid - max_lag_car : mid + max_lag_car + 1])))
                    matrix_xcorr_car[row_idx, col_idx] = r_car
                    
                    menv = mag_env[p]
                    menv_norm = (menv - np.mean(menv)) / (np.std(menv) + 1e-30)
                    xc_e = dsp.correlate(yenv_norm, menv_norm, mode="full") / len(yenv_norm)
                    r_env = float(np.max(np.abs(xc_e[mid - max_lag_env : mid + max_lag_env + 1])))
                    matrix_xcorr_env[row_idx, col_idx] = r_env
                    
                    fc, coh = dsp.coherence(mag_data[p][idx_w], yw, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
                    idx_fpk = np.argmin(np.abs(fc / 1000.0 - f_target))
                    matrix_coh_fpk[row_idx, col_idx] = float(coh[idx_fpk])
                    
                    mb = (fc >= flo) & (fc <= fhi)
                    matrix_coh_max[row_idx, col_idx] = float(np.max(coh[mb]))
                    matrix_coh_mean[row_idx, col_idx] = float(np.mean(coh[mb]))
            
            # Identify max values across probes for each ECE channel
            max_coh_across_probes = np.max(matrix_coh_max, axis=1)
            best_probe_idx = np.argmax(matrix_coh_max, axis=1)
            best_probe_names = [mag_probes[idx] for idx in best_probe_idx]
            
            max_xcorr_across_probes = np.max(matrix_xcorr_car, axis=1)
            best_xcorr_probe_idx = np.argmax(matrix_xcorr_car, axis=1)
            best_xcorr_probe_names = [mag_probes[idx] for idx in best_xcorr_probe_idx]
            
            mp1_col_idx = mag_probes.index("MP1")
            mp1_coh_curve = matrix_coh_max[:, mp1_col_idx]
            mp1_xcorr_curve = matrix_xcorr_car[:, mp1_col_idx]
            
            band_res = {
                "target_mode_freq_khz": f_target,
                "ece12_13_mutual_coh_mean": ece12_13_mean_coh,
                "ece12_13_mutual_coh_max": ece12_13_max_coh,
                "channels": {}
            }
            for row_idx, ch in enumerate(ece_ch_names):
                band_res["channels"][ch] = {
                    "freq_ghz": ece_ghz_list[row_idx],
                    "rms_active_v": ece_rms_active[row_idx],
                    "rms_quiet_v": ece_rms_quiet[row_idx],
                    "active_over_quiet_ratio": ece_ratio[row_idx],
                    "best_probe_max_coh": float(max_coh_across_probes[row_idx]),
                    "best_probe_name": best_probe_names[row_idx],
                    "mp1_coh": float(mp1_coh_curve[row_idx]),
                    "best_probe_max_xcorr": float(max_xcorr_across_probes[row_idx]),
                    "best_xcorr_probe_name": best_xcorr_probe_names[row_idx],
                    "mp1_xcorr": float(mp1_xcorr_curve[row_idx]),
                }
            
            win_results["bands"][bname] = band_res
            plot_data[wkey]["bands"][bname] = {
                "ece_ghz": np.array(ece_ghz_list),
                "ece_ch_names": ece_ch_names,
                "rms_act": np.array(ece_rms_active),
                "rms_quiet": np.array(ece_rms_quiet),
                "ratio": np.array(ece_ratio),
                "max_coh": max_coh_across_probes,
                "best_probe_names": best_probe_names,
                "mp1_coh": mp1_coh_curve,
                "max_xcorr": max_xcorr_across_probes,
                "best_xcorr_probe_names": best_xcorr_probe_names,
                "mp1_xcorr": mp1_xcorr_curve,
                "matrix_coh_max": matrix_coh_max,
                "matrix_xcorr_car": matrix_xcorr_car,
                "matrix_xcorr_env": matrix_xcorr_env,
                "ece12_13_mutual_coh_mean": ece12_13_mean_coh,
                "ece12_13_mutual_coh_max": ece12_13_max_coh,
            }
        
        results_export["windows"][wkey] = win_results
        
    # Export full JSON
    json_path = root_dir / "analysis" / "obj1_mode_identification" / "ece_all_probes_radial_88653.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_export, f, indent=2)
    print(f"Exported results JSON to '{json_path}'")
    
    # -------------------------------------------------------------
    # 3. Generate Publication Figure: 4-Panel Synthesis
    # -------------------------------------------------------------
    # Layout (16:9 Beamer optimized, 2x2 grid):
    # -------------------------------------------------------------
    # 3. Generate Publication Figure: 4-Panel Synthesis
    # -------------------------------------------------------------
    # Layout (16:9 Beamer optimized, 2x2 grid):
    # Top-Left  (a): Radial RMS Fluctuation Profile Te(r) for Primary & Secondary (both windows)
    # Top-Right (b): Radial Cross-Coherence: Best of All 18 Coils vs MP1-only baseline
    # Bot-Left  (c): Matrix Heatmap: Secondary Band (40-80 kHz) - All 16 ECEs vs All 18 Coils
    # Bot-Right (d): Matrix Heatmap: Primary Band (80-100 kHz) - All 16 ECEs vs All 18 Coils
    
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12.5,
        "axes.titlesize": 13.5,
        "axes.labelsize": 13.0,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 10.5,
        "figure.titlesize": 15.0,
    })
    
    fig, axs = plt.subplots(2, 2, figsize=(13.4, 9.5), facecolor="white")
    fig.suptitle(f"Heliotron J #{shot}", fontweight="bold", y=0.985, fontsize=15.0)
    
    w_ech = plot_data["ECH_NBI"]
    ghz = w_ech["bands"]["Primary"]["ece_ghz"]
    
    # --- PANEL (a): Radial Fluctuation Amplitude Profile ---
    ax_a = axs[0, 0]
    ax_a.axvspan(64.0, 67.0, color="#fff2cc", alpha=0.7, label="Resonant Zone", zorder=1)
    
    # Plot Secondary and Primary RMS in ECH+NBI
    rms_sec_ech = w_ech["bands"]["Secondary"]["rms_act"]
    rms_pri_ech = w_ech["bands"]["Primary"]["rms_act"]
    ax_a.plot(ghz, rms_sec_ech, marker="s", color="darkorange", lw=2.2, ms=6.0, label="Sec (ECH+NBI)", zorder=3)
    ax_a.plot(ghz, rms_pri_ech, marker="o", color="crimson", lw=2.4, ms=6.5, label="Pri (ECH+NBI)", zorder=4)
    
    # Plot NBI-only dashed
    w_nbi = plot_data["NBI_only"]
    rms_sec_nbi = w_nbi["bands"]["Secondary"]["rms_act"]
    rms_pri_nbi = w_nbi["bands"]["Primary"]["rms_act"]
    ax_a.plot(ghz, rms_sec_nbi, marker="s", ls="--", color="goldenrod", lw=1.7, ms=5.0, alpha=0.85, label="Sec (NBI-only)", zorder=2)
    ax_a.plot(ghz, rms_pri_nbi, marker="o", ls="--", color="indianred", lw=1.8, ms=5.5, alpha=0.85, label="Pri (NBI-only)", zorder=2)
    
    # Peak channel indices
    idx_12 = list(ghz).index(64.5)
    idx_13 = list(ghz).index(66.5)
    
    # Clean high-contrast summary badge placed in right void (x in [68, 73.5], y in [0.10, 0.72]) where curves are ~0 V
    summary_badge = (
        r"$\mathbf{Core\ Peak}$" + "\n" +
        r"$\mathbf{(ECE12/13)}$" + "\n" +
        f"ECE13 ($\\rho=0.29$):\n" +
        f"Sec: {rms_sec_ech[idx_13]:.2f} V (12×)\n" +
        f"Pri: {rms_pri_ech[idx_13]:.2f} V (11×)\n" +
        f"ECE12 ($\\rho=0.47$):\n" +
        f"Sec: {rms_sec_ech[idx_12]:.2f} V (18×)\n" +
        f"Pri: {rms_pri_ech[idx_12]:.2f} V (21×)"
    )
    ax_a.text(73.9, 0.90, summary_badge, fontsize=11.7, va="top", ha="right",
              bbox=dict(boxstyle="round,pad=0.24", facecolor="#fff9f9", edgecolor="crimson", alpha=0.95),
              zorder=10)
    
    # Vertical drop pins pointing down to the peaks, with labels placed clearly away from badge
    ax_a.annotate("", xy=(66.5, rms_sec_ech[idx_13] + 0.02), xytext=(66.2, rms_sec_ech[idx_13] + 0.18),
                  arrowprops=dict(arrowstyle="->", color="darkred", lw=1.4), zorder=5)
    ax_a.text(66.2, rms_sec_ech[idx_13] + 0.20, "ECE13\n(66.5 GHz)", fontsize=10.2, fontweight="bold",
              color="darkred", ha="center", va="bottom", zorder=10)
    
    ax_a.annotate("", xy=(64.5, rms_sec_ech[idx_12] + 0.02), xytext=(62.3, rms_sec_ech[idx_12] + 0.22),
                  arrowprops=dict(arrowstyle="->", color="darkred", lw=1.4), zorder=5)
    ax_a.text(62.2, rms_sec_ech[idx_12] + 0.23, "ECE12\n(64.5 GHz)", fontsize=10.2, fontweight="bold",
              color="darkred", ha="right", va="bottom", zorder=10)
    
    ax_a.set_xlabel(r"ECE Frequency $f_{\mathrm{ECE}}$ [GHz]", fontsize=13.0)
    ax_a.set_ylabel(r"Fluctuation RMS $\tilde{T}_e$ [V]", fontsize=13.0)
    ax_a.set_title(r"(a) Radial Profile $\tilde{T}_e(r)$ ($\rho \sim 0.29$–$0.47$, centroid $\approx 0.38$)", fontweight="bold", fontsize=12.6, pad=8)
    ax_a.set_xlim(56.8, 74.2)
    ax_a.set_ylim(-0.02, 1.62)
    ax_a.grid(True, ls=":", alpha=0.5)
    
    secax_a = ax_a.secondary_xaxis('top', functions=(freq_to_rho, rho_to_freq))
    secax_a.set_xlabel(r"Normalized Minor Radius $\rho$ [-] (TRAVIS Fit)", fontsize=11.5, fontweight="bold", labelpad=6)
    secax_a.set_ticks([0.9, 0.7, 0.5, 0.3, 0.1, -0.1])
    secax_a.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    secax_a.tick_params(labelsize=11.0)
    
    ax_a.legend(loc="upper left", bbox_to_anchor=(0.015, 0.985), ncol=2, columnspacing=0.8, fontsize=9.8, framealpha=0.92, edgecolor="#cccccc")
    
    # --- PANEL (b): Radial Cross-Coherence & Cross-Correlation ---
    ax_b = axs[0, 1]
    ax_b.axvspan(64.0, 67.0, color="#fff2cc", alpha=0.7, label=r"Resonant ($\rho \sim 0.29$–$0.47$)", zorder=1)
    
    floor_ech = w_ech["floor95"]
    ax_b.axhline(0.70, color="forestgreen", ls="--", lw=1.4, label=r"Mode Gate ($\gamma^2 > 0.70$)", zorder=2)
    ax_b.axhline(floor_ech, color="gray", ls=":", lw=1.2, label=f"95% Floor ({floor_ech:.3f})", zorder=2)
    
    pri_best_coh = w_ech["bands"]["Primary"]["max_coh"]
    pri_mp1_coh = w_ech["bands"]["Primary"]["mp1_coh"]
    sec_best_coh = w_ech["bands"]["Secondary"]["max_coh"]
    sec_mp1_coh = w_ech["bands"]["Secondary"]["mp1_coh"]
    
    ax_b.plot(ghz, sec_best_coh, marker="s", color="darkorange", lw=2.2, ms=6.0, label="Sec: Max 18 Coils", zorder=4)
    ax_b.plot(ghz, pri_best_coh, marker="o", color="crimson", lw=2.4, ms=6.5, label="Pri: Max 18 Coils", zorder=5)
    ax_b.plot(ghz, sec_mp1_coh, marker="s", ls=":", color="goldenrod", lw=1.5, ms=4.5, alpha=0.75, label="Sec: MP1 base", zorder=3)
    ax_b.plot(ghz, pri_mp1_coh, marker="o", ls=":", color="steelblue", lw=1.5, ms=4.5, alpha=0.75, label="Pri: MP1 base", zorder=3)
    
    # Mutual ECE12-13 Coherence displayed cleanly in upper-left headroom
    mutual_box = (
        r"$\mathbf{Mutual\ ECE12}$–$\mathbf{13:}$" + "\n"
        + r"Sec: $\bar{\gamma}^2=" + f"{w_ech['bands']['Secondary']['ece12_13_mutual_coh_mean']:.2f}$ (pk {w_ech['bands']['Secondary']['ece12_13_mutual_coh_max']:.2f})\n"
        + r"Pri: $\bar{\gamma}^2=" + f"{w_ech['bands']['Primary']['ece12_13_mutual_coh_mean']:.2f}$ (pk {w_ech['bands']['Primary']['ece12_13_mutual_coh_max']:.2f})"
    )
    ax_b.text(0.012, 0.982, mutual_box, transform=ax_b.transAxes, va="top", ha="left",
              fontsize=7.8, color="navy",
              bbox=dict(boxstyle="round,pad=0.16", facecolor="#f0f4f8", edgecolor="steelblue", alpha=0.94),
              zorder=10)
    
    # Multi-coil gain callout placed in upper-right headroom
    top_p_p13 = w_ech["bands"]["Primary"]["best_probe_names"][idx_13]
    top_p_s12 = w_ech["bands"]["Secondary"]["best_probe_names"][idx_12]
    gain_box = (
        r"$\mathbf{Top\ Coils\ vs.\ MP1:}$" + "\n"
        + f"ECE12 ({top_p_s12}): {sec_best_coh[idx_12]:.2f} vs {sec_mp1_coh[idx_12]:.2f}\n"
        + f"ECE13 ({top_p_p13}): {pri_best_coh[idx_13]:.2f} vs {pri_mp1_coh[idx_13]:.2f}"
    )
    ax_b.text(0.988, 0.982, gain_box, transform=ax_b.transAxes, va="top", ha="right",
              fontsize=7.8, color="#8a3b00",
              bbox=dict(boxstyle="round,pad=0.16", facecolor="#fff9f0", edgecolor="darkorange", alpha=0.94),
              zorder=10)
    
    ax_b.set_xlabel(r"ECE Frequency $f_{\mathrm{ECE}}$ [GHz]", fontsize=13.0)
    ax_b.set_ylabel(r"Cross-Coherence $\gamma^2$ [-]", fontsize=13.0)
    ax_b.set_title(r"(b) Coherence Gain: 18 Coils vs. MP1 Baseline", fontweight="bold", fontsize=12.6, pad=8)
    ax_b.set_xlim(56.8, 73.8)
    ax_b.set_ylim(0.0, 1.56)
    ax_b.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_b.grid(True, ls=":", alpha=0.5)
    
    secax_b = ax_b.secondary_xaxis('top', functions=(freq_to_rho, rho_to_freq))
    secax_b.set_xlabel(r"Normalized Minor Radius $\rho$ [-] (TRAVIS Fit)", fontsize=11.5, fontweight="bold", labelpad=6)
    secax_b.set_ticks([0.9, 0.7, 0.5, 0.3, 0.1, -0.1])
    secax_b.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    secax_b.tick_params(labelsize=11.0)
    
    ax_b.legend(loc="upper left", bbox_to_anchor=(0.015, 0.72), ncol=2, columnspacing=0.65,
                fontsize=8.8, handlelength=1.4, borderpad=0.22, labelspacing=0.20,
                framealpha=0.95, edgecolor="#cccccc")
    
    # --- PANEL (c): Heatmap Matrix: Secondary Band (40-80 kHz) ---
    ax_c = axs[1, 0]
    mat_sec = w_ech["bands"]["Secondary"]["matrix_coh_max"]
    im_c = ax_c.imshow(mat_sec, cmap="YlOrRd", vmin=0.0, vmax=0.45, aspect="auto", origin="lower")
    ax_c.set_xticks(range(len(mag_probes)))
    ax_c.set_xticklabels(mag_probes, rotation=45, ha="right", fontsize=10.5)
    ax_c.set_yticks(range(16))
    ax_c.set_yticklabels([f"{ch.replace('FAST','')} ({ghz[k]:.1f}G, $\\rho$={freq_to_rho(ghz[k]):.2f})" for k, ch in enumerate(w_ech["bands"]["Secondary"]["ece_ch_names"])], fontsize=13.1)
    ax_c.set_title(r"(c) Secondary (40–80 kHz): $\gamma^2_{\max}$ (16 ECEs $\times$ 18 Coils)", fontweight="bold", fontsize=12.6)
    ax_c.set_xlabel(r"Magnetic Coil Index $j$ (MP1–4, PMP1–14) [-]", fontsize=13.0)
    ax_c.set_ylabel(r"ECE Channel Index $i$ [-]", fontsize=13.0)
    ax_c.axhspan(idx_12 - 0.5, idx_13 + 0.5, color="cyan", fill=False, lw=1.8, ls="--")
    mp2_idx = mag_probes.index("MP2")
    ax_c.axvspan(mp2_idx - 0.5, mp2_idx + 0.5, color="blue", fill=False, lw=1.3, ls=":")
    cbar_c = fig.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.03)
    cbar_c.set_label(r"Max Cross-Coherence $\gamma^2_{\max}$ [-]", fontsize=12.5)
    cbar_c.ax.tick_params(labelsize=11.5)
    
    # --- PANEL (d): Heatmap Matrix: Primary Band (80-100 kHz) ---
    ax_d = axs[1, 1]
    mat_pri = w_ech["bands"]["Primary"]["matrix_coh_max"]
    im_d = ax_d.imshow(mat_pri, cmap="YlOrRd", vmin=0.0, vmax=0.45, aspect="auto", origin="lower")
    ax_d.set_xticks(range(len(mag_probes)))
    ax_d.set_xticklabels(mag_probes, rotation=45, ha="right", fontsize=10.5)
    ax_d.set_yticks(range(16))
    ax_d.set_yticklabels([f"{ch.replace('FAST','')} ({ghz[k]:.1f}G, $\\rho$={freq_to_rho(ghz[k]):.2f})" for k, ch in enumerate(w_ech["bands"]["Primary"]["ece_ch_names"])], fontsize=13.1)
    ax_d.set_title(r"(d) Primary (80–100 kHz): $\gamma^2_{\max}$ (16 ECEs $\times$ 18 Coils)", fontweight="bold", fontsize=12.6)
    ax_d.set_xlabel(r"Magnetic Coil Index $j$ (MP1–4, PMP1–14) [-]", fontsize=13.0)
    ax_d.set_ylabel(r"ECE Channel Index $i$ [-]", fontsize=13.0)
    ax_d.axhspan(idx_12 - 0.5, idx_13 + 0.5, color="cyan", fill=False, lw=1.8, ls="--")
    ax_d.axvspan(mp2_idx - 0.5, mp2_idx + 0.5, color="blue", fill=False, lw=1.3, ls=":")
    cbar_d = fig.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.03)
    cbar_d.set_label(r"Max Cross-Coherence $\gamma^2_{\max}$ [-]", fontsize=12.5)
    cbar_d.ax.tick_params(labelsize=11.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96], w_pad=2.6, h_pad=1.8)
    
    out_fig_path = out_dir / "mhd_ece_all_probes_radial_localization_88653.png"
    safe_savefig(fig, out_fig_path, dpi=300)
    print(f"Saved synthesis figure to '{out_fig_path}'")
    
    # Also save standalone panels for Beamer presentations
    extract_panel_axes(fig, ax_a, panels_dir / "panel_a_radial_fluctuation_profile.png", dpi=200)
    extract_panel_axes(fig, ax_b, panels_dir / "panel_b_coherence_gain_all_probes.png", dpi=200)
    extract_panel_axes(fig, ax_c, panels_dir / "panel_c_secondary_matrix_heatmap.png", dpi=200)
    extract_panel_axes(fig, ax_d, panels_dir / "panel_d_primary_matrix_heatmap.png", dpi=200)
    plt.close(fig)
    print("Saved individual panels to 'panels/'")

if __name__ == "__main__":
    run_analysis()
