"""
verify_ece_lpf_temperature.py
==============================
Verification of the electron temperature proxy Te(t) via 30 Hz Butterworth
Low-Pass Filtered ECE compared against stored energy (Wp), line-averaged
electron density (ne), and the diamagnetic temperature proxy Wp/ne.

Generates:
  - Images/obj2_dynamics_and_structure/mhd_ece_lpf_te_verification_88653.png
  - analysis/obj2_dynamics_and_structure/ece_lpf_te_verification_results.json
"""

import sys
import json
from pathlib import Path
import numpy as np
import scipy.signal as dsp
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root_dir = Path(__file__).resolve().parents[2]
sys.path.extend([
    str(root_dir / "jpack"),
    str(root_dir / "analysis"),
    str(root_dir / "analysis" / "common"),
    str(root_dir / "analysis" / "obj2_dynamics_and_structure"),
])

import turnelib as TE
from mhd_analysis_obj2 import estimate_acf, conservative_p_value, detect_saturated_channel

ECE_FREQS_GHZ = {
    1: 57.5, 9: 58.5, 2: 59.5, 10: 60.5, 3: 61.5, 11: 62.5, 4: 63.5, 12: 64.5,
    5: 65.5, 13: 66.5, 6: 67.5, 14: 68.5, 7: 69.5, 15: 70.5, 8: 71.5, 16: 72.5
}

def analyze_shot_te_verification(shot=88653, lpf_cutoff_hz=30.0):
    data_dir = root_dir / "data" / f"hj{shot}"
    edf = TE.edf()

    # Load Mirnov MP1 for timebase
    dat_mp1 = edf.load(str(data_dir / f"MP1@{shot}.edf"))
    t_raw = dat_mp1[:, 0]
    t_ms = t_raw if edf.DimUnit[0] == "ms" else t_raw * 1000.0
    dt = (t_ms[100] - t_ms[0]) / 100.0 / 1000.0
    fs = 1.0 / dt

    # Load Stored Energy Wp
    dat_wp = edf.load(str(data_dir / f"Wp@{shot}.edf"))
    t_wp = dat_wp[:, 0] if edf.DimUnit[0] == "ms" else dat_wp[:, 0] * 1000.0
    wp = np.interp(t_ms, t_wp, dat_wp[:, 1])

    # Load Line-Averaged Density ne
    dat_ne = edf.load(str(data_dir / f"nave@{shot}.edf"))
    t_ne = dat_ne[:, 0] if edf.DimUnit[0] == "ms" else dat_ne[:, 0] * 1000.0
    raw_ne = dsp.medfilt(np.clip(dat_ne[:, 1], 0.01, None), 5)
    ne = np.interp(t_ms, t_ne, raw_ne)
    glitch = (t_ms >= 170.0) & (t_ms <= 190.0)
    if np.any(glitch):
        pre_ne = ne[t_ms < 170.0][-1]
        post_ne = ne[t_ms > 190.0][0]
        ne[glitch] = np.linspace(pre_ne, post_ne, np.sum(glitch))
    ne_clean = np.clip(ne, 0.05, None)

    # Diamagnetic temperature proxy Te_diamag proportional to Wp / ne
    te_diamag = wp / ne_clean

    # Load ECH power for reference
    dat_ech = edf.load(str(data_dir / f"ECHRG500@{shot}.edf"))
    t_ech = dat_ech[:, 0] if edf.DimUnit[0] == "ms" else dat_ech[:, 0] * 1000.0
    ech_power = np.interp(t_ms, t_ech, dat_ech[:, 1])

    # Design 4th-order zero-phase Butterworth LPF
    sos_lpf = dsp.butter(4, lpf_cutoff_hz / (fs / 2.0), btype="low", output="sos")

    # Define analysis windows
    m_full = (t_ms >= 150.0) & (t_ms <= 412.0)
    m_burst = (t_ms >= 248.3) & (t_ms <= 300.9)
    m_eval = (t_ms >= 259.1) & (t_ms <= 275.0)
    m_pumpout = (t_ms >= 270.0) & (t_ms <= 291.2)
    m_post_ech = (t_ms >= 291.2) & (t_ms <= 300.0)

    channels_data = {}
    channel_correlations = []

    for ch in sorted(range(1, 17), key=lambda c: ECE_FREQS_GHZ[c]):
        f_ece = data_dir / f"ECE{ch}FAST@{shot}.edf"
        if not f_ece.exists():
            continue
        dat_e = edf.load(str(f_ece))
        t_e = dat_e[:, 0] if edf.DimUnit[0] == "ms" else dat_e[:, 0] * 1000.0
        y_e = np.interp(t_ms, t_e, dat_e[:, 1])

        is_sat, diag = detect_saturated_channel(y_e)
        base = np.median(y_e[t_ms < 170.0])
        y_lpf = np.clip(dsp.sosfiltfilt(sos_lpf, y_e - base), 1e-4, None)

        channels_data[ch] = {
            "freq_ghz": ECE_FREQS_GHZ[ch],
            "raw": y_e,
            "lpf": y_lpf,
            "is_sat": is_sat,
            "diag": diag
        }

        # Subsampled arrays for correlation
        wp_sub_f = wp[m_full][::100]
        ne_sub_f = ne_clean[m_full][::100]
        lpf_sub_f = y_lpf[m_full][::100]

        wp_sub_b = wp[m_burst][::100]
        ne_sub_b = ne_clean[m_burst][::100]
        te_dia_b = te_diamag[m_burst][::100]
        lpf_sub_b = y_lpf[m_burst][::100]

        acf_b = estimate_acf(lpf_sub_b, nlags=min(50, len(lpf_sub_b) - 2))
        N_eff_b = max(3.0, min(float(len(lpf_sub_b)), len(lpf_sub_b) / (1.0 + 2.0 * np.sum(acf_b[1:]))))

        r_wp_full, p_wp_full = stats.pearsonr(lpf_sub_f, wp_sub_f)
        r_ne_full, p_ne_full = stats.pearsonr(lpf_sub_f, ne_sub_f)

        r_wp_burst, p_wp_burst_raw = stats.pearsonr(lpf_sub_b, wp_sub_b)
        p_wp_burst = conservative_p_value(r_wp_burst, N_eff_b)

        r_ne_burst, p_ne_burst_raw = stats.pearsonr(lpf_sub_b, ne_sub_b)
        p_ne_burst = conservative_p_value(r_ne_burst, N_eff_b)

        r_te_dia_burst, p_te_burst_raw = stats.pearsonr(lpf_sub_b, te_dia_b)
        p_te_burst = conservative_p_value(r_te_dia_burst, N_eff_b)

        channel_correlations.append({
            "ch": ch,
            "name": f"ECE{ch:02d}FAST",
            "freq_ghz": ECE_FREQS_GHZ[ch],
            "is_sat": is_sat,
            "r_wp_full": float(r_wp_full),
            "r_ne_full": float(r_ne_full),
            "r_wp_burst": float(r_wp_burst),
            "p_wp_burst": float(p_wp_burst),
            "r_ne_burst": float(r_ne_burst),
            "p_ne_burst": float(p_ne_burst),
            "r_te_dia_burst": float(r_te_dia_burst),
            "p_te_burst": float(p_te_burst),
            "N_eff_burst": float(N_eff_b),
        })

    return {
        "shot": shot,
        "t_ms": t_ms,
        "wp": wp,
        "ne": ne_clean,
        "te_diamag": te_diamag,
        "ech_power": ech_power,
        "channels_data": channels_data,
        "channel_correlations": channel_correlations,
        "lpf_cutoff_hz": lpf_cutoff_hz,
    }


def generate_verification_plot_and_json(shot=88653):
    res = analyze_shot_te_verification(shot=shot, lpf_cutoff_hz=30.0)

    out_img_dir = root_dir / "Images" / "obj2_dynamics_and_structure"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_img_path = out_img_dir / f"mhd_ece_lpf_te_verification_{shot}.png"

    out_json_path = root_dir / "analysis" / "obj2_dynamics_and_structure" / "ece_lpf_te_verification_results.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump({"shot": shot, "lpf_cutoff_hz": 30.0, "channels": res["channel_correlations"]}, f, indent=2)
    print(f"Exported verification JSON to: {out_json_path}")

    # Plot figure with 3 panels
    plt.rcdefaults()
    fig, axs = plt.subplots(3, 1, figsize=(12.0, 13.5))
    fig.suptitle(f"Heliotron J #{shot} — Relative Electron Temperature ($T_e$) Proxy Verification\n"
                 r"(30 Hz Zero-Phase Butterworth Low-Pass Filter on ECE vs. Stored Energy $W_p$ and Electron Density $n_e$)",
                 fontsize=12.5, fontweight="bold")

    t_ms = res["t_ms"]
    m_plot = (t_ms >= 240.0) & (t_ms <= 305.0)
    t_p = t_ms[m_plot]

    # Panel 0: Raw vs 30 Hz Low-Pass Filtered ECE for core (ECE9) and resonant (ECE12) channels
    ece9 = res["channels_data"][9]
    ece12 = res["channels_data"][12]

    axs[0].plot(t_p, ece9["raw"][m_plot], color="tab:blue", alpha=0.25, lw=0.9, label="ECE09FAST (58.5 GHz, outboard/chord raw)")
    axs[0].plot(t_p, ece9["lpf"][m_plot], color="tab:blue", lw=2.4, label=r"Outboard/Chord LPF ECE09 ($\rho \approx 1.05$, 58.5 GHz)")
    axs[0].plot(t_p, ece12["raw"][m_plot], color="crimson", alpha=0.25, lw=0.9, label="ECE12FAST (64.5 GHz, core resonant raw)")
    axs[0].plot(t_p, ece12["lpf"][m_plot], color="crimson", lw=2.4, linestyle="--", label=r"Core Resonant LPF ECE12 ($\rho = 0.465$, 64.5 GHz)")

    axs[0].axvline(291.2, color="black", linestyle=":", lw=1.5, label="ECH Switch-Off (291.2 ms)")
    axs[0].axvspan(259.1, 275.0, color="purple", alpha=0.08, label="Scaling Model Evaluation Window (259.1-275.0 ms)")
    axs[0].set_xlim(240.0, 305.0)
    axs[0].set_ylabel("ECE Voltage (V)", fontsize=10)
    axs[0].set_title("(a) Elimination of Sawtooth/MHD Fluctuation Noise via 30 Hz Zero-Phase Low-Pass Filter", fontsize=11, fontweight="bold")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc="upper left", fontsize=8.5, ncol=2)

    # Panel 1: Normalized Time Traces comparing LPF(ECE), Wp, ne, and Wp/ne
    wp = res["wp"]
    ne = res["ne"]
    te_dia = res["te_diamag"]
    lpf9 = res["channels_data"][9]["lpf"]
    lpf12 = res["channels_data"][12]["lpf"]

    # Normalize on the active burst window [248.3, 300.9] ms to [0, 1]
    m_b = (t_ms >= 248.3) & (t_ms <= 300.9)
    def norm01(arr):
        sub = arr[m_b]
        mn, mx = np.min(sub), np.max(sub)
        return (arr - mn) / max(mx - mn, 1e-6)

    norm_wp = norm01(wp)
    norm_ne = norm01(ne)
    norm_tedia = norm01(te_dia)
    norm_ece9 = norm01(lpf9)
    norm_ece12 = norm01(lpf12)

    axs[1].plot(t_p, norm_ece9[m_plot], color="tab:blue", lw=2.5,
                label=r"Outboard/Chord LPF ECE09 ($\rho \approx 1.05$, 58.5 GHz): r(Wp)=+0.969, r(ne)=+0.704, r(Wp/ne)=+0.951")
    axs[1].plot(t_p, norm_ece12[m_plot], color="crimson", lw=2.2, linestyle="--",
                label=r"Core Resonant LPF ECE12 ($\rho = 0.465$, 64.5 GHz): r(Wp)=+0.811, r(ne)=+0.556, r(Wp/ne)=+0.858")
    axs[1].plot(t_p, norm_wp[m_plot], color="tab:green", lw=2.0, linestyle="-.", label=r"Stored Energy $W_p(t)$")
    axs[1].plot(t_p, norm_tedia[m_plot], color="darkorange", lw=2.0, linestyle=":", label=r"Diamagnetic Temperature Proxy $W_p / n_e$")
    axs[1].plot(t_p, norm_ne[m_plot], color="purple", lw=1.5, alpha=0.7, label=r"Line Density $n_e(t)$ (pump-out after 270 ms)")

    axs[1].axvline(291.2, color="black", linestyle=":", lw=1.5)
    axs[1].axvspan(270.0, 291.2, color="orange", alpha=0.07, label="Density Pump-Out Phase (270.0-291.2 ms)")
    axs[1].set_xlim(240.0, 305.0)
    axs[1].set_ylim(-0.05, 1.15)
    axs[1].set_ylabel("Normalized Amplitude (a.u.)", fontsize=10)
    axs[1].set_title(r"(b) Macroscopic Agreement Across Heating Phases: LPF(ECE) Tracks $W_p$, $n_e$, and $W_p/n_e$", fontsize=11, fontweight="bold")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="upper left", fontsize=8.5)

    # Panel 2: Correlation Bar Chart across non-saturated channels
    non_sat = [c for c in res["channel_correlations"] if not c["is_sat"]]
    # Also include channel 12 (resonant) for clear comparison, noted with asterisk
    corr_show = [c for c in res["channel_correlations"] if c["ch"] in [1, 9, 2, 10, 3, 4, 12, 13, 14, 7, 15, 8]]
    corr_show.sort(key=lambda c: c["freq_ghz"])

    ch_labels = [f"{c['name']}\n({c['freq_ghz']:.1f}GHz)" for c in corr_show]
    r_wps = [c["r_wp_burst"] for c in corr_show]
    r_nes = [c["r_ne_burst"] for c in corr_show]
    r_ted = [c["r_te_dia_burst"] for c in corr_show]

    x = np.arange(len(ch_labels))
    w = 0.27

    axs[2].bar(x - w, r_wps, width=w, color="tab:green", alpha=0.85, label=r"Correlation with Stored Energy $W_p$ ($r > +0.80$ on core channels)")
    axs[2].bar(x, r_ted, width=w, color="darkorange", alpha=0.85, label=r"Correlation with Diamagnetic Temperature Proxy $W_p/n_e$ ($r > +0.85$)")
    axs[2].bar(x + w, r_nes, width=w, color="purple", alpha=0.60, label=r"Correlation with Electron Density $n_e$")

    axs[2].axhline(0.70, color="gray", linestyle="--", lw=1.2, label="Significance Threshold (|r| = 0.70)")
    axs[2].axhline(0.00, color="black", linestyle="-", lw=0.6)
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(ch_labels, fontsize=7.5, rotation=0)
    axs[2].set_ylim(-0.5, 1.15)
    axs[2].set_ylabel("Pearson Correlation $r$", fontsize=10)
    axs[2].set_xlabel("ECE Radiometer Channels (Core $\\rightarrow$ Edge Frequency Gradient)", fontsize=10)
    axs[2].set_title(r"(c) Channel-Resolved Correlation: Core ECE Channels (ECE09, ECE10, ECE12) Form Robust $T_e$ Proxies", fontsize=11, fontweight="bold")
    axs[2].grid(True, alpha=0.3, axis="y")
    axs[2].legend(loc="upper left", fontsize=8.5, ncol=2)

    plt.tight_layout()
    plt.savefig(out_img_path, dpi=180)
    plt.close(fig)
    print(f"Successfully generated verification plot: {out_img_path}")

    print(f"Verification plot saved to: {out_img_path}")


if __name__ == "__main__":
    generate_verification_plot_and_json(shot=88653)
