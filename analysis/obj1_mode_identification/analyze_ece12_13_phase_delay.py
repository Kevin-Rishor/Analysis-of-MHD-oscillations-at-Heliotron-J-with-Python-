"""
analyze_ece12_13_phase_delay.py
================================
Calculates the precise radial phase delay, time lag, and propagation direction
between ECE13FAST (rho = 0.294, r ~ 4.99 cm, inner core) and ECE12FAST (rho = 0.465, r ~ 7.91 cm, outer resonant zone)
for Heliotron J discharges #88653 and #88654 across three heating windows:
  1. ECH + NBI (259.1 - 291.3 ms)
  2. Sawtooth window (277.0 - 294.0 ms)
  3. NBI only (291.3 - 300.0 ms)

Evaluates:
  (a) Fast MHD carrier wave phase delay (Secondary: 40-80 kHz; Primary: 80-100 kHz)
  (b) Hilbert envelope group / wavepacket propagation delay
  (c) Mesoscopic sawtooth envelope modulation delay
  (d) Macroscopic baseband thermal heat pulse delay (0.35-2.2 kHz)

Direction Convention:
  - Delta_phi_{13->12} = phi_13 - phi_12 = angle<H(y_13) * conj(H(y_12))>
  - Delta_tau_{13->12} = t_12 - t_13 (peak lag of cross-correlation correlate(y_12, y_13))
  - If Delta_phi > 0 (Delta_tau > 0): OUTWARD flow (13 -> 12, rho: 0.294 -> 0.465)
    * Contributes directly to radial transport and loss of stored energy W_p.
  - If Delta_phi < 0 (Delta_tau < 0): INWARD flow (12 -> 13, rho: 0.465 -> 0.294)

Generates:
  - analysis/obj1_mode_identification/ece12_13_phase_delay_results.json
  - Images/obj1_mode_identification/mhd_ece12_13_phase_delay_flow_88653.png
  - Images/obj1_mode_identification/panels/ (individual panels for presentation slides)
"""

import sys
import json
from pathlib import Path
import numpy as np
import scipy.signal as dsp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.extend([
    str(ROOT_DIR / "jpack"),
    str(ROOT_DIR / "analysis"),
    str(ROOT_DIR / "analysis" / "common"),
    str(ROOT_DIR / "analysis" / "heating_phase_comparison"),
    str(ROOT_DIR / "analysis" / "obj1_mode_identification"),
])

from comparison_common import load_channel, safe_savefig, extract_panel_axes
from utils.heliotron_j_ece import freq_to_rho, rho_to_freq, ECE_CHANNELS

ECE_FREQS_GHZ = {k: v['f_ghz'] for k, v in ECE_CHANNELS.items()}

# Verify convention synthetically
def _verify_synthetic_convention():
    fs_test = 1e6
    t_test = np.arange(20000) / fs_test
    f0 = 50e3
    tau_true_us = +2.0  # x13 leads x12 by +2 us (outward flow 13 -> 12)
    s13 = np.cos(2 * np.pi * f0 * t_test)
    s12 = np.cos(2 * np.pi * f0 * (t_test - tau_true_us * 1e-6))
    h13 = dsp.hilbert(s13)
    h12 = dsp.hilbert(s12)
    dphi = np.degrees(np.angle(np.mean(h13 * np.conj(h12))))
    assert abs(dphi - 36.0) < 0.1, f"Synthetic phase error: {dphi}"
    xc = dsp.correlate(s12, s13, mode='full')
    lags = np.arange(-len(s13) + 1, len(s13)) * (1e6 / fs_test)
    m = np.abs(lags) <= 5.0
    lag_pk = lags[m][np.argmax(xc[m])]
    assert abs(lag_pk - 2.0) < 0.1, f"Synthetic lag error: {lag_pk}"

_verify_synthetic_convention()


def parabolic_peak(xc_sub, lags_sub):
    ipk = np.argmax(xc_sub)
    tau = lags_sub[ipk]
    val = xc_sub[ipk]
    if 0 < ipk < len(xc_sub) - 1:
        y1, y2, y3 = xc_sub[ipk - 1], xc_sub[ipk], xc_sub[ipk + 1]
        denom = 2.0 * y2 - y1 - y3
        if abs(denom) > 1e-12:
            delta = 0.5 * (y3 - y1) / denom
            tau += delta * (lags_sub[1] - lags_sub[0])
            val = y2 + 0.25 * (y3 - y1) * delta
    return float(tau), float(val)


def analyze_shot(shot=88653):
    t, e12, dt, fs = load_channel(shot, "ECE12FAST")
    _, e13, _, _ = load_channel(shot, "ECE13FAST")
    
    # Also load all 16 ECE channels for radial profile
    ece_all = {}
    for ch_idx in range(1, 17):
        ch_name = f"ECE{ch_idx}FAST"
        _, y, _, _ = load_channel(shot, ch_name)
        if y is not None:
            ece_all[ch_idx] = y

    windows = {
        "ECH_NBI": {"t0": 259.1, "t1": 291.3, "label": "ECH + NBI (259.1 - 291.3 ms)"},
        "Sawtooth": {"t0": 277.0, "t1": 294.0, "label": "Sawtooth Window (277.0 - 294.0 ms)"},
        "NBI_only": {"t0": 291.3, "t1": 300.0, "label": "NBI only (291.3 - 300.0 ms)"},
    }

    results = {"shot": shot, "windows": {}}

    for wkey, win_info in windows.items():
        t0, t1 = win_info["t0"], win_info["t1"]
        idx = np.where((t >= t0) & (t <= t1))[0]
        t_sub = t[idx]
        e12_w = e12[idx].astype(np.float64)
        e13_w = e13[idx].astype(np.float64)

        # 1. CSD & Coherence across full spectrum (0 - 150 kHz)
        nperseg = 2048 if len(idx) >= 4096 else 1024
        f_csd, P13_12 = dsp.csd(e13_w, e12_w, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
        _, coh_13_12 = dsp.coherence(e13_w, e12_w, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
        # Note: In scipy.signal.csd(x, y), Pxy = E[X* * Y], so angle(P12_13) = phi_13 - phi_12
        f_csd_rev, P12_13 = dsp.csd(e12_w, e13_w, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
        csd_dphi_deg = np.degrees(np.angle(P12_13))

        # 2. Band-specific metrics
        bands = [
            ("Secondary", 40000.0, 80000.0, 6.0),
            ("Primary", 80000.0, 100000.0, 5.0),
            ("Sawtooth_Baseband", 350.0, 2200.0, 500.0),
        ]

        band_results = {}
        for bname, flo, fhi, max_lag_us in bands:
            sos = dsp.bessel(4, [flo / (fs / 2.0), fhi / (fs / 2.0)], btype="bandpass", output="sos")
            y13_f = dsp.sosfiltfilt(sos, e13_w)
            y12_f = dsp.sosfiltfilt(sos, e12_w)
            h13 = dsp.hilbert(y13_f)
            h12 = dsp.hilbert(y12_f)

            # Carrier phase
            carrier_dphi = float(np.degrees(np.angle(np.mean(h13 * np.conj(h12)))))
            
            # Carrier xcorr
            y13_n = (y13_f - np.mean(y13_f)) / (np.std(y13_f) + 1e-12)
            y12_n = (y12_f - np.mean(y12_f)) / (np.std(y12_f) + 1e-12)
            xc = dsp.correlate(y12_n, y13_n, mode="full") / len(y13_n)
            lags_us = np.arange(-len(y13_n) + 1, len(y13_n)) * (1e6 / fs)
            m_l = np.abs(lags_us) <= max_lag_us
            tau_car_us, r_car_pk = parabolic_peak(xc[m_l], lags_us[m_l])

            # Hilbert envelope
            env13 = np.abs(h13)
            env12 = np.abs(h12)
            env13_n = (env13 - np.mean(env13)) / (np.std(env13) + 1e-12)
            env12_n = (env12 - np.mean(env12)) / (np.std(env12) + 1e-12)
            xce = dsp.correlate(env12_n, env13_n, mode="full") / len(env13_n)
            m_le = np.abs(lags_us) <= (15.0 if flo >= 40000 else 500.0)
            tau_env_us, r_env_pk = parabolic_peak(xce[m_le], lags_us[m_le])

            # Sawtooth-modulated envelope (filtering carrier envelope in sawtooth 0.35-2.2 kHz)
            if flo >= 40000:
                sos_saw = dsp.bessel(4, [350.0 / (fs / 2.0), 2200.0 / (fs / 2.0)], btype="bandpass", output="sos")
                env13_saw = dsp.sosfiltfilt(sos_saw, env13)
                env12_saw = dsp.sosfiltfilt(sos_saw, env12)
                h_es13 = dsp.hilbert(env13_saw)
                h_es12 = dsp.hilbert(env12_saw)
                env_saw_dphi = float(np.degrees(np.angle(np.mean(h_es13 * np.conj(h_es12)))))
                
                es13_n = (env13_saw - np.mean(env13_saw)) / (np.std(env13_saw) + 1e-12)
                es12_n = (env12_saw - np.mean(env12_saw)) / (np.std(env12_saw) + 1e-12)
                xces = dsp.correlate(es12_n, es13_n, mode="full") / len(es13_n)
                m_les = np.abs(lags_us) <= 600.0
                tau_saw_us, r_saw_pk = parabolic_peak(xces[m_les], lags_us[m_les])
            else:
                env_saw_dphi = 0.0
                tau_saw_us = 0.0
                r_saw_pk = 0.0

            # CSD peak in band
            m_fb = (f_csd >= flo) & (f_csd <= fhi)
            if np.any(m_fb):
                id_pk = np.argmax(coh_13_12[m_fb])
                f_pk = float(f_csd[m_fb][id_pk])
                coh_pk = float(coh_13_12[m_fb][id_pk])
                csd_ph_pk = float(csd_dphi_deg[m_fb][id_pk])
            else:
                f_pk = 0.0
                coh_pk = 0.0
                csd_ph_pk = 0.0

            # Determine direction
            flow_car = "OUTWARD (13 -> 12)" if carrier_dphi > 0 else "INWARD (12 -> 13)"
            flow_env = "OUTWARD (13 -> 12)" if tau_env_us > 0 else "INWARD (12 -> 13)"
            flow_saw = "OUTWARD (13 -> 12)" if tau_saw_us > 0 else "INWARD (12 -> 13)"

            # Estimate apparent radial velocities (Delta r = 2.92 cm = 0.0292 m)
            delta_r_m = 0.0292
            v_ph_r = (delta_r_m / (tau_car_us * 1e-6)) if abs(tau_car_us) > 1e-3 else 0.0
            v_group_r = (delta_r_m / (tau_env_us * 1e-6)) if abs(tau_env_us) > 1e-3 else 0.0

            band_results[bname] = {
                "carrier_dphi_deg": carrier_dphi,
                "carrier_tau_us": tau_car_us,
                "carrier_r_pk": r_car_pk,
                "carrier_flow": flow_car,
                "v_phase_radial_km_s": v_ph_r / 1000.0,
                "env_tau_us": tau_env_us,
                "env_r_pk": r_env_pk,
                "env_flow": flow_env,
                "v_group_radial_km_s": v_group_r / 1000.0,
                "sawtooth_env_dphi_deg": env_saw_dphi,
                "sawtooth_env_tau_us": tau_saw_us,
                "sawtooth_env_r_pk": r_saw_pk,
                "sawtooth_env_flow": flow_saw,
                "csd_f_peak_khz": f_pk / 1000.0,
                "csd_coh_peak": coh_pk,
                "csd_phase_peak_deg": csd_ph_pk,
            }

        # 3. Radial profile across all channels relative to ECE13 (Primary & Secondary)
        rad_profile = {}
        for bname, flo, fhi, _ in bands[:2]:
            sos_b = dsp.bessel(4, [flo / (fs / 2.0), fhi / (fs / 2.0)], btype="bandpass", output="sos")
            h13_ref = dsp.hilbert(dsp.sosfiltfilt(sos_b, e13_w))
            ch_list = []
            for ch_i in sorted(ece_all.keys(), key=lambda c: ECE_FREQS_GHZ[c]):
                yi = ece_all[ch_i][idx].astype(np.float64)
                hi = dsp.hilbert(dsp.sosfiltfilt(sos_b, yi))
                dphi_i = float(np.degrees(np.angle(np.mean(hi * np.conj(h13_ref)))))
                # Cross-coherence at mode peak
                fc, coh = dsp.coherence(e13_w, yi, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
                mb = (fc >= flo) & (fc <= fhi)
                c_max = float(np.max(coh[mb])) if np.any(mb) else 0.0
                ch_list.append({
                    "channel": f"ECE{ch_i:02d}FAST",
                    "f_ghz": ECE_FREQS_GHZ[ch_i],
                    "rho": float(freq_to_rho(ECE_FREQS_GHZ[ch_i])),
                    "dphi_rel_ece13_deg": dphi_i,
                    "max_coherence": c_max,
                })
            rad_profile[bname] = ch_list

        results["windows"][wkey] = {
            "label": win_info["label"],
            "bands": band_results,
            "radial_profile": rad_profile,
            "csd_spectrum": {
                "f_khz": (f_csd / 1000.0).tolist(),
                "coherence": coh_13_12.tolist(),
                "phase_deg": csd_dphi_deg.tolist(),
            }
        }

    return results


def generate_figures(results_88653, results_88654):
    out_dir = ROOT_DIR / "Images" / "obj1_mode_identification"
    out_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = out_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10.0,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 8.5,
    })

    # Single-Panel 16:9 Presentation-Optimized Figure
    fig, ax = plt.subplots(1, 1, figsize=(11.5, 5.8), facecolor="white")

    # Reconstruct cross-correlation functions on ECH+NBI (259.1 - 291.3 ms)
    t, e12, dt, fs = load_channel(88653, "ECE12FAST")
    _, e13, _, _ = load_channel(88653, "ECE13FAST")
    idx = np.where((t >= 259.1) & (t <= 291.3))[0]

    sos_pri = dsp.bessel(4, [80e3 / (fs / 2.0), 100e3 / (fs / 2.0)], btype="bandpass", output="sos")
    sos_sec = dsp.bessel(4, [40e3 / (fs / 2.0), 80e3 / (fs / 2.0)], btype="bandpass", output="sos")

    y13_pri = dsp.sosfiltfilt(sos_pri, e13[idx])
    y12_pri = dsp.sosfiltfilt(sos_pri, e12[idx])
    y13_sec = dsp.sosfiltfilt(sos_sec, e13[idx])
    y12_sec = dsp.sosfiltfilt(sos_sec, e12[idx])

    h13_pri = dsp.hilbert(y13_pri)
    h12_pri = dsp.hilbert(y12_pri)

    # Normalized signals
    y13_pn = (y13_pri - np.mean(y13_pri)) / np.std(y13_pri)
    y12_pn = (y12_pri - np.mean(y12_pri)) / np.std(y12_pri)
    y13_sn = (y13_sec - np.mean(y13_sec)) / np.std(y13_sec)
    y12_sn = (y12_sec - np.mean(y12_sec)) / np.std(y12_sec)

    # Carrier cross-correlations
    xc_pri = dsp.correlate(y12_pn, y13_pn, mode="full") / len(y13_pn)
    xc_sec = dsp.correlate(y12_sn, y13_sn, mode="full") / len(y13_sn)

    # Envelope cross-correlation
    e13_env = np.abs(h13_pri)
    e12_env = np.abs(h12_pri)
    e13_en = (e13_env - np.mean(e13_env)) / np.std(e13_env)
    e12_en = (e12_env - np.mean(e12_env)) / np.std(e12_env)
    xc_env = dsp.correlate(e12_en, e13_en, mode="full") / len(e13_en)

    lags_us = np.arange(-len(y13_pn) + 1, len(y13_pn)) * (1e6 / fs)
    m = np.abs(lags_us) <= 6.0
    l_sub = lags_us[m]

    # Shaded direction zones
    ax.axvspan(-6.0, 0.0, color="#eaf2f8", alpha=0.6, label=r"INWARD Phase Propagation ($12 \to 13$, $\Delta\tau < 0$)")
    ax.axvspan(0.0, 6.0, color="#fdf2e9", alpha=0.6, label=r"OUTWARD Energy Flow ($13 \to 12$, $\Delta\tau > 0$)")

    # Plot the 3 key curves
    ax.plot(l_sub, xc_sec[m], color="darkorange", lw=2.2, label=r"Secondary Carrier (40–80 kHz): $\Delta\tau = -0.51\,\mu$s ($\Delta\phi = -10.8^\circ$, $r = +0.84$) $\to$ Inward")
    ax.plot(l_sub, xc_pri[m], color="crimson", lw=2.4, label=r"Primary Carrier (80–100 kHz): $\Delta\tau = -0.39\,\mu$s ($\Delta\phi = -12.7^\circ$, $r = +0.86$) $\to$ Inward")
    ax.plot(l_sub, xc_env[m], color="navy", lw=2.2, ls="--", label=r"Primary Envelope $|\mathcal{H}|$ (Group Velocity): $\Delta\tau_{\mathrm{env}} = +0.82\,\mu$s ($r = +0.77$) $\to$ Outward")

    # Vertical reference lines
    ax.axvline(0.0, color="black", ls="-", lw=1.1)
    ax.axhline(0.0, color="gray", ls=":", lw=0.8)

    # Pin markers and text annotations for the 3 peaks (placed without crossing arrows)
    ax.axvline(-0.514, color="darkorange", ls=":", lw=1.2)
    ax.scatter([-0.514], [0.841], color="darkorange", s=60, zorder=5)
    ax.annotate(r"Sec: -0.51 $\mu$s" + "\n" + r"($v_{\mathrm{ph}} \approx -57\,$km/s)",
                xy=(-0.514, 0.841), xytext=(+0.15, 1.24),
                fontsize=7.8, fontweight="bold", color="darkorange", ha="left",
                arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.2))

    ax.axvline(-0.389, color="crimson", ls=":", lw=1.2)
    ax.scatter([-0.389], [0.863], color="crimson", s=60, zorder=5)
    ax.annotate(r"Pri: -0.39 $\mu$s" + "\n" + r"($v_{\mathrm{ph}} \approx -75\,$km/s)",
                xy=(-0.389, 0.863), xytext=(+1.35, 1.07),
                fontsize=7.8, fontweight="bold", color="crimson", ha="left",
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2))

    ax.axvline(+0.82, color="navy", ls=":", lw=1.2)
    ax.scatter([+0.82], [0.768], color="navy", s=60, zorder=5)
    ax.annotate(r"Env: +0.82 $\mu$s" + "\n" + r"($v_g \approx +36\,$km/s)",
                xy=(+0.82, 0.768), xytext=(+2.05, 0.89),
                fontsize=7.8, fontweight="bold", color="navy", ha="left",
                arrowprops=dict(arrowstyle="->", color="navy", lw=1.2))

    # Clean Callout Box summarizing the physical dichotomy (placed on top left, with generous margins)
    summary_box = (
        "Physical Flow Direction Summary (Shots #88653 & #88654):\n"
        r"• INWARD Carrier Phase (12 $\to$ 13, $\rho$: 0.47 $\to$ 0.29):" + "\n"
        r"   - Primary Mode (89 kHz): $\Delta\phi = -12.7^\circ,\ \Delta\tau = -0.39\,\mu\mathrm{s}\ (v_{\mathrm{ph},r} \approx -75\ \mathrm{km/s})$" + "\n"
        r"   - Secondary Mode (49 kHz): $\Delta\phi = -10.8^\circ,\ \Delta\tau = -0.51\,\mu\mathrm{s}\ (v_{\mathrm{ph},r} \approx -57\ \mathrm{km/s})$" + "\n"
        r"• OUTWARD Energy & Heat Flow (13 $\to$ 12, $\rho$: 0.29 $\to$ 0.47):" + "\n"
        r"   - Wavepacket Envelope: $\Delta\tau_{\mathrm{env}} = +0.82\,\mu\mathrm{s}\ (v_{g,r} \approx +36\ \mathrm{km/s})$" + "\n"
        r"   - Sawtooth Modulation: $\Delta\tau_{\mathrm{saw}} = +8.0\,\mu\mathrm{s}\ (\mathrm{Pri}),\ +40.0\,\mu\mathrm{s}\ (\mathrm{Sec})$" + "\n"
        r"   - Post-ECH Thermal Pulse: $\Delta\tau = +46.6\,\mu\mathrm{s}\ (+15.9^\circ,\ r = +0.78)$"
    )
    ax.text(0.02, 0.97, summary_box, transform=ax.transAxes, fontsize=7.2, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#333333", alpha=0.95, lw=1.1))

    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-0.95, 1.38)
    ax.set_xlabel(r"Radial Time Lag $\Delta\tau = t_{12} - t_{13}$ [$\mu$s]: $[\Delta\tau < 0 \to \mathrm{Inward}\ (12 \to 13);\ \Delta\tau > 0 \to \mathrm{Outward}\ (13 \to 12)]$",
                  fontsize=10.0, fontweight="bold")
    ax.set_ylabel(r"Cross-Correlation Coefficient $r(\Delta\tau)$ [-]", fontsize=10.0, fontweight="bold")
    ax.set_title(r"Radial Phase Delay & Inward/Outward Flow: $\mathrm{ECE13FAST}\ (\rho=0.29)\ \longleftrightarrow\ \mathrm{ECE12FAST}\ (\rho=0.47)$",
                 fontweight="bold", fontsize=11.5, pad=8)
    ax.grid(True, ls=":", alpha=0.45)
    ax.legend(loc="lower left", fontsize=7.8, framealpha=0.94, edgecolor="#cccccc")

    plt.tight_layout()
    out_fig_path = out_dir / "mhd_ece12_13_phase_delay_flow_88653.png"
    safe_savefig(fig, out_fig_path, dpi=250)
    print(f"Saved single-panel figure to '{out_fig_path}'")
    plt.close(fig)


def main():
    print("=" * 80)
    print("CALCULATING ECE12 - ECE13 RADIAL PHASE DELAY & FLOW DIRECTION (SHOTS #88653 & #88654)")
    print("=" * 80)

    res_88653 = analyze_shot(88653)
    res_88654 = analyze_shot(88654)

    # Export complete results JSON
    json_path = ROOT_DIR / "analysis" / "obj1_mode_identification" / "ece12_13_phase_delay_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"shot_88653": res_88653, "shot_88654": res_88654}, f, indent=2)
    print(f"Exported results JSON to '{json_path}'")

    # Generate figures
    generate_figures(res_88653, res_88654)

    print("\n" + "=" * 80)
    print("ECE12 - ECE13 ANALYSIS COMPLETE.")
    print("=" * 80)


if __name__ == "__main__":
    main()
