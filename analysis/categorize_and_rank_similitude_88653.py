"""
categorize_and_rank_similitude_88653.py

Exhaustive cross-diagnostic sawtooth analysis and similitude ranking across all 191 physical channels
for Heliotron J discharge #88653 over the full physical sawtooth window:
    t in [277.0, 294.0] ms (17.0 ms duration, 15 continuous sawtooth cycles on primary mode MP1).

Key Methodological Principles:
1. MP1 is the benchmark primary mode reference, attaining exactly 100.0% similitude and 0.00 ms offset.
2. Evaluates dual pathways across t in [277.0, 294.0] ms:
   - Carrier STFT ridge / Hilbert envelope tracking (70-110 kHz) for fs >= 250 kHz
   - Direct zero-phase Bessel bandpass (0.35 - 2.2 kHz) for thermal, radiative, edge, and actuator signals
3. Computes four calibrated sub-metrics:
   - S_wf: Waveform shape cross-correlation over [277.0, 294.0] ms
   - S_asym: Derivative skewness asymmetry relative to MP1
   - S_coh: Cross-spectral coherence at the sawtooth repetition frequency (Welch)
   - S_comb: Harmonic comb SNR across the sawtooth fundamental and harmonics
   Composite General Similitude: S_sim = 0.45*S_wf + 0.25*S_asym + 0.15*S_coh + 0.15*S_comb
4. Ranks all 191 channels and assigns them to 6 physical behavioral categories:
   - Cat. 1: Carrier-Modulated Magnetic (MP, PMP)
   - Cat. 2: Core Thermal & SXR Sawtooth (ECE, sx2d)
   - Cat. 3: Divertor & Edge Boundary Sawtooth (DivProArr, HAFAST, HAARRAY)
   - Cat. 4: Radiative & Bulk Density Response (Wp, nave, AXUV, FIR)
   - Cat. 5: Current & Actuators (GASPUFF, ECH, NBI, FW, Ip)
   - Cat. 6: Non-Responsive / Low-fs / Saturated
5. Generates publication figures including the 6-panel Figure 8 showing full t in [277.0, 294.0] ms waveforms.
"""

import sys
import json
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import scipy.signal as dsp
from scipy.stats import skew
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path("c:/TFG")
DATA_DIR = ROOT_DIR / "data" / "hj88653"
ANALYSIS_DIR = ROOT_DIR / "analysis"
FIG_DIR = ROOT_DIR / "Images" / "sawtooth_analysis_88653"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT_DIR / "jpack"))
import turnelib as TE


def load_edf(filepath):
    """Safely loads an EDF file into numpy arrays."""
    e = TE.edf()
    dat = e.load(str(filepath))
    t_raw = dat[:, 0]
    t_ms = t_raw if e.DimUnit[0] == "ms" else t_raw * 1000.0
    ys = dat[:, 1:] if dat.shape[1] > 2 else dat[:, 1]
    unit = getattr(e, "ValUnit", ["V"])
    return t_ms, ys, unit


def main():
    print("=" * 80)
    print("CROSS-DIAGNOSTIC SAWTOOTH SIMILITUDE RANKING — [277.0, 294.0] ms (SHOT #88653)")
    print("=" * 80)

    # 1. Load existing results table from master ledger
    csv_in_path = ANALYSIS_DIR / "sawtooth_88653_master_ledger.csv"
    with open(csv_in_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        results_table = list(reader)

    print(f"Loaded {len(results_table)} channel records from {csv_in_path.name}")

    # 2. Extract Primary Mode Reference and Reset Timestamps from MP1 across [277.0, 294.0] ms
    print("\n>>> Extracting Canonical Primary Mode Sawtooth Reference from MP1...")
    t_ref, dat_ref, _ = load_edf(DATA_DIR / "MP1@88653.edf")
    mp1_raw = dat_ref[:, 0] if dat_ref.ndim > 1 else dat_ref
    dt_ref = (t_ref[1] - t_ref[0]) / 1000.0
    fs_ref = 1.0 / dt_ref

    nperseg, noverlap = 1024, 960
    f_stft, t_s, Zxx = dsp.stft(mp1_raw, fs=fs_ref, nperseg=nperseg, noverlap=noverlap, window="hann")
    t_stft_ms = t_ref[0] + t_s * 1000.0
    psd = np.abs(Zxx) ** 2
    f_khz = f_stft / 1000.0
    m_pri = (f_khz >= 70.0) & (f_khz <= 110.0)
    f_pri_track = f_khz[m_pri][np.argmax(psd[m_pri, :], axis=0)]

    fs_stft = 1000.0 / (t_stft_ms[1] - t_stft_ms[0])  # 15.625 kHz
    sos_saw = dsp.bessel(4, [350.0, 2200.0], btype="bandpass", fs=fs_stft, output="sos")
    f_pri_saw = dsp.sosfiltfilt(sos_saw, f_pri_track)

    # Analysis window [277.0, 294.0] ms
    m_win = (t_stft_ms >= 277.0) & (t_stft_ms <= 294.0)
    t_win = t_stft_ms[m_win]
    dt_win = t_win[1] - t_win[0]  # 0.064 ms
    N_pts = len(t_win)

    # Normalized MP1 comparative sawtooth reference
    ref_saw = (f_pri_saw[m_win] - np.mean(f_pri_saw[m_win])) / np.std(f_pri_saw[m_win])
    ref_sk_raw = float(skew(np.diff(ref_saw)))

    # Evaluate reference MP1 parameters
    f_c_ref, Cxy_ref = dsp.coherence(ref_saw, ref_saw, fs=fs_stft, nperseg=128, noverlap=96)
    m_fc = (f_c_ref >= 550.0) & (f_c_ref <= 1100.0)
    ref_coh_raw = float(np.max(Cxy_ref[m_fc]))  # 1.000

    f_p_ref, Pxx_ref = dsp.welch(ref_saw, fs=fs_stft, nperseg=256, noverlap=192)
    m_cb_ref = ((f_p_ref >= 650.0) & (f_p_ref <= 1050.0)) | ((f_p_ref >= 1400.0) & (f_p_ref <= 1900.0))
    m_bg_ref = (f_p_ref >= 350.0) & (f_p_ref <= 2200.0) & (~m_cb_ref)
    ref_snr_raw = float(np.max(Pxx_ref[m_cb_ref]) / (np.median(Pxx_ref[m_bg_ref]) + 1e-12))

    print(f"MP1 Reference on [277.0, 294.0] ms: Skewness={ref_sk_raw:+.3f}, Coherence={ref_coh_raw:.3f}, Comb SNR={ref_snr_raw:.2f}")

    # 3. Load all raw channel signals
    bundled_files = {
        "axuv@88653.edf": ("AXUV", 16, [f"AXUV{i}" for i in range(1, 17)]),
        "haarr@88653.edf": ("HAARRAY", 32, [f"HAARRAY{i:02d}" for i in range(1, 33)]),
        "pmp@88653.edf": ("PMP", 14, [f"PMP{i}" for i in range(1, 15)]),
        "tmp@88653.edf": ("MP", 4, [f"MP{i}" for i in range(1, 5)]),
        "DivProArr@88653.edf": ("DivProArr", 38, [f"DivPro_{i:02d}" for i in range(1, 39)]),
        "sx2d@88653.edf": ("sx2d", 48, [f"sx2d_chord_{i:02d}" for i in range(1, 49)])
    }

    print("\n>>> Loading raw EDF channels...")
    ch_raw_map = {}
    for b_name, (fam, n_ch, ch_names) in bundled_files.items():
        t_b, dat_b, _ = load_edf(DATA_DIR / b_name)
        fs_b = 1.0 / ((t_b[1] - t_b[0]) / 1000.0)
        for c_idx in range(n_ch):
            ch_raw_map[ch_names[c_idx]] = (t_b, dat_b[:, c_idx], fs_b, fam)

    for r in results_table:
        ch_id = r["channel_id"]
        if ch_id not in ch_raw_map:
            t_s, dat_s, _ = load_edf(DATA_DIR / r["source_file"])
            y_s = dat_s[:, 0] if dat_s.ndim > 1 else dat_s
            fs_s = 1.0 / ((t_s[1] - t_s[0]) / 1000.0)
            ch_raw_map[ch_id] = (t_s, y_s, fs_s, r["family"])

    # 4. Helper function to evaluate signal against ref_saw
    def eval_sig(sig_w, is_ece=False):
        sig_norm = (sig_w - np.mean(sig_w)) / (np.std(sig_w) + 1e-12)
        corr_full = dsp.correlate(sig_norm, ref_saw, mode="full") / N_pts
        lags = np.arange(-N_pts + 1, N_pts) * dt_win
        m_lag = np.abs(lags) <= 1.50
        corr_sub = corr_full[m_lag]
        lags_sub = lags[m_lag]

        # For ECE channels, maintain consistent positive-phase lobe tracking
        # to avoid 180 deg lobe jumps between adjacent radiometer channels
        pos_idx = np.argmax(corr_sub)
        neg_idx = np.argmin(corr_sub)
        r_pos, lag_pos = float(corr_sub[pos_idx]), float(lags_sub[pos_idx])
        r_neg, lag_neg = float(corr_sub[neg_idx]), float(lags_sub[neg_idx])

        if is_ece and r_pos > 0.80 * abs(r_neg):
            r_val, lag_ms = r_pos, lag_pos
        else:
            idx = np.argmax(np.abs(corr_sub))
            r_val = float(corr_sub[idx])
            lag_ms = float(lags_sub[idx])

        sk_val = float(skew(np.diff(sig_norm)))
        f_c, Cxy = dsp.coherence(sig_norm, ref_saw, fs=fs_stft, nperseg=128, noverlap=96)
        m_f = (f_c >= 550.0) & (f_c <= 1100.0)
        coh_val = float(np.max(Cxy[m_f])) if np.any(m_f) else 0.0
        f_p, Pxx = dsp.welch(sig_norm, fs=fs_stft, nperseg=256, noverlap=192)
        m_c = ((f_p >= 650.0) & (f_p <= 1050.0)) | ((f_p >= 1400.0) & (f_p <= 1900.0))
        m_bg = (f_p >= 350.0) & (f_p <= 2200.0) & (~m_c)
        snr_val = float(np.max(Pxx[m_c]) / (np.median(Pxx[m_bg]) + 1e-12)) if np.any(m_c) else 1.0
        return r_val, lag_ms, sk_val, coh_val, snr_val, sig_norm

    # 5. Categorization mapping
    cat_names = {
        "I": "Cat. 1: Carrier-Modulated Magnetic",
        "II": "Cat. 2: Core Thermal & SXR Sawtooth",
        "III": "Cat. 3: Divertor & Edge Boundary Sawtooth",
        "IV": "Cat. 4: Radiative & Bulk Density Response",
        "V": "Cat. 5: Plasma Current & Heating Actuators",
        "VI": "Cat. 6: Non-Responsive / Low-fs / Saturated"
    }

    def assign_category(fam, is_sat, fs_k, s_sim):
        if is_sat or fs_k < 2.5 or s_sim < 20.0:
            return "VI"
        if fam in ("MP", "PMP"):
            return "I"
        if fam in ("ECE", "sx2d"):
            return "II"
        if fam in ("DivProArr", "HAFAST", "HAARRAY"):
            return "III"
        if fam in ("AXUV", "nave", "Wp", "FIR"):
            return "IV"
        return "V"

    # 6. Evaluate all 191 channels
    print("\n>>> Evaluating all channels over t in [277.0, 294.0] ms...")
    analyzed_records = []
    plot_waveforms = {}

    for r in results_table:
        ch_id = r["channel_id"]
        fam = r["family"]
        is_sat = (r["is_saturated"] == "True") or (r["is_saturated"] is True)
        fs_k = float(r["fs_khz"])

        t_c, y_c, fs_c, _ = ch_raw_map[ch_id]
        ptp_val = float(np.ptp(y_c))

        if ptp_val < 1e-6 or fs_k < 2.5 or is_sat:
            # Saturated or flat or extremely low fs
            rec = {
                "channel_id": ch_id, "family": fam, "source_file": r["source_file"],
                "fs_khz": fs_k, "is_saturated": is_sat, "mode_type": "None",
                "r_max": 0.0, "tau_star_ms": 0.0, "s_deriv": 0.0, "coherence": 0.0, "comb_snr": 1.0,
                "s_wf": 0.0, "s_asym": 0.0, "s_coh": 0.0, "s_comb": 0.0, "similitude_pct": 0.0,
                "category_id": "VI", "category_name": cat_names["VI"]
            }
            analyzed_records.append(rec)
            flat_sig = np.interp(t_win, t_c, y_c)
            std_f = np.std(flat_sig)
            flat_norm = (flat_sig - np.mean(flat_sig)) / (std_f + 1e-12) if std_f > 1e-12 else np.zeros_like(t_win)
            plot_waveforms[ch_id] = (flat_norm, flat_norm, 0.00)
            continue

        # Benchmark MP1 check
        if ch_id == "MP1":
            rec = {
                "channel_id": "MP1", "family": "MP", "source_file": "tmp@88653.edf",
                "fs_khz": fs_k, "is_saturated": False, "mode_type": "RidgeTrack",
                "r_max": 1.000, "tau_star_ms": 0.00, "s_deriv": ref_sk_raw, "coherence": 1.000, "comb_snr": ref_snr_raw,
                "s_wf": 100.0, "s_asym": 100.0, "s_coh": 100.0, "s_comb": 100.0, "similitude_pct": 100.0,
                "category_id": "I", "category_name": cat_names["I"]
            }
            analyzed_records.append(rec)
            plot_waveforms["MP1"] = (ref_saw, ref_saw, 0.00)
            continue

        # 1. Baseband pathway
        is_ece = (fam == "ECE")
        f_hi = min(2200.0, 0.40 * fs_c)
        sos_c = dsp.bessel(4, [350.0, f_hi], btype="bandpass", fs=fs_c, output="sos")
        y_bp = dsp.sosfiltfilt(sos_c, y_c)
        y_w = np.interp(t_win, t_c, y_bp)
        r_bp, lag_bp, sk_bp, coh_bp, snr_bp, sig_bp_norm = eval_sig(y_w, is_ece=is_ece)
        best = ("Baseband", r_bp, lag_bp, sk_bp, coh_bp, snr_bp, sig_bp_norm)

        # 2. STFT ridge & carrier envelope pathways (if fs_c >= 250 kHz)
        if fs_c >= 250e3:
            nper_c = int(round(1024 * fs_c / fs_ref))
            nov_c = int(round(960 * fs_c / fs_ref))
            fc, tsc, Zc = dsp.stft(y_c, fs=fs_c, nperseg=nper_c, noverlap=nov_c, window="hann")
            tc_ms = t_c[0] + tsc * 1000.0
            fkc = fc / 1000.0
            mpc = (fkc >= 70.0) & (fkc <= 110.0)
            tr_c = fkc[mpc][np.argmax(np.abs(Zc[mpc, :]) ** 2, axis=0)]
            tr_c_saw = dsp.sosfiltfilt(sos_saw, tr_c)
            tr_w = np.interp(t_win, tc_ms, tr_c_saw)
            r_tr, lag_tr, sk_tr, coh_tr, snr_tr, sig_tr_norm = eval_sig(tr_w, is_ece=is_ece)
            if abs(r_tr) > abs(best[1]):
                best = ("RidgeTrack", r_tr, lag_tr, sk_tr, coh_tr, snr_tr, sig_tr_norm)

            sos_car = dsp.butter(4, [75000.0, 102000.0], btype="bandpass", fs=fs_c, output="sos")
            m_pad = (t_c >= 270.0) & (t_c <= 300.0)
            y_car = dsp.sosfiltfilt(sos_car, y_c[m_pad])
            env_car = np.abs(dsp.hilbert(y_car))
            t_sub = t_stft_ms[(t_stft_ms >= 272.0) & (t_stft_ms <= 298.0)]
            env_w_raw = np.interp(t_sub, t_c[m_pad], env_car)
            env_saw = dsp.sosfiltfilt(sos_saw, env_w_raw)
            env_w = np.interp(t_win, t_sub, env_saw)
            r_env, lag_env, sk_env, coh_env, snr_env, sig_env_norm = eval_sig(env_w, is_ece=is_ece)
            if abs(r_env) > abs(best[1]):
                best = ("CarrierEnv", r_env, lag_env, sk_env, coh_env, snr_env, sig_env_norm)

        mode_type, r_v, lag_v, sk_v, coh_v, snr_v, sig_norm = best

        # Calibrated sub-metrics (MP1 = 100.0%)
        s_wf = 100.0 * abs(r_v)
        s_asym = 100.0 * min(1.0, abs(sk_v) / max(abs(ref_sk_raw), 1e-6))
        s_coh = 100.0 * min(1.0, coh_v)
        s_comb = 100.0 * min(1.0, np.log10(max(1.0, snr_v)) / np.log10(max(1.1, ref_snr_raw)))
        s_sim = 0.45 * s_wf + 0.25 * s_asym + 0.15 * s_coh + 0.15 * s_comb

        cid = assign_category(fam, is_sat, fs_k, s_sim)
        rec = {
            "channel_id": ch_id, "family": fam, "source_file": r["source_file"],
            "fs_khz": fs_k, "is_saturated": is_sat, "mode_type": mode_type,
            "r_max": r_v, "tau_star_ms": lag_v, "s_deriv": sk_v, "coherence": coh_v, "comb_snr": snr_v,
            "s_wf": s_wf, "s_asym": s_asym, "s_coh": s_coh, "s_comb": s_comb, "similitude_pct": s_sim,
            "category_id": cid, "category_name": cat_names[cid]
        }
        analyzed_records.append(rec)

        # Shifted waveform
        sig_shifted = np.interp(t_win, t_win - lag_v, sig_norm, left=sig_norm[0], right=sig_norm[-1])
        plot_waveforms[ch_id] = (sig_norm, sig_shifted, lag_v)

    # 7. Sort by Similitude Percentage
    analyzed_records.sort(key=lambda x: x["similitude_pct"], reverse=True)
    for rank_idx, rec in enumerate(analyzed_records, start=1):
        rec["rank"] = rank_idx

    # Export master categorized ranking CSV
    out_csv = ANALYSIS_DIR / "sawtooth_88653_categorized_ranking.csv"
    fieldnames = [
        "rank", "channel_id", "family", "category_id", "category_name", "similitude_pct",
        "s_wf", "s_asym", "s_coh", "s_comb", "mode_type", "r_max", "tau_star_ms",
        "s_deriv", "coherence", "comb_snr", "fs_khz", "is_saturated", "source_file"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in analyzed_records:
            row_dict = {k: f"{rec[k]:.3f}" if isinstance(rec[k], float) else rec[k] for k in fieldnames}
            writer.writerow(row_dict)

    print(f"\nSaved categorized ranking to {out_csv.name}")

    # 8. Print Top Rankings & Groups
    print("\n" + "=" * 80)
    print("TOP 5 OVERALL MAGNETIC PROBES (EXCLUDING MP1):")
    mag_top5 = [r for r in analyzed_records if r["channel_id"] != "MP1" and r["family"] in ("MP", "PMP")][:5]
    for r in mag_top5:
        print(f"  #{r['rank']:3d} {r['channel_id']:12s} ({r['family']:8s}) : S_sim={r['similitude_pct']:5.1f}% | S_wf={r['s_wf']:5.1f}%, S_asym={r['s_asym']:5.1f}%, S_coh={r['s_coh']:5.1f}%, S_comb={r['s_comb']:5.1f}% | Offset={r['tau_star_ms']:+5.2f} ms")

    print("\nTOP 5 NON-MAGNETIC DIAGNOSTICS (EXCLUDING MP & PMP):")
    non_mag_top5 = [r for r in analyzed_records if r["family"] not in ("MP", "PMP")][:5]
    for r in non_mag_top5:
        print(f"  #{r['rank']:3d} {r['channel_id']:12s} ({r['family']:8s}) : S_sim={r['similitude_pct']:5.1f}% | S_wf={r['s_wf']:5.1f}%, S_asym={r['s_asym']:5.1f}%, S_coh={r['s_coh']:5.1f}%, S_comb={r['s_comb']:5.1f}% | Offset={r['tau_star_ms']:+5.2f} ms")

    print("\nTOP REPRESENTATIVE FROM EACH OF THE 6 BEHAVIORAL CATEGORIES:")
    cat_top6 = []
    for cid in ["I", "II", "III", "IV", "V", "VI"]:
        cands = [r for r in analyzed_records if r["category_id"] == cid and r["channel_id"] != "MP1"]
        if cands:
            cat_top6.append(cands[0])
            r = cands[0]
            print(f"  Cat {cid}: {r['channel_id']:14s} ({r['family']:10s}) : S_sim={r['similitude_pct']:5.1f}% | Offset={r['tau_star_ms']:+5.2f} ms")

    # 9. Generate 6-Panel Figure 8 across the full physical window t in [277.0, 294.0] ms
    print("\n>>> Generating 6-Panel Figure 8 (Full 277.0-294.0 ms Time Window)...")
    fig8, axes8 = plt.subplots(2, 3, figsize=(15.0, 8.0), sharex=True, sharey=True)
    cat_palette = {
        "I": "#1f77b4",    # Blue
        "II": "#d62728",   # Red
        "III": "#2ca02c",  # Green
        "IV": "#ff7f0e",   # Orange
        "V": "#9467bd",    # Purple
        "VI": "#8c564b"    # Brown
    }

    for idx, (ax, rec) in enumerate(zip(axes8.flat, cat_top6)):
        ch_id = rec["channel_id"]
        cid = rec["category_id"]
        col = cat_palette[cid]
        sig_raw, sig_shifted, lag_ms = plot_waveforms[ch_id]

        # 1. MP1 Comparative Reference Waveform (Dashed Charcoal)
        ax.plot(t_win, ref_saw, color="#333333", ls="--", lw=1.6, alpha=0.75,
                label=r"$\mathrm{MP1}$ Comparative Ref. ($100.0\%$)")

        # 2. Raw physical waveform (Light Dotted)
        if abs(lag_ms) >= 0.05:
            ax.plot(t_win, sig_raw, color=col, ls=":", lw=1.4, alpha=0.55,
                    label=f"Raw ({ch_id})")

        # 3. Lag-aligned waveform (Solid)
        sign_lbl = rf"Aligned ($\tau^* = {lag_ms:+.2f}$ ms)" if abs(lag_ms) >= 0.05 else f"Aligned (In-Phase, $\tau^* = 0$)"
        ax.plot(t_win, sig_shifted, color=col, ls="-", lw=2.2, alpha=0.95, label=sign_lbl)

        # Baseline guides
        ax.axhline(0.0, color="gray", ls=":", lw=0.6, alpha=0.5)

        # Title
        ax.set_title(
            f"Cat. {cid}: {ch_id} ({rec['family']}) — $\\mathcal{{S}}_{{\\mathrm{{sim}}}} = {rec['similitude_pct']:.1f}\\%$",
            fontsize=10.5, fontweight="bold", pad=6
        )

        # Badge
        badge_text = (
            f"General Sim: {rec['similitude_pct']:.1f}%\n"
            f"Shape (S_wf): {rec['s_wf']:.1f}%\n"
            f"Asymmetry: {rec['s_asym']:.1f}%\n"
            f"Coherence: {rec['s_coh']:.1f}%\n"
            f"Comb SNR: {rec['s_comb']:.1f}%\n"
            f"Offset: {lag_ms:+.2f} ms"
        )
        ax.text(
            0.025, 0.96, badge_text, transform=ax.transAxes,
            fontsize=7.6, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=col, alpha=0.92, lw=1.2)
        )

        ax.set_xlim(277.0, 294.0)
        ax.set_ylim(-3.2, 3.6)
        ax.legend(loc="lower right", fontsize=7.2, framealpha=0.88)
        ax.grid(True, ls=":", alpha=0.5)

        if idx in [0, 3]:
            ax.set_ylabel(r"Normalized Amplitude ($\sigma$)", fontweight="bold", fontsize=9.5)
        if idx in [3, 4, 5]:
            ax.set_xlabel(r"Physical Discharge Time $t$ (ms)", fontweight="bold", fontsize=9.5)

    plt.tight_layout()
    fig8_path = FIG_DIR / "fig8_top6_category_channels_sawtooth_88653.png"
    fig8.savefig(fig8_path, dpi=300)
    plt.close(fig8)
    print(f"Saved {fig8_path.name}")

    # 10. Save JSON summary audit
    summary_data = {
        "analysis_window_ms": [277.0, 294.0],
        "n_cycles": 15,
        "reference_channel": "MP1",
        "reference_similitude_pct": 100.0,
        "top5_magnetic": [
            {"rank": r["rank"], "channel": r["channel_id"], "family": r["family"], "similitude_pct": r["similitude_pct"], "offset_ms": r["tau_star_ms"]}
            for r in mag_top5
        ],
        "top5_non_magnetic": [
            {"rank": r["rank"], "channel": r["channel_id"], "family": r["family"], "similitude_pct": r["similitude_pct"], "offset_ms": r["tau_star_ms"]}
            for r in non_mag_top5
        ],
        "top6_categories": [
            {"category_id": r["category_id"], "category_name": r["category_name"], "channel": r["channel_id"], "family": r["family"], "similitude_pct": r["similitude_pct"], "offset_ms": r["tau_star_ms"]}
            for r in cat_top6
        ]
    }
    with open(ANALYSIS_DIR / "sawtooth_88653_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved summary audit to {ANALYSIS_DIR / 'sawtooth_88653_summary.json'}")

    print("\n" + "=" * 80)
    print("CATEGORIZATION & SIMILITUDE ANALYSIS OVER [277.0, 294.0] ms COMPLETE.")
    print("=" * 80)


if __name__ == "__main__":
    main()
