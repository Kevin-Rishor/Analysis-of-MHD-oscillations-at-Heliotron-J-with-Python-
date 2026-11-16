#!/usr/bin/env python3
"""
compute_bicoherence_maps_and_controls.py

Exact copy of algorithm from analysis/obj3_nonlinear_coupling/make_figures_v2.py.
Computes corrected squared bicoherence map maxima on mask_B and 33 control-window maxima
for shots 88653 and 88654, and outputs structured results to JSON.
"""

import sys
import json
from pathlib import Path
import numpy as np
import scipy.signal as dsp

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
for p in [ROOT_DIR / "jpack", ROOT_DIR / "analysis", ROOT_DIR / "analysis" / "common", ROOT_DIR / "analysis" / "heating_phase_comparison"]:
    if str(p) not in sys.path:
        sys.path.append(str(p))

from comparison_common import load_channel
from abicoh2_corrected import abicoh2_corrected

def run_computation():
    t_ms_53, ys_mp1_53, dt_53, fs_53 = load_channel(88653, "MP1")
    t_ms_54, ys_mp1_54, dt_54, fs_54 = load_channel(88654, "MP1")

    nfft = 512
    nhalf = 256
    hop = 256
    noverlap = 256
    n_ens = 30
    sig_len = 256 * 29 + 512  # 7936 samples
    df_khz = (fs_53 / nfft) / 1e3

    f1_vec = np.arange(nhalf) * df_khz
    f2_vec = (np.arange(nfft) - nhalf) * df_khz
    FF1, FF2 = np.meshgrid(f1_vec, f2_vec, indexing="ij")

    # Construct mask_B exactly as in make_figures_v2.py
    fmin_khz = 5.0
    f1max = 100.0
    f2max = 100.0
    fs_khz = fs_53 / 1e3
    physical_mask_diff = (FF1 > 0) & (FF2 < 0) & (FF1 > -FF2)
    search_diff = (
        physical_mask_diff
        & (FF1 < f1max)
        & (-FF2 < f2max)
        & (FF1 >= fmin_khz)
        & (-FF2 >= fmin_khz)
        & ((FF1 + FF2) >= fmin_khz)
    )
    physical_mask_sum = (
        (FF1 > 0)
        & (FF2 > 0)
        & (FF1 >= FF2)
        & ((FF1 + FF2) <= (fs_khz / 2.0))
    )
    search_sum = (
        physical_mask_sum
        & (FF1 < f1max)
        & (FF2 < f2max)
        & (FF1 >= FF2)
        & (FF1 >= fmin_khz)
        & (FF2 >= fmin_khz)
    )
    mask_B = search_sum | search_diff
    coords_B = np.argwhere(mask_B)
    assert len(coords_B) == 2306, f"Expected 2306 coords in mask_B, got {len(coords_B)}"

    def get_nominal_max_b2(ys, fs, t_ms, t_target):
        idx0 = int(np.where(t_ms >= t_target)[0][0])
        y_w = ys[idx0 : idx0 + sig_len]
        _, _, spec = dsp.spectrogram(
            y_w, fs=fs, window="hann", nperseg=nfft, noverlap=noverlap, nfft=nfft,
            detrend="linear", return_onesided=False, scaling="density", axis=0, mode="complex"
        )
        spec = np.transpose(spec, [1, 0])[:n_ens, :]
        _, b2_raw = abicoh2_corrected(spec)
        b2_grid = np.transpose(np.fft.fftshift(b2_raw, axes=0), [1, 0])
        b2_obs = b2_grid[coords_B[:, 0], coords_B[:, 1]]
        top_i = int(np.argmax(b2_obs))
        max_val = float(b2_obs[top_i])
        ka_t = int(coords_B[top_i, 0])
        kb_t = int(coords_B[top_i, 1] - nhalf)
        return max_val, float(ka_t * df_khz), float(kb_t * df_khz)

    def scan_shot_tiling1(ys, fs, t_ms):
        n_windows = 33
        win_starts = []
        max_b2_list = []
        for w in range(n_windows):
            i_start = w * sig_len
            w_start = float(t_ms[i_start])
            y_w = ys[i_start : i_start + sig_len]
            _, _, spec = dsp.spectrogram(
                y_w, fs=fs, window="hann", nperseg=nfft, noverlap=noverlap, nfft=nfft,
                detrend="linear", return_onesided=False, scaling="density", axis=0, mode="complex"
            )
            spec = np.transpose(spec, [1, 0])[:n_ens, :]
            _, b2_raw = abicoh2_corrected(spec)
            b2_grid = np.transpose(np.fft.fftshift(b2_raw, axes=0), [1, 0])
            b2_obs = b2_grid[coords_B[:, 0], coords_B[:, 1]]
            max_val = float(np.max(b2_obs))
            win_starts.append(w_start)
            max_b2_list.append(max_val)
        return win_starts, max_b2_list

    # Compute for both shots
    max_53_ech, f1_53_ech, f2_53_ech = get_nominal_max_b2(ys_mp1_53, fs_53, t_ms_53, 259.1)
    max_53_nbi, f1_53_nbi, f2_53_nbi = get_nominal_max_b2(ys_mp1_53, fs_53, t_ms_53, 291.3)
    ws_53, ctrl_53 = scan_shot_tiling1(ys_mp1_53, fs_53, t_ms_53)

    max_54_ech, f1_54_ech, f2_54_ech = get_nominal_max_b2(ys_mp1_54, fs_54, t_ms_54, 259.1)
    max_54_nbi, f1_54_nbi, f2_54_nbi = get_nominal_max_b2(ys_mp1_54, fs_54, t_ms_54, 291.3)
    ws_54, ctrl_54 = scan_shot_tiling1(ys_mp1_54, fs_54, t_ms_54)

    # 23-window non-overlapping subset outside heated interval [222.4, 300.4] ms
    ws_53_arr = np.array(ws_53)
    c53_arr = np.array(ctrl_53)
    m53_non_ov = (ws_53_arr + 7.936 <= 222.4) | (ws_53_arr >= 300.4)
    ctrl_53_non_ov = c53_arr[m53_non_ov].tolist()
    ws_53_non_ov = ws_53_arr[m53_non_ov].tolist()

    ws_54_arr = np.array(ws_54)
    c54_arr = np.array(ctrl_54)
    m54_non_ov = (ws_54_arr + 7.936 <= 222.4) | (ws_54_arr >= 300.4)
    ctrl_54_non_ov = c54_arr[m54_non_ov].tolist()
    ws_54_non_ov = ws_54_arr[m54_non_ov].tolist()

    results = {
        "88653": {
            "ech_nbi": {
                "time_window_ms": [259.1, 267.0],
                "max_b2": max_53_ech,
                "f1_khz": f1_53_ech,
                "f2_khz": f2_53_ech
            },
            "nbi_only": {
                "time_window_ms": [291.3, 299.2],
                "max_b2": max_53_nbi,
                "f1_khz": f1_53_nbi,
                "f2_khz": f2_53_nbi
            },
            "control_windows": {
                "n_windows": len(ctrl_53),
                "window_starts_ms": ws_53,
                "max_b2_values": ctrl_53,
                "median": float(np.median(ctrl_53)),
                "min": float(np.min(ctrl_53)),
                "max": float(np.max(ctrl_53)),
                "range": [float(np.min(ctrl_53)), float(np.max(ctrl_53))]
            },
            "control_windows_non_overlapping_23": {
                "n_windows": len(ctrl_53_non_ov),
                "window_starts_ms": ws_53_non_ov,
                "max_b2_values": ctrl_53_non_ov,
                "median": float(np.median(ctrl_53_non_ov)),
                "min": float(np.min(ctrl_53_non_ov)),
                "max": float(np.max(ctrl_53_non_ov)),
                "range": [float(np.min(ctrl_53_non_ov)), float(np.max(ctrl_53_non_ov))]
            }
        },
        "88654": {
            "ech_nbi": {
                "time_window_ms": [259.1, 267.0],
                "max_b2": max_54_ech,
                "f1_khz": f1_54_ech,
                "f2_khz": f2_54_ech
            },
            "nbi_only": {
                "time_window_ms": [291.3, 299.2],
                "max_b2": max_54_nbi,
                "f1_khz": f1_54_nbi,
                "f2_khz": f2_54_nbi
            },
            "control_windows": {
                "n_windows": len(ctrl_54),
                "window_starts_ms": ws_54,
                "max_b2_values": ctrl_54,
                "median": float(np.median(ctrl_54)),
                "min": float(np.min(ctrl_54)),
                "max": float(np.max(ctrl_54)),
                "range": [float(np.min(ctrl_54)), float(np.max(ctrl_54))]
            },
            "control_windows_non_overlapping_23": {
                "n_windows": len(ctrl_54_non_ov),
                "window_starts_ms": ws_54_non_ov,
                "max_b2_values": ctrl_54_non_ov,
                "median": float(np.median(ctrl_54_non_ov)),
                "min": float(np.min(ctrl_54_non_ov)),
                "max": float(np.max(ctrl_54_non_ov)),
                "range": [float(np.min(ctrl_54_non_ov)), float(np.max(ctrl_54_non_ov))]
            }
        }
    }

    out_file = ROOT_DIR / "analysis" / "obj_ech" / "bicoherence_maps_and_controls_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {out_file}")
    print(f"Shot 88653 ECH+NBI max b2: {max_53_ech:.4f} at ({f1_53_ech:.1f}, {f2_53_ech:.1f}) kHz")
    print(f"Shot 88653 NBI-only max b2: {max_53_nbi:.4f} at ({f1_53_nbi:.1f}, {f2_53_nbi:.1f}) kHz")
    print(f"Shot 88653 control median: {np.median(ctrl_53):.4f}, range: [{np.min(ctrl_53):.4f}, {np.max(ctrl_53):.4f}]")
    print(f"Shot 88654 ECH+NBI max b2: {max_54_ech:.4f} at ({f1_54_ech:.1f}, {f2_54_ech:.1f}) kHz")
    print(f"Shot 88654 NBI-only max b2: {max_54_nbi:.4f} at ({f1_54_nbi:.1f}, {f2_54_nbi:.1f}) kHz")
    print(f"Shot 88654 control median: {np.median(ctrl_54):.4f}, range: [{np.min(ctrl_54):.4f}, {np.max(ctrl_54):.4f}]")

if __name__ == "__main__":
    run_computation()
