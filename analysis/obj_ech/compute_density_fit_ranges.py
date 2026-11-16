#!/usr/bin/env python3
"""
analysis/obj_ech/compute_density_fit_ranges.py

Density-following model re-extraction without literals:
1. Reads nave and t_nave from cache/shot_88653_full_record_cache.npz.
2. Smooths nave with the same Savitzky-Golay filter as analyze_task4g_reconcile_and_repair.py:
   - Window: 1.5 ms (pts_15ms = int(round(1.5 / dt_n)); if even, +1)
   - Order: 2
3. Builds 0.5 ms bins from the same STFT (psd_mp1_std, t_stft_std, f_stft_std).
   NOTE: Bins at 0.5 ms steps from 1.024 ms windows overlap.
4. Fits log(f) vs log(n) on [262, 290], [270, 290], and [275, 291.2] ms with np.polyfit.
5. Evaluates f_pred at 290.2 and 293.7 ms.
6. Divides by delta_fc_sec observed_at_t0 read directly from frozen JSON
   (analysis/obj_ech/frozen_json/task4g_results.json).
7. Writes analysis/obj_ech/density_fit_ranges.json.
"""

import json
from pathlib import Path
import numpy as np
import scipy.signal as dsp

ROOT_DIR = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT_DIR / "analysis" / "obj_ech" / "density_fit_ranges.json"
FROZEN_T4G = ROOT_DIR / "analysis" / "obj_ech" / "task4g_results.json"

# Read delta_fc_sec observed_at_t0 from frozen JSON (no literal numbers)
with open(FROZEN_T4G, "r", encoding="utf-8") as f:
    t4g_json = json.load(f)
delta_fc_obs = float(
    t4g_json["88653"]["section_B_transition_matched_null"]["delta_fc_sec"]["observed_at_t0"]
)

# Load cache
cache_path = ROOT_DIR / "cache" / "shot_88653_full_record_cache.npz"
c53 = np.load(cache_path)

t_stft_53 = c53["t_stft_std"]
f_stft_53 = c53["f_stft_std"] / 1000.0  # kHz
psd_53 = c53["psd_mp1_std"]
t_nave_53 = c53["t_nave_ms"]
nave_53 = c53["nave"]

# Savitzky-Golay filter on nave (identical lines from analyze_task4g_reconcile_and_repair.py)
dt_n = float(np.median(np.diff(t_nave_53)))
pts_15ms = int(round(1.5 / dt_n))
if pts_15ms % 2 == 0:
    pts_15ms += 1
nave_smooth_53 = dsp.savgol_filter(nave_53, pts_15ms, 2)

# Binning on 0.5 ms step across [262.0, 300.0] ms
t_bins_c = np.arange(262.0, 300.1, 0.5)
fdom_sec_c, n_at_t_c = [], []

m_sec_c = (f_stft_53 >= 30.0) & (f_stft_53 <= 65.0)
f_sec_axis_c = f_stft_53[m_sec_c]

for tb in t_bins_c:
    m_win = (t_stft_53 >= tb - 0.25) & (t_stft_53 <= tb + 0.25)
    psd_s = np.mean(psd_53[m_sec_c, :][:, m_win], axis=1)
    fdom_sec_c.append(float(f_sec_axis_c[np.argmax(psd_s)]))
    n_at_t_c.append(float(np.interp(tb, t_nave_53, nave_smooth_53)))

fdom_sec_c = np.array(fdom_sec_c)
n_at_t_c = np.array(n_at_t_c)

# Evaluation densities at pre-transition and post-turnaround points
t_eval_pre = 290.2
t_eval_post = 293.7
n_pre = float(np.interp(t_eval_pre, t_nave_53, nave_smooth_53))
n_post = float(np.interp(t_eval_post, t_nave_53, nave_smooth_53))

intervals = [
    ("262-290", 262.0, 290.0),
    ("270-290", 270.0, 290.0),
    ("275-291.2", 275.0, 291.2),
]

output_dict = {
    "shot": "88653",
    "overlap_note": "Bins at 0.5 ms from 1.024 ms windows overlap.",
    "evaluation_times_ms": {
        "t_pre": t_eval_pre,
        "t_post": t_eval_post,
        "nave_pre": n_pre,
        "nave_post": n_post,
    },
    "observed_delta_fc_sec_khz": delta_fc_obs,
    "fit_intervals": {},
}

print("=" * 80)
print("DENSITY-FOLLOWING POWER-LAW RE-EXTRACTION (NO LITERAL PARAMETERS)")
print("Note: Bins at 0.5 ms from 1.024 ms windows overlap.")
print(f"Observed delta_fc_sec at t0 (from frozen task4g): {delta_fc_obs:.6f} kHz")
print(f"nave({t_eval_pre:.1f} ms) = {n_pre:.6f} | nave({t_eval_post:.1f} ms) = {n_post:.6f}")
print("=" * 80)

fractions = []

for label, t_start, t_end in intervals:
    m_fit = (t_bins_c >= t_start) & (t_bins_c <= t_end)
    ln_n = np.log(n_at_t_c[m_fit])
    ln_f = np.log(fdom_sec_c[m_fit])

    # Fit ln(f) = slope * ln(n) + intercept => f = f0 * n^(-gamma)
    p = np.polyfit(ln_n, ln_f, 1)
    gamma = -float(p[0])
    f0 = float(np.exp(p[1]))

    f_pred_pre = float(f0 * (n_pre) ** (-gamma))
    f_pred_post = float(f0 * (n_post) ** (-gamma))
    delta_f_pred = float(f_pred_post - f_pred_pre)
    frac = float(delta_f_pred / delta_fc_obs)
    fractions.append(frac)

    output_dict["fit_intervals"][label] = {
        "t_start_ms": t_start,
        "t_end_ms": t_end,
        "n_points": int(np.sum(m_fit)),
        "gamma": gamma,
        "f0_khz": f0,
        "polyfit_slope": float(p[0]),
        "polyfit_intercept": float(p[1]),
        "f_pred_290p2_khz": f_pred_pre,
        "f_pred_293p7_khz": f_pred_post,
        "delta_f_pred_khz": delta_f_pred,
        "fraction_of_observed": frac,
        "fraction_pct_str": f"{frac * 100.0:.1f}%",
    }

    print(f"Interval [{t_start:5.1f}, {t_end:5.1f}] ms (label: '{label}'):")
    print(f"  gamma = {gamma:.6f}")
    print(f"  f0    = {f0:.6f} kHz")
    print(f"  n({t_eval_pre:.1f} ms) = {n_pre:.6f}")
    print(f"  n({t_eval_post:.1f} ms) = {n_post:.6f}")
    print(f"  f_pred({t_eval_pre:.1f} ms) = {f_pred_pre:.6f} kHz")
    print(f"  f_pred({t_eval_post:.1f} ms) = {f_pred_post:.6f} kHz")
    print(f"  delta_f_pred = {delta_f_pred:.6f} kHz")
    print(f"  Fraction of observed = {frac:.6f} ({frac * 100.0:.2f}%)\n")

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output_dict, f, indent=2)

print(f"Written density_fit_ranges.json successfully to {OUT_JSON}")
