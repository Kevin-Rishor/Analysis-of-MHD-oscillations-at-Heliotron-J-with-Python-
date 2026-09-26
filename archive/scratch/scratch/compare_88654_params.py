"""
Phase 1: Shot 88654 vs 88653 parameter comparison.
Extracts Ip, ne, NBI timing, ECH timing, Wp from both shots' EDF files.
Then re-aligns the analysis window by heating phase and re-runs the reproducibility check.
"""
import sys, os
import numpy as np
from pathlib import Path

sys.path.append(os.path.abspath('c:/TFG/analysis/obj3_nonlinear_coupling'))

from mhd_analysis_obj3 import (load_edf_signal, analyze_channel_bicoherence,
                                 triads_match, detect_hmode_burst_window,
                                 compute_flat_frequency_window, slice_window)

# --- helpers ----------------------------------------------------------------

def load_slow(fpath):
    """Load an EDF, return (t_ms, signal)."""
    t_sec, ys, dt, fs = load_edf_signal(fpath)
    return t_sec * 1000.0, ys

def find_edge(t_ms, y, threshold_frac=0.5, direction='rising', search_start_ms=50.0):
    """Find the first rising or falling edge crossing threshold_frac of (max-baseline)."""
    i0 = np.searchsorted(t_ms, search_start_ms)
    baseline = np.mean(y[:max(1, int(len(y)*0.05))])
    peak = np.max(y[i0:])
    thr = baseline + (peak - baseline) * threshold_frac

    segment = y[i0:]
    t_seg = t_ms[i0:]

    if direction == 'rising':
        crossings = np.where((segment[:-1] < thr) & (segment[1:] >= thr))[0]
    else:
        crossings = np.where((segment[:-1] >= thr) & (segment[1:] < thr))[0]

    if len(crossings) == 0:
        return None
    return float(t_seg[crossings[0]])

def find_last_falling_edge(t_ms, y, threshold_frac=0.5, search_start_ms=50.0):
    """Find the LAST falling edge."""
    i0 = np.searchsorted(t_ms, search_start_ms)
    baseline = np.mean(y[:max(1, int(len(y)*0.05))])
    peak = np.max(y[i0:])
    thr = baseline + (peak - baseline) * threshold_frac

    segment = y[i0:]
    t_seg = t_ms[i0:]

    crossings = np.where((segment[:-1] >= thr) & (segment[1:] < thr))[0]
    if len(crossings) == 0:
        return None
    return float(t_seg[crossings[-1]])

def mean_in_window(t_ms, y, t0, t1):
    """Mean of signal in [t0, t1] ms."""
    mask = (t_ms >= t0) & (t_ms <= t1)
    if not np.any(mask):
        return float('nan')
    return float(np.mean(y[mask]))

# --- main -------------------------------------------------------------------

out_lines = []
def P(s=""):
    out_lines.append(s)
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode('ascii', 'replace').decode('ascii'))

P("=" * 90)
P("Phase 1: Shot 88654 vs 88653 Discharge Parameter Comparison")
P("=" * 90)

for shot in ["88653", "88654"]:
    d = Path(f"c:/TFG/data/hj{shot}")
    P(f"\n{'-'*90}")
    P(f"SHOT #{shot}")
    P(f"{'-'*90}")

    # Wp burst detection
    wp_t, wp_y = load_slow(d / f"Wp@{shot}.edf")
    det = detect_hmode_burst_window(d / f"Wp@{shot}.edf")
    P(f"  Wp burst (auto-detected): {det['start_ms']:.1f} -- {det['end_ms']:.1f} ms   (peak Wp ~ {det.get('peak_wp', np.max(wp_y)):.3f})")

    # Ip
    for ip_name in ["Ip135", "Ip15"]:
        fp = d / f"{ip_name}@{shot}.edf"
        if fp.exists():
            ip_t, ip_y = load_slow(fp)
            ip_mean = mean_in_window(ip_t, ip_y, 200.0, 290.0)
            P(f"  {ip_name} mean (200-290 ms): {ip_mean:.4f}")

    # ne (line-averaged density)
    fp = d / f"nave@{shot}.edf"
    if fp.exists():
        ne_t, ne_y = load_slow(fp)
        ne_mean = mean_in_window(ne_t, ne_y, 260.0, 290.0)
        P(f"  nave mean (260-290 ms): {ne_mean:.4f}")

    # ECH timing
    fp = d / f"ECHRG500@{shot}.edf"
    if fp.exists():
        ech_t, ech_y = load_slow(fp)
        ech_on = find_edge(ech_t, ech_y, 0.5, 'rising', search_start_ms=50.0)
        ech_off = find_last_falling_edge(ech_t, ech_y, 0.5, search_start_ms=50.0)
        ech_mean = mean_in_window(ech_t, ech_y, 200.0, 280.0) if ech_on else float('nan')
        P(f"  ECHRG500 ON:  {ech_on:.1f} ms" if ech_on else "  ECHRG500 ON:  not detected")
        P(f"  ECHRG500 OFF: {ech_off:.1f} ms" if ech_off else "  ECHRG500 OFF: not detected")
        P(f"  ECHRG500 mean power (200-280 ms): {ech_mean:.4f}")

    # NBI timing (all available beamlines)
    for nbi_name in ["NBIS3I", "NBIS4I", "NBIS9I", "NBIS10I"]:
        fp = d / f"{nbi_name}@{shot}.edf"
        if fp.exists():
            nbi_t, nbi_y = load_slow(fp)
            nbi_on = find_edge(nbi_t, nbi_y, 0.5, 'rising', search_start_ms=50.0)
            nbi_off = find_last_falling_edge(nbi_t, nbi_y, 0.5, search_start_ms=50.0)
            nbi_mean = mean_in_window(nbi_t, nbi_y, 200.0, 290.0)
            on_str = f"{nbi_on:.1f} ms" if nbi_on else "not detected"
            off_str = f"{nbi_off:.1f} ms" if nbi_off else "not detected"
            P(f"  {nbi_name}: ON={on_str}, OFF={off_str}, mean(200-290ms)={nbi_mean:.4f}")

# --- Alignment --------------------------------------------------------------

P(f"\n{'='*90}")
P("TIMING ALIGNMENT")
P(f"{'='*90}")

# Load ECH off edge for both shots
ech_off_53 = None
ech_off_54 = None
for shot in ["88653", "88654"]:
    fp = Path(f"c:/TFG/data/hj{shot}/ECHRG500@{shot}.edf")
    if fp.exists():
        ech_t, ech_y = load_slow(fp)
        val = find_last_falling_edge(ech_t, ech_y, 0.5, search_start_ms=50.0)
        if shot == "88653":
            ech_off_53 = val
        else:
            ech_off_54 = val

# Also find NBI off for both
nbi_off_53 = None
nbi_off_54 = None
for shot in ["88653", "88654"]:
    # Use NBIS10I as the reference beamline (highest power, cleanest edge)
    fp = Path(f"c:/TFG/data/hj{shot}/NBIS10I@{shot}.edf")
    if fp.exists():
        nbi_t, nbi_y = load_slow(fp)
        val = find_last_falling_edge(nbi_t, nbi_y, 0.5, search_start_ms=50.0)
        if shot == "88653":
            nbi_off_53 = val
        else:
            nbi_off_54 = val

P(f"  88653 ECH off: {ech_off_53:.1f} ms" if ech_off_53 else "  88653 ECH off: not detected")
P(f"  88654 ECH off: {ech_off_54:.1f} ms" if ech_off_54 else "  88654 ECH off: not detected")
P(f"  88653 NBI off (NBIS10I): {nbi_off_53:.1f} ms" if nbi_off_53 else "  88653 NBI off: not detected")
P(f"  88654 NBI off (NBIS10I): {nbi_off_54:.1f} ms" if nbi_off_54 else "  88654 NBI off: not detected")

# The manual analysis window for 88653 was 259.1-275.0 ms.
# Anchor to ECH off if available, else to NBI off.
anchor_53 = ech_off_53 or nbi_off_53
anchor_54 = ech_off_54 or nbi_off_54
anchor_name = "ECH off" if ech_off_53 else "NBI off"

if anchor_53 and anchor_54:
    win_start_offset = 259.1 - anchor_53
    win_end_offset = 275.0 - anchor_53
    P(f"\n  88653 manual window 259.1--275.0 ms is at {anchor_name} + ({win_start_offset:+.1f}) to {anchor_name} + ({win_end_offset:+.1f}) ms")

    aligned_start = anchor_54 + win_start_offset
    aligned_end = anchor_54 + win_end_offset
    P(f"  -> Translated to 88654 (anchored to {anchor_name}): {aligned_start:.1f} -- {aligned_end:.1f} ms")
else:
    P("  Cannot anchor to any heating edge; using rigid transplant.")
    aligned_start = 259.1
    aligned_end = 275.0

# --- M9 flat-frequency sub-window on the aligned window ----------------------

P(f"\n{'='*90}")
P("M9 FLAT-FREQUENCY SUB-WINDOW FOR SHOT 88654 (within heating-phase-aligned window)")
P(f"{'='*90}")

class Args:
    pass

args = Args()
args.flat_freq_low_khz = 5.0
args.flat_freq_high_khz = 150.0
args.flat_order = 4
args.flat_sg_win = 525
args.nfft = 1024
args.ensemble = 30
args.flat_min_duration_ms = 5.0
args.flat_scan_window_ms = 8.0
args.flat_growth_tolerance = 0.5
args.flat_slope_smooth_ms = 2.0
args.flat_window_start = None
args.flat_window_end = None
args.fmin = 5.0
args.f1max = 150.0
args.f2max = 150.0
args.topn = 10
args.threshold = 0.5
args.peak_window_khz = 5.0
args.repro_tol_khz = 10.0

validation_shot = "88654"
validation_dir = Path("c:/TFG/data/hj88654")
mp1_file = validation_dir / f"MP1@{validation_shot}.edf"

# Run M9 on the heating-phase-aligned window
i0_m9, i1_m9, info_m9 = compute_flat_frequency_window(mp1_file, aligned_start, aligned_end, args)
if info_m9.get('used_fallback'):
    P(f"  M9 fell back on aligned window [{aligned_start:.1f}--{aligned_end:.1f}]: {info_m9.get('reason','?')}")
    m9_start = aligned_start
    m9_end = aligned_end
else:
    m9_start = i0_m9
    m9_end = i1_m9
P(f"  Heating-phase-aligned input window: {aligned_start:.1f} -- {aligned_end:.1f} ms")
P(f"  M9 flat-frequency sub-window:       {m9_start:.1f} -- {m9_end:.1f} ms")

# --- Reproducibility test on the heating-phase-aligned M9 window -------------

P(f"\n{'='*90}")
P(f"REPRODUCIBILITY TEST ON HEATING-PHASE-ALIGNED WINDOW ({m9_start:.1f}--{m9_end:.1f} ms)")
P(f"{'='*90}")

channels = ["MP1", "MP3", "MP4"]
triads_to_test = {
    "PSD-confirmed triad (89.8/41.0 kHz)": (89.8, 41.0, 0.0),
    "non-PSD-confirmed triad (98.6/48.8 kHz), for comparison only": (98.6, 48.8, 0.0)
}

for label, ref_triad in triads_to_test.items():
    P(f"\n  --- {label} ---")
    for ch in channels:
        fpath = validation_dir / f"{ch}@{validation_shot}.edf"
        result_rep = analyze_channel_bicoherence(
            fpath, args.nfft, 0.5, args.ensemble, args.threshold,
            args.f1max, args.f2max, args.topn,
            t_start_ms=m9_start, t_end_ms=m9_end,
            peak_window_khz=args.peak_window_khz, fmin_khz=args.fmin
        )

        max_b2 = result_rep['diff_max_b2']
        best_freqs = result_rep['diff_best_freqs']

        # Local max at reference triad location
        bicoh2 = result_rep['bicoh2']
        f1_arr = result_rep['f1'] / 1000.0
        f2_arr = result_rep['f2'] / 1000.0
        f1_ref, f2_ref, _ = ref_triad

        f1_mask = np.abs(f1_arr - f1_ref) <= args.repro_tol_khz
        f2_mask = np.abs(f2_arr - (-f2_ref)) <= args.repro_tol_khz
        f1_idx = np.where(f1_mask)[0]
        f2_idx = np.where(f2_mask)[0]

        if len(f1_idx) > 0 and len(f2_idx) > 0:
            subgrid = bicoh2[np.ix_(f1_idx, f2_idx)]
            max_ref_b2 = float(np.max(subgrid))
            max_sub_idx = np.unravel_index(np.argmax(subgrid), subgrid.shape)
            found_f1 = float(f1_arr[f1_idx[max_sub_idx[0]]])
            found_f2 = float(-f2_arr[f2_idx[max_sub_idx[1]]])
        else:
            max_ref_b2 = 0.0
            found_f1 = 0.0
            found_f2 = 0.0

        short_circuit = "YES (global max < threshold)" if max_b2 < args.threshold else "NO"
        P(f"  {ch}: global_max_b2={max_b2:.3f} at f1={best_freqs[0]:.1f}/f2={best_freqs[1]:.1f}  |  "
          f"at_ref_triad_b2={max_ref_b2:.3f} at f1={found_f1:.1f}/f2={found_f2:.1f}  |  "
          f"short-circuits: {short_circuit}")

# --- Write output ------------------------------------------------------------

output_path = "c:/TFG/shot_88654_parameter_comparison.log"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines) + "\n")

P(f"\nDone. Output written to {output_path}")
