import sys
import os
import re
import glob
import argparse
import fnmatch
import logging
from pathlib import Path
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from scipy.ndimage import maximum_filter, uniform_filter1d
from scipy.signal import find_peaks

log = logging.getLogger("mhd_obj3")


def setup_logging(verbose, log_file):
    log.setLevel(logging.DEBUG)
    log.handlers.clear()

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(console)

    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(file_handler)
        log.info(f"(Full diagnostic detail for this run is being written to: {log_file})")

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

# -------------------------------------------------------------
# SHOT CONFIGURATION 
# -------------------------------------------------------------
SHOT = 88652

import turnelib as TE
import libana_signal as LAS
from mhd_common import extract_instantaneous_frequency, OPTIMAL_SG_WIN

# Expected pattern for genuine (toroidal) Mirnov coil names, restricted to MP1, MP3, and MP4
MIRNOV_NAME_RE = re.compile(r'^MP[134]$')

def check_channel_identity(channel_name):
    """Checks if a channel captured by the glob follows the expected Mirnov coil pattern.
    """
    if not MIRNOV_NAME_RE.match(channel_name):
        # The log message has been slightly updated to reflect the new strict pattern
        log.info(f"\u26a0\ufe0f WARNING: channel '{channel_name}' does not follow the expected Mirnov coil "
              f"pattern (MP1, MP3, or MP4). Check the diagnostic documentation to "
              f"see what this channel actually measures before interpreting it as a Mirnov "
              f"magnetic signal.")

BICOH_NOVERLAP_FRAC = 0.5


def bicoherence_min_samples(nfft, ensemble, noverlap_frac):
    """Minimum number of samples LAS.abicoh2() needs to form `ensemble` overlapping segments of
    length `nfft` with fractional overlap `noverlap_frac`. Shared by analyze_channel_bicoherence()
    (where it is enforced) and compute_flat_frequency_window() (where it is used to make sure the
    [M9] flat-frequency sub-window is never shrunk below what the bicoherence call itself will
    require -- see the [M9][BICOH-FLOOR] note there)."""
    return nfft + (ensemble - 1) * int(nfft * (1.0 - noverlap_frac))


def statistical_b2_threshold(n_ensemble, alpha=0.05):
    """Rigorous statistical significance threshold for the squared bicoherence.

    Under the null hypothesis (statistically independent phases/signals), Kim, Y.C. &
    Powers, E.J. (IEEE Trans. Plasma Sci., 1979) show that N_ensemble * b^2 approximately
    follows an exponential distribution with mean 1 (equivalently, b^2 is distributed as an
    exponential with mean 1/N_ensemble). The (1 - alpha) percentile of that exponential
    distribution is:

        b^2_alpha = -ln(alpha) / N_ensemble

    This is different from (and more rigorous than) comparing only against the MEAN of that
    distribution (1/N_ensemble), which systematically underestimates the true 95% significance
    level (with alpha=0.05, the factor -ln(0.05) ~= 3.0, i.e. the 95% threshold is ~3 times the
    mean, not the mean itself).
    """
    return -np.log(alpha) / n_ensemble


def load_edf_signal(fpath):
    """Centralized helper to load an .edf channel and return (t_sec, ys, dt, fs).

    NOTE: this always returns COLUMN 1 (dat[:, 1]) only. This is correct for genuine
    single-channel diagnostics (Mirnov probes MP1..MPn, W_p, ECH), but is WRONG for a
    multi-tip probe array file such as DivProArr (ValNo=38: 36 real Langmuir probe tips
    plus a bias and a trigger channel) -- for that kind of file use
    `load_edf_all_channels()` below instead, which reads every column.
    """
    edf = TE.edf()
    dat = edf.load(str(fpath))
    t = dat[:, 0]
    ys = dat[:, 1]
    t_sec = t / 1000.0 if edf.DimUnit[0] == 'ms' else t
    dt = (t_sec[100] - t_sec[0]) / 100.0
    fs = 1.0 / dt
    return t_sec, ys, dt, fs


def load_edf_all_channels(fpath, exclude_names=None):
    """Loads EVERY data column of a multi-channel .edf file (e.g. a Langmuir/divertor
    probe array such as DivProArr, which packs 38 columns -- 36 real probe tips plus a
    bias channel and a trigger channel -- into one file).

    Returns (t_sec, dt, fs, channels), where `channels` is a dict
    {channel_name: ys_array} for every column NOT in `exclude_names` (case-insensitive
    match against the raw pin label reported by the .edf metadata, e.g. '18-1', '27-1').

    If the .edf's channel-name metadata (ValName) is unavailable or doesn't match the
    number of data columns, falls back to naming columns 'col1', 'col2', ... so the
    caller still gets every column instead of silently defaulting to just the first one.
    """
    if exclude_names is None:
        exclude_names = set()
    else:
        exclude_names = {str(n).strip().lower() for n in exclude_names}

    edf = TE.edf()
    dat = edf.load(str(fpath))
    t = dat[:, 0]
    t_sec = t / 1000.0 if edf.DimUnit[0] == 'ms' else t
    dt = (t_sec[100] - t_sec[0]) / 100.0
    fs = 1.0 / dt

    n_cols = dat.shape[1] - 1
    names = list(getattr(edf, 'ValName', []) or [])
    if len(names) != n_cols:
        log.debug(f"    \u26a0\ufe0f {fpath}: edf.ValName length ({len(names)}) does not match the number of "
                  f"data columns ({n_cols}); falling back to generic column names 'col1'..'col{n_cols}'.")
        names = [f"col{i + 1}" for i in range(n_cols)]

    channels = {}
    for i, name in enumerate(names):
        if str(name).strip().lower() in exclude_names:
            continue
        channels[name] = dat[:, i + 1]
    return t_sec, dt, fs, channels


def slice_window(t_sec, ys, t_start_ms, t_end_ms):
    """Trims the signal to a time window [t_start_ms, t_end_ms] in milliseconds.

    If either bound is None, returns the full signal untrimmed (allows this same function to
    be reused for physical-window analysis or for the full discharge).
    """
    if t_start_ms is None or t_end_ms is None:
        return t_sec, ys
    mask = (t_sec * 1000.0 >= t_start_ms) & (t_sec * 1000.0 <= t_end_ms)
    return t_sec[mask], ys[mask]


def _typical_background_score(ys_smooth, duration_bins, step_bins, exclude_lo, exclude_hi):
    """Sample the (std + range) 'flatness score' of many `duration_bins`-long windows tiling the
    WHOLE trace (excluding [exclude_lo, exclude_hi], meant to be the burst region with margin),
    to build an empirical picture of how much a window of this length naturally wiggles in this
    shot's background, at this smoothing setting. Returns the array of scores (may be empty if
    there isn't enough non-excluded data to sample from).
    """
    scores = []
    n = len(ys_smooth)
    start = 0
    while start + duration_bins <= n:
        end = start + duration_bins - 1
        if not (end < exclude_lo or start > exclude_hi):
            start += step_bins
            continue
        window = ys_smooth[start:end + 1]
        scores.append(float(np.std(window) + (np.max(window) - np.min(window))))
        start += step_bins
    return np.array(scores)


def _search_background_window(ys_smooth, anchor_idx, direction, duration_bins, step_bins,
                               level_threshold, score_threshold, min_idx, max_idx):
    """Search outward from `anchor_idx` (adjacent to a burst boundary) for the NEAREST window of
    `duration_bins` samples that genuinely looks like background (quiescent/control), instead of
    just assuming the region immediately next to the burst is flat.

    direction: -1 to search backward in time (for the quiescent window, before the burst),
               +1 to search forward in time (for the control window, after the burst).

    A candidate window is accepted as soon as it satisfies BOTH:
      - mean level <= `level_threshold` (the same end_threshold used to detect the H-L back-
        transition, i.e. the window must sit in the "not elevated" regime, not still on a ramp)
      - (std + range) <= `score_threshold`, where `score_threshold` is calibrated by the caller
        from the empirical spread of same-length windows sampled elsewhere in this shot's own
        background (see `_typical_background_score`), rather than from a fixed noise model --
        this avoids both over- and under-estimating "flat enough" for shots whose W_p measurement
        happens to be cleaner or noisier than assumed.

    If no candidate in [min_idx, max_idx] satisfies both conditions, the fallback prioritizes
    LEVEL over flatness: among all evaluated candidates, it prefers the flattest one that at
    least sits below `level_threshold` (i.e. "not elevated" even if not perfectly flat) over the
    flattest one overall, which could otherwise be a deceptively smooth window that's still
    clearly sitting on an elevated plateau (e.g. right before the burst) rather than genuine
    background -- flatness alone is not a good fallback criterion, since a flat elevated plateau
    can easily out-score a slightly noisier but genuinely low region. 'found' is False in
    either case, so the caller always knows this wasn't a full pass.
    """
    best_overall = None
    best_overall_score = np.inf
    best_level_ok = None
    best_level_ok_score = np.inf
    offset = 0
    while True:
        if direction < 0:
            end_idx = anchor_idx - offset
            start_idx = end_idx - duration_bins + 1
            if start_idx < min_idx:
                break
        else:
            start_idx = anchor_idx + offset
            end_idx = start_idx + duration_bins - 1
            if end_idx > max_idx:
                break

        window = ys_smooth[start_idx:end_idx + 1]
        w_mean = float(np.mean(window))
        w_std = float(np.std(window))
        w_range = float(np.max(window) - np.min(window))
        score = w_std + w_range
        level_ok = w_mean <= level_threshold

        satisfies = (level_ok and score <= score_threshold)
        if satisfies:
            return {'found': True, 'start_idx': start_idx, 'end_idx': end_idx,
                    'mean': w_mean, 'std': w_std}

        if score < best_overall_score:
            best_overall_score = score
            best_overall = {'found': False, 'start_idx': start_idx, 'end_idx': end_idx,
                             'mean': w_mean, 'std': w_std}
        if level_ok and score < best_level_ok_score:
            best_level_ok_score = score
            best_level_ok = {'found': False, 'start_idx': start_idx, 'end_idx': end_idx,
                              'mean': w_mean, 'std': w_std}
        offset += step_bins

    if best_level_ok is not None:
        return best_level_ok
    return best_overall if best_overall is not None else {'found': False, 'start_idx': None,
                                                            'end_idx': None, 'mean': None, 'std': None}


def detect_hmode_burst_window(wp_file, baseline_end_ms=50.0, smooth_ms=2.0,
                               k_on=3.0, k_off=1.5, min_duration_ms=10.0,
                               quiescent_duration_ms=25.0, control_duration_ms=25.0,
                               window_margin_ms=5.0, flat_percentile=50.0, flat_k=1.5,
                               robust_baseline=True, diagnostic_plot_path=None, defer_plot=False):
    """Heuristic, data-driven identification of the H-mode BURST window, and of matching
    QUIESCENT (pre-burst) and CONTROL (post-burst) comparison windows, from the stored
    energy signal W_p, instead of hardcoded [--burst/--quiescent/--control -start/-end] values.

    Method (threshold crossing with hysteresis on a smoothed W_p trace):
      1. A pre-burst BASELINE is estimated from [t0, baseline_end_ms] (assumed to be an
         L-mode / pre-transition segment), giving a mean and std of W_p in that segment.
      2. W_p is smoothed with a moving average of width `smooth_ms` to suppress
         high-frequency noise before thresholding.
      3. ONSET (L-H transition): first time the smoothed W_p sustained-exceeds
         baseline_mean + k_on * baseline_std for at least `min_duration_ms`.
      4. END (H-L transition / burst termination): first time AFTER onset that the smoothed
         W_p sustained-drops back below baseline_mean + k_off * baseline_std (k_off < k_on,
         i.e. hysteresis, so the window doesn't fragment on a single noisy sample).
      5. QUIESCENT window: rather than assuming the region immediately before onset is flat
         (it often isn't -- e.g. a rising precursor phase, or another small transition sitting
         just before the main burst), the code SEARCHES backward in time starting from
         `window_margin_ms` before onset, testing successive `quiescent_duration_ms`-long
         windows until it finds one that is both (a) below the end_threshold (i.e. genuinely
         "background", not still elevated/transitioning) and (b) flat -- its (std + range) is
         within the empirical spread of same-length windows sampled elsewhere in this shot's own
         background (see `flat_percentile`/`flat_k`), rather than a fixed noise model, so it
         self-calibrates whether this particular shot's W_p measurement happens to be clean or
         noisy. The nearest such window to the burst is used. If none qualifies before running
         out of pre-burst data, the best (flattest, lowest) candidate found is used instead and
         flagged as not fully validated.
      6. CONTROL window: the same search, run forward in time from `window_margin_ms` after the
         detected end, over the post-burst decay/relaxation phase.
      Both are marked not-available ('quiescent_ok'/'control_ok' = False) if there wasn't enough
      room to evaluate even one candidate window (e.g. burst starts right at t0, or ends right at
      the end of the recording) -- in that case 'quiescent_start_ms'/'control_start_ms' are None
      and the caller should fall back to a manual window for that one.

    This is a HEURISTIC, not a validated H-mode detector (a proper one would normally also
    use a direct edge-turbulence marker such as D-alpha/H-alpha, if available in the dataset).
    The detected windows and threshold crossings are always printed and should be visually
    cross-checked against the W_p trace (and D-alpha/H-alpha if you have it) before being
    trusted as the physical windows for the bicoherence analysis.

    Returns a dict with 'start_ms'/'end_ms' (burst), 'quiescent_start_ms'/'quiescent_end_ms'/
    'quiescent_ok'/'quiescent_mean'/'quiescent_std', the equivalent 'control_*' fields,
    'baseline_mean', 'baseline_std', 'onset_threshold', 'end_threshold', and 'ok' (False if
    burst detection itself failed, e.g. no sustained onset was found -- in that case the caller
    should fall back to fully manual windows for all three).

    If `diagnostic_plot_path` is given, a PNG is saved there showing the raw and smoothed W_p
    trace, the baseline band, onset/end thresholds, and the three detected windows shaded --
    this is the recommended way to VISUALLY confirm the auto-detected windows rather than
    trusting the printed numbers alone (see caller for details).

    If `defer_plot` is True, the plot is NOT saved here even if `diagnostic_plot_path` is given
    (the failure path below is the one exception -- see [M9-PLOT] note there); instead the raw
    arrays needed to draw it later (`t_ms`, `ys`, `ys_smooth`) are included in the returned dict,
    so the caller can render the PNG itself once it also knows the [M9] mode-active
    (flat-frequency) sub-window -- which is only computed AFTER this function returns, and would
    otherwise never make it onto this diagnostic plot at all (see refine_burst_with_mode_active_
    window() and the deferred _plot_wp_window_detection() call in run_single_shot()).
    """
    t_sec, ys, dt, _ = load_edf_signal(wp_file)
    t_ms = t_sec * 1000.0

    # --- 1. Baseline from the pre-burst segment ---
    mask_base = t_ms <= (t_ms[0] + baseline_end_ms)
    if np.sum(mask_base) < 5:
        return {'ok': False, 'reason': 'baseline window too short / no samples'}
    if robust_baseline:
        # Median/MAD instead of mean/std: robust to a contaminating bump or small secondary
        # transition landing inside the baseline segment (a real risk when shots differ in how
        # early the interesting activity starts), which would otherwise inflate baseline_std
        # and loosen the onset/end thresholds.
        baseline_mean = float(np.median(ys[mask_base]))
        mad = float(np.median(np.abs(ys[mask_base] - baseline_mean)))
        baseline_std = float(mad * 1.4826)  # MAD -> std-equivalent under a Gaussian assumption
        if baseline_std <= 1e-12:
            baseline_std = float(np.std(ys[mask_base]))  # degenerate MAD (near-constant) -> fall back
    else:
        baseline_mean = float(np.mean(ys[mask_base]))
        baseline_std = float(np.std(ys[mask_base]))

    # --- 2. Smooth W_p to suppress noise before thresholding ---
    smooth_bins = max(1, int(round(smooth_ms / (dt * 1000.0))))
    ys_smooth = uniform_filter1d(ys, size=smooth_bins, mode='nearest') if smooth_bins > 1 else ys

    onset_threshold = baseline_mean + k_on * baseline_std
    end_threshold = baseline_mean + k_off * baseline_std
    min_duration_bins = max(1, int(round(min_duration_ms / (dt * 1000.0))))

    above_onset = ys_smooth > onset_threshold

    # --- 3. Onset: first index where `above_onset` stays True for >= min_duration_bins ---
    onset_idx = None
    run_len = 0
    for i, flag in enumerate(above_onset):
        run_len = run_len + 1 if flag else 0
        if run_len >= min_duration_bins:
            onset_idx = i - min_duration_bins + 1
            break

    if onset_idx is None:
        result_fail = {'ok': False, 'reason': 'no sustained W_p rise above baseline_mean + '
                                        f'{k_on}*std was found', 'baseline_mean': baseline_mean,
                'baseline_std': baseline_std, 'onset_threshold': onset_threshold,
                't_ms': t_ms, 'ys': ys, 'ys_smooth': ys_smooth}
        # [M9-PLOT] No burst was found at all, so there is no later [M9] step to wait for --
        # always save immediately here regardless of `defer_plot`.
        if diagnostic_plot_path is not None:
            _plot_wp_window_detection(t_ms, ys, ys_smooth, baseline_end_ms, baseline_mean,
                                       baseline_std, onset_threshold, end_threshold,
                                       result_fail, diagnostic_plot_path)
        return result_fail

    # --- 4. End: after onset, first sustained drop below the (lower) end_threshold ---
    below_end = ys_smooth <= end_threshold
    end_idx = len(ys_smooth) - 1
    run_len = 0
    for i in range(onset_idx, len(ys_smooth)):
        run_len = run_len + 1 if below_end[i] else 0
        if run_len >= min_duration_bins:
            end_idx = i - min_duration_bins + 1
            break

    burst_start_ms = float(t_ms[onset_idx])
    burst_end_ms = float(t_ms[end_idx])
    n_samples = len(ys_smooth)

    # --- 5/6. Quiescent (pre-burst) and Control (post-burst) windows: found by searching
    # outward from the burst edges for the nearest genuinely flat, sub-threshold segment,
    # instead of assuming a fixed offset is automatically representative background. ---
    margin_bins = max(1, int(round(window_margin_ms / (dt * 1000.0))))
    duration_bins_q = max(1, int(round(quiescent_duration_ms / (dt * 1000.0))))
    duration_bins_c = max(1, int(round(control_duration_ms / (dt * 1000.0))))
    step_bins = max(1, int(round(2.0 / (dt * 1000.0))))  # scan in ~2 ms steps

    # Calibrate "how flat is flat enough" from the empirical spread of same-length windows
    # sampled across the WHOLE trace, excluding the burst (+ margin). This self-scales to
    # whatever this shot's own background naturally looks like, instead of assuming a fixed
    # noise model that can be wrong by an order of magnitude between a clean and a noisy shot.
    exclude_lo = max(0, onset_idx - margin_bins)
    exclude_hi = min(n_samples - 1, end_idx + margin_bins)
    q_scores = _typical_background_score(ys_smooth, duration_bins_q, step_bins, exclude_lo, exclude_hi)
    c_scores = (q_scores if duration_bins_c == duration_bins_q else
                _typical_background_score(ys_smooth, duration_bins_c, step_bins, exclude_lo, exclude_hi))
    q_score_threshold = (flat_k * float(np.percentile(q_scores, flat_percentile))
                          if len(q_scores) >= 5 else np.inf)
    c_score_threshold = (flat_k * float(np.percentile(c_scores, flat_percentile))
                          if len(c_scores) >= 5 else np.inf)

    res_q = _search_background_window(
        ys_smooth, anchor_idx=onset_idx - margin_bins, direction=-1,
        duration_bins=duration_bins_q, step_bins=step_bins,
        level_threshold=end_threshold, score_threshold=q_score_threshold,
        min_idx=0, max_idx=onset_idx - 1)
    res_c = _search_background_window(
        ys_smooth, anchor_idx=end_idx + margin_bins, direction=+1,
        duration_bins=duration_bins_c, step_bins=step_bins,
        level_threshold=end_threshold, score_threshold=c_score_threshold,
        min_idx=end_idx + 1, max_idx=n_samples - 1)

    q_start_ms = float(t_ms[res_q['start_idx']]) if res_q['start_idx'] is not None else None
    q_end_ms = float(t_ms[res_q['end_idx']]) if res_q['end_idx'] is not None else None
    c_start_ms = float(t_ms[res_c['start_idx']]) if res_c['start_idx'] is not None else None
    c_end_ms = float(t_ms[res_c['end_idx']]) if res_c['end_idx'] is not None else None

    result = {
        'ok': True,
        'start_ms': burst_start_ms,
        'end_ms': burst_end_ms,
        'quiescent_start_ms': q_start_ms, 'quiescent_end_ms': q_end_ms, 'quiescent_ok': res_q['found'],
        'quiescent_mean': res_q['mean'], 'quiescent_std': res_q['std'],
        'control_start_ms': c_start_ms, 'control_end_ms': c_end_ms, 'control_ok': res_c['found'],
        'control_mean': res_c['mean'], 'control_std': res_c['std'],
        'baseline_mean': baseline_mean, 'baseline_std': baseline_std,
        'quiescent_score_threshold': q_score_threshold, 'control_score_threshold': c_score_threshold,
        'onset_threshold': onset_threshold, 'end_threshold': end_threshold,
        'wp_peak': float(np.max(ys_smooth[onset_idx:end_idx + 1])) if end_idx > onset_idx else float(ys_smooth[onset_idx]),
        # Raw arrays for a caller that wants to (re-)render the diagnostic plot itself later --
        # see `defer_plot` above. Included unconditionally (cheap to carry for a single shot);
        # only whether the plot gets SAVED here depends on `defer_plot`.
        't_ms': t_ms, 'ys': ys, 'ys_smooth': ys_smooth, 'baseline_end_ms': baseline_end_ms,
    }

    if diagnostic_plot_path is not None and not defer_plot:
        _plot_wp_window_detection(t_ms, ys, ys_smooth, baseline_end_ms, baseline_mean,
                                   baseline_std, onset_threshold, end_threshold,
                                   result, diagnostic_plot_path)

    return result


def _plot_wp_window_detection(t_ms, ys, ys_smooth, baseline_end_ms, baseline_mean, baseline_std,
                               onset_threshold, end_threshold, result, out_path, flat_window=None):
    """Save a diagnostic PNG so the auto-detected Burst/Quiescent/Control windows can be
    VISUALLY confirmed against the raw and smoothed W_p trace, instead of trusting the
    printed numbers alone. Shows: raw W_p (thin, transparent), smoothed W_p (used for
    detection), the baseline segment and its mean +/- k*std thresholds, and shaded spans
    for whichever of Burst/Quiescent/Control were successfully placed.

    `flat_window`: optional (start_ms, end_ms, note) tuple for the [M9] mode-active
    (flat-frequency) sub-window -- the actual window used for the bicoherence calculations and
    figure, which can be tighter than the raw Burst window shown here. `note` is an extra string
    appended to its legend label (e.g. to flag that it was extended past the flatness tolerance to
    meet the bicoherence sample floor); pass '' for no extra note. Drawn as a distinct
    purple/magenta span nested inside the (wider, lighter) red Burst span so both remain visible
    at once -- this is the SAME window already annotated on the squared-bicoherence figure's
    title, now cross-referenced here too so a single glance at this plot shows exactly which
    sub-interval of the burst the physics results actually come from.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_ms, ys, color='0.75', linewidth=0.6, label='W_p (raw)')
    ax.plot(t_ms, ys_smooth, color='black', linewidth=1.2, label='W_p (smoothed, used for detection)')

    ax.axvspan(t_ms[0], t_ms[0] + baseline_end_ms, color='gray', alpha=0.15,
               label=f'Baseline segment (first {baseline_end_ms:.0f} ms)')
    ax.axhline(baseline_mean, color='gray', linestyle=':', linewidth=1,
               label=f'Baseline mean = {baseline_mean:.3f}')
    ax.axhline(onset_threshold, color='crimson', linestyle='--', linewidth=1,
               label=f'Onset threshold = {onset_threshold:.3f}')
    ax.axhline(end_threshold, color='darkorange', linestyle='--', linewidth=1,
               label=f'End threshold = {end_threshold:.3f}')

    if result.get('ok'):
        ax.axvspan(result['start_ms'], result['end_ms'], color='red', alpha=0.15, label='Burst window')
        ax.axvline(result['start_ms'], color='red', linewidth=1.5)
        ax.axvline(result['end_ms'], color='red', linewidth=1.5)

        if flat_window is not None:
            fw_start, fw_end, fw_note = flat_window
            fw_label = '[M9] Mode-active (flat-frequency) sub-window'
            ax.axvspan(fw_start, fw_end, facecolor='purple', alpha=0.32, edgecolor='purple',
                       linewidth=1.5, zorder=4, label=fw_label)
            ax.axvline(fw_start, color='purple', linewidth=1.5, zorder=4)
            ax.axvline(fw_end, color='purple', linewidth=1.5, zorder=4)

        if result.get('quiescent_start_ms') is not None:
            q_label = 'Quiescent window' if result.get('quiescent_ok') else \
                      'Quiescent window (best-effort, did NOT pass flatness/level test)'
            q_alpha = 0.18 if result.get('quiescent_ok') else 0.10
            q_hatch = None if result.get('quiescent_ok') else '//'
            ax.axvspan(result['quiescent_start_ms'], result['quiescent_end_ms'], facecolor='royalblue',
                       alpha=q_alpha, hatch=q_hatch, edgecolor='royalblue', label=q_label)
        if result.get('control_start_ms') is not None:
            c_label = 'Control window' if result.get('control_ok') else \
                       'Control window (best-effort, did NOT pass flatness/level test)'
            c_alpha = 0.18 if result.get('control_ok') else 0.10
            c_hatch = None if result.get('control_ok') else '//'
            ax.axvspan(result['control_start_ms'], result['control_end_ms'], facecolor='seagreen',
                       alpha=c_alpha, hatch=c_hatch, edgecolor='seagreen', label=c_label)
        title_suffix = f"Burst detected: {result['start_ms']:.1f}-{result['end_ms']:.1f} ms"
        if flat_window is not None:
            title_suffix += f" | [M9] sub-window: {flat_window[0]:.1f}-{flat_window[1]:.1f} ms"
    else:
        title_suffix = f"Detection FAILED ({result.get('reason', 'unknown')})"

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("W_p (kJ)")
    ax.set_title(f"H-mode window auto-detection (visual check) -- {title_suffix}")
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    log.debug(f"   \U0001f5bc\ufe0f  Diagnostic plot saved to: '{out_path}' -- open this to VISUALLY confirm "
          f"the auto-detected window(s) against the actual W_p trace before trusting them.")


def detect_flat_frequency_subwindow(ifreq_khz, t_ms, dt, i0_domain, i1_domain, args, min_duration_ms=None):
    """
    [M9] FLAT-FREQUENCY SUB-WINDOW DETECTOR, ported verbatim (same math, same defaults) from
    Objective 2's mhd_analysis_obj2.py, where it gates the inter-probe cross-spectral coherence
    panels. Here it gates the bicoherence analysis window instead (see compute_flat_frequency_
    window() below and its call site in run_single_shot()).

    Motivation: even restricted to a genuine burst span (from detect_hmode_burst_window() above),
    the dominant mode's instantaneous FREQUENCY is not necessarily flat throughout -- it can sweep
    near the burst's own edges (onset transient, offset/transition). Bicoherence, like cross-
    spectral coherence, assumes the coupled frequencies are reasonably stationary across the
    analysis window, so restricting to the flattest interior patch avoids diluting the coupling
    estimate with chirping edges.

    Algorithm ("find the flattest patch, then grow it" -- adaptive per shot, no fixed kHz/ms
    cutoff):
      1. Local slope |d(f_inst)/dt| (kHz/ms) via np.gradient over the domain, then smoothed with a
         short moving average (--flat-slope-smooth-ms) to suppress residual per-sample phase noise.
      2. SCAN: slide a window of length --flat-scan-window-ms across the domain and compute its mean
         |slope| (via a 'valid' convolution, i.e. only full windows are considered -- no edge
         padding). The window position with the LOWEST mean |slope| is the flattest contiguous
         patch of that length in the domain.
      3. GROW: extend that window outward, one sample at a time in each direction, for as long as
         the window's mean |slope| stays within --flat-growth-tolerance (relative) of the best mean
         found in step 2. Growth stops the moment either side would pull the mean too high -- i.e.
         right where the mode starts chirping again.
      4. Manual override (--flat-window-start/--flat-window-end) bypasses steps 1-3 entirely.
      5. [M9][BICOH-FLOOR] `min_duration_ms` is the minimum duration this window is actually
         allowed to end up with -- it is NOT just --flat-min-duration-ms any more. The caller
         (compute_flat_frequency_window()) computes it as
         max(--flat-min-duration-ms, the number of samples LAS.abicoh2() itself requires for
         --nfft/--ensemble, converted to ms at this probe's sample rate). This is the actual bug
         fix: previously a window could pass the plain --flat-min-duration-ms check (e.g. 10 ms)
         and still be too short for the bicoherence call downstream (e.g. it needs ~15.9 ms for
         nfft=1024/nensemble=30), which made EVERY channel raise "Time window too short" and
         silently emptied the ranking table and every downstream comparison. If `min_duration_ms`
         is not given, falls back to --flat-min-duration-ms (old behavior).
      6. If the tolerance-grown window in step 3 is still shorter than `min_duration_ms`, instead
         of immediately giving up, a SECOND, forced growth pass extends the window past the
         --flat-growth-tolerance bound -- one sample at a time, always taking whichever side
         currently has the lower local |slope| -- until `min_duration_ms` is reached or the domain
         is exhausted. This keeps the window as tight (as close to genuinely flat) as possible
         while still guaranteeing it is USABLE by the bicoherence call, rather than either (a)
         silently failing downstream, or (b) throwing away the whole flat-window idea and using
         the raw (possibly chirping) full burst. Only if even the full domain cannot reach
         `min_duration_ms` does this fall back to the full domain, with a message that explains
         why (the burst itself is too short for the requested --nfft/--ensemble/overlap).

    Returns (i0_flat, i1_flat, info_dict) with i0_flat/i1_flat as GLOBAL indices into the shot's
    full t_ms array.
    """
    n_domain = i1_domain - i0_domain
    fallback = (i0_domain, i1_domain, {"used_fallback": True, "reason": "domain too short or manual window empty"})
    effective_min_duration_ms = args.flat_min_duration_ms if min_duration_ms is None else min_duration_ms

    if args.flat_window_start is not None and args.flat_window_end is not None:
        lo = max(args.flat_window_start, t_ms[i0_domain])
        hi = min(args.flat_window_end, t_ms[i1_domain - 1])
        if hi <= lo:
            log.info(f"  \u26a0\ufe0f [M9] MANUAL --flat-window-start/--flat-window-end ({args.flat_window_start:.1f}-"
                     f"{args.flat_window_end:.1f} ms) does not overlap the search domain "
                     f"({t_ms[i0_domain]:.1f}-{t_ms[i1_domain-1]:.1f} ms); falling back to the full domain.")
            return fallback
        idx = np.where((t_ms >= lo) & (t_ms <= hi))[0]
        i0f, i1f = int(idx[0]), int(idx[-1]) + 1
        log.debug(f"  [M9] Flat-frequency sub-window: MANUAL override = {t_ms[i0f]:.1f}-{t_ms[i1f-1]:.1f} ms "
                  "(--flat-window-start/--flat-window-end).")
        return i0f, i1f, {"used_fallback": False, "manual": True}

    scan_samples = max(3, int(round(args.flat_scan_window_ms / (dt * 1000.0))))
    if scan_samples >= n_domain:
        log.info(f"  \u26a0\ufe0f [M9] Search domain ({n_domain} samples, {n_domain * dt * 1000.0:.1f} ms) is shorter "
                 f"than --flat-scan-window-ms ({args.flat_scan_window_ms:.1f} ms); the full domain will be used as-is.")
        return fallback

    ifreq_domain = ifreq_khz[i0_domain:i1_domain]
    t_domain = t_ms[i0_domain:i1_domain]

    slope_raw = np.gradient(ifreq_domain, t_domain)
    smooth_samples = max(1, int(round(args.flat_slope_smooth_ms / (dt * 1000.0))))
    if smooth_samples % 2 == 0:
        smooth_samples += 1
    if 3 <= smooth_samples < n_domain:
        kernel = np.ones(smooth_samples) / smooth_samples
        slope_smooth = np.convolve(slope_raw, kernel, mode='same')
    else:
        slope_smooth = slope_raw
    slope_abs = np.abs(slope_smooth)

    # SCAN: mean |slope| of every full scan_samples-long window (valid convolution -> no edge padding).
    scan_kernel = np.ones(scan_samples) / scan_samples
    window_means = np.convolve(slope_abs, scan_kernel, mode='valid')
    best_start = int(np.argmin(window_means))
    best_end = best_start + scan_samples
    best_mean = float(window_means[best_start])

    log.debug(f"  [M9] Flat-frequency scan: flattest {args.flat_scan_window_ms:.1f} ms patch in the "
              f"{n_domain * dt * 1000.0:.1f} ms search domain is at "
              f"{t_domain[best_start]:.1f}-{t_domain[best_end-1]:.1f} ms "
              f"(mean |d(f_inst)/dt| = {best_mean:.3f} kHz/ms).")

    # GROW: extend outward while the window's mean |slope| stays within tolerance of best_mean.
    tol_mean = best_mean * (1.0 + args.flat_growth_tolerance)
    cumsum = np.concatenate(([0.0], np.cumsum(slope_abs)))

    def window_mean(a, b):
        return (cumsum[b] - cumsum[a]) / (b - a)

    start, end = best_start, best_end
    while start > 0 and window_mean(start - 1, end) <= tol_mean:
        start -= 1
    while end < n_domain and window_mean(start, end + 1) <= tol_mean:
        end += 1

    duration_ms = (end - start) * dt * 1000.0
    touches_edge = (start == 0) or (end == n_domain)
    grown_mean = float(window_mean(start, end))

    i0_flat, i1_flat = i0_domain + start, i0_domain + end
    edge_note = " (grew all the way to the search domain's own edge)" if touches_edge else ""
    log.debug(f"  [M9] Flat-frequency sub-window after growth: {t_ms[i0_flat]:.1f}-{t_ms[i1_flat-1]:.1f} ms "
              f"({end - start} samples, {duration_ms:.1f} ms, mean |slope| = {grown_mean:.3f} kHz/ms, "
              f"grown from an {args.flat_scan_window_ms:.1f} ms core within "
              f"{100.0*args.flat_growth_tolerance:.0f}% tolerance){edge_note}.")

    relaxed_for_min_duration = False
    if duration_ms < effective_min_duration_ms:
        floor_note = ""
        if min_duration_ms is not None:
            floor_note = (f" -- set by --nfft/--ensemble, not just --flat-min-duration-ms="
                           f"{args.flat_min_duration_ms:.1f} ms")
        log.info(f"  \u26a0\ufe0f [M9] Tolerance-grown flat-frequency window ({duration_ms:.1f} ms, "
                 f"{end - start} samples) is shorter than the required minimum "
                 f"({effective_min_duration_ms:.1f} ms{floor_note}); extending further, past the "
                 f"{100.0*args.flat_growth_tolerance:.0f}% flatness tolerance, using whichever side is "
                 "least-chirping at each step, to reach that floor...")
        relaxed_for_min_duration = True
        while (end - start) * dt * 1000.0 < effective_min_duration_ms and (start > 0 or end < n_domain):
            left_slope = slope_abs[start - 1] if start > 0 else np.inf
            right_slope = slope_abs[end] if end < n_domain else np.inf
            if left_slope <= right_slope:
                start -= 1
            else:
                end += 1

        duration_ms = (end - start) * dt * 1000.0
        touches_edge = (start == 0) or (end == n_domain)
        grown_mean = float(window_mean(start, end))
        i0_flat, i1_flat = i0_domain + start, i0_domain + end

        if duration_ms < effective_min_duration_ms:
            log.info(f"  \u26a0\ufe0f [M9] Even the full {n_domain * dt * 1000.0:.1f} ms search domain "
                     f"({n_domain} samples) is shorter than the {effective_min_duration_ms:.1f} ms required "
                     "for the requested --nfft/--ensemble/overlap; falling back to the full search domain "
                     "as-is. The downstream bicoherence call will still fail for this shot -- lower --nfft "
                     "and/or --ensemble, or find a longer burst, to fix this for good.")
            return fallback

        log.info(f"  [M9] Flat-frequency sub-window extended (beyond the flatness tolerance) to "
                 f"{t_ms[i0_flat]:.1f}-{t_ms[i1_flat-1]:.1f} ms ({end - start} samples, {duration_ms:.1f} ms, "
                 f"mean |slope| = {grown_mean:.3f} kHz/ms) to meet the bicoherence sample floor.")

    return i0_flat, i1_flat, {
        "used_fallback": False, "manual": False, "scan_mean_khz_per_ms": best_mean,
        "grown_mean_khz_per_ms": grown_mean, "duration_ms": duration_ms, "touches_edge": touches_edge,
        "relaxed_for_min_duration": relaxed_for_min_duration,
    }


def compute_flat_frequency_window(fpath, t_start_ms, t_end_ms, args):
    """
    [M9] Wraps detect_flat_frequency_subwindow() for Objective 3's use case: loads a single Mirnov
    probe (--flat-probe, def: MP1), Bessel-bandpass-filters it (--flat-freq-low-khz to
    --flat-freq-high-khz, order --flat-order) and extracts its Hilbert instantaneous frequency
    (smoothed with the Savitzky-Golay window --flat-sg-win -- same extract_instantaneous_frequency()
    used by Objective 2), then searches WITHIN [t_start_ms, t_end_ms] (the H-mode burst window
    already found by detect_hmode_burst_window()) for the flattest, least-chirping sub-interval.

    Returns (i0_flat_ms, i1_flat_ms, flat_info). On any failure to run the detector at all (window
    too short to even load), returns the original [t_start_ms, t_end_ms] unchanged with
    flat_info['used_fallback'] = True and a 'reason'.
    """
    t_sec, ys, dt, fs = load_edf_signal(fpath)
    t_ms = t_sec * 1000.0

    idx_domain = np.where((t_ms >= t_start_ms) & (t_ms <= t_end_ms))[0]
    if len(idx_domain) < 3:
        return t_start_ms, t_end_ms, {"used_fallback": True, "reason": "burst window has too few samples to search"}
    i0_domain, i1_domain = int(idx_domain[0]), int(idx_domain[-1]) + 1

    fl_hz = args.flat_freq_low_khz * 1000.0
    fu_hz = args.flat_freq_high_khz * 1000.0
    log.debug(f"  [M9] Bandpass-filtering {fpath.name} ({args.flat_freq_low_khz:.1f}-"
              f"{args.flat_freq_high_khz:.1f} kHz, Bessel order {args.flat_order}, sg_win="
              f"{args.flat_sg_win}) and extracting the Hilbert instantaneous frequency for "
              "flat-sub-window detection...")
    _, _, _, ifreq_hz = extract_instantaneous_frequency(ys, fs, fl_hz, fu_hz, args.flat_order, args.flat_sg_win)
    ifreq_khz = ifreq_hz / 1000.0

    # [M9][BICOH-FLOOR] The flat-frequency window must be long enough for the bicoherence call
    # that will actually consume it (--nfft/--ensemble/BICOH_NOVERLAP_FRAC), not just long enough
    # to pass the generic --flat-min-duration-ms sanity floor. Compute that requirement here, in
    # ms, at THIS probe's own sample rate (dt), and take whichever of the two floors is stricter.
    bicoh_min_samples = bicoherence_min_samples(args.nfft, args.ensemble, BICOH_NOVERLAP_FRAC)
    bicoh_min_duration_ms = bicoh_min_samples * dt * 1000.0
    effective_min_duration_ms = max(args.flat_min_duration_ms, bicoh_min_duration_ms)
    log.debug(f"  [M9][BICOH-FLOOR] Bicoherence needs >= {bicoh_min_samples} samples "
              f"(nfft={args.nfft}, nensemble={args.ensemble}, noverlap_frac={BICOH_NOVERLAP_FRAC}) = "
              f"{bicoh_min_duration_ms:.1f} ms at this probe's sample rate; effective [M9] minimum "
              f"duration = max({args.flat_min_duration_ms:.1f}, {bicoh_min_duration_ms:.1f}) = "
              f"{effective_min_duration_ms:.1f} ms.")

    i0_flat, i1_flat, flat_info = detect_flat_frequency_subwindow(
        ifreq_khz, t_ms, dt, i0_domain, i1_domain, args, min_duration_ms=effective_min_duration_ms
    )
    return float(t_ms[i0_flat]), float(t_ms[i1_flat - 1]), flat_info


def refine_burst_with_mode_active_window(raw_start_ms, raw_end_ms, base_dir, shot_id, args):
    """
    [M9] BURST WINDOW DETECTION, STAGE 2: given the raw burst window (start/end in ms) -- however
    it was obtained, either MANUALLY via --burst-start/--burst-end or auto-detected from W_p by
    detect_hmode_burst_window() above -- this finds the MODE-ACTIVE sub-window within it (the
    flattest, least-chirping patch of --flat-probe's instantaneous frequency) and returns THAT as
    the window to actually use for the bicoherence calculations and the squared-bicoherence
    figure. Both callers in run_single_shot() route through this single function immediately
    after the raw burst window is established, so the burst-window step and everything
    downstream of it (ranking + figure) are always in sync -- there is no separate later step
    that could silently fail to update the figure's window.

    Returns (start_ms, end_ms, applied, info):
      - (start_ms, end_ms): the mode-active sub-window if one was found, else the ORIGINAL
        (raw_start_ms, raw_end_ms) unchanged (nothing to apply).
      - applied: True iff the mode-active sub-window was actually used in place of the raw burst.
      - info: the flat_info dict from detect_flat_frequency_subwindow(), or
        {'used_fallback': True, 'reason': ...} if the probe file was missing or the step is
        disabled via --no-flat-subwindow.
    """
    if args.no_flat_subwindow:
        return raw_start_ms, raw_end_ms, False, {"used_fallback": True, "reason": "--no-flat-subwindow"}

    mode_active_file = base_dir / f"{args.flat_probe}@{shot_id}.edf"
    if not mode_active_file.exists():
        return raw_start_ms, raw_end_ms, False, {
            "used_fallback": True,
            "reason": f"mode-active probe file not found ({mode_active_file})",
        }

    log.debug(f"--- [M9] Searching for the mode-active (flat-frequency) sub-window within the "
              f"{raw_start_ms:.1f}-{raw_end_ms:.1f} ms burst, using {mode_active_file.name} ---")
    ma_start_ms, ma_end_ms, ma_info = compute_flat_frequency_window(
        mode_active_file, raw_start_ms, raw_end_ms, args)

    if ma_info.get('used_fallback', True):
        return raw_start_ms, raw_end_ms, False, ma_info
    return ma_start_ms, ma_end_ms, True, ma_info


def analyze_channel_bicoherence(fpath, nfft, noverlap_frac, ensemble, threshold, f1max, f2max, topn,
                                 t_start_ms=None, t_end_ms=None, peak_window_khz=5.0, fmin_khz=5.0, detrend='linear',
                                 stability_frac=0.5, stability_tol_khz=None, preloaded=None):
    """Loads an .edf channel, optionally trims it to a time window, and computes its
    squared auto-bicoherence, returning the ranking of local 2D peaks (already deduplicated
    by f1<->f2 symmetry) and the metadata needed for plotting and for the statistical
    comparisons.

    Used both for the main multichannel scan and for the window comparison (burst/quiescent/
    control), shot-to-shot reproducibility, and cross-validation with Langmuir probes, avoiding
    triplicating the logic (and its bugs).

    `preloaded`: optional (t_sec, ys, dt, fs) tuple. When given, `fpath` is NOT read from
    disk -- this lets the caller load a multi-tip probe array file ONCE (see
    `load_edf_all_channels`) and re-use the already-extracted per-tip arrays for many calls,
    instead of re-opening the file from disk for every probe tip.
    """
    if preloaded is not None:
        t_sec, ys, dt, fs = preloaded
    else:
        t_sec, ys, dt, fs = load_edf_signal(fpath)
    t_sec, ys = slice_window(t_sec, ys, t_start_ms, t_end_ms)

    min_samples_needed = bicoherence_min_samples(nfft, ensemble, noverlap_frac)
    if len(ys) < min_samples_needed:
        raise ValueError(
            f"Time window too short ({len(ys)} samples) for nfft={nfft} and "
            f"nensemble={ensemble} (minimum required: {min_samples_needed}). "
            f"Requested window: {t_start_ms}-{t_end_ms} ms."
        )

    fnyq = fs / 2.0

    f1, f2, bicoh2 = LAS.abicoh2(ys, t_sec, dt=dt, nfft=nfft, noverlap=int(nfft * noverlap_frac),
                                  nensemble=ensemble, detrend=detrend)

    ff1, ff2 = np.meshgrid(f1, f2, indexing='ij')

    # --- SUM COUPLING CHANNELS (f1 + f2 = f3, with f1 > 0, f2 > 0) ---
    physical_mask_sum = (ff1 > 0) & (ff2 > 0) & ((ff1 + ff2) < fnyq)
    non_redundant_mask_sum = physical_mask_sum & (ff1 >= ff2) & (ff1 >= fmin_khz * 1000.0) & (ff2 >= fmin_khz * 1000.0)
    total_physical_bins_sum = int(np.sum(non_redundant_mask_sum))
    significant_count_sum = int(np.sum((bicoh2 > threshold) & non_redundant_mask_sum))

    b2_stat_threshold = statistical_b2_threshold(ensemble, alpha=0.05)
    significant_count_stat_sum = int(np.sum((bicoh2 > b2_stat_threshold) & non_redundant_mask_sum))

    search_mask_sum = physical_mask_sum & (ff1 < f1max * 1000.0) & (ff2 < f2max * 1000.0) & (ff1 >= ff2) & (ff1 >= fmin_khz * 1000.0) & (ff2 >= fmin_khz * 1000.0)

    # --- DIFFERENCE COUPLING CHANNELS (f1 - f2 = f3, with f1 > 0, f2 > 0 represented by ff1 > 0 and ff2 < 0) ---
    physical_mask_diff = (ff1 > 0) & (ff2 < 0) & (ff1 > -ff2)
    non_redundant_mask_diff = physical_mask_diff & (ff1 >= fmin_khz * 1000.0) & (-ff2 >= fmin_khz * 1000.0) & ((ff1 + ff2) >= fmin_khz * 1000.0)
    total_physical_bins_diff = int(np.sum(non_redundant_mask_diff))
    significant_count_diff = int(np.sum((bicoh2 > threshold) & non_redundant_mask_diff))
    significant_count_stat_diff = int(np.sum((bicoh2 > b2_stat_threshold) & non_redundant_mask_diff))

    search_mask_diff = physical_mask_diff & (ff1 < f1max * 1000.0) & (-ff2 < f2max * 1000.0) & (ff1 >= fmin_khz * 1000.0) & (-ff2 >= fmin_khz * 1000.0) & ((ff1 + ff2) >= fmin_khz * 1000.0)

    # Peak search window size defined in kHz, converted to bins according to the ACTUAL
    # spectral resolution of this run (instead of a fixed size in bins).
    df_khz = (f1[1] - f1[0]) / 1000.0
    filter_size_bins = max(3, int(round(peak_window_khz / df_khz)))
    if filter_size_bins % 2 == 0:
        filter_size_bins += 1  # odd size to correctly center the maximum filter

    # --- SUM PEAKS ---
    local_max_mask_sum = (bicoh2 == maximum_filter(bicoh2, size=filter_size_bins)) & search_mask_sum
    peak_coords_sum = np.argwhere(local_max_mask_sum)
    peak_vals_sum = bicoh2[local_max_mask_sum]
    sorted_idx_sum = np.argsort(peak_vals_sum)[::-1]

    sum_peaks = []
    for p_idx in sorted_idx_sum[:topn]:
        coord = peak_coords_sum[p_idx]
        val = bicoh2[coord[0], coord[1]]
        f1_val = f1[coord[0]] / 1000.0
        f2_val = f2[coord[1]] / 1000.0
        f3_val = f1_val + f2_val
        sum_peaks.append((val, f1_val, f2_val, f3_val))

    sum_max_b2, sum_best_freqs = (sum_peaks[0][0], sum_peaks[0][1:]) if sum_peaks else (0.0, (0.0, 0.0, 0.0))

    # --- DIFFERENCE PEAKS ---
    local_max_mask_diff = (bicoh2 == maximum_filter(bicoh2, size=filter_size_bins)) & search_mask_diff
    peak_coords_diff = np.argwhere(local_max_mask_diff)
    peak_vals_diff = bicoh2[local_max_mask_diff]
    sorted_idx_diff = np.argsort(peak_vals_diff)[::-1]

    diff_peaks = []
    for p_idx in sorted_idx_diff[:topn]:
        coord = peak_coords_diff[p_idx]
        val = bicoh2[coord[0], coord[1]]
        f1_val = f1[coord[0]] / 1000.0
        f2_val = -f2[coord[1]] / 1000.0  # Positive representation
        f3_val = f1_val - f2_val
        diff_peaks.append((val, f1_val, f2_val, f3_val))

    diff_max_b2, diff_best_freqs = (diff_peaks[0][0], diff_peaks[0][1:]) if diff_peaks else (0.0, (0.0, 0.0, 0.0))

    # PEAK-STABILITY CHECK: the raw argmax over a large search box is not necessarily a robust
    # physical feature -- with a large fraction of bins already above the 95% statistical
    # threshold in some channels, the single highest bin can be a noise-driven local maximum
    # that depends on how large a box it is allowed to be picked from. To surface this
    # automatically, we independently find the best peak within a smaller "core" box
    # (stability_frac of f1max/f2max, default 50%) and flag whether it agrees with the
    # full-box leader. Disagreement means the reported leader is sensitive to the arbitrary
    # search-box choice and should be treated with extra caution before being reported as a
    # finding.
    if stability_tol_khz is None:
        stability_tol_khz = peak_window_khz

    def _core_box_leader(search_mask, is_diff):
        core_mask = search_mask.copy()
        core_mask &= (ff1 < f1max * stability_frac * 1000.0)
        if is_diff:
            core_mask &= (-ff2 < f2max * stability_frac * 1000.0)
        else:
            core_mask &= (ff2 < f2max * stability_frac * 1000.0)
        if not np.any(core_mask):
            return None
        core_bicoh = np.where(core_mask, bicoh2, -np.inf)
        idx = np.unravel_index(np.argmax(core_bicoh), core_bicoh.shape)
        val = bicoh2[idx]
        f1_val = f1[idx[0]] / 1000.0
        f2_val = (-f2[idx[1]] / 1000.0) if is_diff else (f2[idx[1]] / 1000.0)
        return (val, f1_val, f2_val)

    sum_core_leader = _core_box_leader(search_mask_sum, is_diff=False)
    sum_peak_stable = (sum_core_leader is not None and sum_peaks and
                        abs(sum_core_leader[1] - sum_best_freqs[0]) <= stability_tol_khz and
                        abs(sum_core_leader[2] - sum_best_freqs[1]) <= stability_tol_khz)

    diff_core_leader = _core_box_leader(search_mask_diff, is_diff=True)
    diff_peak_stable = (diff_core_leader is not None and diff_peaks and
                         abs(diff_core_leader[1] - diff_best_freqs[0]) <= stability_tol_khz and
                         abs(diff_core_leader[2] - diff_best_freqs[1]) <= stability_tol_khz)

    return {
        'f1': f1, 'f2': f2, 'bicoh2': bicoh2, 'fs': fs, 'fnyq_khz': fnyq / 1000.0,

        # Backward compatibility aliases (returning sum values)
        'top_peaks': sum_peaks, 'max_b2': sum_max_b2, 'best_freqs': sum_best_freqs,
        'significant_count': significant_count_sum, 'total_bins': total_physical_bins_sum,
        'significant_count_stat': significant_count_stat_sum,

        # Explicit Sum coupling keys
        'sum_peaks': sum_peaks, 'sum_max_b2': sum_max_b2, 'sum_best_freqs': sum_best_freqs,
        'sum_significant_count': significant_count_sum, 'sum_total_bins': total_physical_bins_sum,
        'sum_significant_count_stat': significant_count_stat_sum,

        # Explicit Difference coupling keys
        'diff_peaks': diff_peaks, 'diff_max_b2': diff_max_b2, 'diff_best_freqs': diff_best_freqs,
        'diff_significant_count': significant_count_diff, 'diff_total_bins': total_physical_bins_diff,
        'diff_significant_count_stat': significant_count_stat_diff,

        'b2_stat_threshold': b2_stat_threshold,
        'n_samples': len(ys), 'filter_size_bins': filter_size_bins, 'df_khz': df_khz,

        # Peak-stability check results (leader found within a smaller "core" box, and whether it
        # agrees with the full-box leader reported above).
        'sum_peak_stable': sum_peak_stable, 'sum_core_leader': sum_core_leader,
        'diff_peak_stable': diff_peak_stable, 'diff_core_leader': diff_core_leader,
        'stability_frac': stability_frac, 'stability_tol_khz': stability_tol_khz,
    }


def triads_match(freqs_a, freqs_b, tol_khz):
    """Compares two triads (f1, f2, f3) within a tolerance in kHz. Since the leading peak is
    always reported with f1 >= f2, the direct component-by-component comparison is valid."""
    f1a, f2a, _ = freqs_a
    f1b, f2b, _ = freqs_b
    return abs(f1a - f1b) <= tol_khz and abs(f2a - f2b) <= tol_khz


def chance_match_probability(tol_khz, f1max_khz, f2max_khz, n_candidate_triads):
    """Estimates the probability that AT LEAST ONE of `n_candidate_triads` independently and
    uniformly distributed (f1, f2) points would fall, purely by chance, within a
    +/-tol_khz x +/-tol_khz acceptance box centered on one fixed reference triad, inside the
    rectangular search area [0, f1max_khz] x [0, f2max_khz].

    This is a coarse, order-of-magnitude estimate (it ignores the f1 >= f2 asymmetry and the
    diagonal/redundancy structure of the search space), but it is exactly the kind of "what
    would we expect under the null hypothesis of no real physical relationship" baseline that
    is already applied to the bicoherence bins themselves. It should be applied to the
    cross-diagnostic (Langmuir) matching too, since that step tests MANY candidate triads
    (multiple channels x multiple peaks x two coupling types) against a single reference.

    IMPORTANT: `n_candidate_triads` must already be an EFFECTIVE (not raw) count when the
    candidates are not independent of each other -- e.g. when they come from many spatially
    adjacent Langmuir probe tips, which are physically correlated, not independent random
    draws. See `effective_n_independent_series()` below for how that correction is computed.
    """
    box_area = (2.0 * tol_khz) * (2.0 * tol_khz)
    search_area = max(f1max_khz * f2max_khz, 1e-9)
    p_single = min(1.0, box_area / search_area)
    p_at_least_one = 1.0 - (1.0 - p_single) ** max(n_candidate_triads, 0)
    return p_single, p_at_least_one


def effective_n_independent_series(series_dict, max_lag_bins=None):
    """Estimates the EFFECTIVE number of independent tests among a set of time series that may
    be mutually correlated -- e.g. many Langmuir probe tips on the same array, which are
    physically close together and therefore see overlapping turbulence rather than independent
    signals. Treating them as N independent statistical tests (as a naive Bonferroni-style
    correction would) systematically OVER-penalizes a real, weaker signal, because the
    "effective" number of independent looks is smaller than N.

    Method (Li & Ji 2005 / Nyholt 2004, "effective number of independent tests" from genetics,
    applied here to correlated probe channels instead of correlated genetic markers): build the
    Pearson correlation matrix between every pair of series, eigen-decompose it, and sum a
    per-eigenvalue contribution f(lambda) = min(lambda, 1). Perfectly correlated (redundant)
    series contribute eigenvalues near 0 (after the first) and barely add to the count;
    perfectly independent series each contribute close to 1.

    `series_dict`: {name: 1D np.ndarray}, all series must be the same length (e.g. all sliced to
    the same burst-window time range) and have nonzero variance.

    Returns (m_eff, n_raw, corr_matrix_shape). m_eff is clipped to [1, n_raw].
    """
    names = list(series_dict.keys())
    n_raw = len(names)
    if n_raw <= 1:
        return float(n_raw), n_raw, (n_raw, n_raw)

    # Stack into a (n_raw, n_samples) matrix, dropping any series with ~zero variance (can't
    # correlate a constant signal; treat it as contributing 0 to M_eff rather than crashing).
    mat = np.array([series_dict[n] for n in names], dtype=float)
    valid = np.std(mat, axis=1) > 1e-15
    n_valid = int(np.sum(valid))
    if n_valid <= 1:
        return float(max(n_valid, 1)), n_raw, (n_raw, n_raw)
    mat = mat[valid]

    with np.errstate(invalid='ignore'):
        corr = np.corrcoef(mat)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)

    eigvals = np.linalg.eigvalsh(corr)
    eigvals = np.clip(eigvals, 0.0, None)  # numerical noise can give tiny negative eigenvalues
    m_eff = float(np.sum(np.minimum(eigvals, 1.0)))
    m_eff = min(max(m_eff, 1.0), float(n_valid))
    return m_eff, n_raw, corr.shape


def print_channel_result(channel_name, result, threshold):
    """Prints BOTH the Sum-coupling and Diff-coupling statistics for a channel, each
    normalized by its OWN bin-domain size, and explicitly marks which one is the channel's
    dominant/reported type.

    Printing both, always, keeps visible that the Sum and Diff domains are not necessarily
    built with the same redundancy/deduplication convention and are generally different sizes,
    so a percentage computed on one domain is not automatically comparable to a percentage
    computed on the other.
    """
    dominant = 'Diff' if result['diff_max_b2'] > result['sum_max_b2'] else 'Sum'
    for label, key in (('Sum', 'sum'), ('Diff', 'diff')):
        max_b2 = result[f'{key}_max_b2']
        f1v, f2v, f3v = result[f'{key}_best_freqs']
        sig = result[f'{key}_significant_count']
        sig_stat = result[f'{key}_significant_count_stat']
        total = result[f'{key}_total_bins']
        pct_phys = sig / total * 100.0 if total else 0.0
        pct_stat = sig_stat / total * 100.0 if total else 0.0
        marker = "  <- dominant/reported type" if label == dominant else ""
        sign = '+' if label == 'Sum' else '-'
        log.debug(f"Channel {channel_name:<12} ({label:<4}) -> Max b\u00b2 = {max_b2:.4f} at "
              f"f1={f1v:.1f} {sign} f2={f2v:.1f} kHz | "
              f"{sig}/{total} ({pct_phys:.2f}%) > physical threshold ({threshold}) | "
              f"{sig_stat}/{total} ({pct_stat:.2f}%) > 95% statistical threshold "
              f"({result['b2_stat_threshold']:.3f}){marker}")
    if result['sum_total_bins'] != result['diff_total_bins']:
        ratio = result['diff_total_bins'] / result['sum_total_bins'] if result['sum_total_bins'] else float('nan')
        log.debug(f"     \u26a0\ufe0f [Domain-size caveat] Sum domain = {result['sum_total_bins']} bins vs. "
              f"Diff domain = {result['diff_total_bins']} bins (ratio {ratio:.2f}x). Percentages above "
              f"are each self-normalized; do NOT directly compare a Diff-domain percentage against "
              f"a Sum-domain percentage from another channel without confirming both domains use "
              f"an equivalent non-redundant (deduplicated) convention.")


def run_single_shot(shot_id, args):
    import copy
    args = copy.deepcopy(args)
    shot_str = str(shot_id)

    # Adapt args for this specific shot
    args.pattern = args.pattern.replace("{SHOT}", shot_str).replace("88652", shot_str)
    args.langmuir_pattern = args.langmuir_pattern.replace("{SHOT}", shot_str).replace("88652", shot_str)

    # Avoid self-reproducibility by changing the validation shot if it's the same
    if args.validation_shot == shot_str:
        args.validation_shot = "88654" if shot_str != "88654" else "88652"
    args.validation_dir = args.validation_dir.replace("88654", args.validation_shot)

    # Append shot number to window plot path
    wp_path = Path(args.window_plot_path)
    args.window_plot_path = str(wp_path.with_name(f"{wp_path.stem}_{shot_id}{wp_path.suffix}"))

    noverlap_frac = BICOH_NOVERLAP_FRAC  # [M9][BICOH-FLOOR] shared with compute_flat_frequency_window()

    # Dynamically extract the base directory and shot ID from the main pattern
    pattern_path = Path(args.pattern)
    base_dir = pattern_path.parent
    shot_match = re.search(r'@(\d+)', pattern_path.name)
    shot_id = shot_match.group(1) if shot_match else f"{SHOT}"

    # Effective time windows for the main analysis (ranking) and for the Burst/Quiescent/Control
    # comparison. Burst, Quiescent, and Control are resolved independently: each is taken from
    # its manual --*-start/--*-end pair if BOTH were given, otherwise from a single shared
    # W_p-based auto-detection pass (unless --no-auto-burst forces manual-only mode).
    quiescent_start, quiescent_end = args.quiescent_start, args.quiescent_end
    control_start, control_end = args.control_start, args.control_end

    flat_window_applied = False
    flat_info = None
    det = None
    if args.full_discharge:
        main_t_start, main_t_end = None, None
        log.debug("\u26a0\ufe0f --full-discharge mode active: the main ranking will use the FULL shot, "
              "mixing distinct physical windows.")
        # Quiescent/Control still need concrete bounds for the later window comparison; if they
        # weren't given manually, fall through to auto-detection for those two only.
        need_auto = (quiescent_start is None or quiescent_end is None or
                     control_start is None or control_end is None)
    else:
        need_auto = False
        if args.burst_start is not None and args.burst_end is not None:
            raw_t_start, raw_t_end = args.burst_start, args.burst_end
            log.debug(f"--- Burst window: MANUALLY specified as {raw_t_start:.0f}-{raw_t_end:.0f} ms ---")
            main_t_start, main_t_end, flat_window_applied, flat_info = refine_burst_with_mode_active_window(
                raw_t_start, raw_t_end, base_dir, shot_id, args)
            if flat_window_applied:
                log.info(f"[M9] Mode-active (flat-frequency) sub-window within the manually-specified "
                         f"{raw_t_start:.1f}-{raw_t_end:.1f} ms burst: {main_t_start:.1f}-{main_t_end:.1f} ms "
                         f"({flat_info['duration_ms']:.1f} ms) -- this is the window used below for the "
                         "bicoherence calculations and figure.")
            else:
                log.info(f"[M9] No mode-active sub-window applied ({flat_info.get('reason', 'n/a')}); using "
                         f"the manually-specified burst window as-is: {main_t_start:.1f}-{main_t_end:.1f} ms.")
        else:
            need_auto = True
        if quiescent_start is None or quiescent_end is None or control_start is None or control_end is None:
            need_auto = True

    if need_auto and args.no_auto_burst:
        log.info("\u274c ABORTING: --no-auto-burst was set but at least one of --burst-start/--burst-end, "
                 "--quiescent-start/--quiescent-end, --control-start/--control-end was not fully "
                 "provided. Cannot determine all required windows.")
        return

    if need_auto and not args.no_auto_burst:
        wp_file_detect = base_dir / f"Wp@{shot_id}.edf"
        log.debug(f"--- Burst/Quiescent/Control windows: AUTO-DETECTING from {wp_file_detect} "
              f"(k_on={args.auto_burst_k_on}, k_off={args.auto_burst_k_off}, "
              f"min_duration={args.auto_burst_min_duration_ms:.0f} ms) ---")
        if not wp_file_detect.exists():
            log.info(f"\u274c ABORTING: cannot auto-detect windows, {wp_file_detect} not found. "
                     f"Supply all window bounds manually, or check --pattern/data directory.")
            return
        det = detect_hmode_burst_window(
            wp_file_detect, baseline_end_ms=args.auto_burst_baseline_ms,
            k_on=args.auto_burst_k_on, k_off=args.auto_burst_k_off,
            min_duration_ms=args.auto_burst_min_duration_ms,
            quiescent_duration_ms=args.auto_quiescent_duration_ms,
            control_duration_ms=args.auto_control_duration_ms,
            window_margin_ms=args.auto_window_margin_ms,
            flat_percentile=args.auto_flat_percentile, flat_k=args.auto_flat_k,
            robust_baseline=not args.no_robust_baseline,
            diagnostic_plot_path=None if args.no_window_plot else args.window_plot_path,
            # [M9-PLOT] Defer the actual save: at this point the [M9] mode-active sub-window
            # hasn't been computed yet (it depends on `det['start_ms']`/`det['end_ms']` below), so
            # saving here would produce the same M9-unaware plot as before. The deferred save
            # after the window-resolution block (search for [M9-PLOT] further down) renders it
            # once, WITH the M9 sub-window included, instead of saving it twice.
            defer_plot=True)
        if not det['ok']:
            log.info(f"\u274c ABORTING: auto-detection FAILED ({det.get('reason', 'unknown reason')}). "
                     f"Supply all window bounds manually.")
            return

        log.debug(f"   Baseline W_p (first {args.auto_burst_baseline_ms:.0f} ms): "
              f"mean={det['baseline_mean']:.3f}, std={det['baseline_std']:.3f}")
        log.debug(f"   Onset threshold (k_on={args.auto_burst_k_on}) = {det['onset_threshold']:.3f} | "
              f"End threshold (k_off={args.auto_burst_k_off}) = {det['end_threshold']:.3f}")

        if not args.full_discharge and (args.burst_start is None or args.burst_end is None):
            raw_t_start, raw_t_end = det['start_ms'], det['end_ms']
            log.info(f"Detected burst window: {raw_t_start:.1f}-{raw_t_end:.1f} ms "
                     f"(peak W_p in window \u2248 {det.get('wp_peak', float('nan')):.3f})")
            main_t_start, main_t_end, flat_window_applied, flat_info = refine_burst_with_mode_active_window(
                raw_t_start, raw_t_end, base_dir, shot_id, args)
            if flat_window_applied:
                log.info(f"[M9] Mode-active (flat-frequency) sub-window within the detected "
                         f"{raw_t_start:.1f}-{raw_t_end:.1f} ms burst: {main_t_start:.1f}-{main_t_end:.1f} ms "
                         f"({flat_info['duration_ms']:.1f} ms) -- this is the window used below for the "
                         "bicoherence calculations and figure (NOT the raw W_p burst window above).")
            else:
                log.info(f"[M9] No mode-active sub-window applied ({flat_info.get('reason', 'n/a')}); using "
                         f"the full detected burst window as-is: {main_t_start:.1f}-{main_t_end:.1f} ms.")

        if quiescent_start is None or quiescent_end is None:
            quiescent_start, quiescent_end = det['quiescent_start_ms'], det['quiescent_end_ms']
            if quiescent_start is None:
                log.info("\u274c Quiescent window: NO candidate window could be evaluated at all "
                      "(the burst starts too close to the beginning of the recording for even "
                      "one window to fit before it). Supply --quiescent-start/--quiescent-end "
                      "manually, or reduce --auto-quiescent-duration-ms.")
            elif det['quiescent_ok']:
                log.debug(f"   Detected quiescent window: {quiescent_start:.1f}-{quiescent_end:.1f} ms "
                      f"(nearest flat, sub-threshold segment found before burst onset; "
                      f"mean={det['quiescent_mean']:.3f}, std={det['quiescent_std']:.3f})")
            else:
                log.info(f"\u26a0\ufe0f Quiescent window: NO segment before the burst fully satisfied the "
                      f"flatness/level criteria. Using the best available candidate as a fallback: "
                      f"{quiescent_start:.1f}-{quiescent_end:.1f} ms. Treat with extra caution -- check "
                      f"the diagnostic plot or supply --quiescent-start/--quiescent-end manually.")

        if control_start is None or control_end is None:
            control_start, control_end = det['control_start_ms'], det['control_end_ms']
            if control_start is None:
                log.info("\u274c Control window: NO candidate window could be evaluated at all "
                      "(the burst ends too close to the end of the recording for even one window "
                      "to fit after it). Supply --control-start/--control-end manually, or reduce "
                      "--auto-control-duration-ms.")
            elif det['control_ok']:
                log.debug(f"   Detected control window: {control_start:.1f}-{control_end:.1f} ms "
                      f"(nearest flat, sub-threshold segment found after burst end; "
                      f"mean={det['control_mean']:.3f}, std={det['control_std']:.3f})")
            else:
                log.info(f"\u26a0\ufe0f Control window: NO segment after the burst fully satisfied the "
                      f"flatness/level criteria. Using the best available candidate as a fallback: "
                      f"{control_start:.1f}-{control_end:.1f} ms. Treat with extra caution -- check "
                      f"the diagnostic plot or supply --control-start/--control-end manually.")

        log.debug("   These are HEURISTIC threshold-crossing detections on W_p only (no D-alpha/"
              "H-alpha cross-check). Visually verify all three windows against the W_p trace (and any "
              "edge-turbulence diagnostic you have) before treating them as final -- override any of "
              "them individually with --burst-start/--burst-end, --quiescent-start/--quiescent-end, "
              "--control-start/--control-end if they look wrong.")
    else:
        if quiescent_start is not None and quiescent_end is not None:
            log.debug(f"--- Quiescent window: MANUALLY specified as {quiescent_start:.0f}-{quiescent_end:.0f} ms ---")
        if control_start is not None and control_end is not None:
            log.debug(f"--- Control window: MANUALLY specified as {control_start:.0f}-{control_end:.0f} ms ---")

    # [M9-PLOT] Now that the [M9] mode-active (flat-frequency) sub-window (if any) is known, render
    # and save the W_p window-detection diagnostic PNG WITH it overlaid -- this is the plot
    # detect_hmode_burst_window() itself deferred above via defer_plot=True, specifically so this
    # sub-window could be added to it instead of it being saved one step too early, M9-unaware.
    # Only applies when auto-detection actually ran (an all-manual burst/quiescent/control run
    # never touches Wp@<shot>.edf and has no W_p trace to plot in the first place).
    if det is not None and det.get('ok') and not args.no_window_plot:
        fw_span = None
        if not args.full_discharge and flat_window_applied and flat_info is not None:
            fw_note = "extended past flatness tolerance to meet bicoherence sample floor" \
                if flat_info.get('relaxed_for_min_duration') else ""
            fw_span = (main_t_start, main_t_end, fw_note)
        _plot_wp_window_detection(
            det['t_ms'], det['ys'], det['ys_smooth'], det.get('baseline_end_ms', args.auto_burst_baseline_ms),
            det['baseline_mean'], det['baseline_std'], det['onset_threshold'], det['end_threshold'],
            det, args.window_plot_path, flat_window=fw_span)

    b2_stat_threshold_preview = statistical_b2_threshold(args.ensemble, alpha=0.05)

    log.debug(f"--- Searching for diagnostic channels with pattern: {args.pattern} ---")
    # On Windows, glob.glob() is case-insensitive by default.
    # We explicitly filter to ensure case-sensitive behavior.
    raw_files = glob.glob(args.pattern)
    pattern_filename = Path(args.pattern).name
    files = sorted([f for f in raw_files if fnmatch.fnmatchcase(Path(f).name, pattern_filename)])
    files = sorted([f for f in files if MIRNOV_NAME_RE.match(Path(f).name.split('@')[0])])

    if not files:
        log.info(f"\u26a0\ufe0f No files found for pattern '{args.pattern}' -- falling back to default channel.")
        files = [f"data/hj{SHOT}/MP1@{SHOT}.edf"]
        log.debug(f"Using default channel: {files[0]}")
    else:
        log.debug(f"Channels found ({len(files)}): {[Path(f).name.split('@')[0] for f in files]}")
        log.debug("(Verify that all of them actually correspond to Mirnov coils before interpreting the results.)")

    if not args.full_discharge:
        flat_tag = " [M9 flat sub-window]" if flat_window_applied else ""
        log.info(f"Analysis window (burst){flat_tag}: {main_t_start:.0f}-{main_t_end:.0f} ms "
              f"(use --full-discharge to revert to full-shot behavior).")
    log.debug(f"Rigorous 95% statistical threshold (N_ensemble={args.ensemble}): "
          f"b\u00b2_95 = {b2_stat_threshold_preview:.3f} (vs. physical threshold = {args.threshold})")

    # Multiple-comparisons context: under H0, we would expect by chance that ~alpha% of the
    # bins/points exceed the 95% statistical threshold -by construction, alpha=5%-. All the
    # percentages reported below (main ranking, window comparison) must be read in comparison
    # with that baseline, not in absolute terms.
    alpha_sig = 0.05
    chance_level_pct = alpha_sig * 100.0
    log.debug(f"NOTE (multiple-comparisons context): under the null hypothesis of total")
    log.debug(f"     independence, we would expect by pure chance that ~{chance_level_pct:.1f}% of the bins exceed b\u00b2_95.")
    log.debug(f"     The '>Stat.95%' percentages reported below should be compared against that")
    log.debug(f"     baseline of {chance_level_pct:.1f}%, not interpreted in absolute terms.")

    results_ranking = []
    best_channel_data = None
    best_channel_name = None
    best_b2_max = -1.0
    psd_confirmation_by_peak = {}  # safe default in case best_channel_data ends up None

    log.debug(f"\nProcessing {len(files)} channels simultaneously for bicoherence (nensemble={args.ensemble})...")

    for fpath in files:
        fpath = Path(fpath)
        channel_name = fpath.name.split('@')[0]
        check_channel_identity(channel_name)

        try:
            result = analyze_channel_bicoherence(fpath, args.nfft, noverlap_frac, args.ensemble,
                                                  args.threshold, args.f1max, args.f2max, args.topn,
                                                  t_start_ms=main_t_start, t_end_ms=main_t_end,
                                                  peak_window_khz=args.peak_window_khz,
                                                  fmin_khz=args.fmin, stability_frac=args.stability_frac)

            log.debug(f"Channel {channel_name}: fs = {result['fs']/1e6:.3f} MHz, nfft = {args.nfft}, "
                  f"N_window_samples = {result['n_samples']} "
                  f"-> Spectral resolution df = {result['df_khz']:.3f} kHz")

            coupling_type = 'Diff' if result['diff_max_b2'] > result['sum_max_b2'] else 'Sum'
            if coupling_type == 'Diff':
                max_b2 = result['diff_max_b2']
                freqs_khz = result['diff_best_freqs']
                significant_couplings = result['diff_significant_count']
                significant_couplings_stat = result['diff_significant_count_stat']
                total_bins = result['diff_total_bins']
                top_peaks = result['diff_peaks']
            else:
                max_b2 = result['sum_max_b2']
                freqs_khz = result['sum_best_freqs']
                significant_couplings = result['sum_significant_count']
                significant_couplings_stat = result['sum_significant_count_stat']
                total_bins = result['sum_total_bins']
                top_peaks = result['sum_peaks']

            # RMS amplitude of this channel in the analysis window, used later for the
            # cross-channel amplitude-anomaly sanity check (see after the main loop).
            t_sec_amp, ys_amp, _, _ = load_edf_signal(fpath)
            t_sec_amp, ys_amp = slice_window(t_sec_amp, ys_amp, main_t_start, main_t_end)
            rms_amplitude = float(np.sqrt(np.mean(np.square(ys_amp)))) if len(ys_amp) else float('nan')

            results_ranking.append({
                'channel': channel_name,
                'max_b2': max_b2,
                'freqs_khz': freqs_khz,
                'significant_couplings': significant_couplings,
                'significant_couplings_stat': significant_couplings_stat,
                'total_bins': total_bins,
                'top_peaks': top_peaks,
                'coupling_type': coupling_type,
                'rms_amplitude': rms_amplitude,
                'sum_significant_couplings': result['sum_significant_count'],
                'sum_significant_couplings_stat': result['sum_significant_count_stat'],
                'sum_total_bins': result['sum_total_bins'],
                'diff_significant_couplings': result['diff_significant_count'],
                'diff_significant_couplings_stat': result['diff_significant_count_stat'],
                'diff_total_bins': result['diff_total_bins'],
                'peak_stable': result['diff_peak_stable'] if coupling_type == 'Diff' else result['sum_peak_stable'],
                'core_leader': result['diff_core_leader'] if coupling_type == 'Diff' else result['sum_core_leader'],
                'stability_frac': result['stability_frac'],
            })

            # Print BOTH Sum and Diff statistics transparently (see print_channel_result docstring)
            print_channel_result(channel_name, result, args.threshold)

            # Store the best channel (highest max b^2) to plot it in detail
            if max_b2 > best_b2_max:
                best_b2_max = max_b2
                best_channel_name = channel_name

                # We compute the linear PSD IN THE SAME TIME WINDOW used for the bicoherence,
                # so that the PSD panel is physically consistent with the bicoherence panel.
                t_sec_best, ys_best, dt_best, fs_best = load_edf_signal(fpath)
                t_sec_best, ys_best = slice_window(t_sec_best, ys_best, main_t_start, main_t_end)
                f_psd, Pxx = LAS.psd(ys_best, t_sec_best, dt=dt_best, nfft=args.nfft, noverlap=args.nfft // 2,
                                     nensemble=args.ensemble, detrend='linear')

                best_channel_data = dict(result)
                best_channel_data['channel'] = channel_name
                best_channel_data['fpath'] = fpath
                best_channel_data['f_psd'] = f_psd
                best_channel_data['Pxx'] = Pxx
                best_channel_data['coupling_type'] = coupling_type
                best_channel_data['max_b2'] = max_b2
                best_channel_data['best_freqs'] = freqs_khz
                best_channel_data['significant_count'] = significant_couplings
                best_channel_data['significant_count_stat'] = significant_couplings_stat
                best_channel_data['total_bins'] = total_bins
                best_channel_data['top_peaks'] = top_peaks

        except Exception as e:
            log.debug(f"Error processing channel {channel_name}: {e}")

    # Automated amplitude-anomaly sanity check across the genuine MP channels.
    # A channel whose RMS amplitude in the analysis window differs drastically from its
    # array-mates (e.g. >5x the median) could still be a real physical effect (different
    # coupling to a given mode at different toroidal locations), but it is also exactly the
    # signature of a calibration/gain mismatch on that one channel -- something that should be
    # confirmed against the diagnostic documentation before a channel is singled out as "the"
    # validated result in the thesis, rather than assumed.
    AMPLITUDE_ANOMALY_RATIO = 5.0
    amplitude_outlier_channels = []  # populated below; referenced later in the double-validation summary
    amp_entries = [(r['channel'], r['rms_amplitude']) for r in results_ranking if np.isfinite(r.get('rms_amplitude', np.nan))]
    if len(amp_entries) >= 2:
        amps = np.array([a for _, a in amp_entries])
        median_amp = float(np.median(amps))
        outliers = [(ch, a, a / median_amp) for ch, a in amp_entries
                    if median_amp > 0 and (a / median_amp >= AMPLITUDE_ANOMALY_RATIO or a / median_amp <= 1.0 / AMPLITUDE_ANOMALY_RATIO)]
        amplitude_outlier_channels = [ch for ch, _, _ in outliers]
        for ch, a in amp_entries:
            log.debug(f"   * {ch}: RMS = {a:.4g} V")
        if outliers:
            log.info(f"\u26a0\ufe0f [Amplitude sanity check] the following channel(s) deviate from the array "
                     f"median RMS ({median_amp:.4g} V) by more than {AMPLITUDE_ANOMALY_RATIO:.0f}x -- could be "
                     f"real physics or a calibration/gain difference; verify before treating as validated:")
            for ch, a, ratio in outliers:
                direction = "above" if ratio >= 1.0 else "below"
                log.info(f"     - {ch}: RMS = {a:.4g} V ({ratio:.1f}x the median, {direction} it)")

    # 2a. Print the Nonlinear Coupling Ranking (f1 +/- f2 = f3)
    results_ranking.sort(key=lambda x: x['max_b2'], reverse=True)

    log.info("\n" + "=" * 110)
    log.info("MULTICHANNEL NONLINEAR COUPLING RANKING (f1 \u00b1 f2 = f3)")
    if not args.full_discharge:
        log.info(f"   Window analyzed: BURST ({main_t_start:.0f}-{main_t_end:.0f} ms)")
    log.debug(f"   (Leading peak search restricted to f1 < {args.f1max} kHz and f2 < {args.f2max} kHz)")
    log.info("=" * 110)
    log.info(f"{'Rank':<8}{'Channel':<12}{'Mode':<8}{'Max b\u00b2':<12}{'f1(kHz)':<10}{'f2(kHz)':<10}{'f3(kHz)':<12}"
             f"{'>Phys.(%)':<12}{'>Stat.95%(%)':<14}")
    log.info("-" * 110)
    for idx, r in enumerate(results_ranking):
        pct_phys = (r['significant_couplings'] / r['total_bins']) * 100.0
        pct_stat = (r['significant_couplings_stat'] / r['total_bins']) * 100.0
        log.info(f"{idx+1:<8}{r['channel']:<12}{r['coupling_type']:<8}{r['max_b2']:<12.4f}{r['freqs_khz'][0]:<10.1f}"
                 f"{r['freqs_khz'][1]:<10.1f}{r['freqs_khz'][2]:<12.1f}{pct_phys:<12.2f}{pct_stat:<14.2f}")

        p_strs = []
        for p_idx, p in enumerate(r['top_peaks'][1:]):  # We skip the absolute peak already listed in the main row
            sign = "+" if r['coupling_type'] == 'Sum' else "-"
            p_strs.append(f"#{p_idx+2}: b\u00b2={p[0]:.3f} @ {p[1]:.1f}{sign}{p[2]:.1f}={p[3]:.1f}kHz")
        if p_strs:
            log.debug(f"        \u2514\u2500 Other local peaks: {', '.join(p_strs)}")

        if idx == 0 and r.get('top_peaks'):
            sign_str = "+" if r['coupling_type'] == 'Sum' else "-"
            log.info("\n" + "=" * 115)
            log.info(f"TOP {len(r['top_peaks'])} LOCAL NONLINEAR COUPLINGS ON LEADING CHANNEL ({r['channel']})")
            log.info("=" * 115)
            log.info(f"{'Rank':<6}{'b\u00b2':<8}{'f1 (kHz)':<11}{'f2 (kHz)':<11}{'f3 (kHz)':<11}{'Slope m':<12}{'Triad Equation':<26}{'Physical Classification':<25}")
            log.info("-" * 115)
            for p_rank, p in enumerate(r['top_peaks'], 1):
                pf1, pf2, pf3 = p[1], p[2], p[3]
                m_slope = -pf2 / pf1 if r['coupling_type'] == 'Diff' else pf2 / pf1
                triad_eq = f"{pf1:.1f} {sign_str} {pf2:.1f} = {pf3:.1f} kHz"
                if abs(m_slope - (-0.5 if r['coupling_type'] == 'Diff' else 0.5)) < 0.035:
                    classification = "Subharmonic (m \u2248 -0.5)"
                elif abs(pf1 - 89.0) < 6.0 or abs(pf2 - 89.0) < 6.0:
                    classification = "Primary Pump Decay (89 kHz)"
                elif abs(pf1 - 41.0) < 4.0 or abs(pf2 - 41.0) < 4.0:
                    classification = "Secondary Mode Coupling"
                else:
                    classification = "Broadband Non-Linear Mode"
                log.info(f"#{p_rank:<5}{p[0]:<8.4f}{pf1:<11.1f}{(-pf2 if r['coupling_type']=='Diff' else pf2):<11.1f}{pf3:<11.1f}{m_slope:<12.4f}{triad_eq:<26}{classification:<25}")
            log.info("-" * 115 + "\n")

        # Boundary-proximity warning: if the leading peak sits close to the f1max/f2max edge of
        # the search window, it may simply be the best point INSIDE an artificially truncated
        # box rather than the true global maximum of the (f1,f2) plane.
        boundary_margin_khz = 2.0 * args.peak_window_khz
        f1_r, f2_r = r['freqs_khz'][0], r['freqs_khz'][1]
        near_f1_edge = (args.f1max - f1_r) <= boundary_margin_khz
        near_f2_edge = (args.f2max - f2_r) <= boundary_margin_khz
        if near_f1_edge or near_f2_edge:
            which = " and ".join([n for n, cond in (("f1", near_f1_edge), ("f2", near_f2_edge)) if cond])
            log.info(f"  \u26a0\ufe0f [{r['channel']}] Boundary-proximity warning: the leading peak's {which} is within "
                     f"{boundary_margin_khz:.1f} kHz of the search boundary (f1max={args.f1max:.0f}, "
                     f"f2max={args.f2max:.0f} kHz) -- rerun with a larger --f1max/--f2max to confirm this isn't "
                     f"a boundary artifact before reporting it.")

        # Peak-stability warning: does the leader found inside a smaller "core" box (a fraction
        # of the full search box) agree with the leader reported above? If not, the reported
        # leader is sensitive to the arbitrary choice of --f1max/--f2max rather than being a
        # robust feature.
        if not r.get('peak_stable', True):
            core_leader = r.get('core_leader')
            core_pct = r.get('stability_frac', 0.5) * 100.0
            if core_leader is not None:
                core_sign = "+" if r['coupling_type'] == 'Sum' else "-"
                log.info(f"  \u26a0\ufe0f [{r['channel']}] Peak-stability warning: leading peak does NOT agree with "
                         f"the best peak inside a smaller core box (innermost {core_pct:.0f}% of f1max/f2max), "
                         f"which instead points to f1={core_leader[1]:.1f} {core_sign} f2={core_leader[2]:.1f} kHz "
                         f"(b\u00b2={core_leader[0]:.3f}) -- sensitive to --f1max/--f2max choice, treat with caution.")
            else:
                log.info(f"  \u26a0\ufe0f [{r['channel']}] Peak-stability warning: no comparable peak found inside a "
                         f"smaller core search box; treat the reported leader with extra caution.")

    # 2b. Detailed bicoherence + PSD plot for the leading channel
    if best_channel_data is not None:
        f1_plot = best_channel_data['f1'] / 1000.0
        f2_plot = best_channel_data['f2'] / 1000.0
        ff1_plot, ff2_plot = np.meshgrid(f1_plot, f2_plot, indexing='ij')
        fnyq_khz = best_channel_data['fnyq_khz']
        window_label = ("FULL DISCHARGE" if args.full_discharge
                         else f"Burst {main_t_start:.0f}-{main_t_end:.0f} ms" +
                              (" [M9 flat sub-window]" if flat_window_applied else ""))

        log.debug(f"\n--- Generating Bicoherence Plot for the Best Channel: {best_channel_data['channel']} ({window_label}) ---")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # --- PANEL 1: Squared Auto-Bicoherence b^2 ---
        # [COLOR-FIX v2] contourf(..., levels=100) with no vmin/vmax auto-scales the color axis to
        # the min/max of the ENTIRE bicoh2 array. v1 of this fix only excluded the near-DC/
        # unphysical bins (fmin floor + correct sum/diff half-plane) but did NOT bound the region
        # by frequency -- so a noise-driven bin anywhere between fmin and the Nyquist frequency
        # (including well outside the +/-100 kHz box the ranking/peak-search actually uses, and
        # even outside the +/-150 kHz this panel visually crops to) could still set color_vmax,
        # which is exactly why the colorbar barely moved (0.648 -> 0.6567) between runs. Fix:
        # scale the color axis to the max b^2 found within the EXACT SAME search box
        # analyze_channel_bicoherence() uses to find/rank the peaks shown here (search_mask_sum /
        # search_mask_diff there: fmin <= f1,f2 < f1max/f2max, correct sum/diff half-plane, and
        # f1+f2 < Nyquist for the sum case) -- so color_vmax is guaranteed to be driven by the same
        # peaks that are actually being reported and annotated, not by noise in some unranked
        # corner of the spectrum. Floored by the physical/statistical thresholds so both reference
        # lines stay visible, and anything above color_vmax (including activity outside the search
        # box, e.g. the visible-but-unranked 100-150 kHz band) is clipped into the top color bin
        # via extend='max' (a triangle on the colorbar) instead of rescaling everything else.
        PLOT_FMAX_KHZ = 150.0  # visual crop only (set_xlim/set_ylim below) -- NOT used for color scaling
        is_diff = best_channel_data['coupling_type'] == 'Diff'
        if is_diff:
            display_mask = (ff1_plot > 0) & (ff2_plot < 0) & (ff1_plot > -ff2_plot) & \
                            (ff1_plot >= args.fmin) & (-ff2_plot >= args.fmin) & \
                            (ff1_plot < args.f1max) & (-ff2_plot < args.f2max)
        else:
            display_mask = (ff1_plot > 0) & (ff2_plot > 0) & \
                            (ff1_plot >= args.fmin) & (ff2_plot >= args.fmin) & \
                            (ff1_plot < args.f1max) & (ff2_plot < args.f2max) & \
                            ((ff1_plot + ff2_plot) < fnyq_khz)
        if np.any(display_mask):
            color_vmax = float(np.nanmax(best_channel_data['bicoh2'][display_mask]))
        else:
            color_vmax = float(np.nanmax(best_channel_data['bicoh2']))
        # [BUGFIX] The previous version of this fix floored color_vmax at
        # max(args.threshold * 1.2, b2_stat_threshold * 1.2) "so the reference threshold lines
        # stay visible" -- but args.threshold is the PHYSICAL significance threshold (0.5 by
        # default), and 0.5*1.2=0.6 is well above the actual data in shots like this one where no
        # bin reaches physical significance (leading peak b^2=0.44 < 0.5). That floor silently
        # re-introduced the exact washing-out this fix exists to remove. There is no need to force
        # the color axis to include a threshold value the data never reaches -- if a threshold
        # isn't crossed anywhere in the search box, ax1.contour(levels=[args.threshold]) below
        # simply won't draw a line, which is the scientifically correct outcome. Only pad
        # color_vmax a little above the ACTUAL data max (for headroom against the colorbar's own
        # top tick) and against the statistical threshold ONLY as a sanity floor for degenerate
        # near-zero cases, not as a hard multiplier that can dominate real data.
        color_vmax = max(color_vmax * 1.05, best_channel_data['b2_stat_threshold'], 1e-6)
        color_levels = np.linspace(0.0, color_vmax, 101)

        c = ax1.contourf(ff1_plot, ff2_plot, best_channel_data['bicoh2'], cmap='inferno',
                          levels=color_levels, extend='max')
        ax1.set_title(f"Squared Auto-Bicoherence $b^2$ - Leading Channel: {best_channel_data['channel']} "
                      f"(nensemble={args.ensemble}, {window_label})")
        ax1.set_xlabel("$f_1$ (kHz)")
        ax1.set_ylabel("$f_2$ (kHz)")
        ax1.contour(ff1_plot, ff2_plot, best_channel_data['bicoh2'], levels=[args.threshold], colors='white', linestyles='dashed')
        cbar = fig.colorbar(c, ax=ax1, label="Squared Bicoherence $b^2$")
        cbar.ax.axhline(best_channel_data['b2_stat_threshold'], color='cyan', ls=':', lw=1.5)
        ax1.grid(True, alpha=0.3, linestyle=':')
        ax1.set_xlim(0, PLOT_FMAX_KHZ)
        if best_channel_data['coupling_type'] == 'Diff':
            ax1.set_ylim(-PLOT_FMAX_KHZ, 0.0)
            ax1.plot([0, PLOT_FMAX_KHZ], [0, -0.5 * PLOT_FMAX_KHZ], color='cyan', ls='-.', lw=1.6, alpha=0.95,
                     label=r"Subharmonic Decay $f_2 = -0.5 f_1$ (Slope $-0.5$)")
            ax1.axvline(89.0, color='gold', ls='--', lw=1.2, alpha=0.75, label=r"Pump $f_1 = 89$ kHz")
            ax1.legend(loc='lower left', fontsize=8.0, framealpha=0.88, facecolor='white', edgecolor='gray')
        else:
            ax1.set_ylim(0, PLOT_FMAX_KHZ)
            ax1.plot([0, PLOT_FMAX_KHZ], [PLOT_FMAX_KHZ, 0], color='white', ls='--', lw=1.3, alpha=0.8,
                     label=r"Diagonal $f_1 + f_2 = 150$ kHz (Slope $-1$)")
            ax1.plot([0, 89.0], [89.0, 0], color='cyan', ls='-.', lw=1.4, alpha=0.8,
                     label=r"Sum $f_1 + f_2 = 89$ kHz (Slope $-1$)")
            ax1.axvline(89.0, color='gold', ls='--', lw=1.2, alpha=0.7, label=r"Pump $f_1 = 89$ kHz")
            ax1.legend(loc='lower left', fontsize=8.0, framealpha=0.88, facecolor='white', edgecolor='gray')

        offsets_map = {
            1: (8, 6),
            2: (-22, -10),
            3: (8, 6),
            4: (-20, 10),
            5: (6, 8),
            6: (8, 6),
            7: (-20, 8),
            8: (8, -12),
            9: (-22, 8),
            10: (8, -12)
        }
        best_rank_entry = next((r for r in results_ranking if r['channel'] == best_channel_data['channel']), None)
        best_f = best_channel_data['best_freqs']

        if best_rank_entry:
            for idx, p in enumerate(best_rank_entry['top_peaks']):
                y_coord = -p[2] if best_channel_data['coupling_type'] == 'Diff' else p[2]
                off_x, off_y = offsets_map.get(idx + 1, (8, 6))
                ax1.annotate(f"#{idx+1}", xy=(p[1], y_coord), xytext=(p[1] + off_x, y_coord + off_y),
                             color='white', fontsize=9.0, fontweight='bold', zorder=6,
                             arrowprops=dict(arrowstyle="->", color='white', lw=1.1, alpha=0.85, shrinkA=2, shrinkB=3),
                             path_effects=[path_effects.withStroke(linewidth=2.5, foreground="black")])

        # Compact structured info badge in lower right
        badge_text = (
            f"Leading: f1={best_f[0]:.1f}, f2={best_f[1]:.1f}, f3={best_f[2]:.1f} kHz | b\u00b2={best_channel_data['max_b2']:.3f}\n"
            f"95% Stat. Floor: b\u00b2 = {best_channel_data['b2_stat_threshold']:.3f}\n\n"
            "Non-Linear Triad Clustering:\n"
            "\u2022 Subharmonic Line (Slope -0.5):\n"
            "  Peaks #1, #2, #4, #5, #8, #9, #10\n"
            "\u2022 Pump Column (f1 = 89 kHz):\n"
            "  Peak #3 (89.8 - 41.0 = 48.8 kHz)\n"
            "  Peak #7 (84.0 - 23.4 = 60.5 kHz)"
        )
        ax1.text(0.98, 0.03, badge_text, transform=ax1.transAxes, fontsize=8.2,
                 va="bottom", ha="right", bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray", alpha=0.9))

        # --- PANEL 2: Linear Power Spectral Density (PSD) (same time window) ---
        f_psd_khz = best_channel_data['f_psd'] / 1000.0
        Pxx_val = best_channel_data['Pxx']
        mask_psd = (f_psd_khz >= 0) & (f_psd_khz <= 150.0)

        ax2.semilogy(f_psd_khz[mask_psd], Pxx_val[mask_psd], color='tab:blue', linewidth=1.8,
                     label=f"PSD {best_channel_data['channel']} ({window_label})")
        ax2.set_title(f"Power Spectral Density (PSD) of Leading Channel: {best_channel_data['channel']}", fontsize=11, fontweight="bold", pad=8)
        ax2.set_xlabel("Frequency (kHz)", fontsize=10)
        ax2.set_ylabel(r"Power (V$^2$/Hz)", fontsize=10)
        ax2.grid(True, alpha=0.3, which='both', linestyle=':')
        ax2.set_xlim(0, 150.0)

        colors_picos = ['#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#17becf']
        if best_rank_entry:
            # Highlight Peak #1 (Max mathematical b^2)
            p1 = best_rank_entry['top_peaks'][0]
            ax2.axvline(x=p1[1], color='#d62728', linestyle='--', linewidth=1.4,
                        label=f"Peak #1: $f_1$={p1[1]:.1f} kHz ($b^2$={p1[0]:.2f})")
            ax2.axvline(x=p1[2], color='#d62728', linestyle=':', linewidth=1.4,
                        label=f"Peak #1: $f_2$={p1[2]:.1f} kHz")

            # Check if there is a triad involving the primary mode ~89 kHz (e.g. Peak #4)
            prim_peak = next((p for p in best_rank_entry['top_peaks'] if 85.0 <= p[1] <= 95.0 or 85.0 <= p[2] <= 95.0), None)
            if prim_peak and prim_peak != p1:
                ax2.axvline(x=prim_peak[1], color='darkorange', linestyle='--', linewidth=1.6,
                            label=f"Primary Triad: $f_1$={prim_peak[1]:.1f} kHz ($b^2$={prim_peak[0]:.2f})")
                ax2.axvline(x=prim_peak[2], color='darkorange', linestyle=':', linewidth=1.6,
                            label=f"Primary Triad: $f_2$={prim_peak[2]:.1f} kHz")
                ax2.axvline(x=prim_peak[3], color='darkorange', linestyle='-.', linewidth=1.2, alpha=0.7,
                            label=f"Primary Triad: $f_3$={prim_peak[3]:.1f} kHz")
            elif len(best_rank_entry['top_peaks']) > 1:
                p2 = best_rank_entry['top_peaks'][1]
                ax2.axvline(x=p2[1], color='#2ca02c', linestyle='--', linewidth=1.3,
                            label=f"Peak #2: $f_1$={p2[1]:.1f} kHz ($b^2$={p2[0]:.2f})")
                ax2.axvline(x=p2[2], color='#2ca02c', linestyle=':', linewidth=1.3,
                            label=f"Peak #2: $f_2$={p2[2]:.1f} kHz")

        ax2.legend(loc='upper right', fontsize=8.5, framealpha=0.92)

        # --- Check for f1, f2, and f3 correspondence with real PSD peaks ---
        log.debug("\n--- Checking f1, f2, and f3 correspondence with real local PSD peaks ---")
        psd_peaks_idx, _ = find_peaks(np.log10(Pxx_val[mask_psd]), prominence=0.1)
        psd_peaks_khz = f_psd_khz[mask_psd][psd_peaks_idx]
        df_res_khz = best_channel_data['df_khz']
        # Track, per top peak of the headline channel, whether ALL THREE of its frequencies
        # are PSD-confirmed. Peak #1 (idx 0) is specifically what is later cross-referenced in
        # the reconciliation with the Langmuir cross-check, since a headline peak whose own
        # components aren't corroborated by real spectral power is weaker evidence than one
        # where all three are.
        psd_confirmation_by_peak = {}

        if len(psd_peaks_khz) > 0 and best_rank_entry:
            psd_sign = "+" if best_rank_entry.get('coupling_type', 'Sum') == 'Sum' else "-"
            for idx, p in enumerate(best_rank_entry['top_peaks'][:3]):
                f1_val, f2_val, f3_val = p[1], p[2], p[3]
                log.debug(f"Peak #{idx+1}: Triad ({f1_val:.1f} {psd_sign} {f2_val:.1f} = {f3_val:.1f} kHz) | b\u00b2 = {p[0]:.3f}")
                all_confirmed = True
                for label, f_val in [("f1", f1_val), ("f2", f2_val), ("f3", f3_val)]:
                    closest_peak_idx = np.argmin(np.abs(psd_peaks_khz - f_val))
                    closest_peak = psd_peaks_khz[closest_peak_idx]
                    dist_khz = np.abs(closest_peak - f_val)
                    dist_bins = dist_khz / df_res_khz
                    log.debug(f"  -> {label}={f_val:.1f} kHz | Closest PSD peak: {closest_peak:.1f} kHz "
                          f"(Distance: {dist_khz:.2f} kHz / {dist_bins:.2f} bins)")
                    if dist_bins <= 1.5:
                        log.debug(f"     [CONFIRMED] {label} matches a real local PSD peak (\u2264 1.5 bins)")
                    else:
                        log.debug(f"     [WARNING] {label} does NOT match a prominent local PSD peak (> 1.5 bins)")
                        all_confirmed = False
                psd_confirmation_by_peak[idx] = all_confirmed
            if 0 in psd_confirmation_by_peak:
                status = "CONFIRMED" if psd_confirmation_by_peak[0] else "NOT fully confirmed"
                p0 = best_rank_entry['top_peaks'][0]
                log.info(f"Headline peak PSD check: triad ({p0[1]:.1f} {psd_sign} {p0[2]:.1f} = {p0[3]:.1f} kHz) "
                         f"is {status} by real local PSD peaks (\u2264 1.5 bins).")
        else:
            log.debug("No peaks detected in the PSD or empty ranking.")

        plt.tight_layout()
        output_png = f"mhd_bicoherence_objective3_{shot_id}.png"
        plt.savefig(output_png, dpi=150)
        log.info(f"Bicoherence plot saved to: '{output_png}'")

    # ------------------------------------------------------------------------------------------
    # 3. Explicit comparison of Burst vs. Quiescent vs. Control for the leading channel
    #
    # This directly checks whether the detected nonlinear coupling is a SPECIFIC signature of
    # the H-mode burst window, or whether it appears equally (or even more strongly) in the
    # quiescent/control windows -- in which case it could NOT be attributed exclusively to the
    # phenomenon of interest in the proposal.
    # ------------------------------------------------------------------------------------------
    log.debug("\n" + "=" * 90)
    log.debug("\U0001f4ca BICOHERENCE COMPARISON: BURST vs. QUIESCENT vs. CONTROL (Leading Channel)")
    log.debug("=" * 90)

    if best_channel_data is None or args.full_discharge:
        if args.full_discharge:
            log.debug("This comparison is skipped because --full-discharge is active (no physical windows defined).")
        else:
            log.debug("There is no leading channel with which to perform the window comparison.")
    else:
        fpath_best = best_channel_data['fpath']
        windows = {
            'Burst': (main_t_start, main_t_end),
            'Quiescent': (quiescent_start, quiescent_end),
            'Control': (control_start, control_end),
        }
        skipped_windows = [name for name, (w_start, w_end) in windows.items()
                            if w_start is None or w_end is None]
        if skipped_windows:
            log.debug(f"\u274c Skipping {', '.join(skipped_windows)} from this comparison: no valid time "
                  f"bounds are available for it (see the window-resolution messages above). It will "
                  f"NOT be silently analyzed as the full discharge.")
            windows = {name: bounds for name, bounds in windows.items() if name not in skipped_windows}
        window_results = {}
        for win_name, (w_start, w_end) in windows.items():
            try:
                res_w = analyze_channel_bicoherence(fpath_best, args.nfft, noverlap_frac, args.ensemble,
                                                     args.threshold, args.f1max, args.f2max, args.topn,
                                                     t_start_ms=w_start, t_end_ms=w_end,
                                                     peak_window_khz=args.peak_window_khz,
                                                     fmin_khz=args.fmin)

                # Adapt standard keys to the coupling type of best_channel_data
                if best_channel_data['coupling_type'] == 'Diff':
                    res_w['max_b2'] = res_w['diff_max_b2']
                    res_w['best_freqs'] = res_w['diff_best_freqs']
                    res_w['significant_count'] = res_w['diff_significant_count']
                    res_w['significant_count_stat'] = res_w['diff_significant_count_stat']
                    res_w['total_bins'] = res_w['diff_total_bins']
                    res_w['top_peaks'] = res_w['diff_peaks']
                else:
                    res_w['max_b2'] = res_w['sum_max_b2']
                    res_w['best_freqs'] = res_w['sum_best_freqs']
                    res_w['significant_count'] = res_w['sum_significant_count']
                    res_w['significant_count_stat'] = res_w['sum_significant_count_stat']
                    res_w['total_bins'] = res_w['sum_total_bins']
                    res_w['top_peaks'] = res_w['sum_peaks']

                window_results[win_name] = res_w
                pct_phys = res_w['significant_count'] / res_w['total_bins'] * 100.0
                pct_stat = res_w['significant_count_stat'] / res_w['total_bins'] * 100.0
                log.info(f"{win_name:<12} ({w_start:.0f}-{w_end:.0f} ms): Max b\u00b2 = {res_w['max_b2']:.4f} at "
                      f"f1={res_w['best_freqs'][0]:.1f}, f2={res_w['best_freqs'][1]:.1f} kHz | "
                      f"{pct_stat:.2f}% of bins > 95% statistical threshold")
            except ValueError as e:
                log.info(f"{win_name:<12} ({w_start:.0f}-{w_end:.0f} ms): Could not analyze -> {e}")

        if 'Burst' in window_results and 'Quiescent' in window_results and 'Control' in window_results:
            f1 = best_channel_data['f1']
            f2 = best_channel_data['f2']
            ff1, ff2 = np.meshgrid(f1, f2, indexing='ij')
            fnyq = best_channel_data['fs'] / 2.0

            if best_channel_data['coupling_type'] == 'Diff':
                physical_mask_diff = (ff1 > 0) & (ff2 < 0) & (ff1 > -ff2)
                non_redundant_mask = physical_mask_diff & (ff1 >= args.fmin * 1000.0) & (-ff2 >= args.fmin * 1000.0) & ((ff1 + ff2) >= args.fmin * 1000.0)
            else:
                physical_mask = (ff1 > 0) & (ff2 > 0) & ((ff1 + ff2) < fnyq)
                non_redundant_mask = physical_mask & (ff1 >= ff2) & (ff1 >= args.fmin * 1000.0) & (ff2 >= args.fmin * 1000.0)

            burst_b2 = window_results['Burst']['bicoh2'][non_redundant_mask]
            quiescent_b2 = window_results['Quiescent']['bicoh2'][non_redundant_mask]
            control_b2 = window_results['Control']['bicoh2'][non_redundant_mask]

            log.debug("\n  - Formal Statistical Inference Assessment:")
            try:
                # 1. Kruskal-Wallis H-test (non-parametric ANOVA)
                kw_stat, kw_p = stats.kruskal(burst_b2, quiescent_b2, control_b2)
                log.debug(f"    * Kruskal-Wallis H-test (compares all three full populations): H-stat = {kw_stat:.3f}, p-value = {kw_p:.4e}")

                # 2. Mann-Whitney U (post-hoc, one-sided greater)
                u_bq, p_bq = stats.mannwhitneyu(burst_b2, quiescent_b2, alternative='greater')
                u_bc, p_bc = stats.mannwhitneyu(burst_b2, control_b2, alternative='greater')
                log.debug(f"    * Mann-Whitney U post-hoc (Burst > Quiescent?): U = {u_bq:.1f}, p-value = {p_bq:.4e}")
                log.debug(f"    * Mann-Whitney U post-hoc (Burst > Control?):    U = {u_bc:.1f}, p-value = {p_bc:.4e}")

                # 3. Chi-squared test for proportions of significant bins (> stat. 95%)
                k_b = window_results['Burst']['significant_count_stat']
                n_b = window_results['Burst']['total_bins']
                k_q = window_results['Quiescent']['significant_count_stat']
                n_q = window_results['Quiescent']['total_bins']
                k_c = window_results['Control']['significant_count_stat']
                n_c = window_results['Control']['total_bins']
                obs = np.array([[k_b, n_b - k_b], [k_q, n_q - k_q], [k_c, n_c - k_c]])
                chi2, chi2_p, _, _ = stats.chi2_contingency(obs)
                log.debug(f"    * Chi-squared test of proportions (bins > b\u00b2_95): chi2-stat = {chi2:.3f}, p-value = {chi2_p:.4e}")

                alpha_bonf = 0.05 / 2  # Correction for 2 post-hoc comparisons against Burst
                is_significant = (kw_p < 0.05) and (p_bq < alpha_bonf) and (p_bc < alpha_bonf)

                # We also check whether the maximum b2 is higher outside the burst
                b2_max_burst = window_results['Burst']['max_b2']
                b2_max_quiescent = window_results['Quiescent']['max_b2']
                b2_max_control = window_results['Control']['max_b2']
                max_outside = max(b2_max_quiescent, b2_max_control)

                if is_significant and b2_max_burst > max_outside:
                    log.info("Statistical test (Kruskal-Wallis + Mann-Whitney, Bonferroni-corrected): burst "
                             "bicoherence is significantly higher than quiescent/control, and the absolute "
                             f"max b\u00b2 also falls in the burst window (p < {alpha_bonf:.4f}).")
                else:
                    reasons = []
                    if not is_significant:
                        reasons.append("burst is NOT statistically superior to quiescent/control "
                                        f"(p_bq={p_bq:.4f}, p_bc={p_bc:.4f})")
                    if b2_max_burst <= max_outside:
                        reasons.append(f"max b\u00b2 in burst ({b2_max_burst:.4f}) is lower than outside it "
                                        f"(quiescent={b2_max_quiescent:.4f}, control={b2_max_control:.4f}), "
                                        f"though the Mann-Whitney U population-level test still favors the "
                                        f"burst (p_bq={p_bq:.4e}, p_bc={p_bc:.4e}) -- the single extreme value "
                                        f"and the whole-distribution shift are different questions.")
                    log.info("\u26a0\ufe0f Statistical nuance: " + "; ".join(reasons) + ". No rigorous basis to "
                             "call the coupling an exclusive/dominant H-mode signature from the max value alone.")
            except Exception as e_test:
                log.debug(f"    \u26a0\ufe0f Could not complete the formal inference analysis: {e_test}")

        # --- Final confirmation criterion: agreement with enhanced heat transport (Wp) ---
        # main.md: "Agreement between the enhanced heat transport periods and the temporal
        # windows where strong bicoherence is detected will serve as the final confirmation."
        # This is a plain comparison of W_p / dW_p/dt across the same Burst/Quiescent/Control
        # windows already used for the bicoherence comparison above -- no additional transport
        # model is fitted, since the proposal only calls for this window-level agreement check.
        wp_file = base_dir / f"Wp@{shot_id}.edf"
        ech_file = base_dir / f"ECHRG500@{shot_id}.edf"
        if wp_file.exists():
            log.debug("\n\U0001f4c8 ENERGY TRANSPORT WINDOW COMPARISON (W_p) -- FINAL CONFIRMATION CRITERION:")
            try:
                # Verify and print the actual units of the Wp channel from the EDF metadata
                edf_wp = TE.edf()
                edf_wp.load(str(wp_file))
                wp_val_unit = edf_wp.ValUnit[0] if (hasattr(edf_wp, 'ValUnit') and len(edf_wp.ValUnit) > 0) else "kJ"
                wp_dim_unit = edf_wp.DimUnit[0] if (hasattr(edf_wp, 'DimUnit') and len(edf_wp.DimUnit) > 0) else "ms"
                log.debug(f"  - Verified metadata for W_p ({wp_file.name}): DimUnit (time) = '{wp_dim_unit}', ValUnit (energy) = '{wp_val_unit}'")

                t_sec_wp, ys_wp, dt_wp, _ = load_edf_signal(wp_file)
                t_ms_wp = t_sec_wp * 1000.0

                # dWp/dt derivative to measure the rate of energy loss/gain
                dwp_dt = np.diff(ys_wp) / np.diff(t_sec_wp)
                t_ms_wp_mid = 0.5 * (t_ms_wp[:-1] + t_ms_wp[1:])

                # Load the ECH heating power only to report the heating status alongside W_p,
                # for context when reading the table below (not used in any fit or model).
                has_ech = False
                if ech_file.exists():
                    try:
                        t_sec_ech, ys_ech, _, _ = load_edf_signal(ech_file)
                        t_ms_ech = t_sec_ech * 1000.0
                        has_ech = True
                    except Exception as e_ech:
                        log.debug(f"    \u26a0\ufe0f Could not load the ECH signal: {e_ech}")

                log.debug(f"  - Average values in the burst / quiescent / control windows:")
                for win_name, (w_start, w_end) in windows.items():
                    mask_wp = (t_ms_wp >= w_start) & (t_ms_wp <= w_end)
                    mean_wp = np.mean(ys_wp[mask_wp]) if np.sum(mask_wp) > 0 else 0.0

                    mask_dwp = (t_ms_wp_mid >= w_start) & (t_ms_wp_mid <= w_end)
                    mean_dwp = np.mean(dwp_dt[mask_dwp]) if np.sum(mask_dwp) > 0 else 0.0

                    mean_ech_v = 0.0
                    if has_ech:
                        mask_ech = (t_ms_ech >= w_start) & (t_ms_ech <= w_end)
                        mean_ech_v = np.mean(ys_ech[mask_ech]) if np.sum(mask_ech) > 0 else 0.0

                    res_win = window_results.get(win_name)
                    b2_max_win = res_win['max_b2'] if res_win else 0.0
                    pct_stat_win = (res_win['significant_count_stat'] / res_win['total_bins'] * 100.0) if res_win else 0.0

                    ech_str = f" | <ECH> = {mean_ech_v:.2f} V" if has_ech else ""
                    log.info(f"  Window {win_name:<10} ({w_start:.0f}-{w_end:.0f} ms): <W_p> = {mean_wp:.2f} {wp_val_unit} | <dW_p/dt> = {mean_dwp:+.2f} {wp_val_unit}/s{ech_str} | "
                          f"Max b\u00b2 = {b2_max_win:.3f} | Bins > Stat.95% = {pct_stat_win:.2f}%")

                log.debug("Read this table alongside the Burst vs. Quiescent vs. Control comparison above: "
                          "agreement between elevated bicoherence and a degraded/negative <dW_p/dt> specifically "
                          "during the burst window is what the proposal defines as the final confirmation criterion.")

            except Exception as e:
                log.info(f"\u26a0\ufe0f Could not perform the W_p window comparison: {e}")
    log.debug("=" * 90)

    # ------------------------------------------------------------------------------------------
    # 4. Shot-to-Shot Reproducibility Validation (Discharge Reproducibility)
    #
    # Uses the SAME channel name in both shots (with an explicit fallback and warning if it does
    # not exist), requires the peak to exceed the significance threshold, and compares the found
    # triad against that of the main shot within a configurable tolerance. Uses the validation
    # shot's OWN burst window, auto-detected the same way as the main shot's (unless
    # --validation-burst-start/end was given explicitly), since burst timing genuinely differs
    # between discharges -- comparing the main shot's auto-detected burst against a fixed window
    # in the validation shot would silently compare two different physical phases of the
    # discharge and produce a meaningless PASS/FAIL.
    # ------------------------------------------------------------------------------------------
    log.debug("\n" + "=" * 80)
    log.debug("\U0001f504 SHOT-TO-SHOT REPRODUCIBILITY VALIDATION (DISCHARGE REPRODUCIBILITY)")
    log.debug("=" * 80)

    # Validate ALL genuine MP channels from the ranking, not just the leading one, so a channel
    # flagged with a boundary/stability warning above is still put through this test rather than
    # silently skipped.
    key_channels = [r['channel'] for r in results_ranking if MIRNOV_NAME_RE.match(r['channel'])]

    repro_confirmed = {}  # Dynamically records the reproducibility result for each key channel
    if best_channel_data is None:
        log.debug("There is no leading channel from the main shot to compare against; validation is skipped.")
    else:
        # Resolve the validation shot's own burst window: manual override if both bounds were
        # given, otherwise auto-detect from ITS OWN W_p trace (same detector as the main shot).
        val_t_start, val_t_end = args.validation_burst_start, args.validation_burst_end
        if val_t_start is not None and val_t_end is not None:
            log.debug(f"Validation shot burst window: MANUALLY specified as {val_t_start:.0f}-{val_t_end:.0f} ms")
        else:
            val_wp_file = Path(args.validation_dir) / f"Wp@{args.validation_shot}.edf"
            val_det = None
            if val_wp_file.exists():
                val_det = detect_hmode_burst_window(
                    val_wp_file, baseline_end_ms=args.auto_burst_baseline_ms,
                    k_on=args.auto_burst_k_on, k_off=args.auto_burst_k_off,
                    min_duration_ms=args.auto_burst_min_duration_ms,
                    quiescent_duration_ms=args.auto_quiescent_duration_ms,
                    control_duration_ms=args.auto_control_duration_ms,
                    window_margin_ms=args.auto_window_margin_ms,
                    flat_percentile=args.auto_flat_percentile, flat_k=args.auto_flat_k,
                    robust_baseline=not args.no_robust_baseline,
                    diagnostic_plot_path=None)
            if val_det is not None and val_det.get('ok'):
                val_t_start, val_t_end = val_det['start_ms'], val_det['end_ms']
                log.info(f"Validation shot {args.validation_shot} burst window (auto-detected): "
                         f"{val_t_start:.1f}-{val_t_end:.1f} ms")
            else:
                # Fall back to the old fixed default, but say so loudly: this is now a degraded
                # mode, not the normal path, and any reproducibility result obtained this way
                # should be treated with caution until the window is confirmed manually.
                val_t_start, val_t_end = 188.0, 236.0
                reason = (val_det.get('reason', 'unknown reason') if val_det is not None
                          else f"{val_wp_file} not found")
                log.info(f"\u26a0\ufe0f Could not auto-detect the burst window for validation shot "
                         f"{args.validation_shot} ({reason}). Falling back to a fixed default "
                         f"({val_t_start:.0f}-{val_t_end:.0f} ms) which may NOT correspond to this shot's "
                         f"actual H-mode burst -- verify manually with --validation-burst-start/end before "
                         f"trusting the reproducibility result below.")

        # We extend to validate the reproducibility of the key channels
        channels_to_validate = [best_channel_name]
        for ch in key_channels:
            if ch not in channels_to_validate:
                channels_to_validate.append(ch)

        for val_channel_name in channels_to_validate:
            repro_confirmed[val_channel_name] = False  # Default initialization
            log.debug(f"\n--- Testing reproducibility for channel: '{val_channel_name}' ---")

            # We look up the reference in results_ranking from the main shot
            ref_entry = next((r for r in results_ranking if r['channel'] == val_channel_name), None)
            if not ref_entry:
                log.debug(f"No reference data for '{val_channel_name}' from the main shot; skipping.")
                continue

            ref_triad = ref_entry['freqs_khz']
            file_rep = Path(args.validation_dir) / f"{val_channel_name}@{args.validation_shot}.edf"

            if not file_rep.exists():
                log.debug(f"\u26a0\ufe0f Channel '{val_channel_name}' not found for shot {args.validation_shot} "
                      f"in {args.validation_dir}; skipping.")
                continue

            log.debug(f"Loading validation channel: {file_rep} (Shot {args.validation_shot}, channel {val_channel_name})")
            log.debug(f"Burst window of the validation shot: {val_t_start:.0f}-{val_t_end:.0f} ms")
            try:
                result_rep = analyze_channel_bicoherence(file_rep, args.nfft, noverlap_frac, args.ensemble,
                                                          args.threshold, args.f1max, args.f2max, args.topn,
                                                          t_start_ms=val_t_start,
                                                          t_end_ms=val_t_end,
                                                          peak_window_khz=args.peak_window_khz,
                                                          fmin_khz=args.fmin)

                # Adapt reproducibility result keys to match reference coupling type
                if ref_entry['coupling_type'] == 'Diff':
                    result_rep['max_b2'] = result_rep['diff_max_b2']
                    result_rep['best_freqs'] = result_rep['diff_best_freqs']
                    result_rep['significant_count'] = result_rep['diff_significant_count']
                    result_rep['significant_count_stat'] = result_rep['diff_significant_count_stat']
                    result_rep['total_bins'] = result_rep['diff_total_bins']
                    result_rep['top_peaks'] = result_rep['diff_peaks']
                else:
                    result_rep['max_b2'] = result_rep['sum_max_b2']
                    result_rep['best_freqs'] = result_rep['sum_best_freqs']
                    result_rep['significant_count'] = result_rep['sum_significant_count']
                    result_rep['significant_count_stat'] = result_rep['sum_significant_count_stat']
                    result_rep['total_bins'] = result_rep['sum_total_bins']
                    result_rep['top_peaks'] = result_rep['sum_peaks']

                if not result_rep['top_peaks']:
                    log.info(f"[{val_channel_name}] Reproducibility: no coupling peaks found in the validation shot.")
                elif result_rep['max_b2'] < args.threshold:
                    log.info(f"[{val_channel_name}] Reproducibility NOT confirmed: max peak "
                          f"(b\u00b2 = {result_rep['max_b2']:.3f}) does NOT exceed threshold ({args.threshold}).")
                else:
                    f1_rep, f2_rep, f3_rep = result_rep['best_freqs']
                    log.debug(f"Significant coupling detected: b\u00b2 = {result_rep['max_b2']:.4f} at "
                          f"f1={f1_rep:.1f}, f2={f2_rep:.1f} -> f3={f3_rep:.1f} kHz")

                    # We attempt to match against the reference leading peak of the main shot
                    if triads_match(ref_triad, result_rep['best_freqs'], args.repro_tol_khz):
                        log.info(f"[{val_channel_name}] Reproducibility CONFIRMED: triad matches shot "
                              f"{args.validation_shot} within \u00b1{args.repro_tol_khz} kHz "
                              f"(main: f1={ref_triad[0]:.1f}, f2={ref_triad[1]:.1f} | "
                              f"{args.validation_shot}: f1={f1_rep:.1f}, f2={f2_rep:.1f} kHz).")
                        repro_confirmed[val_channel_name] = True
                    else:
                        # We check whether it matches any of its other secondary top_peaks
                        matched_sec = False
                        for p_idx, peak in enumerate(ref_entry['top_peaks']):
                            _, ref_f1, ref_f2, ref_f3 = peak
                            if triads_match((ref_f1, ref_f2, ref_f3), result_rep['best_freqs'], args.repro_tol_khz):
                                log.info(f"[{val_channel_name}] Reproducibility CONFIRMED: matches secondary "
                                      f"peak #{p_idx+1} of the main shot (main: f1={ref_f1:.1f}, f2={ref_f2:.1f} | "
                                      f"{args.validation_shot}: f1={f1_rep:.1f}, f2={f2_rep:.1f} kHz).")
                                matched_sec = True
                                repro_confirmed[val_channel_name] = True
                                break

                        if not matched_sec:
                            log.info(f"[{val_channel_name}] Reproducibility NOT confirmed: significant coupling "
                                  f"found in shot {args.validation_shot} but at different frequencies "
                                  f"(main: f1={ref_triad[0]:.1f}, f2={ref_triad[1]:.1f} | "
                                  f"{args.validation_shot}: f1={f1_rep:.1f}, f2={f2_rep:.1f} kHz; "
                                  f"outside \u00b1{args.repro_tol_khz} kHz tolerance).")
            except Exception as e:
                log.info(f"\u26a0\ufe0f Error processing validation shot {args.validation_shot} for {val_channel_name}: {e}")

    # ------------------------------------------------------------------------------------------
    # 5. Cross-Diagnostic Consistency Validation: magnetic vs. electrostatic
    #
    # Required by section 2.4 of the proposal: confirm that the same nonlinear coupling
    # signature appears simultaneously in the magnetic (Mirnov) channel and in the electrostatic
    # (Langmuir) channel.
    #
    # Instead of comparing only against the single absolute leading channel, we compare the
    # Langmuir probe's bicoherence signatures against the peaks of ALL genuine Mirnov channels.
    # This allows identifying cross-consistency for coexisting phenomena in different frequency
    # bands.
    #
    # We explicitly track whether the match corresponds to the LEADING peak of the headline
    # channel (best_channel_name) or to a SECONDARY peak of another channel, so both results can
    # be reconciled at the end instead of being presented as if they were the same confirmed
    # finding.
    # ------------------------------------------------------------------------------------------
    log.debug("\n" + "=" * 80)
    log.debug("\U0001f500 CROSS-DIAGNOSTIC VALIDATION (MAGNETIC vs. ELECTROSTATIC)")
    log.debug("=" * 80)

    langmuir_files = sorted(glob.glob(args.langmuir_pattern))
    xdiag_titular_match = False
    xdiag_secondary_matches = []  # list of (langmuir_probe_tip, mp_channel, peak_number)
    xdiag_confirmed = {}  # Dynamically records whether each magnetic channel has cross-consistency with Langmuir
    xdiag_max_chance_match_p = 0.0  # worst-case (highest) chance-match probability seen across all Langmuir probes
    xdiag_tips_matched_per_mp = {}  # mp_channel -> set of probe-tip names that matched it (for the summary)

    # We dynamically initialize for all MP channels in the ranking
    for r in results_ranking:
        if MIRNOV_NAME_RE.match(r['channel']):
            xdiag_confirmed[r['channel']] = False
            xdiag_tips_matched_per_mp[r['channel']] = set()

    if not langmuir_files:
        log.info(f"\u26a0\ufe0f No Langmuir probe files found with pattern '{args.langmuir_pattern}' -- "
                 f"cross-diagnostic consistency (Objective 3, criterion 2) CANNOT be tested, so it will "
                 f"show as FAIL by default below. This is a missing-data/config issue, not a physics result: "
                 f"adjust --langmuir-pattern to the actual Langmuir probe names in your dataset (e.g. LP1, "
                 f"DLP1, ESP1, etc.) before trusting the double-validation FAIL verdict.")
    elif not results_ranking:
        log.debug("There are no analyzed magnetic channels to compare against; cross-validation is skipped.")
    else:
        exclude_tips = [s.strip() for s in args.langmuir_exclude_channels.split(",") if s.strip()]
        for lp_file in langmuir_files:
            lp_array_name = Path(lp_file).name.split('@')[0]
            try:
                t_sec_lp, dt_lp, fs_lp, lp_channels = load_edf_all_channels(lp_file, exclude_names=exclude_tips)
            except Exception as e:
                log.info(f"\u26a0\ufe0f [{lp_array_name}] Could not read probe-tip columns from this file: {e}")
                continue

            if not lp_channels:
                log.info(f"\u26a0\ufe0f [{lp_array_name}] No usable probe-tip columns found after excluding "
                         f"{exclude_tips} -- check --langmuir-exclude-channels against this file's actual "
                         f"bias/trigger pin labels.")
                continue

            log.debug(f"\n[{lp_array_name}] Probe array file: testing {len(lp_channels)} individual probe "
                      f"tip(s) (excluded: {exclude_tips}) against all genuine magnetic channels.")

            n_tips_significant = 0
            n_tips_matched_raw = 0
            n_candidates_tested_file = 0  # accumulated over ALL tips in this array file (RAW, treats tips as independent)
            best_tip_overall = None  # (tip_name, b2, freqs, coupling_type) for reporting the array's own best triad
            raw_matches = []  # (lp_name, tip_name, mp_channel, p_idx, mp_f1, mp_f2, mp_peak_val, is_titular)
            significant_tip_series = {}  # tip_name -> windowed ys, for the M_eff correlation correction below

            # Fixed set of genuine magnetic channels with a usable leading peak -- computed once,
            # since it doesn't depend on the tip (this is the "candidates per tip" denominator).
            mp_channels_with_leading_peak = [r for r in results_ranking
                                              if MIRNOV_NAME_RE.match(r['channel']) and r['top_peaks']]

            for tip_name, ys_tip in lp_channels.items():
                lp_name = f"{lp_array_name}[{tip_name}]"
                try:
                    result_lp = analyze_channel_bicoherence(
                        None, args.nfft, noverlap_frac, args.ensemble,
                        args.threshold, args.f1max, args.f2max, args.topn,
                        t_start_ms=main_t_start, t_end_ms=main_t_end,
                        peak_window_khz=args.peak_window_khz,
                        fmin_khz=args.fmin, preloaded=(t_sec_lp, ys_tip, dt_lp, fs_lp))
                except ValueError as e_short:
                    log.debug(f"  [{lp_name}] Skipped: {e_short}")
                    continue

                # IMPORTANT: 'max_b2'/'top_peaks' are backward-compatibility aliases that only
                # reflect SUM coupling (f1+f2=f3). Since the magnetic channels' leading coupling
                # type is frequently 'Diff' (f1-f2=f3) in this dataset, gating on the Sum-only
                # alias here would silently ignore genuine Diff-type coupling in the Langmuir
                # probe. Use whichever of Sum/Diff is stronger for this probe instead.
                lp_dominant_type = 'Diff' if result_lp['diff_max_b2'] > result_lp['sum_max_b2'] else 'Sum'
                lp_overall_max_b2 = max(result_lp['sum_max_b2'], result_lp['diff_max_b2'])
                lp_overall_peaks = result_lp['diff_peaks'] if lp_dominant_type == 'Diff' else result_lp['sum_peaks']
                lp_overall_best_freqs = (result_lp['diff_best_freqs'] if lp_dominant_type == 'Diff'
                                          else result_lp['sum_best_freqs'])

                if best_tip_overall is None or lp_overall_max_b2 > best_tip_overall[1]:
                    best_tip_overall = (tip_name, lp_overall_max_b2, lp_overall_best_freqs, lp_dominant_type)

                if not lp_overall_peaks or lp_overall_max_b2 < args.threshold:
                    # Sub-threshold tip: skip the verbose per-tip report (would be extremely noisy
                    # across 36 tips) but still keep it out of the matching loop below.
                    continue

                n_tips_significant += 1
                # Keep the tip's own time series (sliced to the same analysis window used for the
                # bicoherence comparison) so we can later estimate how CORRELATED the significant
                # tips are with each other -- adjacent probe tips see overlapping turbulence, so
                # they are not independent tests, and the multiple-comparisons correction below
                # needs the EFFECTIVE (not raw) tip count to avoid over-penalizing a real signal.
                ys_tip_windowed = slice_window(t_sec_lp, ys_tip, main_t_start, main_t_end)[1]
                significant_tip_series[tip_name] = ys_tip_windowed

                f1_lp_sum, f2_lp_sum, f3_lp_sum = result_lp['sum_best_freqs']
                f1_lp_diff, f2_lp_diff, f3_lp_diff = result_lp['diff_best_freqs']
                log.debug(f"  [{lp_name}] Sum leader: f1={f1_lp_sum:.1f}, f2={f2_lp_sum:.1f}, f3={f3_lp_sum:.1f} kHz | b\u00b2={result_lp['sum_max_b2']:.3f}")
                log.debug(f"  [{lp_name}] Diff leader: f1={f1_lp_diff:.1f}, f2={f2_lp_diff:.1f}, f3={f3_lp_diff:.1f} kHz | b\u00b2={result_lp['diff_max_b2']:.3f}")

                # We only compare the tip's OWN leading peak (already selected above via
                # lp_dominant_type/lp_overall_best_freqs) against each MP channel's LEADING peak
                # (p_idx == 0 only, i.e. r['top_peaks'][0]) -- NOT every secondary peak of every
                # channel. Secondary-peak "fishing" is what produced most of the raw matches in
                # the previous run (up to 300 candidates/shot) and, as the earlier caveat showed,
                # secondary peaks are already flagged elsewhere as weaker evidence on their own.
                # Restricting to leading-vs-leading is also the standard comparison in the
                # bicoherence cross-validation literature ("does the dominant coupling
                # reproduce"), not "does anything anywhere match anything else".
                matched_any_tip = False
                for r in mp_channels_with_leading_peak:
                    lp_triad = ((f1_lp_diff, f2_lp_diff, f3_lp_diff) if r['coupling_type'] == 'Diff'
                                else (f1_lp_sum, f2_lp_sum, f3_lp_sum))
                    mp_peak_val, mp_f1, mp_f2, mp_f3 = r['top_peaks'][0]
                    n_candidates_tested_file += 1
                    if triads_match(lp_triad, (mp_f1, mp_f2, mp_f3), args.xdiag_tol_khz):
                        is_titular = (r['channel'] == best_channel_name)
                        raw_matches.append((lp_name, tip_name, r['channel'], 0, mp_f1, mp_f2,
                                             mp_peak_val, is_titular))
                        matched_any_tip = True

                if matched_any_tip:
                    n_tips_matched_raw += 1

            # --- Per-array summary of the RAW (uncorrected) scan ---
            if best_tip_overall is not None:
                bt_name, bt_b2, bt_freqs, bt_type = best_tip_overall
                sign = '+' if bt_type == 'Sum' else '-'
                log.info(f"[{lp_array_name}] Scanned {len(lp_channels)} probe tip(s): {n_tips_significant} "
                         f"exceeded the physical threshold (b\u00b2>{args.threshold}), {n_tips_matched_raw} of those "
                         f"had a RAW (uncorrected) leading-peak match to a magnetic channel's leading triad. "
                         f"Best tip overall: {bt_name} (b\u00b2={bt_b2:.3f} at f1={bt_freqs[0]:.1f} {sign} "
                         f"f2={bt_freqs[1]:.1f} kHz, {bt_type}).")

            # ------------------------------------------------------------------------------------
            # MULTIPLE-COMPARISONS SIGNIFICANCE GATE (now on an EFFECTIVE candidate count).
            #
            # Restricting to leading-peak-vs-leading-peak comparisons already cuts the raw
            # candidate count roughly 5x (no more secondary-peak fishing). But testing many
            # probe tips independently would STILL over-penalize a real signal, because adjacent
            # tips are physically correlated, not independent -- so instead of using the raw
            # "n_tips_significant" tip count in the correction, we estimate the EFFECTIVE number
            # of independent tips (M_eff <= n_tips_significant) from how correlated their actual
            # signals are during this analysis window (see effective_n_independent_series()).
            # Perfectly redundant tips barely raise M_eff at all; genuinely independent-looking
            # tips raise it close to 1-for-1. The corrected candidate count used for the
            # significance test is M_eff * (number of MP channels compared), NOT
            # n_tips_significant * (number of MP channels compared).
            # ------------------------------------------------------------------------------------
            if n_candidates_tested_file > 0:
                n_mp_compared = len(mp_channels_with_leading_peak)
                m_eff_tips, n_raw_tips, _corr_shape = effective_n_independent_series(significant_tip_series)
                n_candidates_effective = m_eff_tips * n_mp_compared

                p_single, p_any_raw = chance_match_probability(args.xdiag_tol_khz, args.f1max, args.f2max,
                                                                 n_candidates_tested_file)
                _, p_any = chance_match_probability(args.xdiag_tol_khz, args.f1max, args.f2max,
                                                      n_candidates_effective)
                xdiag_max_chance_match_p = max(xdiag_max_chance_match_p, p_any)
                significant = p_any < args.xdiag_alpha

                log.debug(f"  \u2139\ufe0f Multiple-comparisons context: {n_candidates_tested_file} RAW candidate "
                      f"triad(s) tested (leading peaks only, {n_raw_tips} significant tip(s) x {n_mp_compared} MP "
                      f"channel(s)) across {lp_array_name} (\u00b1{args.xdiag_tol_khz} kHz box, "
                      f"{args.f1max:.0f}x{args.f2max:.0f} kHz search area). Since adjacent probe tips are "
                      f"physically correlated rather than independent, the {n_raw_tips} significant tip(s) are "
                      f"estimated to correspond to only M_eff\u2248{m_eff_tips:.2f} EFFECTIVELY independent tip(s) "
                      f"(from the eigenspectrum of their pairwise correlation over this window) -- giving "
                      f"{n_candidates_effective:.2f} effective candidate(s). Chance of >=1 match: "
                      f"{p_any_raw*100:.1f}% treating tips as independent (over-conservative) vs. "
                      f"{p_any*100:.1f}% using the effective count (the one actually used for the gate below).")

                if not raw_matches:
                    if n_tips_matched_raw == 0:
                        log.info(f"[{lp_array_name}] Cross-consistency NOT confirmed: no probe tip matched any "
                              f"genuine magnetic channel's leading triads.")
                elif significant:
                    # p_any (effective) is small: the raw matches found are unlikely to be chance
                    # coincidences even after accounting for tip correlation, so promote to CONFIRMED.
                    for (lp_name, tip_name, mp_ch, p_idx, mp_f1, mp_f2, mp_peak_val, is_titular) in raw_matches:
                        if is_titular:
                            xdiag_titular_match = True
                        else:
                            xdiag_secondary_matches.append((lp_name, mp_ch, p_idx + 1))
                        xdiag_confirmed[mp_ch] = True
                        xdiag_tips_matched_per_mp[mp_ch].add(tip_name)
                        log.info(f"[{lp_name}] CONFIRMED match with magnetic channel {mp_ch} "
                              f"(leading peak: f1={mp_f1:.1f}, f2={mp_f2:.1f} kHz, b\u00b2={mp_peak_val:.3f}) "
                              f"within tolerance (\u00b1{args.xdiag_tol_khz} kHz) -- statistically significant even "
                              f"after correcting for tip correlation (M_eff\u2248{m_eff_tips:.2f} effective tips, "
                              f"p\u2248{p_any:.4f} < \u03b1={args.xdiag_alpha}).")
                else:
                    # p_any (effective) is still large: raw matches exist but remain consistent
                    # with pure chance even after crediting for tip correlation. Report them, but
                    # do NOT count them as confirmed.
                    involved_channels = sorted({mp_ch for (_, _, mp_ch, *_rest) in raw_matches})
                    log.info(f"\u26a0\ufe0f [{lp_array_name}] {len(raw_matches)} raw match(es) were found (involving "
                             f"{', '.join(involved_channels)}), but NONE are counted as confirmed cross-"
                             f"diagnostic consistency: even after crediting tip correlation "
                             f"(M_eff\u2248{m_eff_tips:.2f} effective tips instead of {n_raw_tips} raw), the "
                             f"probability of finding at least one such match BY PURE CHANCE alone is "
                             f"\u2248{p_any*100:.1f}% (\u2265 \u03b1={args.xdiag_alpha*100:.0f}%) -- statistically "
                             f"indistinguishable from noise, should NOT be reported as physical confirmation.")
                    for (lp_name, tip_name, mp_ch, p_idx, mp_f1, mp_f2, mp_peak_val, is_titular) in raw_matches:
                        log.debug(f"    (unconfirmed, chance-consistent) [{lp_name}] vs {mp_ch} leading peak: "
                                  f"f1={mp_f1:.1f}, f2={mp_f2:.1f} kHz, b\u00b2={mp_peak_val:.3f}")

        # Explicit reconciliation: is the "headline" result (highest b^2 in the ranking) the same
        # as the one cross-validated with Langmuir, or are they two distinct findings?
        log.debug("--- Reconciliation: is the 'headline' result the same as the cross-validated one? ---")
        if xdiag_titular_match:
            log.info(f"Reconciliation: the LEADING peak of headline channel {best_channel_name} WAS cross-confirmed with Langmuir.")
        elif xdiag_secondary_matches:
            matches_str = ', '.join(f"{ch} (peak #{pidx})" for _, ch, pidx in xdiag_secondary_matches)
            log.info(f"Reconciliation: the headline peak of {best_channel_name} was NOT the one cross-validated -- "
                     f"Langmuir instead confirmed secondary peak(s): {matches_str}. These are two distinct "
                     f"findings: (a) the highest-b\u00b2 coupling ({best_channel_name}), reported as 'headline' solely "
                     f"on that basis, and (b) a weaker-b\u00b2 but independently cross-confirmed coupling.")
        else:
            log.debug("No cross-match was found (neither for the headline leading peak nor for secondary peaks).")

        # Cross-reference the PSD-correspondence check performed on the headline channel's own
        # top peaks: a headline peak whose f1/f2/f3 aren't themselves corroborated by real
        # spectral power is weaker evidence than one where all three are, independent of whatever
        # Langmuir does or doesn't confirm.
        if psd_confirmation_by_peak:
            headline_psd_ok = psd_confirmation_by_peak.get(0)
            if headline_psd_ok is False:
                confirmed_alt = [idx for idx, ok in psd_confirmation_by_peak.items() if ok]
                alt_str = (f" Peak(s) #{', #'.join(str(i+1) for i in confirmed_alt)} of {best_channel_name} "
                           f"DO have all three frequencies PSD-confirmed, if you need a more robust "
                           f"alternative to lead with." if confirmed_alt else "")
                log.info(f"\u26a0\ufe0f CAVEAT: the headline peak (#1) of {best_channel_name} has at least one "
                      f"frequency component that does NOT correspond to a real local PSD peak -- treat it as "
                      f"less certain than its b\u00b2 alone suggests.{alt_str}")
            elif headline_psd_ok is True:
                log.debug(f"\n  \u2705 The headline peak (#1) of {best_channel_name} has all three frequencies "
                      f"PSD-confirmed, which is separate, supporting evidence alongside whatever the Langmuir "
                      f"cross-validation above found.")

    # ------------------------------------------------------------------------------------------
    # 6. Statistical threshold justification, and final double-validation summary (Objective 3)
    #
    # The full statistical derivation (Kim & Powers 1979 exponential threshold, spectral
    # resolution, etc.) is kept in the log file at DEBUG level; the console only gets the
    # threshold NUMBER (already used throughout the run) and the final PASS/FAIL table, which
    # is the actual result the proposal (main.md, Third Specific Objective) asks for.
    # ------------------------------------------------------------------------------------------
    log.debug("\n" + "=" * 90)
    log.debug("METHODOLOGICAL DETAIL: THRESHOLD JUSTIFICATION")
    log.debug("=" * 90)
    log.debug("Under the null hypothesis of independent signals, b\u00b2 is approximately exponential with "
               "mean 1/N_ensemble (Kim & Powers, 1979).")
    log.debug(f"Rigorous 95% threshold: b\u00b2_95 = -ln(0.05)/N_ensemble = {b2_stat_threshold_preview:.3f} "
               f"(N_ensemble={args.ensemble}); NOT simply the mean 1/N_ensemble = {1.0/args.ensemble:.3f}.")
    log.debug(f"Physical threshold {args.threshold} is above b\u00b2_95 = {b2_stat_threshold_preview:.3f} "
               f"-> conservative even under the stricter statistical criterion.")
    log.debug(f"Under H0, ~{chance_level_pct:.1f}% of bins are expected to exceed b\u00b2_95 by pure chance.")

    fs_val = best_channel_data['fs'] if (best_channel_data is not None) else 1e6
    df_val = fs_val / args.nfft / 1000.0
    log.debug(f"Spectral resolution: nfft={args.nfft} -> df = {df_val:.3f} kHz.")

    passed_both = []
    for ch in key_channels:
        passed_repro = repro_confirmed.get(ch, False)
        passed_xdiag = xdiag_confirmed.get(ch, False)
        if passed_repro and passed_xdiag:
            passed_both.append(ch)

    log.info("\n" + "=" * 90)
    log.info("DOUBLE-VALIDATION SUMMARY (Objective 3: reproducibility AND cross-diagnostic consistency)")
    log.info("=" * 90)
    for ch in key_channels:
        passed_repro = repro_confirmed.get(ch, False)
        passed_xdiag = xdiag_confirmed.get(ch, False)
        repro_status = "PASS" if passed_repro else "FAIL"
        xdiag_status = "PASS" if passed_xdiag else "FAIL"
        flag = "  <- BOTH CRITERIA MET" if (passed_repro and passed_xdiag) else ""
        log.info(f"  {ch:<6} reproducibility({args.validation_shot})={repro_status:<4}  "
                 f"cross-consistency(Langmuir)={xdiag_status:<4}{flag}")

    if not passed_both:
        log.info("\n\u26a0\ufe0f RESULT: Objective 3 is NOT validated under the strict double criterion -- no "
                 "channel simultaneously passed reproducibility AND cross-diagnostic consistency. "
                 "See the log file for the per-channel breakdown of which criterion each channel failed.")
    else:
        flagged_passed = [ch for ch in passed_both if ch in amplitude_outlier_channels]
        log.info(f"\n\u2705 RESULT: Objective 3 double-validation criterion MET by: {', '.join(passed_both)}")
        if flagged_passed:
            log.info(f"   \u26a0\ufe0f CAVEAT: {', '.join(flagged_passed)} was also flagged as an RMS amplitude "
                     f"outlier above -- confirm its calibration/gain before reporting it as validated.")
        CHANCE_MATCH_CAUTION_THRESHOLD = 0.20
        if xdiag_max_chance_match_p > CHANCE_MATCH_CAUTION_THRESHOLD:
            log.info(f"   \u26a0\ufe0f CAVEAT: up to a {xdiag_max_chance_match_p*100:.0f}% probability that at least "
                     f"one Langmuir-MP triad match arose by pure chance (multiple-comparisons check) -- disclose "
                     f"this alongside the result, don't present the match as unambiguous confirmation.")
    log.debug("=" * 90)


def run_shot_worker(task):
    """Runs one shot inside a multiprocessing worker and captures its logged output.

    Each worker is a separate process, so it gets its own private `log` handler writing to an
    in-memory buffer (rather than the console/file handlers set up in the main process by
    `setup_logging`) -- this keeps concurrent shots' output from interleaving on the console
    and lets the caller print each shot's full report as one block, in order.
    """
    shot_id, args = task
    import io
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.handlers.clear()
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    try:
        run_single_shot(shot_id, args)
    except Exception as e:
        import traceback
        buf.write(f"\n\u274c ERROR processing shot {shot_id}: {e}\n")
        buf.write(traceback.format_exc())
    return shot_id, buf.getvalue()


def main():
    parser = argparse.ArgumentParser(description="Multichannel Comparative Bicoherence Analysis (Objective 3)")
    parser.add_argument("--shots", type=str, default="88653",
                        help="Comma-separated list of shot numbers to run (def: 88653)")
    parser.add_argument("-p", "--pattern", type=str, default=f"data/hj{SHOT}/MP[0-9]*@{SHOT}.edf",
                        help="Pattern for the genuine Mirnov (MP) coil channels analyzed for Objective 3")
    parser.add_argument("-n", "--nfft", type=int, default=1024, help="FFT size for bicoherence (def: 1024)")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="PHYSICAL threshold for significant bicoherence (def: 0.5)")
    parser.add_argument("-e", "--ensemble", type=int, default=30, help="Number of ensembles to average bicoherence (def: 30)")
    # f1max/f2max restricted to <100 kHz: Objective 3 (main.md) is specifically about
    # LOW-FREQUENCY electromagnetic fluctuations during H-mode coupled to edge turbulence, so
    # the peak search is kept within that physically motivated band rather than scanning
    # broadband and reporting whatever the global argmax happens to be.
    parser.add_argument("--f1max", type=float, default=150.0, help="Maximum f1 frequency for peak search in kHz (def: 150)")
    parser.add_argument("--f2max", type=float, default=150.0, help="Maximum f2 frequency for peak search in kHz (def: 150)")
    parser.add_argument("--topn", type=int, default=10, help="Number of top local 2D peaks to report per channel (def: 10)")
    parser.add_argument("--peak-window-khz", type=float, default=5.0,
                        help="Width (in kHz) of the local 2D peak search window, "
                             "converted to bins according to the actual spectral resolution (def: 5.0 kHz)")
    parser.add_argument("--stability-frac", type=float, default=0.5,
                        help="Fraction of f1max/f2max defining the inner 'core' box used to "
                             "check whether the reported leading peak is stable against the search "
                             "box size (def: 0.5, i.e. half of f1max/f2max)")

    # --- Physical time window arguments (burst / quiescent / control) ---
    parser.add_argument("--burst-start", type=float, default=None,
                        help="Start (ms) of the burst window used for bicoherence. If omitted, the "
                             "burst window is AUTO-DETECTED from the W_p signal (see --no-auto-burst "
                             "to disable and require --burst-start/--burst-end to be given manually).")
    parser.add_argument("--burst-end", type=float, default=None,
                        help="End (ms) of the burst window used for bicoherence. If omitted, "
                             "auto-detected together with --burst-start (see above).")
    parser.add_argument("--no-auto-burst", action="store_true",
                        help="Disable W_p-based auto-detection of the burst/quiescent/control windows; "
                             "requires --burst-start/--burst-end (and, for the window comparison, "
                             "--quiescent-start/--quiescent-end/--control-start/--control-end) to be "
                             "supplied explicitly.")
    parser.add_argument("--auto-burst-baseline-ms", type=float, default=50.0,
                        help="Length (ms) of the pre-burst baseline segment used by the W_p auto-"
                             "detector to estimate the L-mode reference level (def: 50)")
    parser.add_argument("--auto-burst-k-on", type=float, default=3.0,
                        help="Onset threshold for W_p auto-detection, in baseline std above the "
                             "baseline mean (def: 3.0)")
    parser.add_argument("--auto-burst-k-off", type=float, default=1.5,
                        help="End threshold for W_p auto-detection, in baseline std above the "
                             "baseline mean; kept lower than --auto-burst-k-on as hysteresis "
                             "against noise (def: 1.5)")
    parser.add_argument("--auto-burst-min-duration-ms", type=float, default=10.0,
                        help="Minimum sustained duration (ms) required before declaring an onset "
                             "or end transition, to avoid triggering on a single noisy sample (def: 10)")
    parser.add_argument("--quiescent-start", type=float, default=None,
                        help="Start (ms) of the quiescent (pre-burst) comparison window. If omitted "
                             "(and auto-detection is active), it is placed automatically right before "
                             "the detected/given burst onset.")
    parser.add_argument("--quiescent-end", type=float, default=None,
                        help="End (ms) of the quiescent comparison window. See --quiescent-start.")
    parser.add_argument("--control-start", type=float, default=None,
                        help="Start (ms) of the control (post-burst) comparison window. If omitted "
                             "(and auto-detection is active), it is placed automatically right after "
                             "the detected/given burst end.")
    parser.add_argument("--control-end", type=float, default=None,
                        help="End (ms) of the control comparison window. See --control-start.")
    parser.add_argument("--auto-quiescent-duration-ms", type=float, default=25.0,
                        help="Duration (ms) of the auto-placed quiescent window (def: 25)")
    parser.add_argument("--auto-control-duration-ms", type=float, default=25.0,
                        help="Duration (ms) of the auto-placed control window (def: 25)")
    parser.add_argument("--auto-window-margin-ms", type=float, default=5.0,
                        help="Gap (ms) left between the burst boundaries and the auto-placed "
                             "quiescent/control windows, so they don't overlap the L-H/H-L "
                             "transition ramps themselves (def: 5)")
    parser.add_argument("--no-window-plot", action="store_true",
                        help="Disable saving the W_p window-detection diagnostic PNG (saved by "
                             "default whenever burst/quiescent/control auto-detection runs).")
    parser.add_argument("--window-plot-path", type=str, default="mhd_burst_window_detection.png",
                        help="Output path for the W_p window-detection diagnostic PNG "
                             "(def: mhd_burst_window_detection.png)")
    parser.add_argument("--auto-flat-percentile", type=float, default=50.0,
                        help="Calibrates 'flat enough' for an auto-placed quiescent/control window "
                             "against the empirical distribution of (std+range) scores of same-"
                             "length windows sampled across this shot's own non-burst background "
                             "(def: 50, i.e. the median). Lower = stricter (only the flattest "
                             "fraction of background windows qualify); higher = more permissive. "
                             "This self-calibrates per shot, so it should not usually need tuning.")
    parser.add_argument("--auto-flat-k", type=float, default=1.5,
                        help="Extra multiplicative slack applied on top of --auto-flat-percentile "
                             "(def: 1.5). Lower = stricter; higher = more permissive.")
    parser.add_argument("--no-robust-baseline", action="store_true",
                        help="Use plain mean/std for the W_p baseline instead of the default "
                             "median/MAD (robust to a contaminating bump or secondary transition "
                             "landing inside the baseline segment).")
    parser.add_argument("--full-discharge", action="store_true",
                        help="If enabled, ignores the physical windows and uses the FULL discharge "
                             "for the main ranking (not recommended: mixes distinct physical regimes).")

    # --- [M9] Flat-frequency sub-window (ported from Objective 2's coherence-window algorithm;
    # tightens the burst window used for the main bicoherence ranking + figure, see
    # detect_flat_frequency_subwindow()/compute_flat_frequency_window() above) ---
    parser.add_argument("--no-flat-subwindow", action="store_true",
                        help="[M9] Disable the flat-frequency sub-window step entirely and use the full "
                             "burst window (from --burst-start/--burst-end or auto-detection) for the "
                             "bicoherence calculations and figure, as before this was added.")
    parser.add_argument("--flat-probe", type=str, default="MP1",
                        help="[M9] Mirnov channel (matched against '<name>@<shot>.edf' in the same data "
                             "directory as --pattern) whose instantaneous frequency is used to find the "
                             "flat sub-window (def: MP1).")
    parser.add_argument("--flat-freq-low-khz", type=float, default=None,
                        help="[M9] Lower edge (kHz) of the Bessel bandpass filter applied to --flat-probe "
                             "before extracting its Hilbert instantaneous frequency. Default: falls back "
                             "to --fmin.")
    parser.add_argument("--flat-freq-high-khz", type=float, default=None,
                        help="[M9] Upper edge (kHz) of the Bessel bandpass filter applied to --flat-probe. "
                             "Default: falls back to --f1max.")
    parser.add_argument("--flat-order", type=int, default=4, help="[M9] Bessel filter order (def: 4).")
    parser.add_argument("--flat-sg-win", type=int, default=OPTIMAL_SG_WIN,
                        help=f"[M9] Savitzky-Golay smoothing window used inside the instantaneous-frequency "
                             f"extraction (def: {OPTIMAL_SG_WIN}, Objective 2's optimal value).")
    parser.add_argument("--flat-slope-smooth-ms", type=float, default=2.0,
                        help="[M9] Moving-average smoothing window (ms) applied to |d(f_inst)/dt| before "
                             "flat-frequency-region detection (def: 2.0). Larger = smoother slope estimate, "
                             "less sensitive to per-sample phase noise, but can blur short genuine sweeps.")
    parser.add_argument("--flat-scan-window-ms", type=float, default=8.0,
                        help="[M9] Length (ms) of the sliding 'core' window scanned across the burst to "
                             "locate its flattest patch (lowest mean |d(f_inst)/dt|), def: 8.0. This core "
                             "is then grown outward (see --flat-growth-tolerance) into the final sub-window.")
    parser.add_argument("--flat-growth-tolerance", type=float, default=0.5,
                        help="[M9] Relative tolerance (def: 0.5, i.e. 50%%) used to grow the flattest-patch "
                             "core outward: the window keeps extending in each direction as long as its "
                             "mean |d(f_inst)/dt| stays within (1 + this) x the core's own mean.")
    parser.add_argument("--flat-min-duration-ms", type=float, default=5.0,
                        help="[M9] Minimum duration (ms) for the grown flat-frequency window to be accepted "
                             "(def: 5.0); shorter falls back to the full burst window.")
    parser.add_argument("--flat-window-start", type=float, default=None,
                        help="[M9] Optional manual override (ms) for the flat-frequency sub-window start, "
                             "instead of the adaptive scan-and-grow detector. Must be paired with "
                             "--flat-window-end. Clipped to the burst window if it extends outside it.")
    parser.add_argument("--flat-window-end", type=float, default=None,
                        help="[M9] Optional manual override (ms) for the flat-frequency sub-window end.")

    # --- Discharge reproducibility (Objective 3, criterion 1) ---
    parser.add_argument("--validation-shot", type=str, default="88654", help="Shot used to validate reproducibility (def: 88654)")
    parser.add_argument("--validation-dir", type=str, default="data/hj88654", help="Data directory for the validation shot (def: data/hj88654)")
    parser.add_argument("--repro-tol-khz", type=float, default=10.0,
                        help="Tolerance in kHz to consider that a triad was reproduced in the other shot (def: 10)")
    parser.add_argument("--validation-burst-start", type=float, default=None,
                        help="Start (ms) of the validation shot's burst window. Default: auto-detect from "
                             "the validation shot's own W_p trace (same method as the main shot), so it's "
                             "compared like-for-like rather than against a fixed window that may not match "
                             "its actual H-mode burst timing. Set this (with --validation-burst-end) to "
                             "override with a manual window.")
    parser.add_argument("--validation-burst-end", type=float, default=None,
                        help="End (ms) of the validation shot's burst window. See --validation-burst-start.")

    # --- Cross-diagnostic consistency with Langmuir probes (Objective 3, criterion 2) ---
    parser.add_argument("--langmuir-pattern", type=str, default=f"data/hj{SHOT}/DivProArr@{SHOT}.edf",
                        help="Glob pattern for Langmuir probe ARRAY file(s) (e.g. DivProArr divertor probe "
                             "array), required by the proposal (section 2.4) to verify magnetic-electrostatic "
                             "cross-consistency (def: DivProArr@<shot>.edf). NOTE: each matched file is itself "
                             "a MULTI-COLUMN array (e.g. DivProArr packs 36 individual probe tips into one "
                             ".edf) -- every non-excluded column is analyzed individually (see "
                             "--langmuir-exclude-channels), not just the first one.")
    parser.add_argument("--langmuir-exclude-channels", type=str, default="18-1,27-1",
                        help="Comma-separated raw pin labels (as reported in the .edf's channel-name metadata, "
                             "e.g. '18-1', '27-1') to EXCLUDE from the per-tip cross-diagnostic scan because "
                             "they are not physical probe-tip signals -- e.g. for DivProArr, pin '18-1' is the "
                             "bias supply channel (DP-BIAS) and pin '27-1' is the trigger channel (TRIG), not "
                             "edge density/potential fluctuation measurements (def: '18-1,27-1', matching the "
                             "DivProArr pin map used at Heliotron J; adjust if your array's bias/trigger pins "
                             "differ, or set to '' to scan every column).")
    parser.add_argument("--xdiag-tol-khz", type=float, default=10.0,
                        help="Tolerance in kHz for the magnetic-electrostatic (Langmuir) cross-diagnostic "
                             "match (def: 10)")
    parser.add_argument("--xdiag-alpha", type=float, default=0.05,
                        help="Significance level for the multiple-comparisons gate on cross-diagnostic "
                             "matches (def: 0.05). A raw Langmuir-Mirnov triad match is only promoted to "
                             "'CONFIRMED' (and counted towards the Objective 3 double-validation PASS) if "
                             "the family-wise probability of seeing at least one such match by pure chance, "
                             "given how many candidate (tip x peak x MP-channel) comparisons were actually "
                             "tested, is below this value. Raw matches that don't clear this bar are still "
                             "logged for transparency but are NOT counted as confirmed physics.")
    parser.add_argument("--fmin", type=float, default=5.0,
                        help="Minimum frequency (kHz) for coupling search (avoids spectral leakage near DC, def: 5.0)")

    # --- Output verbosity ---
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print the FULL diagnostic trail to the console (window auto-detection "
                             "steps, per-channel Sum+Diff stats, per-probe matching detail, statistical "
                             "justification, etc.). By default, the console only shows the essential "
                             "results and any warning/error; the full trail always goes to --log-file.")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Path to write the full diagnostic trail to, regardless of --verbose "
                             "(def: mhd_obj3_<shot>.log for a single shot; omitted for multi-shot runs, "
                             "where each shot's captured report is what would otherwise go there).")
    args = parser.parse_args()

    if args.flat_freq_low_khz is None:
        args.flat_freq_low_khz = args.fmin
    if args.flat_freq_high_khz is None:
        args.flat_freq_high_khz = args.f1max

    shot_list = [int(s.strip()) for s in args.shots.split(",") if s.strip()]

    if len(shot_list) == 1:
        log_file = args.log_file if args.log_file is not None else f"mhd_obj3_{shot_list[0]}.log"
        setup_logging(args.verbose, log_file)
        run_single_shot(shot_list[0], args)
    else:
        from multiprocessing import Pool
        # Console-level filtering happens once already, inside each worker (see run_shot_worker);
        # the parent process just needs a plain handler to print the assembled per-shot reports.
        setup_logging(args.verbose, log_file=None)
        log.info(f"Starting concurrent analysis for shots: {shot_list}...")
        with Pool() as pool:
            results = pool.map(run_shot_worker, [(shot, args) for shot in shot_list])
        for s_id, out in results:
            log.info("\n" + "=" * 110)
            log.info(f"REPORT FOR SHOT {s_id}")
            log.info("=" * 110)
            log.info(out)


if __name__ == "__main__":
    main()