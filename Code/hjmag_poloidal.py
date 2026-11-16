#!/usr/bin/env python3
"""
hjmag_poloidal.py

Poloidal mode number (m) extraction via Non-Uniform Discrete Fourier Transform
(NUDFT) and dual-harmonic circular phase validation for Heliotron J arrays.

Adapted from the Heliotron J turnelib/jpack spectral workflow to incorporate:
1. Voltage-to-(T/s) hardware calibration (NA = 4.5e-3 m^2, G = 2.0).
2. Software polarity inversion on reversed Mirnov coils PMP1--PMP4.
3. Non-uniform spatial Fourier decomposition across PMP1--PMP14.
4. Exhaustive two-mode (m1, m2) in [-20, +20] grid search (820 pairs) and
   Mardia--Jupp circular concentration (r_circ) validation.
"""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy.signal as dsp
import turnelib as TE

# Hardware calibration constants and probe geometry (PMP1--PMP14)
NA_EFF = 4.5e-3  # Effective coil area-turns [m^2]
GAIN = 2.0       # Amplifier gain [dimensionless]
PMP_INVERT = {"PMP1", "PMP2", "PMP3", "PMP4"}
PMP_RAW_DEG = [0., 10., 20., 30., 40., 50., 60.,
               80., 90., 100., 110., 120., 150., 180.]
PMP_ANGLES_DEG = [(360.0 - a) % 360.0 for a in PMP_RAW_DEG]


def load_poloidal_array(shot, data_dir):
    """Load, calibrate to T/s, and polarity-correct PMP1--PMP14 records."""
    signals = {}
    time_ms = fs = None
    for idx in range(1, 15):
        ch = f"PMP{idx}"
        fpath = Path(data_dir) / f"{ch}@{shot}.edf"
        edf = TE.edf()
        dat = edf.load(str(fpath))
        t_col = dat[:, 0]
        t_ms = t_col if edf.DimUnit[0] == "ms" else t_col * 1000.0
        sign = -1.0 if ch in PMP_INVERT else 1.0
        signals[ch] = sign * dat[:, 1] / (NA_EFF * GAIN)
        if time_ms is None:
            time_ms = t_ms
            fs = 1000.0 / (t_ms[1] - t_ms[0])
    return time_ms, fs, signals


def extract_nudft_and_phase(time_ms, fs, signals, fl_hz, fu_hz,
                            t_start, t_end, nfft_max=512):
    """Compute 2D NUDFT peak frequency and cross-spectral phases vs PMP1."""
    channels = [f"PMP{i}" for i in range(1, 15)]
    angles_rad = np.deg2rad(PMP_ANGLES_DEG)
    theta = np.arctan2(np.sin(angles_rad), np.cos(angles_rad))

    mask_t = (time_ms >= t_start) & (time_ms <= t_end)
    mat = np.array([signals[ch][mask_t] for ch in channels])
    nfft = min(nfft_max, mat.shape[1])

    freqs, _, S_pmp = dsp.spectrogram(
        mat, fs=fs, window="hann", nperseg=nfft, noverlap=nfft // 2,
        nfft=nfft, detrend="constant", return_onesided=False,
        scaling="density", axis=-1, mode="complex",
    )
    band = (freqs >= fl_hz) & (freqs <= fu_hz)
    k_grid = np.arange(-20, 21)
    E = np.exp(-1j * k_grid[:, None] * theta[None, :]) / len(theta)
    P2d = np.mean(np.abs(np.tensordot(E, S_pmp[:, band, :], axes=(1, 0))) ** 2,
                  axis=2)
    f_band_khz = freqs[band] / 1000.0
    f_peak_khz = float(f_band_khz[int(np.argmax(np.sum(P2d, axis=0)))])

    idx_f = int(np.argmin(np.abs(freqs - f_peak_khz * 1000.0)))
    S_pk = S_pmp[:, idx_f, :]
    avg_cross = np.mean(S_pk * np.conj(S_pk[0, :])[None, :], axis=1)
    raw_phi = np.angle(avg_cross)
    meas_phase = np.arctan2(np.sin(raw_phi - raw_phi[0]),
                            np.cos(raw_phi - raw_phi[0]))
    return f_peak_khz, theta, meas_phase


def _circular_metrics(residuals_rad):
    """Evaluate Mardia--Jupp r_circ and MAE in degrees."""
    n = len(residuals_rad)
    rc = float(np.hypot(np.sum(np.cos(residuals_rad)),
                        np.sum(np.sin(residuals_rad))) / n)
    mae = float(np.rad2deg(np.mean(np.abs(residuals_rad))))
    return rc, mae


def search_poloidal_modes(theta, meas_phase, m_min=-20, m_max=20):
    """Evaluate all single modes and dual-branch pairs (820 pairs)."""
    single_best = None
    for m in range(m_min, m_max + 1):
        res = np.arctan2(np.sin(meas_phase - m * theta),
                         np.cos(meas_phase - m * theta))
        rc, mae = _circular_metrics(res)
        cand = {"m": (m,), "r_circ": rc, "mae_deg": mae,
                "supported": (rc >= 0.70 and mae <= 45.0)}
        if single_best is None or (rc, -mae) > (single_best["r_circ"],
                                               -single_best["mae_deg"]):
            single_best = cand

    dual_best = None
    for m1, m2 in combinations(range(m_min, m_max + 1), 2):
        r1 = np.arctan2(np.sin(meas_phase - m1 * theta),
                        np.cos(meas_phase - m1 * theta))
        r2 = np.arctan2(np.sin(meas_phase - m2 * theta),
                        np.cos(meas_phase - m2 * theta))
        res = np.where(np.abs(r2) < np.abs(r1), r2, r1)
        rc, mae = _circular_metrics(res)
        cand = {"m": (m1, m2), "r_circ": rc, "mae_deg": mae,
                "supported": (rc >= 0.70 and mae <= 45.0)}
        if dual_best is None or (rc, -mae) > (dual_best["r_circ"],
                                             -dual_best["mae_deg"]):
            dual_best = cand
    return single_best, dual_best


def main():
    parser = argparse.ArgumentParser(
        description="NUDFT and dual-harmonic poloidal mode extractor.")
    parser.add_argument("--shot", type=int, default=88653)
    parser.add_argument("--data-dir", type=str, default="data/hj88653")
    parser.add_argument("--fmin", type=float, default=40e3)
    parser.add_argument("--fmax", type=float, default=80e3)
    parser.add_argument("--tstart", type=float, default=259.1)
    parser.add_argument("--tend", type=float, default=275.0)
    args = parser.parse_args()

    t_ms, fs, sigs = load_poloidal_array(args.shot, args.data_dir)
    f_pk, theta, phi = extract_nudft_and_phase(
        t_ms, fs, sigs, args.fmin, args.fmax, args.tstart, args.tend)
    s_best, d_best = search_poloidal_modes(theta, phi)
    print(f"Shot #{args.shot} | Peak={f_pk:.2f} kHz")
    print("Best single mode:", s_best)
    print("Best dual-harmonic pair:", d_best)


if __name__ == "__main__":
    main()
