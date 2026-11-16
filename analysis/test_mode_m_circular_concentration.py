"""
Verification test for poloidal mode number m and circular concentration r_circ.

Tests:
1. Computes r_circ(m) across all integer candidates m in [-20, +20] for both
   ECH+NBI (48.8 kHz) and NBI-only (41.0 kHz) secondary mode bands in shot #88653.
2. Asserts that m = +2 is the top-ranked mode number (Rank 1) among ALL 41 candidates
   in both heating phases.
3. Compares r_circ for m = +2, m = -1, and m = +6, confirming that m = +2 dominates
   both m = -1 (the dual branch partner) and m = +6 (the raw NUDFT power sidelobe).
"""

import sys
from pathlib import Path
import numpy as np
import scipy.signal as dsp

# Setup paths
root_dir = Path(__file__).resolve().parents[1]
sys.path.extend([str(root_dir / "analysis"), str(root_dir / "analysis" / "common"), str(root_dir / "analysis" / "heating_phase_comparison")])

from comparison_common import load_signals, PMP_ANGLES_DEG, T_W1_START, T_W1_END, T_W2_START, T_W2_END


def compute_phase_and_rcirc(time_ms, fs, pmp_signals, pmp_angles_deg, fl_hz, fu_hz, tw_start, tw_end):
    pmp_channels = sorted(pmp_signals.keys(), key=lambda c: int(c[3:]))
    angles_rad = np.array([np.deg2rad(pmp_angles_deg[int(ch[3:]) - 1]) for ch in pmp_channels])
    
    idx_win = np.where((time_ms >= tw_start) & (time_ms <= tw_end))[0]
    n_samples = len(idx_win)
    sig_pmp_win = np.array([pmp_signals[ch][idx_win] for ch in pmp_channels])
    nfft_pmp = min(512, n_samples)
    
    # 2D NUDFT to find peak frequency in band
    f_pmp, _, S_pmp = dsp.spectrogram(
        sig_pmp_win, fs=fs, window='hann', nperseg=nfft_pmp, noverlap=nfft_pmp // 2,
        nfft=nfft_pmp, detrend='constant', return_onesided=False, scaling='density', axis=-1, mode='complex'
    )
    band_mask = (f_pmp >= fl_hz) & (f_pmp <= fu_hz)
    k_grid = np.arange(-6, 7)
    S_band = S_pmp[:, band_mask, :]
    E = np.exp(-1j * k_grid[:, None] * angles_rad[None, :]) / len(angles_rad)
    Sk = np.tensordot(E, S_band, axes=(1, 0))
    P2d = np.mean((Sk * Sk.conj()).real, axis=2)
    f_band_khz = f_pmp[band_mask] / 1000.0
    f_peak_khz = float(f_band_khz[int(np.argmax(np.sum(P2d, axis=0)))])
    
    # Cross-spectral phases relative to PMP1
    f_spec, _, S_cross = dsp.spectrogram(
        sig_pmp_win, fs=fs, window='hann', nperseg=nfft_pmp, noverlap=nfft_pmp // 2,
        nfft=nfft_pmp, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='complex'
    )
    idx_f_peak = int(np.argmin(np.abs(f_spec - f_peak_khz * 1000.0)))
    S_pk = S_cross[:, idx_f_peak, :]
    ref_pk = S_pk[0, :]
    avg_cross = np.mean(S_pk * np.conj(ref_pk)[None, :], axis=1)
    meas_phase = np.arctan2(np.sin(np.angle(avg_cross) - np.angle(avg_cross)[0]),
                            np.cos(np.angle(avg_cross) - np.angle(avg_cross)[0]))
    
    th_in_pi = np.arctan2(np.sin(angles_rad), np.cos(angles_rad))
    all_m = np.arange(-20, 21)
    results = {}
    for m in all_m:
        th_theo = np.arctan2(np.sin(m * th_in_pi), np.cos(m * th_in_pi))
        diff = np.arctan2(np.sin(meas_phase - th_theo), np.cos(meas_phase - th_theo))
        rc = float(np.sqrt(np.sum(np.cos(diff))**2 + np.sum(np.sin(diff))**2) / len(th_in_pi))
        err_deg = float(np.rad2deg(np.mean(np.abs(diff))))
        results[m] = (rc, err_deg)
        
    return f_peak_khz, results


def test_mode_m_ranking():
    print("Loading signals for Shot 88653...")
    time_ms, dt_s, fs, sigs, pmp_signals = load_signals(shot="88653")
    
    phases = [
        ("ECH+NBI (259.1 - 291.3 ms)", T_W1_START, T_W1_END),
        ("NBI-only (291.3 - 300.0 ms)", T_W2_START, T_W2_END)
    ]
    
    for phase_name, t_start, t_end in phases:
        f_pk, res = compute_phase_and_rcirc(time_ms, fs, pmp_signals, PMP_ANGLES_DEG, 40000.0, 80000.0, t_start, t_end)
        
        # Sort candidates by r_circ descending
        ranked = sorted(res.items(), key=lambda item: item[1][0], reverse=True)
        top_m, (top_rc, top_err) = ranked[0]
        
        rc_p2 = res[2][0]
        rc_m1 = res[-1][0]
        rc_p6 = res[6][0]
        
        print(f"\n========================================================")
        print(f"Phase: {phase_name} | Peak Frequency: {f_pk:.2f} kHz")
        print(f"========================================================")
        print(f"  Top Candidate across all m in [-20, +20]: m = {top_m:+d} (r_circ = {top_rc:.4f}, mean_err = {top_err:.1f} deg)")
        print(f"  Comparison of specific candidates:")
        print(f"    m = +2: r_circ = {rc_p2:.4f} (Rank {1 + [r[0] for r in ranked].index(2)})")
        print(f"    m = -1: r_circ = {rc_m1:.4f} (Rank {1 + [r[0] for r in ranked].index(-1)})")
        print(f"    m = +6: r_circ = {rc_p6:.4f} (Rank {1 + [r[0] for r in ranked].index(6)})")
        
        # Assertions
        assert top_m == 2, f"Expected m = +2 to rank #1 across all [-20, 20], but got m = {top_m}"
        assert rc_p2 > rc_m1, f"Expected r_circ(+2) > r_circ(-1), got {rc_p2:.4f} <= {rc_m1:.4f}"
        assert rc_p2 > rc_p6, f"Expected r_circ(+2) > r_circ(+6), got {rc_p2:.4f} <= {rc_p6:.4f}"
        assert rc_m1 > rc_p6, f"Expected r_circ(-1) > r_circ(+6), got {rc_m1:.4f} <= {rc_p6:.4f}"
        print("  -> ALL ASSERTIONS PASSED!")


if __name__ == "__main__":
    test_mode_m_ranking()
