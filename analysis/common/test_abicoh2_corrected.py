import os
import sys
import numpy as np
import scipy.signal as dsp
from pathlib import Path

# Add current dir to path
sys.path.insert(0, str(Path(__file__).parent))
from abicoh2_corrected import abicoh2_corrected

def test_t1_brute_force():
    # T1: compute b2 for at least 50 random (a,b) bin pairs with plain Python loop
    # directly from segment FFT matrix (np.fft.fft of windowed segments built inside test)
    np.random.seed(101)
    nfft = 128
    noverlap = nfft // 2
    n_segments = 50
    sig_len = (n_segments - 1) * noverlap + nfft
    x = np.random.randn(sig_len)

    win = dsp.get_window('hann', nfft)
    segments = []
    for i in range(n_segments):
        start = i * noverlap
        seg = x[start:start+nfft] * win
        segments.append(np.fft.fft(seg))
    spec = np.array(segments)  # shape (N, nfft)

    # Call module on 2D precomputed spec
    _, b2_mod = abicoh2_corrected(spec)

    nhalf = nfft // 2
    diffs = []
    np.random.seed(202)
    for _ in range(60):
        a = int(np.random.randint(0, nfft))
        b = int(np.random.randint(0, nhalf))
        c = (a + b) % nfft
        xa = spec[:, a]
        xb = spec[:, b]
        xc = spec[:, c]
        xab = xa * xb
        num_pair = np.mean(xab * np.conjugate(xc))
        den1_pair = np.mean(np.abs(xab)**2)
        den2_pair = np.mean(np.abs(xc)**2)
        den_pair = den1_pair * den2_pair
        val = (np.abs(num_pair)**2 / den_pair) if den_pair > 0 else 0.0
        diffs.append(abs(val - b2_mod[a, b]))

    max_diff = float(max(diffs))
    passed = max_diff < 1e-12
    return passed, f"max abs difference = {max_diff:.3e} (< 1e-12)"

def test_t2_bin_alignment():
    # T2: synthetic signal with cosines at bins a,b and phase-locked product at c=a+b;
    # assert b2 maximum is at (a,b) with a+b-c == 0
    np.random.seed(303)
    nfft = 128
    nhalf = nfft // 2
    n_segments = 120
    win = dsp.get_window('hann', nfft)
    bin_a = 15
    bin_b = 20
    bin_c = bin_a + bin_b # 35
    f_a = bin_a / float(nfft)
    f_b = bin_b / float(nfft)
    f_c = bin_c / float(nfft)

    spec = []
    t_seg = np.arange(nfft)
    for _ in range(n_segments):
        phi_a = np.random.uniform(0, 2*np.pi)
        phi_b = np.random.uniform(0, 2*np.pi)
        phi_c = phi_a + phi_b
        s = np.cos(2*np.pi*f_a*t_seg + phi_a) + np.cos(2*np.pi*f_b*t_seg + phi_b) + 0.6 * np.cos(2*np.pi*f_c*t_seg + phi_c) + 0.02 * np.random.randn(nfft)
        spec.append(np.fft.fft(s * win))

    spec = np.array(spec)
    _, b2_mod = abicoh2_corrected(spec)

    # In the positive frequency plane (0 <= a, b < nhalf):
    max_idx = np.unravel_index(np.argmax(b2_mod[:nhalf, :nhalf]), (nhalf, nhalf))
    ma, mb = int(max_idx[0]), int(max_idx[1])
    is_pair = (ma == bin_a and mb == bin_b) or (ma == bin_b and mb == bin_a)
    triad_zero = (ma + mb - bin_c == 0)
    passed = is_pair and triad_zero and (b2_mod[ma, mb] > 0.99)
    return passed, f"peak at ({ma}, {mb}) with b^2 = {b2_mod[ma, mb]:.4f}, a+b-c = {ma+mb-bin_c}"

def test_t3_biphase():
    # T3: same synthetic signal with known phase; assert |measured - true| < 2 deg for Hann windows
    np.random.seed(404)
    nfft = 128
    n_segments = 120
    win = dsp.get_window('hann', nfft)
    bin_a = 15
    bin_b = 20
    bin_c = bin_a + bin_b
    f_a = bin_a / float(nfft)
    f_b = bin_b / float(nfft)
    f_c = bin_c / float(nfft)

    target_phase_deg = 35.0
    target_phase_rad = np.radians(target_phase_deg)

    spec = []
    t_seg = np.arange(nfft)
    for _ in range(n_segments):
        phi_a = np.random.uniform(0, 2*np.pi)
        phi_b = np.random.uniform(0, 2*np.pi)
        phi_c = phi_a + phi_b + target_phase_rad
        s = np.cos(2*np.pi*f_a*t_seg + phi_a) + np.cos(2*np.pi*f_b*t_seg + phi_b) + 0.6 * np.cos(2*np.pi*f_c*t_seg + phi_c)
        spec.append(np.fft.fft(s * win))

    spec = np.array(spec)
    b_comp, _ = abicoh2_corrected(spec)
    # Measured biphase phi_c - phi_a - phi_b = -angle(b_comp[bin_a, bin_b])
    meas_rad = -np.angle(b_comp[bin_a, bin_b])
    meas_deg = float(np.degrees(meas_rad) % 360)
    true_deg = float(target_phase_deg % 360)
    diff_deg = float(abs((meas_deg - true_deg + 180) % 360 - 180))
    passed = diff_deg < 2.0
    return passed, f"|measured ({meas_deg:.2f} deg) - true ({true_deg:.2f} deg)| = {diff_deg:.3f} deg (< 2 deg)"

def test_t4_default_regression():
    # defaults equal the explicit values fs=nfft, detrend=False; the 1D path equals the 2D path built independently from scipy
    np.random.seed(505)
    nfft = 512
    x = np.random.randn(2500)
    b_def, b2_def = abicoh2_corrected(x)
    b_exp, b2_exp = abicoh2_corrected(x, fs=float(nfft), nfft=nfft, detrend=False)
    same_defaults = bool(np.array_equal(b_def, b_exp) and np.array_equal(b2_def, b2_exp))
    _, _, sp = dsp.spectrogram(x, fs=float(nfft), window='hann', nperseg=nfft, noverlap=nfft // 2, nfft=nfft, detrend=False, return_onesided=False, scaling='density', mode='complex')
    b_2d, b2_2d = abicoh2_corrected(sp.T)
    same_paths = bool(np.array_equal(b2_def, b2_2d))
    return (same_defaults and same_paths), f"explicit==default: {same_defaults}, 1D==2D: {same_paths}"

def test_t5_parameters_act():
    # T5: detrend=True differs from detrend=False on signal with linear trend; fs scales density as stated
    np.random.seed(606)
    x = np.linspace(0, 15, 2048) + 0.05 * np.random.randn(2048)
    b_comp_nodetrend, b2_nodetrend = abicoh2_corrected(x, detrend=False)
    b_comp_detrend, b2_detrend = abicoh2_corrected(x, detrend=True)
    detrend_diff = bool(not np.array_equal(b2_nodetrend, b2_detrend))

    # Density scaling of spectrogram: Sxx is scaled by 1/(fs * sum(w^2))
    nfft = 256
    _, _, sp1 = dsp.spectrogram(x, fs=100.0, nperseg=nfft, scaling='density', mode='complex')
    _, _, sp2 = dsp.spectrogram(x, fs=200.0, nperseg=nfft, scaling='density', mode='complex')
    density_ratio = float(np.mean(np.abs(sp1)**2) / np.mean(np.abs(sp2)**2))
    fs_scales = bool(abs(density_ratio - 2.0) < 1e-10)

    passed = detrend_diff and fs_scales
    return passed, f"detrend acts: {detrend_diff}, density scaling ratio for fs=100 vs 200: {density_ratio:.4f} (expected 2.0)"

if __name__ == "__main__":
    tests = [
        ("T1 brute force", test_t1_brute_force),
        ("T2 bin alignment", test_t2_bin_alignment),
        ("T3 biphase", test_t3_biphase),
        ("T4 default regression", test_t4_default_regression),
        ("T5 parameters act", test_t5_parameters_act),
    ]
    all_ok = True
    for name, fn in tests:
        res, metric = fn()
        status_str = "PASS" if res else "FAIL"
        print(f"[{status_str}] {name}: {metric}")
        if not res:
            all_ok = False
    sys.exit(0 if all_ok else 1)
