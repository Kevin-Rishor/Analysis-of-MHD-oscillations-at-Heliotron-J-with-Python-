# Shared module of common functions and constants for MHD analysis

import sys
from pathlib import Path

# -------------------------------------------------------------
# SHOT CONFIGURATION (Change this to configure the default shot)
# -------------------------------------------------------------
DEFAULT_SHOT = 88652

# Add the jpack library path to Python's path for direct imports
jpack_path = str(Path(__file__).parent.parent.resolve() / "jpack")
if jpack_path not in sys.path:
    sys.path.append(jpack_path)

import numpy as np
import scipy.signal as dsp
import anahilbert as AH

# Optimal Savitzky-Golay smoothing value for f_inst found in obj2.py and shared with mhd_wavelet.py
OPTIMAL_SG_WIN = 525

# Default parameters for EPM instantaneous frequency extraction
DEFAULT_BP_LO = 40000.0   # 40 kHz in Hz
DEFAULT_BP_HI = 80000.0   # 80 kHz in Hz
DEFAULT_FILTER_ORDER = 4   # 4th order Bessel filter

def morlet_cwt(x, fs, freqs, w=6.0, max_half_width_frac=10, verbose=False, wavelets_fft=None, n_fft=None):
    """
    Continuous Morlet Wavelet Transform (CWT) with adaptive time support.
    Unlike Welch coherence (fixed FFT bins over a static window), each
    wavelet scale LOCALLY follows the signal in time, so it does not suffer the
    artificial decorrelation introduced by chirping (continuous frequency shift)
    within a static Fourier window.
    Optionally supports precomputation of wavelets in frequency for maximum optimization (Filter Bank CWT).
    """
    if wavelets_fft is not None:
        import scipy.fft as fft
        n = len(x)
        out = np.zeros((len(freqs), n), dtype=np.complex64)
        X_fft = fft.fft(x, n=n_fft).astype(np.complex64)
        for i, (W_fft, half_width) in enumerate(wavelets_fft):
            CWT_full = fft.ifft(X_fft * W_fft.astype(np.complex64))
            out[i, :] = CWT_full[half_width : half_width + n]
        return out

    scales = w * fs / (2.0 * np.pi * freqs)
    n = len(x)
    out = np.zeros((len(freqs), n), dtype=np.complex64)
    max_half_width = max(1, n // max_half_width_frac)
    for i, scale in enumerate(scales):
        half_width = int(np.round(5.0 * scale))
        if half_width > max_half_width:
            if verbose:
                print(f"  ⚠️ Warning: Wavelet support truncated by defensive low-frequency cap "
                      f"(half_width = {half_width} limited to max_half_width = {max_half_width}) for scale = {scale:.1f} "
                      f"(frequency = {freqs[i]/1000.0:.1f} kHz).")
            half_width = max_half_width
        t_wav = np.arange(-half_width, half_width + 1)
        scaled_t = t_wav / scale
        wavelet = np.exp(1j * w * scaled_t) * np.exp(-scaled_t ** 2 / 2.0)
        wavelet /= np.sqrt(scale)
        out[i, :] = dsp.fftconvolve(x, wavelet, mode='same')
    return out

def extract_instantaneous_frequency(ys, fs, fl_hz=DEFAULT_BP_LO, fu_hz=DEFAULT_BP_HI, order=DEFAULT_FILTER_ORDER, sg_win=OPTIMAL_SG_WIN, sg_poly_order=2):
    """
    Extracts the instantaneous frequency and envelope via jpack's Hilbert transform
    pre-filtered with a Bessel filter, followed by numerical differentiation with
    Savitzky-Golay smoothing on the phase.
    """
    # Use jpack.anahilbert's native function
    envelope, phase, filtered = AH.Hilbert_Envelope(ys, fs, fl_hz, fu_hz, order)

    # Ensure the window size is odd and less than or equal to the signal length
    n_samples = len(ys)
    if sg_win >= n_samples:
        sg_win = n_samples - 1 if n_samples % 2 == 0 else n_samples - 2
        # The absolute minimum size must be greater than the polynomial order
        if sg_win <= sg_poly_order:
            sg_win = sg_poly_order + 1
            if sg_win % 2 == 0:
                sg_win += 1

    if sg_win % 2 == 0:
        sg_win += 1

    dt = 1.0 / fs
    ifreq_hz = dsp.savgol_filter(phase, sg_win, sg_poly_order, deriv=1) / dt
    return envelope, phase, filtered, ifreq_hz

def anti_alias_decimate(x, q):
    """
    Decimates signal x by a factor q by applying an anti-aliasing filter (scipy.signal.decimate,
    zero_phase=True) in cascade with factors <= 10, instead of:
      (a) direct slicing x[::q], which does NOT filter spectral content above the new
          Nyquist frequency (fs/q) and can therefore introduce aliasing (high-frequency
          components folding onto low frequencies and contaminating the resulting Pearson r), or
      (b) a single scipy.signal.decimate(x, q) call with a large q, since scipy's
          documentation recommends not exceeding ~10-13 per call to maintain IIR filter stability.
    If any segment of the signal is too short for the filter (e.g. small active
    sub-windows), it falls back to direct slicing with an explicit warning.
    """
    factors = []
    remaining = int(q)
    for f in (13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2):
        while remaining % f == 0 and remaining > 13:
            factors.append(f)
            remaining //= f
    if remaining > 1:
        factors.append(remaining)

    y = np.asarray(x, dtype=float)
    try:
        for f in factors:
            y = dsp.decimate(y, f, zero_phase=True)
        return y
    except ValueError:
        print(f"  ⚠️ Warning: signal too short for anti-aliasing decimate() with q={q}; "
              f"falling back to direct slicing (no filtering).")
        return np.asarray(x)[::q]
