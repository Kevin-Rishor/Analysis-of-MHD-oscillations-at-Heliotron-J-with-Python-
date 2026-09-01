import sys
from pathlib import Path

DEFAULT_SHOT = 88652

current = Path(__file__).resolve().parent
root_dir = None
for p in [current] + list(current.parents):
    if (p / "jpack").exists():
        root_dir = p
        break
if root_dir is None:
    root_dir = Path("c:/TFG")

jpack_path = str(root_dir / "jpack")
if jpack_path not in sys.path:
    sys.path.append(jpack_path)

import numpy as np
import scipy.signal as dsp
import scipy.fft as fft
import anahilbert as AH

OPTIMAL_SG_WIN = 525
DEFAULT_BP_LO = 40000.0
DEFAULT_BP_HI = 80000.0
DEFAULT_FILTER_ORDER = 4


def morlet_cwt(x, fs, freqs, w=6.0, max_half_width_frac=10, verbose=False, wavelets_fft=None, n_fft=None):
    """Continuous Morlet Wavelet Transform (CWT) with adaptive time support.

    Computes the scalogram using a single forward FFT of the input signal
    and batched frequency-domain convolutions.
    """
    n = len(x)
    scales = w * fs / (2.0 * np.pi * freqs)
    max_half_width = max(1, n // max_half_width_frac)
    half_widths = [min(int(np.round(5.0 * s)), max_half_width) for s in scales]

    if wavelets_fft is not None:
        out = np.zeros((len(freqs), n), dtype=np.complex64)
        x_fft = fft.fft(x, n=n_fft).astype(np.complex64)
        for i, (w_fft, hw) in enumerate(wavelets_fft):
            conv = fft.ifft(x_fft * w_fft.astype(np.complex64))
            out[i, :] = conv[hw:hw + n]
        return out

    max_hw = max(half_widths)
    n_fft_opt = fft.next_fast_len(n + 2 * max_hw + 1)
    x_fft = fft.fft(x, n=n_fft_opt)
    out = np.zeros((len(freqs), n), dtype=np.complex64)

    for i, (scale, hw) in enumerate(zip(scales, half_widths)):
        if hw == max_half_width and verbose:
            print(f"Wavelet support capped at max_half_width={hw} for f={freqs[i] / 1e3:.1f} kHz")

        t_wav = np.arange(-hw, hw + 1)
        scaled_t = t_wav / scale
        wavelet = np.exp(1j * w * scaled_t) * np.exp(-scaled_t ** 2 / 2.0) / np.sqrt(scale)
        w_fft = fft.fft(wavelet, n=n_fft_opt)
        conv = fft.ifft(x_fft * w_fft)
        out[i, :] = conv[hw:hw + n]

    return out


def extract_instantaneous_frequency(
    ys, fs, fl_hz=DEFAULT_BP_LO, fu_hz=DEFAULT_BP_HI,
    order=DEFAULT_FILTER_ORDER, sg_win=OPTIMAL_SG_WIN, sg_poly_order=2
):
    """Extracts instantaneous frequency and envelope via Bessel pre-filtered Hilbert transform."""
    envelope, phase, filtered = AH.Hilbert_Envelope(ys, fs, fl_hz, fu_hz, order)

    n_samples = len(ys)
    if sg_win >= n_samples:
        sg_win = n_samples - 1 if n_samples % 2 == 0 else n_samples - 2
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
    """Cascaded anti-aliasing decimation to maintain IIR filter stability."""
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
        print(f"Warning: signal too short for cascaded decimation (q={q}); fallback to slicing.")
        return np.asarray(x)[::q]

