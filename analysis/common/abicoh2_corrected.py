"""
corrected copy of jpack/libana_signal.py abicoh2 (a+b-c = 0 instead of -1 bin); biphase offset about pi for Hann windows in the library routine; units raw FFT^2 / density = fs*sum(w^2).
"""

import numpy as np
import scipy.signal as dsp


def abicoh2_corrected(x, fs=None, nfft=512, window='hann', noverlap=None, detrend=False):
    """
    Compute auto-bicoherence using the aligned-index formulation.

    Parameters
    ----------
    x : ndarray
        Either:
        - 2D array of shape (N_ensemble, nfft): precomputed complex spectra.
        - 1D array of time-domain samples: spectrogram will be computed.
    fs : float, optional
        Sampling rate in Hz (used if x is 1D).
    nfft : int, optional
        FFT size (default 512).
    window : str or array-like, optional
        Window for spectrogram (default 'hann').
    noverlap : int, optional
        Overlap samples for spectrogram (default nfft // 2).

    detrend : bool or str, optional
        Detrending applied to segments for 1D spectrogram (default False).
        If True, 'linear' detrending is used.
    Returns
    -------
    b_comp : ndarray of shape (nf, nhalf), complex128
        Complex bicoherence matrix b = E[X_a X_b X_c*] / sqrt(E[|X_a X_b|^2] * E[|X_c|^2]).
    b2 : ndarray of shape (nf, nhalf), float64
        Squared auto-bicoherence matrix |b|^2.
    """
    if np.ndim(x) == 2:
        spec = x
        N, nf = spec.shape
    elif np.ndim(x) == 1:
        if fs is None:
            fs = float(nfft)
        if noverlap is None:
            noverlap = int(nfft // 2)
        detrend_spec = 'linear' if detrend is True else detrend
        _, _, spec = dsp.spectrogram(
            x, fs=fs, window=window, nperseg=nfft, noverlap=noverlap, nfft=nfft,
            detrend=detrend_spec, return_onesided=False, scaling='density', mode='complex'
        )
        spec = spec.T
        N, nf = spec.shape
    else:
        raise ValueError(f"Expected 1D or 2D input array, got shape {np.shape(x)}")

    nhalf = nf // 2 if nf % 2 == 0 else (nf + 1) // 2
    fcol = np.arange(nf, dtype=int)[:, None]
    brow = np.arange(nhalf, dtype=int)[None, :]
    ha = (fcol + brow) % nf

    # Vectorized expectation:
    # X_ab = spec[:, fcol, None] * spec[:, None, brow] -> shape (N, nf, nhalf)
    X_ab = spec[:, :, None] * spec[:, None, :nhalf]
    X_c = spec[:, ha]

    num = np.mean(X_ab * np.conjugate(X_c), axis=0)
    den1 = np.mean(np.abs(X_ab) ** 2, axis=0)
    den2 = np.mean(np.abs(X_c) ** 2, axis=0)
    den = den1 * den2

    with np.errstate(divide='ignore', invalid='ignore'):
        b2 = np.where(den == 0, 0.0, np.abs(num) ** 2 / den)
        b_comp = np.where(den == 0, 0.0 + 0.0j, num / np.sqrt(den))

    return b_comp, b2
