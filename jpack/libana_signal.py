# -*- coding: utf-8 -*-
"""
    Signal processing tools

    function:
       nfreqreal
       numofensembles
       normalize (obsolated)
       normmean
       normave
       psd   (obsolated)
       xcoh2
       xphase
       csd   (obsolated)
       psd2
       running  (obsolated)
       bicoherence_simple
       abicoh2
       xbicoh2
       summedbicoh2
       periodogram --> spectrogram? 
       xpower
       eave
       xbispct
       iqdemod
       tdiffp (obsolated)
       regdiff_down
       irrdiff_down
       create_basephase
       create_triangle
       create_sawtooth
       create_square

    Dependence
    ----------

    Status
    ------
    Version 0.9.6

    Authour
    -------
    Shigeru Inagaki                                       
    Institute of Advanced Energy, Kyoto University
    s-inagaki@iae.kyoto-u.ac.jp
    
    Revision History
    ----------------
    [30-Mar-2015] Creation (v0.8)
    [18-Oct-2016] Function "normalize" added (v0.8.1)
                  psd(sigs, t, nfft=256, dt=None, noverlap=None, nensemble=1, istart=0, tstart=None, window='hann', detrend=normalize)
    [17-Apr-2017] Bugs in dt-evaluation fixed (v0.8.2)
    [18-Apr-2017] Function "psd2d" added (v0.8.3)
    [05-May-2017] Function "bicoherence_simple, abicoh2, xbicoh2, summedbicoh2" added (v0.9.0)
    [17-May-2017] Function xcoh2 is fixed (Thanks Kin!)
    [19-May-2017] comb key word has benn added on xbicoh2 (v0.9.1)
    [01-Jun-2017] axisf1 option added in summedbicoh2 (v0.9.2)
    [19-Oct-2017] Function "normmean" and "normstd" added. Function "normalize" is obsolated (v0.9.3)
    [24-Oct-2018] Function "iqdemod" and "tdiffp" added (v0.9.4)
    [02-Nov-2018] Function "regdiff_down", "irrdiff_down", "create_basephaseand" and its family added (v0.9.5)
    [14-Dec-2022] Keyword "hanning" is replaced to "hann" (v0.9.6)

    Copyright
    ---------
    2022 Shigeru Inagaki (s-inagaki@iae.kyoto-u.ac.jp)
    Released under the MIT, BSD, and GPL Licenses.

"""

import warnings
import numpy as np
import scipy.fftpack as fft
import scipy.signal as dsp
import scipy.linalg as la

def nfreqreal(nfft): 
    if nfft % 2 == 0 :
       nfreq = nfft//2 + 1
    else:
       nfreq = (nfft+1)//2
    return nfreq
    
def psd(sigs, t, dt=None, nfft=256, noverlap=None, nensemble=1, istart=0, tstart=None, window='hann', detrend='constant'):
    """
    Estimate auto power spectral density using Welch's method.
    Welch's method [1]_ computes an estimate of the power spectral density
    by dividing the data into overlapping segments, computing a modified
    periodogram for each segment and averaging the periodograms.
    Parameters
    ----------
    sigs : array_like
        Time series of measurement values
    t : 1D-array
        Time for sigs
    x : 1D-array
        Time series of reference value
    dt : float, optional
        Sampling time of the `x` time series in units of sec. Defaults
        to None.
    nfft : int, optional
        Length of each segment and  Length of the FFT used. 
    noverlap: int, optional
        Number of points to overlap between segments. If None,
        ``noverlap = nfft / 2``.  Defaults to None.
    nensemble: int, optional
        Number of ensembles  Defaults to 1.
    istart: int, optional
        Start index in the specified axis of sigs. Defaults to 0.
    tstart: float, optional
        Start time of sigs. If None, t[istart] is used. Defaults to None.
    window : str or tuple or array_like, optional
        Desired window to use. See `get_window` for a list of windows and
        required parameters. Defaults to 'Hann'.
    detrend : str or function or False, optional
        Specifies how to detrend each segment. If `detrend` is a string,
        it is passed as the ``type`` argument to `detrend`.  If it is a
        function, it takes a segment and returns a detrended segment.
        If `detrend` is False, no detrending is done.  Defaults to 'constant'.
    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    Pxx : ndarray
        Power spectral density or power spectrum of sigs.
    See Also
    --------
    psd:power spectrum density
    running: running FFT
    Notes
    -----
    An appropriate amount of overlap will depend on the choice of window
    and on your requirements.  For the default 'hann' window an
    overlap of 50% is a reasonable trade off between accurately estimating
    the signal power, while not over counting any of the data.  Narrower
    windows may require a larger overlap.
    If `noverlap` is 0, this method is equivalent to Bartlett's method [2]_.
    .. versionadded:: 0.12.0
    References
    ----------
    .. [1] P. Welch, "The use of the fast Fourier transform for the
           estimation of power spectra: A method based on time averaging
           over short, modified periodograms", IEEE Trans. Audio
           Electroacoust. vol. 15, pp. 70-73, 1967.
    .. [2] M.S. Bartlett, "Periodogram Analysis and Continuous Spectra",
           Biometrika, vol. 37, pp. 1-16, 1950.
    """
   
    if dt is None: 
        dt = (t[-1] - t[0]) / (len(t)-1) 

    if tstart is not None:
        indx = np.where(t >= tstart)[0]
        istart = indx[0]

    if noverlap is None:
        noverlap = nfft // 2
        
    step = nfft - noverlap
    iend = step*(nensemble-1) + nfft + istart

    if iend > len(t) :
       print("Index is out of bounds for axis 0")
       return np.empty(t.shape), np.empty(aigs.shape)

    f, Pxx = dsp.welch(sigs[istart:iend, ...], fs=1.0/dt, window=window, nperseg=nfft, noverlap=noverlap, nfft=None, detrend=detrend, return_onesided=True, scaling='density', axis=0)
    return f, Pxx


def numofensembles(t, nfft=256, noverlap=None, istart=0, iend=-1, tstart=None, tend=None, check=False):
    if noverlap is None:
        noverlap = nfft//2

    if tstart is not None:
        indx = np.where(t >= tstart)[0]
        istart = indx[0]

    if tend is not None:
        if tend >= t[-1]:
            iend = len(t)
        else:
            indx = np.where(t > tend)[0]
            iend = indx[0]
   
    if iend == -1:
        iend = len(t)

    nsig = iend - istart
    step = nfft - noverlap

    if check :
        indices = np.arange(0, nsig-nfft+1, step) + istart
        for k, ind in enumerate(indices):
            print("{0}: {1}---{2}".format(k,t[ind],t[ind+nfft-1]))

    return (nsig-nfft)//step + 1


def running(x, t, dt=None, nfft=256, noverlap=None, istart=0, iend=-1, tstart=None, tend=None, window='hann', detrend='constant'):
    """
    Estimate temporal evolution of power spectral density (running FFT) by dividing 
    the data into segments, computing a modified periodogram for each segment.
    ----------
    x : 1D-array
        Time series of measurement value
    t : 1D-array
        Time for x
    dt : float, optional
        Sampling time of the `x` time series in units of sec. Defaults
        to None.
    nfft : int, optional
        Length of each segment and  Length of the FFT used. 
    noverlap : int, optional
        Number of points to overlap between segments. If None,
        ``noverlap = nperseg // 8``.  Defaults to None.
    istart: int, optional
        Start index in the specified axis of sigs. Defaults to 0.
    tstart: float, optional
        Start time of sigs. If None, t[istart] is used. Defaults to None.
    iend: int, optional
        Last index in the specified axis of sigs. Defaults to -1.
    tend: float, optional
        End time of sigs. If None, t[iend] is used. Defaults to None.
    window : str or tuple or array_like, optional
        Desired window to use. See `get_window` for a list of windows and
        required parameters. Defaults to 'hann'.
    detrend : str or function or False, optional
        Specifies how to detrend each segment. If `detrend` is a string,
        it is passed as the ``type`` argument to `detrend`.  If it is a
        function, it takes a segment and returns a detrended segment.
        If `detrend` is False, no detrending is done.  Defaults to 'constant'.
    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    tave : ndarray
        Array of sample averaged times.
    Pxx : ndarray (2D)
        Power spectral density or power spectrum of x. Pxx(f, tave).
    See Also
    --------
    Notes
    -----
        see welch
    References
    ----------
    .. [1] P. Welch, "The use of the fast Fourier transform for the
           estimation of power spectra: A method based on time averaging
           over short, modified periodograms", IEEE Trans. Audio
           Electroacoust. vol. 15, pp. 70-73, 1967.
    .. [2] M.S. Bartlett, "Periodogram Analysis and Continuous Spectra",
           Biometrika, vol. 37, pp. 1-16, 1950.
    """
    if dt is None: 
        dt = (t[-1]-t[0]) / (len(t)-1) 

    if tstart is not None:
        indx = np.where(t >= tstart)[0]
        istart = indx[0]

    if tend is not None:
        if tend >= t[-1]:
            iend = len(t)
        else:
            indx = np.where(t > tend)[0]
            iend = indx[0] 
   
    if iend == -1:
        iend = len(t)

    f, tave, Pxx = dsp.spectrogram(x[istart:iend,...], fs=1.0/dt, window=window, nperseg=nfft, noverlap=noverlap, nfft=nfft, detrend=detrend,
                                   return_onesided=True, scaling='density', axis=0, mode='psd')
    tave = tave + t[0] - dt
    return f, tave, Pxx

def csd(sigs, t, ref, dt=None, nfft=256, noverlap=None, nensemble=1, istart=0, tstart=None, window='hann', detrend='constant'):

    """
    Estimate cross power spectral density using Welch's method.
    Welch's method [1]_ computes an estimate of the power spectral density
    by dividing the data into overlapping segments, computing a modified
    periodogram for each segment and averaging the periodograms.
    Parameters
    ----------
    sigs : array_like
        Time series of measurement values
    t : 1D-array
        Time for sigs
    ref : 1D-array
        Time series of reference value
    dt : float, optional
        Sampling time of the `x` time series in units of sec. Defaults
        to None.
    nfft : int, optional
        Length of each segment and  Length of the FFT used. 
    noverlap: int, optional
        Number of points to overlap between segments. If None,
        ``noverlap = nfft / 2``.  Defaults to None.
    nensemble: int, optional
        Number of ensembles  Defaults to 1.
    istart: int, optional
        Start index in the specified axis of sigs. Defaults to 0.
    tstart: float, optional
        Start time of sigs. If None, t[istart] is used. Defaults to None.
    window :  str or tuple or array_like, optional
        Desired window to use. See `get_window` for a list of windows and
        required parameters. Defaults to 'hann'.
    detrend : str or function or False, optional
        Specifies how to detrend each segment. If `detrend` is a string,
        it is passed as the ``type`` argument to `detrend`.  If it is a
        function, it takes a segment and returns a detrended segment.
        If `detrend` is False, no detrending is done.  Defaults to 'constant'.
    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    Pxy : ndarray
        Cross power spectral density or cross power spectrum between ref and sigs.
    Pyy : ndarray
        Power spectral density or power spectrum of sigs.
    Pxx : ndarray
        Power spectral density or power spectrum of reference.
    See Also
    --------
    psd:power spectrum density
    running: running FFT
    Notes
    -----
    An appropriate amount of overlap will depend on the choice of window
    and on your requirements.  For the default 'hann' window an
    overlap of 50% is a reasonable trade off between accurately estimating
    the signal power, while not over counting any of the data.  Narrower
    windows may require a larger overlap.
    If `noverlap` is 0, this method is equivalent to Bartlett's method [2]_.
    .. versionadded:: 0.12.0
    References
    ----------
    .. [1] P. Welch, "The use of the fast Fourier transform for the
           estimation of power spectra: A method based on time averaging
           over short, modified periodograms", IEEE Trans. Audio
           Electroacoust. vol. 15, pp. 70-73, 1967.
    .. [2] M.S. Bartlett, "Periodogram Analysis and Continuous Spectra",
           Biometrika, vol. 37, pp. 1-16, 1950.
    """
   
    if dt is None: 
        dt = (t[-1]-t[0]) / (len(t)-1) 

    if tstart is not None:
        indx = np.where(t >= tstart)[0]
        istart = indx[0]

    if noverlap is None:
        noverlap = nfft // 2
        
    step = nfft - noverlap
    iend = step*(nensemble-1) + nfft + istart

    f, Pxy = dsp.csd(sigs[istart:iend,...], ref[istart:iend], fs=1.0/dt, window=window, nperseg=nfft, noverlap=noverlap, nfft=None,
                     detrend=detrend, return_onesided=False, scaling='density', axis=0)
    _, Pyy = dsp.welch(sigs[istart:iend, ...], fs=1.0/dt, window=window, nperseg=nfft, noverlap=noverlap, nfft=None, detrend=detrend,
                       return_onesided=False, scaling='density', axis=0)
    _, Pxx = dsp.welch(ref[istart:iend], fs=1.0/dt, window=window, nperseg=nfft, noverlap=noverlap, nfft=None, detrend=detrend,
                       return_onesided=False, scaling='density', axis=0)
    f = fft.fftshift(f)
    Pxx = fft.fftshift(Pxx)
    Pyy = fft.fftshift(Pyy,axes=0)
    Pxy = fft.fftshift(Pxy,axes=0)
    return f, Pxy, Pyy, Pxx

def xcoh2(Pxy, Pyy, Pxx):
    pwr = (Pxy*Pxy.conj()).real
    coh2 = pwr/Pyy
    
    if Pxy.ndim == 1:
        return coh2/Pxx
    
    for j in range(coh2.shape[1]):
       coh2[:,j] = coh2[:,j]/Pxx[:]
       
    return coh2

def xphase(Pxy, normalize=False, unsigned=False):
    phi = np.angle(Pxy)

    if unsigned:
       indx = np.where(phi < 0.0)
       phi[indx] = phi[indx] + 2*np.pi

    if normalize:
       phi = phi/(np.pi*2)
    
    return phi

def normalize(data, axis=0):
    data = np.asarray(data)
    ave = np.expand_dims(np.mean(data, axis), axis)
    ret = (data - ave)/ave        
    return ret

def normmean(data, axis=0):
    data = np.asarray(data)
    ave = np.expand_dims(np.mean(data, axis), axis)
    ret = (data - ave)/ave        
    return ret

def normstd(data, axis=0):
    data = np.asarray(data)
    ave = np.expand_dims(np.mean(data, axis), axis)
    std = np.expand_dims(np.std(data, axis=axis, ddof = 1), axis)
    ret = (data - ave)/std    
    return ret    

def psd2d(sigs, t, x, dt=None, dx=None, nfft=256, noverlap=None, nensemble=1, istart=0, tstart=None, window='hann', detrend='constant'):
    """
    Estimate 2D cross power spectral density using Welch's method.
    ----------
    sigs : 2D-array
        Time series of measurement values
    t : 1D-array
        Time for sigs
    x : 1D-array
        Locations for sigs
    dt : float, optional
        Sampling time of the `sigs` time series in units of sec. Defaults
        to None.
    dx : float, optional
        Sampling space of the `sigs` time series in units of m. Defaults
        to None.
    nfft : int, optional
        Length of each segment and  Length of the FFT used. 
    noverlap : int, optional
        Number of points to overlap between segments. If None,
        ``noverlap = nperseg // 8``.  Defaults to None.
    nensemble: int, optional
        Number of ensembles  Defaults to 1.
    istart: int, optional
        Start index in the specified axis of sigs. Defaults to 0.
    tstart: float, optional
        Start time of sigs. If None, t[istart] is used. Defaults to None.
    window : str or tuple or array_like, optional
        Desired window to use. See `get_window` for a list of windows and
        required parameters. Defaults to 'hann'.
    detrend : str or function or False, optional
        Specifies how to detrend each segment. If `detrend` is a string,
        it is passed as the ``type`` argument to `detrend`.  If it is a
        function, it takes a segment and returns a detrended segment.
        If `detrend` is False, no detrending is done.  Defaults to 'constant'.
    Returns
    -------
    ft : ndarray
        Array of sample frequencies in time domain.
    fx : ndarray
        Array of sample frequencies in space domain.
    Ptx : ndarray (2D)
        Power spectral density or power spectrum of x. Pxx(f, tave).
    See Also
    --------
    Notes
    -----
        see welch
    """

    nt = len(t)
    nx = len(x)
    if dt is None: 
        dt = (t[-1]-t[0]) / (nt-1)

    if dx is None: 
        dx = (x[-1]-x[0]) / (nx-1) 

    if len(t) < nfft:
        warnings.warn('nfft = {0}, is greater than len(t) = {1}, using nfft = {2}'.format(nfft, nt, nt))
        nfft = len(t)
        
    if tstart is not None:
        indx = np.where(t >= tstart)[0]
        istart = indx[0]

    if noverlap is None:
        noverlap = nfft // 2
    elif noverlap >= nfft:
        raise ValueError('noverlap must be less than nfft.')
    
    step = nfft - noverlap
    iend = step*(nensemble-1) + nfft + istart
    if iend > nt:
        nnew = (nt-nfft-istart)/step + 1  
        warnings.warn('nensemble = {0} exceed boundary, using nensemble = {1}'.format(nt,nnew))
        nfft = nnew        

    Ptx = np.zeros((nfft, nx))

    if sigs.size == 0:
        return np.empty(nfreq), tave, Pxx

    win_t = dsp.get_window(window, nfft)
    scale_t = dt / (win_t*win_t).sum()

    win_x = dsp.get_window(window, nx)    
    scale_x = dx / (win_x*win_x).sum()
    scale = scale_t*scale_x

    if not detrend:
        detrend_func = lambda seg: seg
    elif not hasattr(detrend, '__call__'):
        detrend_func = lambda seg: dsp.detrend(seg, type=detrend)
    else:
        detrend_func = detrend

    work = np.empty((nfft, nx), dtype=complex)
    spc = np.zeros((nfft, nx), dtype=complex)
    for ne in range(nensemble):
        i0 = istart + noverlap*ne
        i1 = i0 + nfft
        for j in range(nx):
            work[:,j] = win_x[j]*win_t*detrend_func(sigs[i0:i1,j])
        spc = fft.fft2(work)
        Ptx = Ptx + (spc * spc.conj()).real
    Ptx = scale*Ptx/nensemble
    Ptx = fft.fftshift(Ptx)
    ft = fft.fftfreq(nfft, dt)
    ft = fft.fftshift(ft)
    fx = fft.fftfreq(nx, dx)
    fx = fft.fftshift(fx)
                   
    return ft, fx, Ptx

def bicoherence_simple(s1, s2, t, dt=None, nfft=256, noverlap=None, nensemble=1, istart=0, tstart=None, window='hann', detrend='constant',vood=0.0):
    """
    Compute the bicoherence between two signals of the same lengths s1 and s2
    using the function scipy.signal.spectrogram.
    This culculates bi-coherence as defined and thus is not optimized.
    ----------
    s1, s2 : 1D-array
        Time series of measurement values should have identical dimensions
    t : 1D-array
        Time for s1 and s2
    dt : float, optional
        Sampling time of the `sigs` time series in units of sec. Defaults
        to None.
    nfft : int, optional
        Length of the FFT used. 
    noverlap : int, optional
        Number of points to overlap between segments. If None,
        ``noverlap = nperseg // 8``.  Defaults to None.
    nensemble: int, optional
        Number of ensembles  Defaults to 1.
    istart: int, optional
        Start index in the specified axis of sigs. Defaults to 0.
    tstart: float, optional
        Start time of sigs. If None, t[istart] is used. Defaults to None.
    window : str or tuple or array_like, optional
        Desired window to use. See `get_window` for a list of windows and
        required parameters. Defaults to 'hann'.
    detrend : str or function or False, optional
        Specifies how to detrend each segment. If `detrend` is a string,
        it is passed as the ``type`` argument to `detrend`.  If it is a
        function, it takes a segment and returns a detrended segment.
        If `detrend` is False, no detrending is done.  Defaults to 'constant'.
    vood : float, optional
        Value of out of domain (f1 + f2 > fnyq). Defaults to 0.
    Returns
    -------
    f1 : ndarray
        Array of sample frequencies from 0 to fnyq.
    f2 : ndarray
        Array of sample frequencies from -fnyq to fnyq.
    bicoh2 : ndarray (2D)
        Squared bi-coherence of sig. bicoh2(f1, f2).
    See Also
    --------
    Notes
    -----
        see scipy.signal.spectrogram
    """

    nt = len(t)
    if dt is None: 
        dt = (t[-1]-t[0]) / (nt-1)

    if len(t) < nfft:
        warnings.warn('nfft = {0}, is greater than len(t) = {1}, using nfft = {2}'.format(nfft, nt, nt))
        nfft = len(t)
        
    if tstart is not None:
        indx = np.where(t >= tstart)[0]
        istart = indx[0]

    if noverlap is None:
        noverlap = nfft // 2
    elif noverlap >= nfft:
        raise ValueError('noverlap must be less than nfft.')
    
    step = nfft - noverlap
    iend = step*(nensemble-1) + nfft + istart
    if iend > nt:
        nnew = (nt-nfft-istart)/step + 1  
        warnings.warn('nensemble = {0} exceed boundary, using nensemble = {1}'.format(nt,nnew))
        nfft = nnew        

    # compute the stft
    f, t, spec1 = dsp.spectrogram(s1[istart:iend,...], fs=1.0/dt, window=window, nperseg=nfft, noverlap=noverlap, nfft=nfft, detrend=detrend,
                                  return_onesided=False, scaling='density', axis=0, mode='complex')
    _, _, spec2 = dsp.spectrogram(s2[istart:iend,...], fs=1.0/dt, window=window, nperseg=nfft, noverlap=noverlap, nfft=nfft, detrend=detrend,
                                   return_onesided=False, scaling='density', axis=0, mode='complex')

    # transpose (f, t) -> (t, f)
    spec1 = np.transpose(spec1, [1, 0])
    spec2 = np.transpose(spec2, [1, 0])

    # compute the bicoherence
    nf = f.size
    if nf % 2 == 0 :
       nhalf = nf//2
    else:
       nhalf = (nf+1)//2
    out_shape = list(spec1.shape)
    out_shape[0] = nhalf
    out_shape[1] = nf
    num = np.ones(out_shape)*vood
    denum = np.ones(out_shape)
        
    for i in range(nhalf):
        for j in range(nf):
            k = i + j
            if j < nhalf :
                if k <= nhalf-1:
                    num[i,j,...] =  np.abs(np.mean(spec1[:,i,...] * spec1[:,j,...] * np.conjugate(spec2[:,k,...]), axis=0))**2
                    denum[i,j,...] = np.mean(np.abs(spec1[:,i,...] * spec1[:,j,...])**2, axis=0) * np.mean(np.abs(np.conjugate(spec2[:,k,...]))**2, axis=0)
                continue
            if k >= nf :
                k = k - nf
            num[i,j,...] =  np.abs(np.mean(spec1[:,i,...] * spec1[:,j,...] * np.conjugate(spec2[:,k,...]), axis=0))**2
            denum[i,j,...] = np.mean(np.abs(spec1[:,i,...] * spec1[:,j,...])**2, axis=0) * np.mean(np.abs(np.conjugate(spec2[:,k,...]))**2, axis=0)
    bicoh2 = num / denum
                    
    bicoh2 = fft.fftshift(bicoh2, axes = 1)
    f1 = f[0:nhalf]
    f2 = fft.fftshift(f)
    return f1, f2, bicoh2

def abicoh2(sig, t, dt=None, nfft=256, noverlap=None, nensemble=1, istart=0, tstart=None, window='hann', detrend='constant',vood=0.0,axisf1=0):
    """
    Compute the squared-auto-bicoherence using the function scipy.signal.spectrogram
    ----------
    sig : 1D-array
        Time series of measurement values
    t : 1D-array
        Time for sig
    dt : float, optional
        Sampling time of the `sigs` time series in units of sec. Defaults
        to None.
    nfft : int, optional
        Length of the FFT used. 
    noverlap : int, optional
        Number of points to overlap between segments. If None,
        ``noverlap = nperseg // 8``.  Defaults to None.
    nensemble: int, optional
        Number of ensembles  Defaults to 1.
    istart: int, optional
        Start index in the specified axis of sigs. Defaults to 0.
    tstart: float, optional
        Start time of sigs. If None, t[istart] is used. Defaults to None.
    window : str or tuple or array_like, optional
        Desired window to use. See `get_window` for a list of windows and
        required parameters. Defaults to 'hann'.
    detrend : str or function or False, optional
        Specifies how to detrend each segment. If `detrend` is a string,
        it is passed as the ``type`` argument to `detrend`.  If it is a
        function, it takes a segment and returns a detrended segment.
        If `detrend` is False, no detrending is done.  Defaults to 'constant'.
    vood : float, optional
        Value of out of domain (f1 + f2 > fnyq). Defaults to 0.
    axisf1 : int, optional
        Axis for f1. Defaults to 0. If axisf1 /= 0, axises [0, 1] of bicoh2 are transposed.
    Returns
    -------
    f1 : ndarray
        Array of sample frequencies from 0 to fnyq.
    f2 : ndarray
        Array of sample frequencies from -fnyq to fnyq.
    bicoh2 : ndarray (2D)
        Squared bicoherence of sig. bicoh2(f1, f2).
    See Also
    --------
    Notes
    -----
        see scipy.signal.spectrogram
    """

    nt = len(t)
    if dt is None: 
        dt = (t[-1]-t[0]) / (nt-1)

    if len(t) < nfft:
        warnings.warn('nfft = {0}, is greater than len(t) = {1}, using nfft = {2}'.format(nfft, nt, nt))
        nfft = len(t)
        
    if tstart is not None:
        indx = np.where(t >= tstart)[0]
        istart = indx[0]

    if noverlap is None:
        noverlap = nfft // 2
    elif noverlap >= nfft:
        raise ValueError('noverlap must be less than nfft.')
    
    step = nfft - noverlap
    iend = step*(nensemble-1) + nfft + istart
    if iend > nt:
        nnew = (nt-nfft-istart)/step + 1  
        warnings.warn('nensemble = {0} exceed boundary, using nensemble = {1}'.format(nt,nnew))
        nfft = nnew        

    # compute the stft
    f, t, spec = dsp.spectrogram(sig[istart:iend,...], fs=1.0/dt, window=window, nperseg=nfft, noverlap=noverlap, nfft=nfft, detrend=detrend,
                                  return_onesided=False, scaling='density', axis=0, mode='complex')

    # transpose (f, t) -> (t, f)
    spec = np.transpose(spec, [1, 0])

    # compute the bicoherence
    nf = f.size
    if nf % 2 == 0 :
       nhalf = nf//2
    else:
       nhalf = (nf+1)//2
    fcol = np.arange(nf, dtype=int)
    lrow = np.arange(nhalf, dtype=int)
    lrow = np.roll(lrow,1)
    ha = la.hankel(fcol,lrow)
    num = np.abs(
                 np.mean(spec[:, fcol, None] * spec[:, None, lrow] * np.conjugate(spec[:, ha]), axis=0)
                ) ** 2
    denum = np.mean(
                    np.abs(spec[:, fcol, None] * spec[:, None, lrow]) ** 2, axis=0) * np.mean(np.abs(np.conjugate(spec[:, ha])) ** 2, axis=0
                    )
    bicoh2 = num / denum

    for i in range(nhalf):
        for j in range(nhalf):
            k = i + j
            if k > nhalf-1:
                bicoh2[i,j,...] = vood    
               
    bicoh2 = fft.fftshift(bicoh2, axes = 0)
    if axisf1 == 0 :
        bicoh2 = np.transpose(bicoh2, [1, 0])
    f1 = f[0:nhalf]
    f2 =  fft.fftshift(f)
    return f1, f2, bicoh2

def xbicoh2(s1, s2, t, dt=None, nfft=256, noverlap=None, nensemble=1, istart=0, tstart=None,  comb='112', window='hann', detrend='constant',vood=0.0,axisf1=0):
    """
    Compute the squared cross-bicoherence between two signals of the same lengths s1 and s2
    using the function scipy.signal.spectrogram
    ----------
    s1, s2 : 1D-array
        Time series of measurement values should have identical dimensions
    t : 1D-array
        Time for s1 and s2
    dt : float, optional
        Sampling time of the `sigs` time series in units of sec. Defaults
        to None.
    nfft : int, optional
        Length of the FFT used. 
    noverlap : int, optional
        Number of points to overlap between segments. If None,
        ``noverlap = nperseg // 8``.  Defaults to None.
    nensemble: int, optional
        Number of ensembles  Defaults to 1.
    istart: int, optional
        Start index in the specified axis of sigs. Defaults to 0.
    tstart: float, optional
        Start time of sigs. If None, t[istart] is used. Defaults to None.
    comb: str, optional
        Deefinition of cross-bicoherence.
        "112": B(f1,f2) = S1(f1)S1(f2)S2(f1+f2)
        "121": B(f1,f2) = S1(f1)S2(f2)S1(f1+f2) 
        "122": B(f1,f2) = S1(f1)S2(f2)S2(f1+f2) 
    window : str or tuple or array_like, optional
        Desired window to use. See `get_window` for a list of windows and
        required parameters. Defaults to 'hann'.
    detrend : str or function or False, optional
        Specifies how to detrend each segment. If `detrend` is a string,
        it is passed as the ``type`` argument to `detrend`.  If it is a
        function, it takes a segment and returns a detrended segment.
        If `detrend` is False, no detrending is done.  Defaults to 'constant'.
    vood : float, optional
        Value of out of domain (f1 + f2 > fnyq). Defaults to 0.
    axisf1 : int, optional
        Axis for f1. Defaults to 0. If axisf1 /= 0, axises [0, 1] of bicoh2 are transposed.
    Returns
    -------
    f1 : ndarray
        Array of sample frequencies from 0 to fnyq.
    f2 : ndarray
        Array of sample frequencies from -fnyq to fnyq.
    bicoh2 : ndarray (2D)
        Squared bi-coherence of sig. bicoh2(f1, f2).
    See Also
    --------
    Notes
    -----
        see scipy.signal.spectrogram
    """

    nt = len(t)
    if dt is None: 
        dt = (t[-1]-t[0]) / (nt-1)

    if len(t) < nfft:
        warnings.warn('nfft = {0}, is greater than len(t) = {1}, using nfft = {2}'.format(nfft, nt, nt))
        nfft = len(t)
        
    if tstart is not None:
        indx = np.where(t >= tstart)[0]
        istart = indx[0]

    if noverlap is None:
        noverlap = nfft // 2
    elif noverlap >= nfft:
        raise ValueError('noverlap must be less than nfft.')
    
    step = nfft - noverlap
    iend = step*(nensemble-1) + nfft + istart
    if iend > nt:
        nnew = (nt-nfft-istart)/step + 1  
        warnings.warn('nensemble = {0} exceed boundary, using nensemble = {1}'.format(nt,nnew))
        nfft = nnew        

    # compute the stft
    f, t, spec1 = dsp.spectrogram(s1[istart:iend,...], fs=1.0/dt, window=window, nperseg=nfft, noverlap=noverlap, nfft=nfft, detrend=detrend,
                                  return_onesided=False, scaling='density', axis=0, mode='complex')
    _, _, spec2 = dsp.spectrogram(s2[istart:iend,...], fs=1.0/dt, window=window, nperseg=nfft, noverlap=noverlap, nfft=nfft, detrend=detrend,
                                   return_onesided=False, scaling='density', axis=0, mode='complex')

    # transpose (f, t) -> (t, f)
    spec1 = np.transpose(spec1, [1, 0])
    spec2 = np.transpose(spec2, [1, 0])

    # compute the bicoherence
    nf = f.size
    if nf % 2 == 0 :
       nhalf = nf//2
    else:
       nhalf = (nf+1)//2
    fcol = np.arange(nf, dtype=int)
    lrow = np.arange(nhalf, dtype=int)
    lrow = np.roll(lrow,1)
    ha = la.hankel(fcol,lrow)

    X = spec1
    if comb == "121":
        Y = spec2
        Z = spec1
    elif comb == "122":
        Y = spec2
        Z = spec2
    else:       
        Y = spec1
        Z = spec2
        
    num = np.abs(
                 np.mean(X[:, fcol, None] * Y[:, None, lrow] * np.conjugate(Z[:, ha]), axis=0)
#                 np.mean(spec1[:, fcol, None] * spec1[:, None, lrow] * np.conjugate(spec2[:, ha]), axis=0)
                ) ** 2
    denum = np.mean(
                    np.abs(X[:, fcol, None] * Y[:, None, lrow]) ** 2, axis=0) * np.mean(np.abs(np.conjugate(Z[:, ha])) ** 2, axis=0
#                    np.abs(spec1[:, fcol, None] * spec1[:, None, lrow]) ** 2, axis=0) * np.mean(np.abs(np.conjugate(spec2[:, ha])) ** 2, axis=0
                    )
    bicoh2 = num / denum

    for i in range(nhalf):
        for j in range(nhalf):
            k = i + j
            if k > nhalf-1:
                bicoh2[i,j,...] = vood    
               
    bicoh2 = fft.fftshift(bicoh2, axes = 0)
    if axisf1 == 0 :
        bicoh2 = np.transpose(bicoh2, [1, 0])
    f1 = f[0:nhalf]
    f2 =  fft.fftshift(f)
    return f1, f2, bicoh2

def summedbicoh2(bicoh2, f1, f2, axisf1=0):
    """
    Compute the summed bicoherence
    ----------
    f1 : ndarray
        Array of sample frequencies from 0 to fnyq.
    f2 : ndarray
        Array of sample frequencies from -fnyq to fnyq.
    bicoh2 : ndarray (2D)
        Squared bi-coherence. bicoh2(f1, f2).
    Returns
    -------
    summed : ndarray
        Summed bi-coherence
    """

    nf = f1.size
    out_shape = list(bicoh2.shape)
    if axisf1 == 0 :
        del out_shape[0]
    else :
        del out_shape[1]
    out_shape[0] = nf
    summed = np.zeros(out_shape)
    nzero = f2.size - nf
       
    for k in range(nf):
        for i in range(nf):
            j = k - i + nzero
            if axisf1 == 0 :
                summed[k,...] = summed[k,...] + bicoh2[i,j,...]
            else:
                summed[k,...] = summed[k,...] + bicoh2[j,i,...]
                
    return summed

def periodogram(sig, t, dt=None, nfft=256, noverlap=None, istart=0, iend=-1, tstart=None, tend=None, window='hann', detrend='constant',fshift=True):
    """
    Estimate temporal evolution of spectral density (Short Time Fourie Transform) 
    by dividing the data into segments, computing a modified periodogram for each segment.

    Parameters
    ----------
    sig : array
        Time series of measurement values. sig[:,n] means temporal evolution of (n-1)th signal.
    t : 1D-array
        Time for x
    dt : float, optional
        Sampling time of the `sig` time series in units of sec. Defaults
        to None.
    nfft : int, optional
        Length of each segment and  Length of the FFT used. 
    noverlap : int, optional
        Number of points to overlap between segments. If None,
        ``noverlap = nfft // 2``.  Defaults to None.
    istart: int, optional
        Start index in the specified axis of sigs. Defaults to 0.
    tstart: float, optional
        Start time of sigs. If None, t[istart] is used. Defaults to None.
    iend: int, optional
        Last index in the specified axis of sigs. Defaults to -1.
    tend: float, optional
        End time of sigs. If None, t[iend] is used. Defaults to None.
    window : str or tuple or array_like, optional
        Desired window to use. See `get_window` for a list of windows and
        required parameters. Defaults to 'hann'.
    detrend : str or function or False, optional
        Specifies how to detrend each segment. If `detrend` is a string,
        it is passed as the ``type`` argument to `detrend`.  If it is a
        function, it takes a segment and returns a detrended segment.
        If `detrend` is False, no detrending is done.  Defaults to 'constant'.
    fshift : logical, optional
        Shift the zero-frequency component to the center of the spectrum. Defaults to True.

    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    tave : ndarray
        Array of sample averaged times.
    S : ndarray
        Complex spectrum density of sig. (f, tave).

    See Also
    --------
    stft

    Notes
    -----

    References
    ----------

    """
    if dt is None: 
        dt = (t[-1]-t[0]) / (len(t)-1) 

    if tstart is not None:
        indx = np.where(t >= tstart)[0]
        istart = indx[0]

    if tend is not None:
        if tend >= t[-1]:
            iend = len(t)
        else:
            indx = np.where(t > tend)[0]
            iend = indx[0] 
   
    if iend == -1:
        iend = len(t)

    if noverlap is None:
        noverlap = nfft//2
        
    f, tave, S = dsp.spectrogram(sig[istart:iend,...], fs=1.0/dt, window=window, nperseg=nfft, noverlap=noverlap, nfft=nfft, detrend=detrend,
                                   return_onesided=False, scaling='density', axis=0, mode='complex')
    tave = tave + t[0] - dt
    if fshift :
        S = fft.fftshift(S, axes = 0)
        f = fft.fftshift(f)
    return f, tave, S


def xpower(Sx, Sy):
    """
    Estimate cross power spectral density.
    Auto power spectral density can be obtained by xpower(Sx, Sx).real.

    Parameters
    ----------
    Sx : 2-D array
        Spectral density of reference values. Output from stft.
    Sy : 2-D array
        Spectral density of measurement values. Output from stft.

    Returns
    -------
    Pxy : array
        Cross power spectral density between Sx and Sy.

    See Also
    --------
    stft

    Notes
    -----

    References
    ----------

    """
    out_shape = list(Sy.shape)
    Pxy = np.zeros(out_shape)
    Pxy = Sx*Sy.conj()
    
    return Pxy

def eave(P):
    """
    Ensemble average of time series of power spectral density.

    Parameters
    ----------
    P : array
        Time series of power spectral density. Output from xpower.

    Returns
    -------
    Pav : array
        Ensemble averaged power spectral density of P.

    See Also
    --------
    psd:power spectrum density
    running: running FFT

    Notes
    -----

    References
    ----------

    """ 

    # S(f, t, ...) 

    Pav = np.mean(P, axis=1)

    return Pav

def xbispct(S1, S2, f, normalize=True, fshifted=True, comb = "112", vood=0.0, axisf1=0):
    """
    Compute ensemble averaged bi-spectrum between two signals of the same lengths s1 and s2.
   
    Parameters   
    ----------
    S1, S2　: 2-D Array
        Temporal evolution of spectral density of signals of s1 and s2. Output from stft. 
        They should have identical dimensions.
    f : 1D-array
        Array of sample frequencies
    normalize : logical, optional
    fshifted : logical, optional
    comb: str, optional
        Deefinition of cross-bicoherence.
        "112": B(f1,f2) = S1(f1)S1(f2)S2(f1+f2)
        "121": B(f1,f2) = S1(f1)S2(f2)S1(f1+f2) 
        "122": B(f1,f2) = S1(f1)S2(f2)S2(f1+f2) 
    vood : float, optional
        Value of out of domain (f1 + f2 > fnyq). Defaults to 0.
    axisf1 : int, optional
        Axis for f1. Defaults to 0. If axisf1 /= 0, axises [0, 1] of bicoh2 are transposed.

    Returns
    -------
    f1 : array
        Array of sample frequencies from 0 to fnyq.
    f2 : array
        Array of sample frequencies from -fnyq to fnyq.
    bispct : ndarray (2D)
        Bi-spectrum S1(f1)S2(f2)S3^(f1+f2).

    See Also
    --------

    Notes
    -----
        see scipy.signal.spectrogram
    """

    # transpose (f, t) -> (t, f)
    spec1 = np.transpose(S1, [1, 0])
    if fshifted :
        spec1 = fft.fftshift(spec1, axes=1)

    if comb == "121":
        spec2 = np.transpose(S2, [1, 0])
        if fshifted :
            spec2 = fft.fftshift(spec2, axes=1)
        spec3 = spec1
    elif comb == "122":
        spec2 = np.transpose(S2, [1, 0])
        if fshifted :
            spec2 = fft.fftshift(spec2, axes=1) 
        spec3 = spec2
    else:       
        spec3 = np.transpose(S2, [1, 0])
        if fshifted :
            spec3 = fft.fftshift(spec3, axes=1) 
        spec2 = spec1
        
    # compute the bicoherence
    nf = f.size
    if nf % 2 == 0 :
       nhalf = nf//2
    else:
       nhalf = (nf+1)//2
    fcol = np.arange(nf, dtype=int)
    lrow = np.arange(nhalf, dtype=int)
    lrow = np.roll(lrow,1)
    ha = la.hankel(fcol,lrow)

    bispct =  np.mean(spec1[:, fcol, None] * spec2[:, None, lrow] * np.conjugate(spec3[:, ha]), axis=0)

    if normalize : 
        norm = np.mean(
                    np.abs(spec1[:, fcol, None] * spec2[:, None, lrow]) ** 2, axis=0) * np.mean(np.abs(np.conjugate(spec3[:, ha])) ** 2, axis=0
                    )
        bispct = bispct / np.sqrt(norm) 

    for i in range(nhalf):
        for j in range(nhalf):
            k = i + j
            if k > nhalf-1:
                bispct[i,j,...] = vood    

    if fshifted:
        bispct = fft.fftshift(bispct, axes=0)
        
    if axisf1 == 0 :
        bispct = np.transpose(bispct, [1, 0])

    fdummy = fft.fftshift(f)
    if fshifted :
        f1 = fdummy[0:nhalf]
        f2 = f
    else:
        f1 = f[0:nhalf]
        f2 = fdummy
        
    return f1, f2, bispct

def iqdemod(sig, lo, lowpass_n, lowpass_d, axist=0, analytic=False):
    """
    Realize digital quadrature demodulator.
    In-phase (I) and quadrature (Q) signals are provided.
    A signal is multiplied by a local oscillator signal (for the I channel) and the local signal shifted by 90° (for the Q channel). 
    After low-pass-filtering, I and Q signals are obtained.
    
    Parameters   
    ----------
    sig : ndarray
        Array of time-series-data.
    lo　: ndarray
        local signal
    lowpass_n : (N,) array_like
        The numerator coefficient vector of the low-pass filter.
    lowpass_d : (N,) array_like
        The denominator coefficient vector of the low-pass filter. 
        If lowpass_d[0] is not 1, then both lowpass_n and lowpass_d are normalized by lowpass_d[0].
    axist : int, optional
        Axis for time. Defaults to 0.
    analytic : logical, optional
        Whether lo is an analytic signal or not. Defaults to False.

    Returns
    -------
    demod: complex ndarray
        Array of time-series-data. 

    See Also
    --------

    Notes
    -----
        see scipy.signal.hilbert
    """

    if analytic :
        analytic_lo = lo
    else:
        analytic_lo = dsp.hilbert(lo)
    demod = dsp.filtfilt(lowpass_n, lowpass_d, sig*analytic_lo, axis=axist)

    return demod

def upconv(sig, lo, axist=0, analytic=False):
    """
    Realize frequency up-convertor.
    
    
    Parameters   
    ----------
    sig : ndarray
        Array of time-series-data.
    lo　: ndarray
        local signal
    axist : int, optional
        Axis for time. Defaults to 0.
    analytic : logical, optional
        Whether lo and sig are analytic signals or not. Defaults to False.

    Returns
    -------
    up: ndarray
        Array of time-series-data. 

    See Also
    --------

    Notes
    -----
        see scipy.signal.hilbert
    """

    if analytic :
        analytic_lo = lo
        analytic_sig = sig
    else:
        analytic_lo = dsp.hilbert(lo)
        analytic_sig = dsp.hilbert(sig, axis=axist)
    analytic_sig = analytic_sig.conjugate()
    if axist == 0 :       
        up = np.real(analytic_sig.T*analytic_lo)
        up = up.transpose()
    else:
        up = np.real(analytic_sig*analytic_lo)
    return up

def tdiffp(analytic, len_win, tsamp, axist = 0):
   """
    Compute time difference of phase where phase is defined by analytic_signal = A(t)cos(phi(t))+iA(t)sin(phi(t))=I(t)+iQ(t).
    Usually phi(t) is given by ATAN(Q/I). Then dphi/dt = xdy/dt - ydx/dt, x = I/A(t), y = Q/A(t).
    Caution very high cost!
    
    Parameters   
    ----------
    analytic : ndarray
        Analytic signal for time-seies data.
    len_win　: int
        The length of the filter window (i.e. the number of coefficients). len_win must be a positive odd integer.
    tsamp : float
        Sampling time.
    axist : int, optional
        Axis for time. Defaults to 0.

    Returns
    -------
    dphi: ndarray
        Array of dphi/dt. 

    See Also
    --------

    Notes
    -----
        see scipy.signal.savgol_filter
   """    

   A = np.abs(analytic)
   normalized = analytic/A
   deriv_real = dsp.savgol_filter(normalized.real, len_win, 1, deriv=1, delta=tsamp, axis=axist)
   deriv_imag = dsp.savgol_filter(normalized.imag, len_win, 1, deriv=1, delta=tsamp, axis=axist)
   dphi = normalized.real*deriv_imag - normalized.imag*deriv_real

   return dphi

def regdiff_down(x, y, dx, len_win):
   """
    Compute differentiate y with respect to x, where x and y are regularly spaced data.
    Apply a Savitzky-Golay filter to y but data are down-sampled to a multiple of about 2/(window length).
    
    Parameters   
    ----------
    x: ndarray
        The independentt variable x.
    y: ndarray
        The data to be filltered. 
    dx: float, optional
        The spacing of the samples. 
    len_win　: int
        The length of the filter window (i.e. the number of coefficients). len_win must be a positive odd integer.

    Returns
    -------
    xdown: ndarray
        Down-sampled x-data.
    ydiff: ndarray
        Down-sampled dy/dx-data
    yerr: ndarray
        Standard deviations of the dy/dx 

    See Also
    --------

    Notes
    -----
        see scipy.signal.savgol_filter
   """
   m = (len_win-1)//2
   dlen = len(x)
   dlen_down = dlen//(m+1) - 1
   ydiff = np.zeros(dlen_down)
   xdown = np.zeros(dlen_down)
   ydifferr = np.zeros(dlen_down)
   dummy = np.arange(1.0,len_win+1)
   sx = np.sum(dummy)
   sxx = np.sum(dummy*dummy)
   denomi = len_win*sxx - sx*sx

   for i in range(dlen_down):
       j0 = i*(m+1)
       jc = j0 + m
       j1 = jc + m + 1
       xdown[i] = x[jc]
       sxy = np.sum(dummy*y[j0:j1])
       sy = np.sum(y[j0:j1])
       a = (len_win*sxy-sx*sy)/denomi
       b = (sxx*sy-sx*sxy)/denomi
       res = y[j0:j1] - a*dummy - b
       stdev = np.sqrt(np.sum(res*res)/(len_win-2))
       ydiff[i] = a/dx
       ydifferr[i] = stdev*np.sqrt(len_win/denomi)/dx

   return xdown, ydiff, ydifferr


def irrdiff_down(x, y, len_win):
   """
    Compute differentiate y with respect to x, where x and y are irregularly spaced data.
    Apply a Savitzky-Golay filter to y but data are down-sampled to a multiple of about 2/(window length).
    
    Parameters   
    ----------
    x: ndarray
        The independentt variable x.
    y: ndarray
        The data to be filltered. 
    len_win　: int
        The length of the filter window (i.e. the number of coefficients). len_win must be a positive odd integer.

    Returns
    -------
    xdown: ndarray
        Down-sampled x-data.
    ydiff: ndarray
        Down-sampled dy/dx-data
    yerr: ndarray
        Standard deviations of the dy/dx 

    See Also
    --------

    Notes
    -----
        see scipy.signal.savgol_filter
   """
   m = (len_win-1)//2
   dlen = len(x)
   dlen_down = dlen//(m+1) - 1
   ydiff = np.zeros(dlen_down)
   xdown = np.zeros(dlen_down)
   ydifferr = np.zeros(dlen_down)
   for i in range(dlen_down):
       j0 = i*(m+1)
       jc = j0 + m
       j1 = jc + m + 1
       popt, pconv = np.polyfit(x[j0:j1], y[j0:j1], 1, cov=True)
       xdown[i] = np.sum(x[j0:j1])/len_win
       ydiff[i] = popt[0]
       ydifferr[i] = np.sqrt(np.diag(pconv))[0]
   return xdown, ydiff, ydifferr

def create_basephase(t, f, delay):
    """
    Create a normalized phase of monocyclic signal below.

       phase
         |
       1 +
         |  /|  /|  /|
         | / | / | / |
         |/  |/  |/  |
       0 +---+---+---+-- t - delay
         0   T  2T  3T  


    Parameters   
    ----------
    t: ndarray
        Time of the time-series data. Typically t = (1/fs)*np.arrange(n), where fs is sampling frequency.
    f: float
        Frequency of the time-series data, f = 1/T (T is period of the signal).
    delay: float
        Phase delay converted into time. Phase becomes zero at t = delay.

    Returns
    -------
    phase: ndarray
        Normalized phase (0 ~ 1).

    See Also
    --------

    Notes
    -----

    """
    phase = f*(t - delay)
    phase = phase - np.floor(phase)
    return phase

def create_triangle(t, f, delay, Amp, offset):
    """
    Create a triangle wave as a function of t.

            triangle
              |
          Amp +
              |  /\    /\
              | /  \  /  \
              |/    \/    \ 
       offset +------+-----+-- t - delay
              0      T    2T  


    Parameters   
    ----------
    t: ndarray
        Time of the time-series data.
    f: float
        Frequency of the time-series data, f = 1/T (T is period of the signal).
    delay: float
        Phase delay converted into time. Phase becomes zero at t = delay.
    Amp: float
        Amplitude of the triangle wave.
    offset: float
        DC Offset bias at phase = 0.

    Returns
    -------
    triangle: ndarray
        Triangle wave.

    See Also
    --------
        Create_basephase 
       
    Notes
    -----

    """
    phase = create_basephase(t, f, delay)
    triangle = np.where(phase >= 0.5,1.0-phase, phase)
    triangle = 2.0*triangle
    triangle = Amp*triangle + offset
    return triangle

def create_sawtooth(t, f, delay, Amp, offset):
    """
    Create a sawtooth wave as a function of t.

           sawtooth
              |
          Amp +
              |  /|  /|  /|
              | / | / | / |
              |/  |/  |/  |
       offset +---+---+---+-- t - delay
              0   T  2T  3T  


    Parameters   
    ----------
    t: ndarray
        Time of the time-series data.
    f: float
        Frequency of the time-series data, f = 1/T (T is period of the signal).
    delay: float
        Phase delay converted into time. Phase becomes zero at t = delay.
    Amp: float
        Amplitude of the sawtooth wave.
    offset: float
        DC Offset bias at phase = 0.

    Returns
    -------
    sawtooth: ndarray
        Sawtooth wave.

    See Also
    --------
        Create_basephase 
       
    Notes
    -----

    """
    sawtooth = create_basephase(t, f, delay)
    sawtooth = Amp*sawtooth + offset
    return sawtooth

def create_square(t, f, delay, Amp, offset):
    """
        Create a square wave as a function of t.

            square
              |
          Amp +  ----  ---- 
              |  |  |  |  |
              |  |  |  |  |
              |  |  |  |  | 
       offset +-----+-----+-- t - delay
              0     T    2T  


    Parameters   
    ----------
    t: ndarray
        Time of the time-series data.
    f: float
        Frequency of the time-series data, f = 1/T (T is period of the signal).
    delay: float
        Phase delay converted into time. Phase becomes zero at t = delay.
    Amp: float
        Amplitude of the square wave.
    offset: float
        DC Offset bias at phase = 0.

    Returns
    -------
    square: ndarray
        Square wave.

    See Also
    --------
        Create_basephase 
       
    Notes
    -----

    """
    phase = create_basephase(t, f, delay)
    square = np.where(phase >= 0.5,1.0, 0.0)    
    square  = Amp*square + offset
    return square 

     
