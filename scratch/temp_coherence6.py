
import sys
import numpy as np
import scipy.signal as dsp
sys.path.append('c:/TFG')
sys.path.append('c:/TFG/jpack')
import turnelib as TE

def get_signal(fname):
    edf = TE.edf()
    dat = edf.load(fname)
    t = np.linspace(edf.DimSta[0], edf.DimSta[0] + edf.DimLen[0] * edf.DimRes[0], edf.DimLen[0], endpoint=False)
    return t, np.array(dat)

t, v_mp1 = get_signal('c:/TFG/data/hj88653/MP1@88653.edf')
_, v_ece03 = get_signal('c:/TFG/data/hj88653/ECE3FAST@88653.edf')
_, v_ece12 = get_signal('c:/TFG/data/hj88653/ECE12FAST@88653.edf')
_, v_ece13 = get_signal('c:/TFG/data/hj88653/ECE13FAST@88653.edf')

idx = (t >= 259.1) & (t <= 275.0)

for ece_name, v_ece in [('ECE03FAST', v_ece03[idx]), ('ECE12FAST', v_ece12[idx]), ('ECE13FAST', v_ece13[idx])]:
    freqs, coh = dsp.coherence(v_mp1[idx], v_ece, fs=1e6, nperseg=1024, noverlap=512)
    mask_band = (freqs >= 80000) & (freqs <= 100000)
    band_coh = coh[mask_band].mean()
    idx_89 = abs(freqs - 89000).argmin()
    bin_coh = coh[idx_89]
    print(f'{ece_name}: Band (80-100kHz) mean = {band_coh:.4f}, Single bin 89kHz = {bin_coh:.4f}')
