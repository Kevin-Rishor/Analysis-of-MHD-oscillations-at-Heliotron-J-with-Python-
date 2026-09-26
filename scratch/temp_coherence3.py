
import sys
from pathlib import Path
sys.path.append('c:/TFG')
sys.path.append('c:/TFG/jpack')
import turnelib as TE
import numpy as np
import scipy.signal as dsp

edf = TE.edf()
edf.read('c:/TFG/data/hj88653/MP1@88653.edf')
t = edf.T()
v_mp1 = edf.V()

edf.read('c:/TFG/data/hj88653/ECE03FAST@88653.edf')
v_ece03 = edf.V()

edf.read('c:/TFG/data/hj88653/ECE12FAST@88653.edf')
v_ece12 = edf.V()

edf.read('c:/TFG/data/hj88653/ECE13FAST@88653.edf')
v_ece13 = edf.V()

idx = (t >= 259.1/1000) & (t <= 275.0/1000)
if np.sum(idx) == 0:
    idx = (t >= 259.1) & (t <= 275.0)

v_mp1_w = v_mp1[idx]
print(f'Window size: {len(v_mp1_w)}')

for ece_name, v_ece in [('ECE03FAST', v_ece03[idx]), ('ECE12FAST', v_ece12[idx]), ('ECE13FAST', v_ece13[idx])]:
    freqs, coh = dsp.coherence(v_mp1_w, v_ece, fs=1e6, nperseg=1024, noverlap=512)
    mask_band = (freqs >= 80000) & (freqs <= 100000)
    band_coh = coh[mask_band].mean()
    idx_89 = abs(freqs - 89000).argmin()
    bin_coh = coh[idx_89]
    print(f'{ece_name}: Band (80-100kHz) mean = {band_coh:.4f}, Single bin 89kHz = {bin_coh:.4f}')
