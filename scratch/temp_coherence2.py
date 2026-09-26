
import sys
import numpy as np
from scipy import signal
sys.path.append('c:/TFG')
from jpack.turnelib import myjread

def get_signal(name):
    t, v, _ = myjread('c:/TFG/data/hj88653/'+name+'@88653.edf')
    return t, v

t_mp1, v_mp1 = get_signal('MP1')
_, v_ece03 = get_signal('ECE03FAST')
_, v_ece12 = get_signal('ECE12FAST')
_, v_ece13 = get_signal('ECE13FAST')

idx = (t_mp1 >= 259.1) & (t_mp1 <= 275.0)
v_mp1_w = v_mp1[idx]

for ece_name, v_ece in [('ECE03FAST', v_ece03[idx]), ('ECE12FAST', v_ece12[idx]), ('ECE13FAST', v_ece13[idx])]:
    freqs, coh = signal.coherence(v_mp1_w, v_ece, fs=1e6, nperseg=1024, noverlap=512)
    mask_band = (freqs >= 80000) & (freqs <= 100000)
    band_coh = coh[mask_band].mean()
    idx_89 = abs(freqs - 89000).argmin()
    bin_coh = coh[idx_89]
    print(f'{ece_name}: Band (80-100kHz) mean = {band_coh:.4f}, Single bin 89kHz = {bin_coh:.4f}')
