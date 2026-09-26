import sys
import numpy as np
import scipy.signal as dsp
from pathlib import Path

# Add paths for TE and mhd_common
sys.path.append(r"c:\TFG\jpack")
sys.path.append(r"c:\TFG\analysis")
import turnelib as TE
from mhd_common import extract_instantaneous_frequency

edf_mag = TE.edf()
dat_mag = edf_mag.load(r"c:\TFG\data\hj88653\MP1@88653.edf")
t = dat_mag[:, 0]
ys = dat_mag[:, 1]
if edf_mag.DimUnit[0] == "s":
    t *= 1000.0

dt = (t[-1] - t[0]) / (len(t) - 1)
fs = 1.0 / (dt / 1000.0)

t0, t1 = 259.1, 275.0
mask = (t >= t0) & (t <= t1)

for fl, fu in [(40000.0, 80000.0), (80000.0, 120000.0)]:
    print(f"--- Band: {fl/1000}-{fu/1000} kHz ---")
    for sg_win in [525, 88653]:
        _, _, _, ifreq_hz = extract_instantaneous_frequency(ys, fs, fl, fu, 4, sg_win)
        mean_ifreq = np.nanmean(ifreq_hz[mask]) / 1000.0
        print(f"sg_win = {sg_win}: mean f_inst = {mean_ifreq:.2f} kHz")

