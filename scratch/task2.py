import sys
from pathlib import Path
import numpy as np
import scipy.signal as dsp

root_dir = Path("c:/TFG")
for p_add in [root_dir / "jpack", root_dir / "analysis", root_dir / "analysis" / "common"]:
    if str(p_add) not in sys.path:
        sys.path.append(str(p_add))

import turnelib as TE

shot = 88653
data_dir = root_dir / "data" / f"hj{shot}"
t_start, t_end = 259.1, 275.0
edf = TE.edf()

# Load MP1
fpath_mp1 = data_dir / f"MP1@{shot}.edf"
dat_mp1 = edf.load(str(fpath_mp1))
t_ms = dat_mp1[:, 0]
if edf.DimUnit[0] != "ms":
    t_ms = t_ms * 1000.0
x_mp1 = dat_mp1[:, 1]
dt = (t_ms[1] - t_ms[0]) / 1000.0
fs = 1.0 / dt
idx_win = np.where((t_ms >= t_start) & (t_ms <= t_end))[0]
x_mp1_win = x_mp1[idx_win]

nperseg = 2048
noverlap = 1024

print("=== Task 2a: 95% floor for ECE-MP1 ===")
n_seg = int(np.floor((len(x_mp1_win) - nperseg) / (nperseg - noverlap))) + 1
sig_floor = 1.0 - (0.05) ** (1.0 / max(1, n_seg - 1))
print(f"95% floor: {sig_floor:.4f}")
print("ECE9 (0.105), ECE10 (0.045), ECE12 (0.112), ECE13 (0.1055) vs floor:")
for ch, val in [("ECE9", 0.105), ("ECE10", 0.045), ("ECE12", 0.112), ("ECE13", 0.1055)]:
    print(f"{ch}: {val:.4f} -> {'Exceeds' if val > sig_floor else 'Below'} floor")

print("\n=== Task 2c: RMS for ECE11, ECE10, ECE12 ===")
quiet_start, quiet_end = 400.0, 410.0
idx_quiet = np.where((t_ms >= quiet_start) & (t_ms <= quiet_end))[0]
mode_freq = 89.0
b_band, a_band = dsp.bessel(4, [80000.0 / (fs / 2.0), 100000.0 / (fs / 2.0)], btype="bandpass")

for ch in [10, 11, 12]:
    f_ece = data_dir / f"ECE{ch}FAST@{shot}.edf"
    dat_ece = edf.load(str(f_ece))
    comm = edf.comments
    y_ece = dat_ece[:, 1]
    
    y_win = y_ece[idx_win]
    y_quiet = y_ece[idx_quiet]
    
    y_win_fluc = dsp.filtfilt(b_band, a_band, y_win)
    y_quiet_fluc = dsp.filtfilt(b_band, a_band, y_quiet)
    
    rms_win = np.std(y_win_fluc)
    rms_quiet = np.std(y_quiet_fluc)
    
    print(f"ECE{ch}FAST: RMS(259-275) = {rms_win:.4f} V, RMS(400-410) = {rms_quiet:.4f} V")
    if ch == 11:
        print("Comments:")
        for c in comm:
            print("  ", c)
