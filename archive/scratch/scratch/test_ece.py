import sys
from pathlib import Path
import numpy as np
import scipy.signal as dsp

root_dir = Path("c:/TFG")
sys.path.append(str(root_dir / "analysis" / "common"))
sys.path.append(str(root_dir / "analysis" / "heating_phase_comparison"))

from comparison_common import load_signals, load_channel

def test_ece():
    shot = 88653
    time_ms, dt_s, fs, signals_dict, _ = load_signals(shot=shot)
    
    tw_start, tw_end = 259.1, 275.0
    mode_freq = 89.0
    idx_win = np.where((time_ms >= tw_start) & (time_ms <= tw_end))[0]
    n_samples = len(idx_win)
    nperseg = min(2048, n_samples // 2 if n_samples > 1000 else n_samples)
    noverlap = nperseg // 2

    b_band, a_band = dsp.bessel(4, [(80.0) * 1e3 / (fs / 2.0), (120.0) * 1e3 / (fs / 2.0)], btype="bandpass")
    
    for i in range(1, 17):
        t_ece, y_ece, _, _ = load_channel(shot, f"ECE{i}FAST")
        if y_ece is None:
            print(f"ECE{i}FAST: None")
            continue
        y_win = y_ece[idx_win]
        
        # In slide-19 script:
        _, coh_e = dsp.coherence(signals_dict["MP1"][idx_win], y_win, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
        f_e, _ = dsp.csd(signals_dict["MP1"][idx_win], y_win, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
        idx_e = np.argmin(np.abs(f_e / 1000.0 - mode_freq))
        gam = float(coh_e[idx_e])
        
        y_fluc = dsp.filtfilt(b_band, a_band, y_win)
        rms = float(np.std(y_fluc))
        
        # Let's see if there are zeros.
        if np.all(y_ece == 0):
            print(f"ECE{i}FAST: All zeros!")
        else:
            print(f"ECE{i}FAST: RMS = {rms:.5f}, Coherence = {gam:.5f}")

if __name__ == "__main__":
    test_ece()
