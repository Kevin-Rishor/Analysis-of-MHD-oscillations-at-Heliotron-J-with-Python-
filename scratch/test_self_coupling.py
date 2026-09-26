import sys
import numpy as np
import scipy.signal as dsp
from pathlib import Path

root_dir = Path("c:/TFG")
sys.path.append(str(root_dir / "analysis"))
sys.path.append(str(root_dir / "analysis" / "common"))
sys.path.append(str(root_dir / "analysis" / "heating_phase_comparison"))

from mhd_common import extract_instantaneous_frequency
from comparison_common import load_signals

def run_test():
    shot = 88653
    time_ms, dt_s, fs, signals_dict, _ = load_signals(shot=shot)
    
    tw_start, tw_end = 259.1, 275.0
    fl_hz, fu_hz = 80000.0, 120000.0 # From ECH+NBI phase
    idx_win = np.where((time_ms >= tw_start) & (time_ms <= tw_end))[0]
    n_samples = len(idx_win)
    
    probes = ["MP1", "MP3", "MP4"]
    
    print("--- 1x nperseg ---")
    for mult in [1, 2, 4]:
        print(f"\nMultiplier: {mult}x nperseg")
        nperseg = min(2048, n_samples // 2 if n_samples > 1000 else n_samples)
        nperseg *= mult
        noverlap = int(nperseg * 0.75)
        nfft = max(2048, nperseg * 2)
        
        n_seg = int(np.floor((len(idx_win) - nperseg) / (nperseg - noverlap))) + 1
        sig_floor = 1.0 - (0.05) ** (1.0 / max(1, n_seg - 1))
        df = fs / nfft
        print(f"Floor: {sig_floor:.4f}, df: {df:.2f} Hz, nperseg: {nperseg}")
        
        for p in probes:
            ys = signals_dict[p]
            env, _, _, _ = extract_instantaneous_frequency(ys, fs, fl_hz, fu_hz, 4, 325)
            ys_win = ys[idx_win]
            env_win = env[idx_win]
            
            f, coh2 = dsp.coherence(ys_win, env_win, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, nfft=nfft)
            
            f_khz = f / 1000.0
            
            # Find coherence at 1.22 kHz and 2.20 kHz
            idx_122 = np.argmin(np.abs(f_khz - 1.22))
            idx_220 = np.argmin(np.abs(f_khz - 2.20))
            
            print(f"{p}: 1.22 kHz (actual {f_khz[idx_122]:.3f}) = {coh2[idx_122]:.4f} | 2.20 kHz (actual {f_khz[idx_220]:.3f}) = {coh2[idx_220]:.4f}")

if __name__ == "__main__":
    run_test()
