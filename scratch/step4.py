import sys
from pathlib import Path
import numpy as np
import scipy.signal as dsp

root_dir = Path("c:/TFG")
sys.path.append(str(root_dir / "jpack"))
sys.path.append(str(root_dir / "analysis"))
sys.path.append(str(root_dir / "analysis" / "common"))
sys.path.append(str(root_dir / "analysis" / "heating_phase_comparison"))

from mhd_common import extract_instantaneous_frequency
from comparison_common import (
    DEFAULT_SHOT, load_signals,
    T_W1_START, T_W1_END, T_W2_START, T_W2_END
)

def run_step4():
    shot = 88653
    time_ms, dt_s, fs, signals_dict, _ = load_signals(shot=shot)
    if "MP1" not in signals_dict:
        print("Error: MP1 not found.")
        return

    windows = [
        ("ECH + NBI Phase", T_W1_START, T_W1_END, 80000.0, 120000.0),
        ("Just NBI Phase", T_W2_START, T_W2_END, 75000.0, 105000.0)
    ]
    
    probes = ["MP1", "MP3", "MP4"]
    
    out_lines = ["\nStep 4:"]
    
    for title_win, tw_start, tw_end, fl_hz, fu_hz in windows:
        out_lines.append(f"\nWindow: {title_win} ({tw_start:.1f}-{tw_end:.1f} ms)")
        idx_win = np.where((time_ms >= tw_start) & (time_ms <= tw_end))[0]
        n_samples = len(idx_win)
        
        nperseg_base = min(2048, n_samples // 2 if n_samples > 1000 else n_samples)
        
        for mult in [1, 4]:
            nperseg = min(nperseg_base * mult, n_samples)
            noverlap = int(nperseg * 0.75)
            nfft = max(2048, nperseg * 2)
            
            df = fs / nfft / 1000.0 # in kHz
            
            n_seg = int(np.floor((n_samples - nperseg) / (nperseg - noverlap))) + 1
            sig_floor = 1.0 - (0.05) ** (1.0 / max(1, n_seg - 1))
            
            out_lines.append(f"  Multiplier: {mult}x (nperseg={nperseg}, df={df:.4f} kHz, 95% floor={sig_floor:.4f})")
            
            for p in probes:
                ys = signals_dict[p]
                env, _, _, _ = extract_instantaneous_frequency(ys, fs, fl_hz, fu_hz, 4, 325)
                
                ys_win = ys[idx_win]
                env_win = env[idx_win]
                
                f, coh2 = dsp.coherence(ys_win, env_win, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, nfft=nfft)
                f_khz = f / 1000.0
                
                idx_1_22 = np.argmin(np.abs(f_khz - 1.22))
                idx_2_20 = np.argmin(np.abs(f_khz - 2.20))
                
                coh_1_22 = coh2[idx_1_22]
                coh_2_20 = coh2[idx_2_20]
                
                act_f_1_22 = f_khz[idx_1_22]
                act_f_2_20 = f_khz[idx_2_20]
                
                out_lines.append(f"    {p}:")
                out_lines.append(f"      Coh @ ~1.22 kHz (actual {act_f_1_22:.3f}): {coh_1_22:.4f}")
                out_lines.append(f"      Coh @ ~2.20 kHz (actual {act_f_2_20:.3f}): {coh_2_20:.4f}")

    with open(root_dir / "scratch" / "round_final2.txt", "a") as f:
        f.write("\n".join(out_lines) + "\n")

if __name__ == "__main__":
    run_step4()
