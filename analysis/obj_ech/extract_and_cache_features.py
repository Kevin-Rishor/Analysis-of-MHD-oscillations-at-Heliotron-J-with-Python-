"""
extract_and_cache_features.py
Extracts and caches multi-channel diagnostic time series, envelopes,
and instantaneous frequencies for Heliotron J shots #88653 and #88654.
Saves cached arrays as .npz in cache/ for efficient downstream reuse.
"""

import sys
from pathlib import Path
import numpy as np
import scipy.signal as dsp

ROOT_DIR = Path(__file__).resolve().parents[2]
JPACK_DIR = ROOT_DIR / "jpack"
CACHE_DIR = ROOT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

if str(JPACK_DIR) not in sys.path:
    sys.path.append(str(JPACK_DIR))

import turnelib as TE


def load_raw_edf(shot, channel_name):
    fpath = ROOT_DIR / "data" / f"hj{shot}" / f"{channel_name}@{shot}.edf"
    if not fpath.exists():
        return None, None, None
    edf = TE.edf()
    dat = edf.load(str(fpath))
    t_raw = dat[:, 0]
    t_ms = t_raw if edf.DimUnit[0] == "ms" else t_raw * 1000.0
    ys = dat[:, 1]
    unit = edf.ValUnit[0] if edf.ValUnit else "V"
    return t_ms, ys, unit


def extract_hilbert_features(ys, fs, f_low, f_high, order=4, sg_win=125):
    sos = dsp.bessel(order, [f_low, f_high], btype="bandpass", fs=fs, output="sos")
    filtered = dsp.sosfiltfilt(sos, ys)
    analytic = dsp.hilbert(filtered)
    envelope = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    
    dt = 1.0 / fs
    if sg_win % 2 == 0:
        sg_win += 1
    if sg_win >= len(ys):
        sg_win = len(ys) - 1 if len(ys) % 2 == 0 else len(ys) - 2
    ifreq = dsp.savgol_filter(phase, sg_win, 2, deriv=1) / (2.0 * np.pi * dt)
    return envelope, phase, filtered, ifreq


def process_shot(shot):
    print(f"\n{'='*80}\nPROCESSING AND CACHING SHOT #{shot}\n{'='*80}")
    cache_file = CACHE_DIR / f"shot_{shot}_features.npz"
    
    # 1. Mirnov toroidal coils
    t_mp1, y_mp1, _ = load_raw_edf(shot, "MP1")
    t_mp3, y_mp3, _ = load_raw_edf(shot, "MP3")
    t_mp4, y_mp4, _ = load_raw_edf(shot, "MP4")
    
    if y_mp1 is None:
        raise FileNotFoundError(f"MP1 missing for shot {shot}")
        
    fs = 1.0 / ((t_mp1[1] - t_mp1[0]) * 1e-3)
    print(f"Loaded MP1: {len(y_mp1)} points, fs = {fs/1e3:.1f} kHz, t = [{t_mp1[0]:.1f}, {t_mp1[-1]:.1f}] ms")
    
    # 2. Extract Primary (80-110 kHz) and Secondary (30-65 kHz) features
    print("Extracting Primary Mode (75-110 kHz) Hilbert features...")
    env_pri, ph_pri, flt_pri, ifreq_pri = extract_hilbert_features(y_mp1, fs, 75000.0, 110000.0, order=4, sg_win=125)
    
    print("Extracting Secondary Mode (30-65 kHz) Hilbert features...")
    env_sec, ph_sec, flt_sec, ifreq_sec = extract_hilbert_features(y_mp1, fs, 30000.0, 65000.0, order=4, sg_win=125)
    
    # Also extract MP3 and MP4 envelopes for Primary mode
    env_pri_mp3, _, _, _ = extract_hilbert_features(y_mp3, fs, 75000.0, 110000.0, order=4, sg_win=125)
    env_pri_mp4, _, _, _ = extract_hilbert_features(y_mp4, fs, 75000.0, 110000.0, order=4, sg_win=125)
    
    # 3. Heating signals
    t_ech, y_ech, _ = load_raw_edf(shot, "ECHRG500")
    t_nbi, y_nbi_tot = None, None
    for n in [3, 4, 9, 10]:
        tn, yn, _ = load_raw_edf(shot, f"NBIS{n}I")
        if yn is not None:
            if y_nbi_tot is None:
                t_nbi = tn
                y_nbi_tot = np.copy(yn)
            else:
                y_nbi_tot += yn
                
    # Interpolate heating onto MP1 time base
    y_ech_interp = np.interp(t_mp1, t_ech, y_ech) if y_ech is not None else np.zeros_like(t_mp1)
    y_nbi_interp = np.interp(t_mp1, t_nbi, y_nbi_tot) if y_nbi_tot is not None else np.zeros_like(t_mp1)
    
    # 4. Macroscopic plasma signals
    t_wp, y_wp, _ = load_raw_edf(shot, "Wp")
    t_ne, y_ne, _ = load_raw_edf(shot, "nave")
    t_ip, y_ip, _ = load_raw_edf(shot, "Ip15")
    t_ha, y_ha, _ = load_raw_edf(shot, "HAFAST7.5")
    
    y_wp_interp = np.interp(t_mp1, t_wp, y_wp) if y_wp is not None else np.zeros_like(t_mp1)
    y_ne_interp = np.interp(t_mp1, t_ne, y_ne) if y_ne is not None else np.zeros_like(t_mp1)
    y_ip_interp = np.interp(t_mp1, t_ip, y_ip) if y_ip is not None else np.zeros_like(t_mp1)
    y_ha_interp = np.interp(t_mp1, t_ha, y_ha) if y_ha is not None else np.zeros_like(t_mp1)
    
    # 5. Core ECE channels
    ece_dict = {}
    for ch_num in [9, 10, 12, 13, 15]:
        ch_name = f"ECE{ch_num}FAST"
        t_e, y_e, _ = load_raw_edf(shot, ch_name)
        if y_e is not None:
            ece_dict[f"ece_{ch_num}"] = y_e
            
    # 6. Soft X-ray (sx2d): load center chords (e.g. chords 20-25)
    f_sx = ROOT_DIR / "data" / f"hj{shot}" / f"sx2d@{shot}.edf"
    sx_center = np.zeros(1)
    t_sx_ms = np.zeros(1)
    if f_sx.exists():
        edf_sx = TE.edf()
        d_sx = edf_sx.load(str(f_sx))
        t_sx_ms = d_sx[:, 0] * 1000.0 if edf_sx.DimUnit[0] == "s" else d_sx[:, 0]
        # Average core chords (columns 20 to 25)
        sx_center = np.mean(d_sx[:, 20:26], axis=1)
        print(f"Loaded sx2d core proxy: shape={d_sx.shape}, t=[{t_sx_ms[0]:.1f}, {t_sx_ms[-1]:.1f}] ms")
        
    # Save into NPZ
    save_dict = {
        "shot": shot,
        "fs": fs,
        "t_ms": t_mp1,
        "mp1": y_mp1,
        "mp3": y_mp3,
        "mp4": y_mp4,
        "env_pri": env_pri,
        "ifreq_pri": ifreq_pri,
        "flt_pri": flt_pri,
        "env_sec": env_sec,
        "ifreq_sec": ifreq_sec,
        "flt_sec": flt_sec,
        "env_pri_mp3": env_pri_mp3,
        "env_pri_mp4": env_pri_mp4,
        "ech": y_ech_interp,
        "nbi": y_nbi_interp,
        "wp": y_wp_interp,
        "ne": y_ne_interp,
        "ip": y_ip_interp,
        "ha": y_ha_interp,
        "t_sx_ms": t_sx_ms,
        "sx_center": sx_center,
    }
    save_dict.update(ece_dict)
    
    np.savez_compressed(cache_file, **save_dict)
    print(f"Successfully cached features to: {cache_file} ({cache_file.stat().st_size / 1e6:.2f} MB)")


def main():
    for shot in [88653, 88654]:
        process_shot(shot)


if __name__ == "__main__":
    main()
