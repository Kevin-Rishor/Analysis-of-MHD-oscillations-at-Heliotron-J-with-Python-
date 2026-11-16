"""
build_full_record_cache.py
Builds complete whole-discharge cache for Heliotron J shots #88653 and #88654:
  1. Full record MP1, MP3, MP4 (150.0 - 412.14 ms, 1 MHz).
  2. STFT with standard settings: nperseg=1024, noverlap=960 (Hann).
  3. STFT with high-time-resolution settings: nperseg=256, noverlap=240 (Hann).
  4. Primary (75-110 kHz) and Secondary (30-65 kHz) Bessel-filtered signals,
     Hilbert analytic envelopes, and instantaneous frequencies.
  5. Peripheral diagnostics: nave, Wp, HAFAST (3.5, 7.5, 11.5, 15.5), AXUV (all chords),
     ECHRG500, NBIS*I monitors, and sx2d camera chords.
Saves to cache/shot_{shot}_full_record_cache.npz.
"""

import sys
from pathlib import Path
import numpy as np
import scipy.signal as dsp

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = ROOT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OBJ_ECH_DIR = ROOT_DIR / "analysis" / "obj_ech"

sys.path.append(str(ROOT_DIR / "jpack"))
import turnelib as TE


def load_edf(filepath):
    e = TE.edf()
    dat = e.load(str(filepath))
    t_raw = dat[:, 0]
    t_ms = t_raw if e.DimUnit[0] == "ms" else t_raw * 1000.0
    ys = dat[:, 1:] if dat.shape[1] > 2 else dat[:, 1]
    return t_ms, ys


def process_shot_cache(shot):
    print(f"\n==========================================")
    print(f"Building Full Record Cache for Shot #{shot}")
    print(f"==========================================")
    
    shot_dir = DATA_DIR / f"hj{shot}"
    
    # 1. Mirnov Probes MP1, MP3, MP4
    t_ms, mp1 = load_edf(shot_dir / f"MP1@{shot}.edf")
    _, mp3 = load_edf(shot_dir / f"MP3@{shot}.edf")
    _, mp4 = load_edf(shot_dir / f"MP4@{shot}.edf")
    fs = 1e6
    dt = 1.0 / fs
    print(f"Loaded MP1, MP3, MP4: N = {len(mp1)}, t = [{t_ms[0]:.2f}, {t_ms[-1]:.2f}] ms")
    
    # 2. STFT standard (1024 / 960)
    print("Computing STFT standard (1024 / 960)...")
    f_stft_std, t_rel_std, zxx_std = dsp.stft(mp1, fs=fs, nperseg=1024, noverlap=960, window="hann")
    t_stft_std = t_ms[0] + t_rel_std * 1000.0  # ms
    psd_mp1_std = (np.abs(zxx_std)**2).astype(np.float32)
    
    # 3. STFT high-time-res (256 / 240)
    print("Computing STFT high-time-res (256 / 240)...")
    f_stft_fast, t_rel_fast, zxx_fast = dsp.stft(mp1, fs=fs, nperseg=256, noverlap=240, window="hann")
    t_stft_fast = t_ms[0] + t_rel_fast * 1000.0  # ms
    psd_mp1_fast = (np.abs(zxx_fast)**2).astype(np.float32)
    
    # 4. Hilbert Envelopes & Instantaneous Frequencies
    print("Computing Hilbert envelopes for Primary & Secondary modes...")
    # Primary Mode (75 - 110 kHz)
    sos_pri = dsp.bessel(4, [75e3, 110e3], btype="bandpass", fs=fs, output="sos")
    flt_pri = dsp.sosfiltfilt(sos_pri, mp1)
    ana_pri = dsp.hilbert(flt_pri)
    env_pri = np.abs(ana_pri).astype(np.float32)
    phase_pri = np.unwrap(np.angle(ana_pri))
    ifreq_pri = (dsp.savgol_filter(phase_pri, 125, 2, deriv=1) / (2.0 * np.pi * dt)).astype(np.float32)
    
    # Secondary Mode (30 - 65 kHz)
    sos_sec = dsp.bessel(4, [30e3, 65e3], btype="bandpass", fs=fs, output="sos")
    flt_sec = dsp.sosfiltfilt(sos_sec, mp1)
    ana_sec = dsp.hilbert(flt_sec)
    env_sec = np.abs(ana_sec).astype(np.float32)
    phase_sec = np.unwrap(np.angle(ana_sec))
    ifreq_sec = (dsp.savgol_filter(phase_sec, 125, 2, deriv=1) / (2.0 * np.pi * dt)).astype(np.float32)
    
    # 5. Core diagnostics
    print("Loading plasma parameters and heating signals...")
    t_nave_ms, nave = load_edf(shot_dir / f"nave@{shot}.edf")
    t_wp_ms, wp = load_edf(shot_dir / f"Wp@{shot}.edf")
    t_ech_ms, ech = load_edf(shot_dir / f"ECHRG500@{shot}.edf")
    t_nbis9_ms, nbis9 = load_edf(shot_dir / f"NBIS9I@{shot}.edf")
    t_nbis10_ms, nbis10 = load_edf(shot_dir / f"NBIS10I@{shot}.edf")
    
    # H-alpha fast channels
    hafast = {}
    for pos in ["3.5", "7.5", "11.5", "15.5"]:
        f_ha = shot_dir / f"HAFAST{pos}@{shot}.edf"
        if f_ha.exists():
            t_ha_ms, ha_val = load_edf(f_ha)
            hafast[pos] = ha_val.astype(np.float32)
            
    # AXUV channels
    axuv = {}
    t_axuv_ms = None
    for i in range(1, 17):
        f_ax = shot_dir / f"AXUV{i}@{shot}.edf"
        if f_ax.exists():
            t_ax, v_ax = load_edf(f_ax)
            axuv[f"AXUV{i}"] = v_ax.astype(np.float32)
            if t_axuv_ms is None:
                t_axuv_ms = t_ax
                
    # All 16 ECE channels
    print("Loading all 16 ECE channels...")
    ece_channels = {}
    for i in range(1, 17):
        f_ece = shot_dir / f"ECE{i}FAST@{shot}.edf"
        if f_ece.exists():
            _, v_ece = load_edf(f_ece)
            ece_channels[f"ECE{i}"] = v_ece.astype(np.float32)
            
    # sx2d camera chords
    print("Loading sx2d 48-chord camera data...")
    t_sx_ms, sx2d_chords = load_edf(shot_dir / f"sx2d@{shot}.edf")
    sx2d_chords = sx2d_chords.astype(np.float32)
    
    # Save to NPZ
    out_file = CACHE_DIR / f"shot_{shot}_full_record_cache.npz"
    print(f"Compressing and saving to {out_file}...")
    np.savez_compressed(
        out_file,
        shot=shot,
        fs=fs,
        t_ms=t_ms.astype(np.float32),
        mp1=mp1.astype(np.float32),
        mp3=mp3.astype(np.float32),
        mp4=mp4.astype(np.float32),
        # STFT std
        t_stft_std=t_stft_std.astype(np.float32),
        f_stft_std=f_stft_std.astype(np.float32),
        psd_mp1_std=psd_mp1_std,
        # STFT high-res
        t_stft_fast=t_stft_fast.astype(np.float32),
        f_stft_fast=f_stft_fast.astype(np.float32),
        psd_mp1_fast=psd_mp1_fast,
        # Hilbert
        env_pri=env_pri,
        ifreq_pri=ifreq_pri,
        flt_pri=flt_pri.astype(np.float32),
        env_sec=env_sec,
        ifreq_sec=ifreq_sec,
        flt_sec=flt_sec.astype(np.float32),
        # Diagnostics
        t_nave_ms=t_nave_ms.astype(np.float32),
        nave=nave.astype(np.float32),
        t_wp_ms=t_wp_ms.astype(np.float32),
        wp=wp.astype(np.float32),
        t_ech_ms=t_ech_ms.astype(np.float32),
        ech=ech.astype(np.float32),
        t_nbis9_ms=t_nbis9_ms.astype(np.float32),
        nbis9=nbis9.astype(np.float32),
        t_nbis10_ms=t_nbis10_ms.astype(np.float32),
        nbis10=nbis10.astype(np.float32),
        t_axuv_ms=t_axuv_ms.astype(np.float32) if t_axuv_ms is not None else np.array([]),
        t_sx_ms=t_sx_ms.astype(np.float32),
        sx2d_chords=sx2d_chords,
        **{f"hafast_{k}": v for k, v in hafast.items()},
        **{k: v for k, v in axuv.items()},
        **{k: v for k, v in ece_channels.items()}
    )
    print(f"SUCCESS: Saved {out_file.name} ({out_file.stat().st_size / 1e6:.2f} MB)")


def main():
    for shot in [88653, 88654]:
        process_shot_cache(shot)


if __name__ == "__main__":
    main()
