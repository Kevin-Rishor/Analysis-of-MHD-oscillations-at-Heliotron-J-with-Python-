import sys
from pathlib import Path
import numpy as np

current = Path(__file__).resolve().parent
root_dir = None
for p in [current] + list(current.parents):
    if (p / "jpack").exists():
        root_dir = p
        break
if root_dir is None:
    root_dir = Path("c:/TFG")

for p_add in [root_dir / "jpack", root_dir / "analysis", root_dir / "analysis" / "common"]:
    if str(p_add) not in sys.path:
        sys.path.append(str(p_add))

import libana_signal as LAS
from mhd_analysis_obj3 import load_edf_signal, slice_window

def determine_energy_transfer_direction(fpath, t_start, t_end, target_triads, nfft=1024, noverlap=512, nensemble=30):
    """
    Implements a single-point energy transfer direction algorithm based on the 
    imaginary part of the complex bispectrum (Biphase asymmetry).
    
    References:
    1. Kim, Y. C., & Powers, E. J. (1979). Digital bispectral analysis and its applications to wave interactions.
    2. Ritz, C. P., et al. (1989). Experimental measurement of three-wave coupling and energy cascading.
    
    Algorithm:
    1. Map any triad into a canonical sum interaction: f_L1 + f_L2 = f_H.
    2. Compute the complex normalized bispectrum (bicoherence) B_sum(f_L1, f_L2).
    3. Extract Im[B_sum].
    4. If Im[B_sum] > 0 => Upward Cascade (f_L1, f_L2 -> f_H)
       If Im[B_sum] < 0 => Downward Cascade / Decay (f_H -> f_L1, f_L2)
    """
    print(f"Loading {fpath} for window {t_start}-{t_end} ms...")
    t_sec, ys, _, _ = load_edf_signal(fpath)
    dt = t_sec[1] - t_sec[0]
    t_sec, ys = slice_window(t_sec, ys, t_start, t_end)
    
    # 1. Compute the Complex Spectrogram
    print("Computing complex spectrogram and cross-bispectrum...")
    f, tave, S1 = LAS.periodogram(ys, t_sec, dt=dt, nfft=nfft, noverlap=noverlap, detrend='constant', fshift=True)
    
    # 2. Compute the Complex Bispectrum (xbispct preserves the complex phase!)
    f1, f2, bispct = LAS.xbispct(S1, S1, f, normalize=True, fshifted=True)
    
    print("\n" + "="*80)
    print("NON-LINEAR ENERGY TRANSFER DIRECTION ANALYSIS")
    print("="*80)
    print("Rule: Im[B_sum] > 0 implies energy flows LOW -> HIGH (Upward Cascade / Coalescence)")
    print("Rule: Im[B_sum] < 0 implies energy flows HIGH -> LOW (Downward Cascade / Decay Instability)")
    print("-" * 80)
    
    for triad in target_triads:
        f1_t, f2_t = triad
        
        # Find closest bins
        r = np.argmin(np.abs(f1/1000.0 - f1_t))
        c = np.argmin(np.abs(f2/1000.0 - f2_t))
        
        b_val = bispct[r, c]
        im_b = np.imag(b_val)
        
        actual_f1 = f1[r]/1000.0
        actual_f2 = f2[c]/1000.0
        actual_f3 = actual_f1 + actual_f2
        
        # Map to Canonical Sum Triad (f_L1 + f_L2 = f_H)
        freqs = np.array([abs(actual_f1), abs(actual_f2), abs(actual_f3)])
        freqs.sort()
        f_L1, f_L2, f_H = freqs
        
        # Reconstruct the imaginary part for the Sum Triad
        # If the input was a Difference triad (f2 < 0), B(f_H, -f_L1) = B_sum*(f_L1, f_L2)
        # Therefore, Im[B_sum] = -Im[B_diff]
        is_sum = (actual_f1 > 0 and actual_f2 > 0)
        im_b_sum = im_b if is_sum else -im_b
        
        print(f"Target Triad : {actual_f1:5.1f} kHz + ({actual_f2:5.1f} kHz) = {actual_f3:5.1f} kHz")
        print(f"Canonical Sum: {f_L1:5.1f} kHz +  {f_L2:5.1f} kHz  = {f_H:5.1f} kHz")
        print(f"  -> Bicoherence |b|^2 = {np.abs(b_val)**2:.4f}")
        print(f"  -> Imaginary Im[B_sum] = {im_b_sum:.4f}")
        
        if im_b_sum > 0:
            print(f"  -> Direction : {f_L1:.1f} & {f_L2:.1f}  ===>  {f_H:.1f}  [UPWARD CASCADE / COALESCENCE]")
        else:
            print(f"  -> Direction : {f_H:.1f}  ===>  {f_L1:.1f} & {f_L2:.1f}  [DOWNWARD CASCADE / DECAY]")
        print("-" * 80)

if __name__ == '__main__':
    # Analyze the most significant triads from the NBI-only phase (291.3 - 300.0 ms)
    # Triads given as (f1, f2).
    nbi_triads = [
        (101.6, -50.8),  # 101.6 - 50.8 = 50.8
        (148.4, -50.8),  # 148.4 - 50.8 = 97.7
        (148.4, -95.7),  # 148.4 - 95.7 = 52.7
        (89.8, -48.8),   # 89.8 - 48.8 = 41.0 (ECH+NBI phase)
    ]
    
    fpath = root_dir / "data" / "hj88653" / "MP1@88653.edf"
    determine_energy_transfer_direction(
        fpath=str(fpath),
        t_start=291.3,
        t_end=300.0,
        target_triads=nbi_triads,
        nfft=1024,
        noverlap=512,
        nensemble=30
    )

