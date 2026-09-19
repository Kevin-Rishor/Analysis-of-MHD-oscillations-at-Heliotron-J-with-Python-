import sys
import argparse
import warnings
from pathlib import Path
import numpy as np
import scipy.signal as dsp
import scipy.linalg as la
import scipy.fft as fft
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

current = Path(__file__).resolve().parent
root_dir = None
for p in [current] + list(current.parents):
    if (p / "jpack").exists():
        root_dir = p
        break
if root_dir is None:
    root_dir = Path("c:/TFG")

for p_add in [root_dir / "jpack", root_dir / "analysis", root_dir / "analysis" / "common", current]:
    if str(p_add) not in sys.path:
        sys.path.append(str(p_add))

from comparison_common import DEFAULT_SHOT, DEFAULT_OUT_DIR, load_channel, safe_savefig


def custom_abicoh2_complex(sig, dt, nfft=512, nensemble=30):
    noverlap = int(nfft * 0.5)
    f, _, spec = dsp.spectrogram(sig, fs=1.0 / dt, window='hann', nperseg=nfft, noverlap=noverlap, nfft=nfft,
                                 return_onesided=False, scaling='density', axis=0, mode='complex')
    spec = np.transpose(spec, [1, 0])
    if spec.shape[0] > nensemble:
        spec = spec[:nensemble, :]
    nf = f.size
    nhalf = nf // 2 if nf % 2 == 0 else (nf + 1) // 2
    fcol = np.arange(nf, dtype=int)
    lrow = np.roll(np.arange(nhalf, dtype=int), 1)
    ha = la.hankel(fcol, lrow)

    num = np.mean(spec[:, fcol, None] * spec[:, None, lrow] * np.conjugate(spec[:, ha]), axis=0)
    denum = np.mean(np.abs(spec[:, fcol, None] * spec[:, None, lrow]) ** 2, axis=0) * np.mean(np.abs(np.conjugate(spec[:, ha])) ** 2, axis=0)
    bicoh2 = np.abs(num) ** 2 / denum
    b_complex = num / np.sqrt(denum)

    bicoh2 = np.transpose(fft.fftshift(bicoh2, axes=0), [1, 0])
    b_complex = np.transpose(fft.fftshift(b_complex, axes=0), [1, 0])
    denum = np.transpose(fft.fftshift(denum, axes=0), [1, 0])
    f1 = f[0:nhalf]
    f2 = fft.fftshift(f)
    return f1, f2, bicoh2, b_complex, denum


def run_sliding_bicoherence_trace(shot=DEFAULT_SHOT, out_dir=None, verbose=False):
    if out_dir is None:
        out_dir = DEFAULT_OUT_DIR
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Sliding Window Bicoherence & Biphase Trace (Shot {shot}) ---")

    t_ms, ys, dt, fs = load_channel(shot, "MP1")
    if ys is None or t_ms is None:
        print("Error: MP1 signal not found.")
        return

    win_len = 8.0  # ms
    step = 0.25    # ms
    t_centers, b2_vals, im_vals, den_vals = [], [], [], []

    for t_start in np.arange(291.3, 310.0, step):
        t_end = t_start + win_len
        if t_end > t_ms[-1]:
            break

        idx = np.where((t_ms >= t_start) & (t_ms <= t_end))[0]
        y_sub = ys[idx]
        if len(y_sub) < 512:
            continue

        f1, f2, b2, b_comp, den = custom_abicoh2_complex(y_sub, dt=dt, nfft=512, nensemble=30)
        ff1_k, ff2_k = np.meshgrid(f1 / 1000.0, f2 / 1000.0, indexing='ij')

        mask_diff = (ff1_k > 95) & (ff1_k < 105) & (ff2_k < -40) & (np.abs(ff2_k / ff1_k + 0.5) < 0.1)
        if np.any(mask_diff):
            best_c = np.argwhere(b2 == np.max(b2[mask_diff]))[0]
            t_centers.append(t_start + win_len / 2.0)
            b2_vals.append(b2[best_c[0], best_c[1]])
            im_vals.append(-np.imag(b_comp[best_c[0], best_c[1]]))
            den_vals.append(den[best_c[0], best_c[1]])

    t_centers = np.array(t_centers)
    b2_vals = np.array(b2_vals)
    im_vals = np.array(im_vals)
    den_vals = np.array(den_vals)

    snr_thresh = 2.5e-23
    valid = den_vals > snr_thresh
    invalid = ~valid

    plt.rcdefaults()
    fig = plt.figure(figsize=(10, 8))

    # Panel 1: Bicoherence
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(t_centers[valid], b2_vals[valid], 'ko-', lw=1.5, markersize=4, label='Valid (High SNR)')
    if np.sum(invalid) > 0:
        ax1.plot(t_centers[invalid], b2_vals[invalid], 's', color='silver', markersize=3, label='Excluded (Low SNR Floor)')
    ax1.axhline(0.20, color='r', linestyle='--', label=r'95% Threshold ($N_{\mathrm{eff}}=15$)')
    ax1.set_ylabel(r'Bicoherence $b^2$', fontsize=11)
    ax1.set_title(f'Heliotron J #{shot} — Sliding Bicoherence Trace ($8\\,\\mathrm{{ms}}$ window, step $0.25\\,\\mathrm{{ms}}$)', fontsize=12, fontweight='bold')
    ax1.set_xlim(295, 310)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)

    # Panel 2: Im[B_sum] Direction
    ax2 = plt.subplot(2, 1, 2)
    ax2.axhline(0, color='k', linestyle='-', alpha=0.5)

    if np.sum(invalid) > 0:
        ax2.plot(t_centers[invalid], im_vals[invalid], 's', color='silver', markersize=3, alpha=0.7)

    ax2.plot(t_centers[valid], im_vals[valid], 'bo-', lw=1.5, markersize=4, label=r'$\mathrm{Im}[B_{\mathrm{sum}}]$ (Valid)')

    for i in range(len(t_centers) - 1):
        if valid[i] and valid[i + 1]:
            if im_vals[i] > 0 and im_vals[i + 1] > 0:
                ax2.fill_between(t_centers[i:i + 2], 0, im_vals[i:i + 2], color='g', alpha=0.3)
            elif im_vals[i] < 0 and im_vals[i + 1] < 0:
                ax2.fill_between(t_centers[i:i + 2], 0, im_vals[i:i + 2], color='r', alpha=0.3)

    ax2.set_xlim(295, 310)
    ax2.set_xlabel('Window Center Time (ms)', fontsize=11)
    ax2.set_ylabel(r'$\mathrm{Im}[B_{\mathrm{sum}}]$', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right', fontsize=9)

    plt.tight_layout()
    out_path = out_dir / 'bicoherence_sliding_trace_snr.png'
    safe_savefig(fig, out_path, dpi=300)
    plt.close(fig)
    print(f"    Saved figure: {out_path.name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sliding Bicoherence Trace with SNR Cutoff")
    parser.add_argument("-s", "--shots", type=int, default=DEFAULT_SHOT, help="Shot number to analyze")
    parser.add_argument("-o", "--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    run_sliding_bicoherence_trace(shot=args.shots, out_dir=args.out_dir, verbose=args.verbose)
