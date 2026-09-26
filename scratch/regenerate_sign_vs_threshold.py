import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

root_dir = Path("c:/TFG")
sys.path.append(str(root_dir / "jpack"))
sys.path.append(str(root_dir / "analysis"))
sys.path.append(str(root_dir / "analysis" / "common"))
sys.path.append(str(root_dir / "analysis" / "heating_phase_comparison"))

from comparison_common import load_channel
from mhd_plot_bicoherence_sliding_trace import custom_abicoh2_complex

def run_step2():
    shot = 88653
    t_ms, ys, dt, fs = load_channel(shot, "MP1")
    if ys is None or t_ms is None:
        print("Error: MP1 signal not found.")
        return

    win_len = 8.0  # ms
    step = 0.25    # ms
    t_centers, b2_vals, im_vals, den_vals = [], [], [], []

    # t_start from 291.3 ms to end so that center reaches up to 313
    for t_start in np.arange(291.3, 313.0, step):
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

    # Plot Im[B_sum] vs window-center time for ALL windows with a candidate (window centers 295.3-313 ms)
    mask_plot = (t_centers >= 295.3) & (t_centers <= 313.0)
    t_plot = t_centers[mask_plot]
    im_plot = im_vals[mask_plot]
    den_plot = den_vals[mask_plot]

    snr_thresh = 2.5e-23
    print("Valid points:")
    for t_val, im_val, den_val in zip(t_centers, im_vals, den_vals):
        if den_val > snr_thresh:
            print(f"{t_val:.2f} ms {im_val:.3f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # colour = log10(den)
    c_vals = np.log10(den_plot)
    
    # black edge where den > 2.5e-23
    edge_colors = ['black' if d > 2.5e-23 else 'none' for d in den_plot]
    
    sc = ax.scatter(t_plot, im_plot, c=c_vals, cmap='viridis', edgecolors=edge_colors, linewidths=1.5, s=50)
    plt.colorbar(sc, label='log10(den)')
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Window Center Time (ms)')
    ax.set_ylabel('Im[B_sum]')
    ax.set_title('Sign vs Threshold - 88653')
    
    out_path = root_dir / "Images" / "sign_vs_threshold_88653.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    # Print results
    valid_mask = den_plot > 2.5e-23
    num_valid = np.sum(valid_mask)
    
    with open(root_dir / "scratch" / "round_final2.txt", "a") as f:
        f.write(f"Step 2: Number of valid windows: {num_valid}\n")
        f.write(f"Valid points (Time, Im[B_sum]):\n")
        for t, im in zip(t_plot[valid_mask], im_plot[valid_mask]):
            f.write(f"  {t:.2f} ms: {im:.4f}\n")

if __name__ == "__main__":
    run_step2()
