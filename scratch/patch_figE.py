import sys; sys.path.append('c:/TFG/jpack'); sys.path.append('c:/TFG/analysis')
import numpy as np
import matplotlib.pyplot as plt
from heating_phase_comparison.comparison_common import load_signals
from heating_phase_comparison.mhd_plot_bicoherence_sliding_trace import custom_abicoh2_complex

shot = 88653
time_ms, dt_s, fs, signals_dict, _ = load_signals(shot=shot)
ys_mp1 = signals_dict['MP1']

win_len = 8.0
step_ms = 0.25
nfft = 512
n_ens = 30
f_pump = 86.0

t_start_scan = 295.0
t_end_scan = 313.0
t_starts = np.arange(t_start_scan, t_end_scan - win_len, step_ms)

t_centers, im_vals, den_vals = [], [], []

for t_start in t_starts:
    idx_win = np.where((time_ms >= t_start) & (time_ms <= t_start + win_len))[0]
    y_sub = ys_mp1[idx_win]
    
    f1, f2, b2, b_comp, den = custom_abicoh2_complex(y_sub, dt=dt_s, nfft=nfft, nensemble=n_ens)
    
    ff1_k, ff2_k = np.meshgrid(f1 / 1000.0, f2 / 1000.0, indexing='ij')
    mask_diff = (ff1_k > 0) & (ff2_k < 0) & (ff1_k > -ff2_k) & (ff1_k >= 5.0) & (-ff2_k >= 5.0) & ((ff1_k + ff2_k) >= 5.0) & (ff1_k < 150.0) & (-ff2_k < 150.0)
    
    from scipy.ndimage import maximum_filter
    df_khz = (f1[1] - f1[0]) / 1000.0
    fsz = max(3, int(round(5.0 / df_khz)))
    if fsz % 2 == 0: fsz += 1
    
    local_max = (b2 == maximum_filter(b2, size=fsz)) & mask_diff
    coords = np.argwhere(local_max)
    vals = b2[local_max]
    
    order = np.argsort(vals)[::-1]
    best_c = None
    for p_idx in order[:10]:
        r, c = coords[p_idx]
        if abs(f1[r]/1000.0 - f_pump) < 6.0:
            best_c = (r, c)
            break
            
    if best_c:
        t_centers.append(t_start + win_len / 2.0)
        im_vals.append(-np.imag(b_comp[best_c[0], best_c[1]]))
        den_vals.append(den[best_c[0], best_c[1]])

t_centers = np.array(t_centers)
im_vals = np.array(im_vals)
den_vals = np.array(den_vals)

plt.figure(figsize=(10, 6))
plt.axhline(0, color='k', linestyle=':', lw=1.5)

log_den = np.log10(den_vals)
valid = den_vals > 2.5e-23

sc = plt.scatter(t_centers, im_vals, c=log_den, cmap='viridis', 
                 edgecolors=np.where(valid, 'black', 'none'),
                 linewidths=np.where(valid, 1.2, 0),
                 s=40, zorder=3)

plt.colorbar(sc, label=r'$\log_{10}(\mathrm{den})$ (Power Normalization)')
plt.title(f'Sign of $\\mathrm{{Im}}[B_{{\\mathrm{{sum}}}}]$ vs SNR cut (Shot {shot})', fontweight='bold')
plt.xlabel('Window Center Time (ms)')
plt.ylabel(r'$\mathrm{Im}[B_{\mathrm{sum}}]$')
plt.grid(True, alpha=0.3)
plt.xlim(295, 313)
plt.tight_layout()
plt.savefig(r'c:\TFG\Images\sign_vs_threshold_88653.png', dpi=300, bbox_inches='tight')
