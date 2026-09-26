import numpy as np

with open(r'c:\TFG\analysis\heating_phase_comparison\mhd_plot_bicoherence_sliding_trace.py', 'r') as f:
    content = f.read()

old_plot1 = \"    ax1.plot(t_centers[valid], b2_vals[valid], 'ko-', lw=1.5, markersize=4, label='Valid (High SNR)')\"
new_plot1 = \"\"\"    t_v = t_centers[valid].copy()
    b2_v = b2_vals[valid].copy()
    im_v = im_vals[valid].copy()
    insert_idx = np.where(np.diff(t_v) > 0.5)[0] + 1
    t_v_nan = np.insert(t_v, insert_idx, np.nan)
    b2_v_nan = np.insert(b2_v, insert_idx, np.nan)
    im_v_nan = np.insert(im_v, insert_idx, np.nan)
    ax1.plot(t_v_nan, b2_v_nan, 'ko-', lw=1.5, markersize=4, label='Valid (High SNR)')\"\"\"
content = content.replace(old_plot1, new_plot1)

old_plot2 = \"    ax2.plot(t_centers[valid], im_vals[valid], 'bo-', lw=1.5, markersize=4, label=r'\\mathrm{Im}[B_{\\mathrm{sum}}] (Valid)')\"
# I must be careful with raw strings in replace, it's safer to just search a shorter string
