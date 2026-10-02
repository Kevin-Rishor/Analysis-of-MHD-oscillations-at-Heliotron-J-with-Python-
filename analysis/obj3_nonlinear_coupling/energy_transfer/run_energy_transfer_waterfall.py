import sys
from pathlib import Path
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

current = Path(__file__).resolve().parent
root_dir = None
for p in [current] + list(current.parents):
    if (p / "jpack").exists():
        root_dir = p
        break
if root_dir is None:
    root_dir = Path("c:/TFG")

for p_add in [root_dir / "jpack", root_dir / "analysis", root_dir / "analysis" / "common", root_dir / "analysis" / "obj3_nonlinear_coupling", root_dir / "analysis" / "heating_phase_comparison", current]:
    if str(p_add) not in sys.path:
        sys.path.append(str(p_add))

from energy_transfer_common import (
    evaluate_energy_transfer_windows,
    WIN_LEN_MS, SNR_THRESHOLD, T_NBI_BURST_START_MS, T_NBI_BURST_END_MS, T_BEAM_OFF_MS
)

def run_waterfall_analysis(staging_dir=None):
    if staging_dir is None:
        staging_dir = root_dir / "scratch" / "staging"
    else:
        staging_dir = Path(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    windows, ref_core = evaluate_energy_transfer_windows(shot=88653)
    n_total = len(windows)

    print("=" * 80)
    print("OBJECTIVE 3: NON-LINEAR ENERGY TRANSFER WINDOW FILTERING (WATERFALL)")
    print("=" * 80)
    print(f"Total candidate windows scanned: {n_total} (t_start in [291.3, 310.0] ms, win_len=8.0 ms, step=0.25 ms)")
    print(f"Reference 8 ms core in [{T_NBI_BURST_START_MS:.1f}, {T_NBI_BURST_END_MS:.1f}] ms:")
    print(f"  Flattest patch: [{ref_core['core_start_ms']:.2f}, {ref_core['core_end_ms']:.2f}] ms")
    print(f"  Core mean |d(f_inst)/dt|: {ref_core['best_mean']:.4f} kHz/ms")
    print(f"  Flatness tolerance (1.5x core): {ref_core['tol_mean']:.4f} kHz/ms")
    print(f"SNR threshold: den > {SNR_THRESHOLD:.1e}")

    # --- WATERFALL 1: Strict Containment [t, t+8.0] in [291.3, 300.0] ms ---
    print("\n" + "-" * 80)
    print("1. WATERFALL: STRICT NBI-BURST CONTAINMENT ([t, t+8.0] in [291.3, 300.0] ms)")
    print("-" * 80)
    w_contain_strict = [w for w in windows if w['pass_containment_strict']]
    w_flat_strict = [w for w in w_contain_strict if w['pass_flatness']]
    w_survive_strict = [w for w in w_flat_strict if w['pass_snr']]

    print(f"Stage 0 (Total candidate windows):       {n_total}")
    print(f"Stage 1 (Containment <= 300.0 ms):       {len(w_contain_strict)}")
    print(f"Stage 2 (Flatness <= {ref_core['tol_mean']:.3f} kHz/ms):     {len(w_flat_strict)}")
    print(f"Stage 3 (SNR den > {SNR_THRESHOLD:.1e}):        {len(w_survive_strict)}")

    print(f"\nStrict Survivors Sign Tally (Im[B_sum]):")
    pos_strict = [w for w in w_survive_strict if w['im_val'] > 0]
    neg_strict = [w for w in w_survive_strict if w['im_val'] < 0]
    print(f"  Positive (Im > 0, Upward cascade): {len(pos_strict)} / {len(w_survive_strict)}")
    print(f"  Negative (Im < 0, Downward decay): {len(neg_strict)} / {len(w_survive_strict)}")
    for i, w in enumerate(w_survive_strict, 1):
        print(f"  Win #{i}: [{w['t_start']:.2f}, {w['t_end']:.2f}] ms (center {w['t_center']:.2f} ms) | "
              f"slope={w['slope']:.3f} kHz/ms | den={w['den']:.2e} | Im[B_sum]={w['im_val']:+.4f} | "
              f"f50={w['f50_peak']/1e3:.2f} kHz, f100={w['f100_peak']/1e3:.2f} kHz")

    # --- WATERFALL 2: Sensitivity at Beam Turn-off ([t, t+8.0] <= 300.6 ms) ---
    print("\n" + "-" * 80)
    print("2. WATERFALL: BEAM TURN-OFF SENSITIVITY ([t, t+8.0] <= 300.6 ms)")
    print("-" * 80)
    w_contain_beam = [w for w in windows if w['pass_containment_beam']]
    w_flat_beam = [w for w in w_contain_beam if w['pass_flatness']]
    w_survive_beam = [w for w in w_flat_beam if w['pass_snr']]

    print(f"Stage 0 (Total candidate windows):       {n_total}")
    print(f"Stage 1 (Containment <= 300.6 ms):       {len(w_contain_beam)}")
    print(f"Stage 2 (Flatness <= {ref_core['tol_mean']:.3f} kHz/ms):     {len(w_flat_beam)}")
    print(f"Stage 3 (SNR den > {SNR_THRESHOLD:.1e}):        {len(w_survive_beam)}")

    print(f"\nBeam Turn-off Survivors Sign Tally (Im[B_sum]):")
    pos_beam = [w for w in w_survive_beam if w['im_val'] > 0]
    neg_beam = [w for w in w_survive_beam if w['im_val'] < 0]
    print(f"  Positive (Im > 0, Upward cascade): {len(pos_beam)} / {len(w_survive_beam)}")
    print(f"  Negative (Im < 0, Downward decay): {len(neg_beam)} / {len(w_survive_beam)}")
    for i, w in enumerate(w_survive_beam, 1):
        print(f"  Win #{i}: [{w['t_start']:.2f}, {w['t_end']:.2f}] ms (center {w['t_center']:.2f} ms) | "
              f"slope={w['slope']:.3f} kHz/ms | den={w['den']:.2e} | Im[B_sum]={w['im_val']:+.4f} | "
              f"f50={w['f50_peak']/1e3:.2f} kHz, f100={w['f100_peak']/1e3:.2f} kHz")

    # --- FREQUENCY DRIFT IN BINS ---
    print("\n" + "-" * 80)
    print("3. MODE FREQUENCY DRIFT ACROSS SURVIVORS")
    print("-" * 80)
    # FFT bin resolution
    fs = 1e6
    nfft_psd = 1024
    df_psd_khz = (fs / nfft_psd) / 1000.0  # 0.9766 kHz
    nfft_bicoh = 512
    df_bicoh_khz = (fs / nfft_bicoh) / 1000.0  # 1.9531 kHz

    f50_beam = [w['f50_peak'] / 1e3 for w in w_survive_beam]
    f100_beam = [w['f100_peak'] / 1e3 for w in w_survive_beam]
    drift_f50 = max(f50_beam) - min(f50_beam)
    drift_f100 = max(f100_beam) - min(f100_beam)

    print(f"FFT resolution: PSD df = {df_psd_khz:.4f} kHz (N_fft=1024), Bicoherence df = {df_bicoh_khz:.4f} kHz (N_fft=512)")
    print(f"Beam survivors (N=5):")
    print(f"  ~50 kHz mode range: [{min(f50_beam):.2f}, {max(f50_beam):.2f}] kHz -> Drift = {drift_f50:.2f} kHz "
          f"({drift_f50 / df_psd_khz:.2f} PSD bins, {drift_f50 / df_bicoh_khz:.2f} bicoh bins)")
    print(f"  ~100 kHz mode range: [{min(f100_beam):.2f}, {max(f100_beam):.2f}] kHz -> Drift = {drift_f100:.2f} kHz "
          f"({drift_f100 / df_psd_khz:.2f} PSD bins, {drift_f100 / df_bicoh_khz:.2f} bicoh bins)")

    # Full 16 valid SNR windows drift for comparison:
    w_valid_16 = [w for w in windows if w['pass_snr']]
    f50_16 = [w['f50_peak'] / 1e3 for w in w_valid_16]
    f100_16 = [w['f100_peak'] / 1e3 for w in w_valid_16]
    drift_f50_16 = max(f50_16) - min(f50_16)
    drift_f100_16 = max(f100_16) - min(f100_16)
    print(f"Full 16 valid SNR windows (spans post-turnoff slowing down):")
    print(f"  ~50 kHz mode range: [{min(f50_16):.2f}, {max(f50_16):.2f}] kHz -> Drift = {drift_f50_16:.2f} kHz "
          f"({drift_f50_16 / df_psd_khz:.2f} PSD bins, {drift_f50_16 / df_bicoh_khz:.2f} bicoh bins)")
    print(f"  ~100 kHz mode range: [{min(f100_16):.2f}, {max(f100_16):.2f}] kHz -> Drift = {drift_f100_16:.2f} kHz "
          f"({drift_f100_16 / df_psd_khz:.2f} PSD bins, {drift_f100_16 / df_bicoh_khz:.2f} bicoh bins)")

    # --- RECOMPUTED REGRESSIONS ---
    print("\n" + "-" * 80)
    print("4. RECOMPUTED AMPLITUDE SCALING REGRESSIONS")
    print("-" * 80)
    
    def eval_reg(subset, name):
        if len(subset) < 2:
            print(f"{name}: N={len(subset)}, insufficient points for regression.")
            return
        x = np.log10([w['A50_band'] for w in subset])
        y = np.log10([w['A100_band'] for w in subset])
        res = stats.linregress(x, y)
        r2 = res.rvalue**2 if not np.isnan(res.rvalue) else np.nan
        print(f"{name:32s}: N={len(subset):2d} | Slope = {res.slope:6.3f} | Intercept = {res.intercept:6.3f} | "
              f"R² = {r2:5.3f} | p-value = {res.pvalue:.4e}")

    eval_reg(w_survive_strict, "Strict Survivors (N=2)")
    eval_reg(w_survive_beam, "Beam Survivors (N=5)")
    eval_reg([w for w in windows if w['pass_snr'] and w['t_center'] <= 297.5], "Episode 1 (N=6, t_center<=297.5)")
    eval_reg([w for w in windows if w['pass_snr'] and w['t_center'] > 300.0], "Episode 2 (N=10, t_center>300.0)")
    eval_reg([w for w in windows if w['pass_snr']], "Pooled 16 Valid Windows (Old)")
    eval_reg(windows, "All 75 Windows (Old)")

    # --- PLOT 1: sign_vs_threshold with 3-rule coloring (excluded in grey) ---
    fig_sign, ax_sign = plt.subplots(figsize=(8, 6), dpi=300)
    
    # Plot excluded windows in grey
    t_all = np.array([w['t_center'] for w in windows])
    im_all = np.array([w['im_val'] for w in windows])
    den_all = np.array([w['den'] for w in windows])
    pass_beam = np.array([w['pass_all_beam'] for w in windows])
    
    # Excluded
    ax_sign.scatter(t_all[~pass_beam], im_all[~pass_beam], color='grey', alpha=0.35, s=40,
                    label='Excluded by 3 rules (N=70)')
    
    # Survivors (colored by log10(den) with prominent edge)
    sc = ax_sign.scatter(t_all[pass_beam], im_all[pass_beam], c=np.log10(den_all[pass_beam]),
                         cmap='viridis', edgecolors='black', linewidths=1.8, s=80, zorder=5,
                         label='Survivors (N=5, all rules met)')
    cbar = plt.colorbar(sc, ax=ax_sign)
    cbar.set_label('log10(den)', fontsize=10)
    ax_sign.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax_sign.set_xlabel('Window Center Time (ms)', fontsize=11)
    ax_sign.set_ylabel('Im[B_sum]', fontsize=11)
    ax_sign.set_title('Sign vs Threshold - 88653 (3-Rule Filtered)', fontsize=12)
    ax_sign.legend(loc='lower left', fontsize=9.5)
    ax_sign.grid(True, alpha=0.3)
    fig_sign.tight_layout()
    out_sign = staging_dir / "sign_vs_threshold_3rules_88653.png"
    fig_sign.savefig(out_sign, dpi=300)
    plt.close(fig_sign)
    print(f"\nSaved staged 3-rule sign_vs_threshold plot to: {out_sign}")

    # --- PLOT 2: amplitude scaling with 3-rule coloring (excluded in grey) ---
    fig_sc, ax_sc = plt.subplots(figsize=(7, 5), dpi=300)
    log_a50_all = np.array([np.log10(w['A50_band']) for w in windows])
    log_a100_all = np.array([np.log10(w['A100_band']) for w in windows])
    
    ax_sc.scatter(log_a50_all[~pass_beam], log_a100_all[~pass_beam], color='grey', alpha=0.35, s=35,
                  label=f'Excluded windows (N={np.sum(~pass_beam)})')
    ax_sc.scatter(log_a50_all[pass_beam], log_a100_all[pass_beam], color='blue', edgecolors='black',
                  linewidths=1.2, s=55, zorder=5, label=f'Survivors (N={np.sum(pass_beam)})')
    
    # Regression on survivors
    res_b = stats.linregress(log_a50_all[pass_beam], log_a100_all[pass_beam])
    x_line = np.linspace(-1.55, -1.25, 100)
    ax_sc.plot(x_line, res_b.slope * x_line + res_b.intercept, color='blue', lw=1.8,
               label=f'Survivor slope = {res_b.slope:.2f}')
    
    # Reference slopes passing through centroid
    xm = np.mean(log_a50_all[pass_beam])
    ym = np.mean(log_a100_all[pass_beam])
    ax_sc.plot(x_line, 1.0 * (x_line - xm) + ym, color='green', ls=':', lw=1.8, label='Ref slope = 1.0')
    ax_sc.plot(x_line, 2.0 * (x_line - xm) + ym, color='red', ls=':', lw=1.8, label='Ref slope = 2.0')
    
    ax_sc.set_title('Amplitude Scaling: 3-Rule Survivors (t_end <= 300.6 ms)', fontsize=12)
    ax_sc.set_xlabel('log10(A_50)', fontsize=11)
    ax_sc.set_ylabel('log10(A_100)', fontsize=11)
    ax_sc.grid(True)
    ax_sc.legend(loc='upper left', fontsize=9.5)
    fig_sc.tight_layout()
    out_scale = staging_dir / "amplitude_scaling_3rules_survivors.png"
    fig_sc.savefig(out_scale, dpi=300)
    plt.close(fig_sc)
    print(f"Saved staged 3-rule amplitude scaling plot to: {out_scale}")

if __name__ == "__main__":
    staging_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_waterfall_analysis(staging_dir=staging_arg)
