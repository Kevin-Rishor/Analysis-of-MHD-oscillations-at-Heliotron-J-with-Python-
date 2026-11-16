# Objective 3 — Nonlinear Wave-Wave Coupling & Bispectral Convergence

This directory implements **Objective 3** of the thesis (*Thesis* Sections 2.7, 3.4.3, and 4.4): distinguishing true quadratic phase coupling from finite-sample estimator bias across shots `#88653` and `#88654` (`MP1`).

---

## Summary of Methodology & Key Findings

1. **Index-Corrected Auto-Bicoherence (`analysis/common/abicoh2_corrected.py`)**:
   - Fixes the legacy `LAS.abicoh2` convention error (where the sum-frequency index was evaluated asymmetrically) with explicit `idx_sum = k1 + k2` and `detrend=False` FFT segments (`nfft = 512`, 50% Hann overlap, Δf = 1.953 kHz).
   - Verified by 5 synthetic/analytical unit tests in `analysis/common/test_abicoh2_corrected.py` (`T1`–`T5`).

2. **Single-Window Control Benchmark (`plot_control_windows.py`, `compute_bicoherence_maps_and_controls.py`)**:
   - Evaluates 33 contiguous `7.936 ms` (`K = 30` sub-ensembles) windows across pre-ECH, ECH-only, ECH+NBI, NBI-only, and post-NBI segments (`Thesis` Section 4.4.1, Fig. 4.17).
   - Demonstrates why single-window peak statistics (`b^2_max`) can be dominated by `~1/K` finite-sample variance rather than genuine physical coupling.

3. **Asymptotic `1/K` Extrapolation & 500 Phase-Randomized Surrogates (`run_convergence_500_surrogates.py`, `generate_two_panel_convergence_fig.py`)**:
   - Extends to the full `32.1 ms` `ECH+NBI` steady window (`259.1–291.2 ms`, `K_eff = K/2`, `1/K <= 0.055`, 8 ensemble counts `K in [18, 21, 24, 27, 29, 32, 35, 38]` within-shot and `K <= 112` cross-shot) and fits `b^2(K) = b0 + C/K` across `500` subsamples without replacement against `500` Fourier phase-randomized surrogates (`Thesis` Tables 4.7–4.8, Figs. 4.18–4.19).
   - **Confirmed Positive Couplings (p_surr < 0.05, CIs strictly excluding zero)**:
     - **Primary harmonic steepening**: `(88, 88) -> 176 kHz` (`b0 = +0.546`, `95% CI [+0.369, +0.728]`, `p_surr = 0.002` in `#88653`) and cross-shot (`b0 = +0.163`, `p_surr = 0.002` in `#88654`); mirrored in difference form `(180 - 90) -> 90 kHz` in `#88654` (`b0 = +0.440`, `p_surr = 0.002`).
     - **Sideband difference triad**: `(115 - 33) -> 82 kHz` (`b0 = +0.146`, `95% CI [+0.059, +0.233]`, `p_surr = 0.014` in `#88653`).
     - **Marginal cross-mode interaction (primary + secondary)**: `(47 + 80) -> 127 kHz` (`b0 = +0.092`, `95% CI [+0.008, +0.179]`, `p_surr = 0.044` in `#88653`).
     - **Off-diagonal null controls**: `(33 + 80) -> 113 kHz` (`b0 = +0.017`, `p_surr = 0.315` in `#88653`) and `(72 + 76) -> 148 kHz` (`b0 = +0.005`, `p_surr = 0.365` in `#88654`) extrapolate to zero.

---

## Script Inventory

| Script | Role | Requires `.npz` Cache? |
| --- | --- | --- |
| `run_obj3.py` | Master rectified Objective 3 runner (`--recompute-surrogates`, `--legacy-library-estimator`) | Only for maps/surrogate recompute |
| `plot_control_windows.py` | Generates `fig_control_windows_v2.png` (Fig. 4.17) from JSON controls | No (reads `analysis/obj_ech/bicoherence_maps_and_controls_results.json`) |
| `update_bicoherence_maps_converged.py` | Generates `fig_bicoherence_maps_both_shots.png` (Fig. 4.18) using `abicoh2_corrected` | Yes |
| `run_convergence_500_surrogates.py` | Runs 500-subset / 500-surrogate `1/K` sweep & exports `t41_convergence_500surr.json` | Yes |
| `generate_two_panel_convergence_fig.py` | Generates `fig_convergence_cumulative_v3_general.png` (Fig. 4.19) | No (reads `gap_inventory/*.json`) |
| `energy_transfer/` | Phase-resolved nonlinear amplitude-modulation & energy-transfer auxiliary diagnostics | Varies |
| `SUPERSEDED.md` | Inventory of early `LAS.abicoh2` exploratory scripts kept for provenance | — |

## Quick Usage

```bash
python analysis/obj3_nonlinear_coupling/run_obj3.py
```
