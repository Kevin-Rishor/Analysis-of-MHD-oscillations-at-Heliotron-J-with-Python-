import os
import re

def main():
    log = []
    
    # 1. t-statistic degrees of freedom
    log.append("### 1. Partial-Correlation t-statistic df")
    log.append("The code **does** explicitly subtract `n_control`, exactly matching the $N_{eff} - 2 - n_{control}$ formula. This preserves the reported p-values accurately without bias.")
    log.append("```python")
    log.append("- `C:/TFG/analysis/mhd_analysis_obj2.py`, Line 117: `df = N_eff - 2 - n_control`")
    log.append("- `C:/TFG/analysis/mhd_analysis_obj2.py`, Line 120: `t_stat = r * np.sqrt(df / (1.0 - r ** 2))`")
    log.append("```")
    log.append("")

    # 2. Toroidal n-fit
    log.append("### 2. Toroidal n-fit (RMSE and Phase Error)")
    log.append("CONFIRMED. The code implements the theoretical phase and RMSE exactly as typeset.")
    log.append("```python")
    log.append("- `C:/TFG/analysis/obj1_mode_identification/mhd_primary_mode_id.py`, Line 119: `theo_rad = (n * d_tor_rad + np.pi) % (2 * np.pi) - np.pi`")
    log.append("- `C:/TFG/analysis/obj1_mode_identification/mhd_primary_mode_id.py`, Line 120: `diff_rad = (pdict[\"phase_rad\"] - theo_rad + np.pi) % (2 * np.pi) - np.pi`")
    log.append("- `C:/TFG/analysis/obj1_mode_identification/mhd_primary_mode_id.py`, Line 123: `rmse_deg = float(np.degrees(np.sqrt(sq_err / len(pairs))))`")
    log.append("```")
    log.append("")

    # 3. Poloidal m-fit
    log.append("### 3. Poloidal m-fit (Circular Concentration and Phase Uncertainty)")
    log.append("CONFIRMED. The circular concentration $r_{circ}$ exactly matches the Mardia-Jupp formulation. The phase uncertainty correctly implements the Bendat & Piersol formula.")
    log.append("```python")
    log.append("- `C:/TFG/analysis/mhd_mode_mixing_analysis.py`, Line 205: `r_circ = float(np.sqrt(np.sum(np.cos(diff))**2 + np.sum(np.sin(diff))**2) / len(angles_rad))`")
    log.append("- `C:/TFG/analysis/mhd_analysis_obj2.py`, Line 498: `sigma_phase = np.sqrt(np.maximum(0.0, 1.0 - coherence_sq) / (2.0 * np.maximum(coherence_sq, 1e-4) * n_seg))`")
    log.append("```")
    log.append("")

    # 4. Self-coupling coherence theory
    log.append("### 4. Self-Coupling Coherence Theory")
    log.append("CONFIRMED. The coherence calculation is performed directly between the raw signal and its Hilbert envelope using standard cross-spectral density. The noise floor calculation precisely matches $\\gamma_{th}^2 = 1 - \\alpha^{1/(N-1)}$.")
    log.append("```python")
    log.append("- `C:/TFG/analysis/obj2_dynamics_and_structure/mhd_plot_self_coupling_coherence.py`, Line 60: `f, coh2 = dsp.coherence(ys_win, env_win, fs=fs, window=\"hann\", nperseg=nperseg, noverlap=noverlap, nfft=nfft)`")
    log.append("- `C:/TFG/analysis/obj2_dynamics_and_structure/mhd_plot_self_coupling_coherence.py`, Line 73: `sig_floor = 1.0 - (0.05) ** (1.0 / max(1, n_seg - 1))`")
    log.append("```")
    log.append("")

    # 5. NUDFT phase extraction
    log.append("### 5. NUDFT Phase Extraction Formula")
    log.append("CONFIRMED. The code implements the discrete poloidal transform exactly matching $\\frac{1}{N} \\sum_{j} S_j e^{-i m \\theta_j}$.")
    log.append("```python")
    log.append("- `C:/TFG/analysis/mhd_analysis_obj2.py`, Line 278: `return (1.0 / n) * np.dot(complex_values, np.exp(sign * 1j * k_grid * angles_rad[:, None]))`")
    log.append("```")
    log.append("")

    # 6. Sum/diff domain sign convention
    log.append("### 6. Sum/Diff Domain Sign Convention (CONTRADICTION FOUND)")
    log.append("The text defines the domains as follows:")
    log.append("```latex")
    log.append(r"\begin{equation}")
    log.append(r"\label{eqn:sum_domain}")
    log.append(r"f_1 + f_2 = f_3 \quad (f_1 > 0, f_2 > 0)")
    log.append(r"\end{equation}")
    log.append(r"\begin{equation}")
    log.append(r"\label{eqn:diff_domain}")
    log.append(r"f_1 - f_2 = f_3 \quad (f_1 > 0, f_2 < 0)")
    log.append(r"\end{equation}")
    log.append("```")
    log.append("**FLAG:** There is a mathematical contradiction (typo) in the `Theoretical_framework.tex` text. If $f_2 < 0$, then $f_1 - f_2 = f_1 + |f_2|$, which evaluates to a sum coupling, not a difference coupling! The standard definition for the difference domain should be either $f_1 + f_2 = f_3 \\quad (f_2 < 0)$ OR $f_1 - f_2 = f_3 \\quad (f_1, f_2 > 0)$.")
    log.append("Because of this typo, it incorrectly breaks the $\\text{Im}[B_{sum}] = -\\text{Im}[B_{diff}]$ symmetry relationship established during the predator-prey Lissajous analysis. The equation needs to be fixed to correctly imply the conjugate mapping $B(f_1, -f_d) = B(f_1-f_d, f_d)^*$.")
    log.append("")

    # 7 & 8. LaTeX and Bib Checks
    log.append("### 7 & 8. Mechanical LaTeX and Citation Checks")
    log.append("- **Duplicate Labels:** None found.")
    log.append("- **Unreferenced Labels (Defined but never `\\ref{}`):** `eqn:MHD_Ohm`, `eqn:MHD_State`, `eqn:MHD_mass_continuity`, `eqn:MHD_Ampere`")
    log.append("- **Missing Labels (Referenced but never defined):** `annex:data`, `appendix:github`, `annex:library`, `annex:poloidalm` (these are likely defined in separate appendix/annex `.tex` files not included in this batch).")
    log.append("- **Bibliography `.bib` file:** **NOT FOUND.** A deep search `ls` and `Get-ChildItem -Filter *.bib` across the entire `C:\\TFG` repository confirms there is no `.bib` file present in the project tree to validate the `\\cite{}` keys against.")

    with open('C:/TFG/thesis_audit_v6.log', 'w', encoding='utf-8') as f:
        f.write('\n'.join(log))

if __name__ == '__main__':
    main()
