# Bispectral Signatures in Heliotron J: Degenerate Parametric Decay vs. Harmonic Self-Coupling

## 1. The Mathematical Identity of the $-0.5$ Slope
In our analysis of magnetic fluctuations (Shot #88653), the squared auto-bicoherence $b^2(f_1, f_2)$ reveals a strong, statistically significant triad clustering along the locus $f_2 \approx -0.5 f_1$ in the difference-coupling domain (specifically at $f_1 = 98.6$ kHz, $f_2 = -48.8$ kHz).

Using the standard resonant triad matching condition $f_3 = f_1 + f_2$, substituting the observed locus yields:
$$ f_3 = f_1 + (-0.5 f_1) = 0.5 f_1 $$
This implies that $|f_2| = |f_3| = 0.5 f_1$. Mathematically, this bispectral geometry is consistent with a process where a higher frequency ($f_1$) and its exact half-harmonic ($f_1/2$) are phase-locked.

## 2. Physical Interpretations

While the algebra dictates a phase-locked $f \leftrightarrow f/2$ relationship, distinguishing its physical origin requires careful scrutiny of two competing mechanisms.

### a) The Parsimonious Explanation: Harmonic Self-Coupling
A well-known caveat in bispectral analysis is that a single coherent mode with an anharmonic waveform (or a mode that chirps across the sampling window) naturally generates a bicoherence peak at $f_2 = -0.5 f_1$ *by construction*. 
If a fundamental mode exists at $f_0$, its non-linear waveform will produce a second harmonic at $2f_0$ that is trivially phase-locked to the fundamental. In the difference-coupling domain, the algorithm interprets the second harmonic as the "pump" ($f_1 = 2f_0$) and the fundamental as the interacting wave ($f_2 = -f_0 = -0.5 f_1$). This peak does not represent two independent modes interacting, but merely the spectral footprint of a single oscillator.

### b) Degenerate Parametric Decay Instability (PDI)
Alternatively, this signature is consistent with a primary pump wave decaying into two identical subharmonic daughter waves ($f_1 \rightarrow f_1/2 + f_1/2$). While non-degenerate PDI of Alfvén Eigenmodes into GAM and kinetic-TAE branches has been proposed theoretically (building on the Alfvén-wave nonlinear theory framework of Chen & Zonca 2016), strictly *degenerate* decay in the low-frequency MHD regime remains an uncommon theoretical extrapolation. It is seen more typically in the high-frequency microwave/upper-hybrid regime (like Two-Plasmon Decay), a distinct physical setting from kHz-range MHD activity — so it should be treated as an analogy for the math, not a direct precedent.

## 3. Methodological Caveats and Conclusion

While the $f_2 = -0.5 f_1$ feature is mathematically suggestive of degenerate parametric decay, empirical validation against the dual-criterion pipeline necessitates extreme caution:

1. **Frequency Mismatch**: The objective dual-criterion pipeline validated highly coherent primary modes at 89.0 kHz and 95.5 kHz. The unvalidated bicoherence "pump" peak at 98.6 kHz differs from the established 89.0 kHz primary by $\sim 10\%$. This discrepancy suggests the 98.6 kHz peak is likely highly sensitive to spectral search boundaries or transient chirping, rather than acting as a stable, physical pump wave.
2. **Failure of Double-Validation**: The isolated triad failed critical verification tests. Cross-diagnostic corroboration (e.g., Langmuir probe array) came up empty, and the triad's power is not statistically superior to quiescent or control periods. Furthermore, shot-to-shot reproducibility checks were hindered by missing data in the control discharge (Shot #88654).

**Conclusion:** The exact $-0.5$ slope in the bicoherence plane confirms phase-locking between a frequency and its half-harmonic. While the asymptotic convergence check (Section 2.6.2) verifies this peak is a genuine quadratic phase coupling rather than a finite-sample statistical artifact, it cannot distinguish between independent three-wave coupling and harmonic self-coupling. Given the lack of cross-diagnostic corroboration, the failure to pass objective reproducibility tests, and the frequency mismatch with established modes, this cannot be claimed as a direct measurement of independent Degenerate Parametric Decay. The most parsimonious and empirically defensible explanation is the harmonic self-coupling of a single anharmonic or chirping mode.

