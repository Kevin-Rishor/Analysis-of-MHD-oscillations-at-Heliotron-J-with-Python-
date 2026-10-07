# Superseded analysis (library bicoherence routine)

The scripts below call `LAS.abicoh2` from `jpack/libana_signal.py`. That routine has an index error: the cell (f1, f2) evaluates X(f2) X(f1-df) X*(f1+f2) instead of X(f1) X(f2) X*(f1+f2) (`np.roll` is applied to the index used for the second factor). Results obtained with it are withdrawn and are not used in the thesis or the slides. The corrected estimator is `analysis/common/abicoh2_corrected.py`.

Direct callers of `LAS.abicoh2`:
- analysis/obj3_nonlinear_coupling/mhd_analysis_obj3.py
- analysis/obj3_nonlinear_coupling/mhd_plot_bicoherence_lines.py
- analysis/obj3_nonlinear_coupling/mhd_plot_bicoherence_top10.py
- analysis/heating_phase_comparison/mhd_compare_bicoherence.py

Scripts that depend on these, or on other library bispectrum routines, and have not been audited individually: analysis/obj3_nonlinear_coupling/run_obj3.py, analysis/obj3_nonlinear_coupling/mhd_plot_asymptotic_bias_scaling.py, the folder analysis/obj3_nonlinear_coupling/energy_transfer/, analysis/heating_phase_comparison/mhd_plot_bicoherence_sliding_trace.py, analysis/analyze_energy_transfer.py.

`run_obj3.py` refuses to run unless the environment variable ALLOW_LEGACY_LIBRARY_ESTIMATOR=1 is set. Figures produced only by these scripts were removed from the working tree; they remain in the git history.
