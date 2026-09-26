import subprocess, time, os, glob, shutil

tasks = [
    ("mhd_primary_mode_id.py", r"c:\TFG\analysis\obj1_mode_identification\mhd_primary_mode_id.py", ["-s", "88653"]),
    ("mhd_plot_2d_torus_maps.py", r"c:\TFG\analysis\obj2_dynamics_and_structure\mhd_plot_2d_torus_maps.py", ["-s", "88653"]),
    ("mhd_analysis_obj1.py", r"c:\TFG\analysis\mhd_analysis_obj1.py", ["-s", "88653"]),
    ("heating_vs_mirnov.py", r"c:\TFG\analysis\obj2_dynamics_and_structure\heating_vs_mirnov.py", ["-s", "88653"]),
    ("mhd_compare_spectrograms.py", r"c:\TFG\analysis\heating_phase_comparison\mhd_compare_spectrograms.py", ["-s", "88653"]),
    ("mhd_compare_mode_identification.py", r"c:\TFG\analysis\heating_phase_comparison\mhd_compare_mode_identification.py", ["-s", "88653"]),
    ("mhd_analysis_obj2.py_40_80", r"c:\TFG\analysis\obj2_dynamics_and_structure\mhd_analysis_obj2.py", ["--shots", "88653", "-l", "40", "-u", "80"]),
    ("mhd_analysis_obj2.py_80_120", r"c:\TFG\analysis\obj2_dynamics_and_structure\mhd_analysis_obj2.py", ["--shots", "88653", "-l", "80", "-u", "120"]),
    ("mhd_compare_mode_structure.py", r"c:\TFG\analysis\heating_phase_comparison\mhd_compare_mode_structure.py", ["-s", "88653"]),
    ("mhd_plot_self_coupling_coherence.py", r"c:\TFG\analysis\obj2_dynamics_and_structure\mhd_plot_self_coupling_coherence.py", ["-s", "88653"]),
    ("mhd_plot_dual_mode_envelope_vs_ip.py", r"c:\TFG\analysis\obj2_dynamics_and_structure\mhd_plot_dual_mode_envelope_vs_ip.py", ["-s", "88653"]),
    ("mhd_plot_hysteresis_5panel.py", r"c:\TFG\analysis\obj2_dynamics_and_structure\mhd_plot_hysteresis_5panel.py", ["-s", "88653"]),
    ("mhd_plot_modulator_dual_mode.py", r"c:\TFG\analysis\obj2_dynamics_and_structure\mhd_plot_modulator_dual_mode.py", ["-s", "88653"]),
    ("mhd_plot_b_tilde_vs_envelope.py", r"c:\TFG\analysis\obj2_dynamics_and_structure\mhd_plot_b_tilde_vs_envelope.py", ["-s", "88653"]),
    ("mhd_plot_snr_halpha_correlation.py", r"c:\TFG\analysis\obj2_dynamics_and_structure\mhd_plot_snr_halpha_correlation.py", ["-s", "88653"]),
    ("mhd_analysis_obj3.py", r"c:\TFG\analysis\mhd_analysis_obj3.py", ["-s", "88653", "--burst-start", "259.1", "--burst-end", "275.0"]),
    ("mhd_plot_bicoherence_sliding_trace.py", r"c:\TFG\analysis\heating_phase_comparison\mhd_plot_bicoherence_sliding_trace.py", ["-s", "88653"]),
    ("phase1_r3.py", r"c:\TFG\scratch\phase1_r3.py", []),
    ("phase1_r4.py", r"c:\TFG\scratch\phase1_r4.py", []),
    ("regenerate_sign_vs_threshold.py", r"c:\TFG\scratch\regenerate_sign_vs_threshold.py", []),
    ("mhd_plot_asymptotic_bias_scaling.py", r"c:\TFG\analysis\obj3_nonlinear_coupling\mhd_plot_asymptotic_bias_scaling.py", ["-s", "88653"]),
    ("mhd_compare_bicoherence.py", r"c:\TFG\analysis\heating_phase_comparison\mhd_compare_bicoherence.py", ["-s", "88653"])
]

print("Starting runs...")
for name, script, args in tasks:
    print(f"Running {name}...")
    cmd = ["python", script] + args + ["-o", r"c:\TFG\Images"]
    # try running without -o first if it fails because of unrecognized arguments
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        with open(f"c:\\TFG\\scratch\\log_{name}.txt", "w") as f:
            f.write(proc.stdout)
            f.write(proc.stderr)
        if proc.returncode != 0:
            print(f"Failed {name} with -o, retrying without -o...")
            cmd = ["python", script] + args
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            with open(f"c:\\TFG\\scratch\\log_{name}.txt", "w") as f:
                f.write(proc.stdout)
                f.write(proc.stderr)
    except subprocess.TimeoutExpired:
        print(f"Timeout on {name}")
print("Done running all scripts.")
