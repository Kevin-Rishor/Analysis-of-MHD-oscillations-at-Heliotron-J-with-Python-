#!/usr/bin/env python
"""
Panel Decomposer — Runs original multi-panel scripts via subprocess with a monkeypatch
injected at the top of a temporary copy. Each copy is saved alongside the original
in a scratch directory. Original files are NEVER modified.

This approach:
1. For each original script, creates a COPY in a scratch/ directory
2. Prepends a monkeypatch to the copy that hooks savefig to also save individual panels
3. Runs the copy via subprocess
4. Reports results

All panel images are saved to panels/ subfolders under the appropriate Images/ directory.
"""

import subprocess
import sys
import os
import textwrap
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRATCH_DIR = ROOT_DIR / "analysis" / "_panel_decompose_scratch"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

PYTHON = str(ROOT_DIR / ".venv" / "Scripts" / "python.exe")

# Monkeypatch code to prepend to each script copy
MONKEYPATCH_HEADER = textwrap.dedent(r'''
# === PANEL DECOMPOSER MONKEYPATCH (injected, does not modify originals) ===
import matplotlib as _mpl_mp
import matplotlib.pyplot as _plt_mp
import matplotlib.figure as _fig_mp
import matplotlib.transforms as _transforms_mp
import re as _re_mp
import os as _os_mp
from pathlib import Path as _Path_mp

_PANELS_DIR = _Path_mp(r"{panels_dir}")
_PANELS_DIR.mkdir(parents=True, exist_ok=True)
_ORIGINAL_FIG_SAVEFIG = _fig_mp.Figure.savefig
_ORIGINAL_PLT_SAVEFIG = _plt_mp.savefig

def _slugify(text):
    if not text:
        return ""
    text = _re_mp.sub(r'\$[^$]*\$', '', text)
    text = _re_mp.sub(r'[^\w\s-]', '', text)
    text = text.strip()
    text = _re_mp.sub(r'[-\s]+', '_', text)
    if len(text) > 40:
        text = text[:40].rstrip('_')
    return text.lower()

def _extract_panels(fig, fname, dpi):
    fname_path = _Path_mp(str(fname))
    stem = fname_path.stem
    all_axes = fig.get_axes()
    primary_axes = []
    for ax in all_axes:
        if hasattr(ax, '_colorbar_info') or getattr(ax, '_colorbar', False):
            continue
        bbox = ax.get_position()
        w, h = bbox.width, bbox.height
        aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 1
        if aspect < 0.12 and (w < 0.06 or h < 0.06):
            continue
        primary_axes.append(ax)
    if not primary_axes or len(primary_axes) < 2:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    print(f"\n  Extracting {{len(primary_axes)}} panels from: {{fname_path.name}}")
    for i, ax in enumerate(primary_axes, 1):
        bbox_display = ax.get_tightbbox(renderer)
        if bbox_display is None:
            bbox_display = ax.get_window_extent(renderer)
        bbox_inches = bbox_display.transformed(fig.dpi_scale_trans.inverted())
        pad = 0.08
        bbox_padded = _transforms_mp.Bbox.from_extents(
            bbox_inches.x0 - pad, bbox_inches.y0 - pad,
            bbox_inches.x1 + pad, bbox_inches.y1 + pad
        )
        title = ax.get_title()
        slug = _slugify(title) if title else ""
        if slug:
            panel_name = f"{{stem}}_panel{{i}}_{{slug}}.png"
        else:
            panel_name = f"{{stem}}_panel{{i}}.png"
        panel_path = _PANELS_DIR / panel_name
        try:
            _ORIGINAL_FIG_SAVEFIG(fig, str(panel_path), dpi=dpi, bbox_inches=bbox_padded,
                                   facecolor=fig.get_facecolor(), edgecolor='none', pad_inches=0)
            print(f"    Panel {{i}}: {{panel_name}}")
        except Exception as e:
            print(f"    Panel {{i}} FAILED: {{e}}")

def _patched_fig_savefig(self, fname, *args, **kwargs):
    _ORIGINAL_FIG_SAVEFIG(self, fname, *args, **kwargs)
    n_axes = len([ax for ax in self.get_axes()])
    if n_axes > 1:
        dpi = kwargs.get('dpi', self.dpi)
        if isinstance(dpi, str):
            dpi = 200
        _extract_panels(self, fname, dpi)

def _patched_plt_savefig(fname, *args, **kwargs):
    fig = _plt_mp.gcf()
    _patched_fig_savefig(fig, fname, *args, **kwargs)

_fig_mp.Figure.savefig = _patched_fig_savefig
_plt_mp.savefig = _patched_plt_savefig
# === END PANEL DECOMPOSER MONKEYPATCH ===

''')


# Registry: (script_relative_path, panels_dir, output_dir_override, extra_argv)
# output_dir_override: if the script saves to root_dir by default, redirect via --out-dir or function arg
REGISTRY = [
    # === OBJ1 ===
    {
        "script": "obj1_mode_identification/mhd_analysis_obj1.py",
        "panels_dir": "Images/obj1_mode_identification/panels",
        "images": ["mhd_analysis_objective1_88653.png"],
        "cwd": "analysis/obj1_mode_identification",
        "argv": ["--shots", "88653", "--data-dir-pattern", str(ROOT_DIR / "data" / "hj{shot}")],
    },
    {
        "script": "obj1_mode_identification/mhd_primary_mode_id.py",
        "panels_dir": "Images/obj1_mode_identification/panels",
        "images": ["mhd_primary_mode_toroidal_n_shot_88653.png", "mhd_primary_mode_radial_profile_shot_88653.png"],
        "cwd": "analysis/obj1_mode_identification",
        "argv": ["--shots", "88653"],
    },
    # === OBJ2 standalone plots ===
    {
        "script": "obj2_dynamics_and_structure/heating_vs_mirnov.py",
        "panels_dir": "Images/obj2_dynamics_and_structure/panels",
        "images": ["heating_vs_mirnov (Shot_88653).png"],
        "cwd": "analysis/obj2_dynamics_and_structure",
        "argv": [],
    },
    {
        "script": "obj2_dynamics_and_structure/mhd_plot_2d_torus_maps.py",
        "panels_dir": "Images/obj2_dynamics_and_structure/panels",
        "images": ["mhd_mode_spatial_map_2d_88653.png"],
        "cwd": "analysis/obj2_dynamics_and_structure",
        "argv": [],
    },
    {
        "script": "obj2_dynamics_and_structure/mhd_plot_b_tilde_vs_envelope.py",
        "panels_dir": "Images/obj2_dynamics_and_structure/panels",
        "images": ["mhd_b_tilde_vs_envelope_shot_88653.png"],
        "cwd": "analysis/obj2_dynamics_and_structure",
        "argv": [],
    },
    {
        "script": "obj2_dynamics_and_structure/mhd_plot_dual_mode_envelope_vs_ip.py",
        "panels_dir": "Images/obj2_dynamics_and_structure/panels",
        "images": ["mhd_mode_envelopes_and_ip_evolution_shot_88653.png"],
        "cwd": "analysis/obj2_dynamics_and_structure",
        "argv": [],
    },
    {
        "script": "obj2_dynamics_and_structure/mhd_plot_hysteresis_5panel.py",
        "panels_dir": "Images/obj2_dynamics_and_structure/panels",
        "images": ["mhd_hysteresis_ip_vs_envelope_dual_modes_shot_88653.png"],
        "cwd": "analysis/obj2_dynamics_and_structure",
        "argv": [],
    },
    {
        "script": "obj2_dynamics_and_structure/mhd_plot_modulator_dual_mode.py",
        "panels_dir": "Images/obj2_dynamics_and_structure/panels",
        "images": ["mhd_modulator_dual_mode_coherence_shot_88653.png"],
        "cwd": "analysis/obj2_dynamics_and_structure",
        "argv": [],
    },
    {
        "script": "obj2_dynamics_and_structure/mhd_plot_self_coupling_coherence.py",
        "panels_dir": "Images/obj2_dynamics_and_structure/panels",
        "images": ["mhd_self_coupling_coherence_shot_88653.png"],
        "cwd": "analysis/obj2_dynamics_and_structure",
        "argv": [],
    },
    {
        "script": "obj2_dynamics_and_structure/mhd_plot_snr_halpha_correlation.py",
        "panels_dir": "Images/obj2_dynamics_and_structure/panels",
        "images": ["mhd_snr_halpha_correlation_shot_88653.png"],
        "cwd": "analysis/obj2_dynamics_and_structure",
        "argv": [],
    },
    {
        "script": "obj2_dynamics_and_structure/mhd_plot_hysteresis_ip_vs_envelope.py",
        "panels_dir": "Images/obj2_dynamics_and_structure/panels",
        "images": [],  # This saves to root but we need to check the actual image location
        "cwd": "analysis/obj2_dynamics_and_structure",
        "argv": [],
    },
    {
        "script": "obj2_dynamics_and_structure/mhd_plot_hysteresis_both_modes.py",
        "panels_dir": "Images/obj2_dynamics_and_structure/panels",
        "images": [],
        "cwd": "analysis/obj2_dynamics_and_structure",
        "argv": [],
    },
    # === OBJ3 ===
    {
        "script": "obj3_nonlinear_coupling/mhd_plot_asymptotic_bias_scaling.py",
        "panels_dir": "Images/obj3_nonlinear_coupling/panels",
        "images": ["mhd_asymptotic_bias_scaling_shot_88653.png"],
        "cwd": "analysis/obj3_nonlinear_coupling",
        "argv": [],
    },
    {
        "script": "obj3_nonlinear_coupling/mhd_plot_bicoherence_lines.py",
        "panels_dir": "Images/obj3_nonlinear_coupling/panels",
        "images": [],
        "cwd": "analysis/obj3_nonlinear_coupling",
        "argv": [],
    },
    {
        "script": "obj3_nonlinear_coupling/mhd_plot_bicoherence_top10.py",
        "panels_dir": "Images/obj3_nonlinear_coupling/panels",
        "images": [],
        "cwd": "analysis/obj3_nonlinear_coupling",
        "argv": [],
    },
    # === HEATING COMPARISON ===
    {
        "script": "heating_phase_comparison/mhd_compare_spectrograms.py",
        "panels_dir": "Images/heating_phase_comparison/panels",
        "images": ["zoomed_spectrogram_3sigma_ECH_NBI_259.1_291.3ms.png", "zoomed_spectrogram_3sigma_NBI_only_291.3_300.0ms.png"],
        "cwd": "analysis/heating_phase_comparison",
        "argv": ["-o", str(ROOT_DIR / "Images" / "heating_phase_comparison")],
    },
    {
        "script": "heating_phase_comparison/mhd_compare_mode_identification.py",
        "panels_dir": "Images/heating_phase_comparison/panels",
        "images": ["mhd_analysis_objective1_88653_ECH_NBI.png", "mhd_analysis_objective1_88653_NBI_only.png"],
        "cwd": "analysis/heating_phase_comparison",
        "argv": ["-o", str(ROOT_DIR / "Images" / "heating_phase_comparison")],
    },
    {
        "script": "heating_phase_comparison/mhd_compare_mode_structure.py",
        "panels_dir": "Images/heating_phase_comparison/panels",
        "images": [],
        "cwd": "analysis/heating_phase_comparison",
        "argv": ["-o", str(ROOT_DIR / "Images" / "heating_phase_comparison")],
    },
    {
        "script": "heating_phase_comparison/mhd_compare_self_coupling.py",
        "panels_dir": "Images/heating_phase_comparison/panels",
        "images": [],
        "cwd": "analysis/heating_phase_comparison",
        "argv": ["-o", str(ROOT_DIR / "Images" / "heating_phase_comparison")],
    },
    {
        "script": "heating_phase_comparison/mhd_compare_bicoherence.py",
        "panels_dir": "Images/heating_phase_comparison/panels",
        "images": ["mhd_bicoherence_objective3_88653_ECH_NBI.png", "mhd_bicoherence_objective3_88653_NBI_only.png"],
        "cwd": "analysis/heating_phase_comparison",
        "argv": ["-o", str(ROOT_DIR / "Images" / "heating_phase_comparison")],
    },
    {
        "script": "heating_phase_comparison/mhd_compare_toroidal_and_radial.py",
        "panels_dir": "Images/heating_phase_comparison/panels",
        "images": [],
        "cwd": "analysis/heating_phase_comparison",
        "argv": ["-o", str(ROOT_DIR / "Images" / "heating_phase_comparison")],
    },
]


def make_patched_copy(script_path, panels_dir):
    """Create a copy of the script with the monkeypatch header prepended."""
    original_content = script_path.read_text(encoding='utf-8')
    
    # Format the monkeypatch with the correct panels directory
    header = MONKEYPATCH_HEADER.format(panels_dir=str(panels_dir).replace('\\', '\\\\'))
    
    # Insert the monkeypatch AFTER any shebang/encoding lines but BEFORE imports
    lines = original_content.split('\n')
    insert_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#!') or stripped.startswith('# -*-') or stripped.startswith('# coding'):
            insert_idx = i + 1
        elif stripped.startswith('#') or stripped == '':
            insert_idx = i + 1
        else:
            break
    
    patched_content = '\n'.join(lines[:insert_idx]) + '\n' + header + '\n'.join(lines[insert_idx:])
    
    # Also force Agg backend before any matplotlib import
    patched_content = patched_content.replace(
        'import matplotlib.pyplot as plt',
        'import matplotlib\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt',
        1  # Only replace the first occurrence
    )
    
    # Save the copy
    copy_name = f"_decompose_{script_path.name}"
    copy_path = SCRATCH_DIR / copy_name
    copy_path.write_text(patched_content, encoding='utf-8')
    return copy_path


def run_entry(entry):
    """Run a single registry entry."""
    script_rel = entry["script"]
    script_path = ROOT_DIR / "analysis" / script_rel
    panels_dir = ROOT_DIR / entry["panels_dir"]
    cwd = ROOT_DIR / entry["cwd"]
    argv = entry.get("argv", [])
    
    if not script_path.exists():
        return {"script": script_rel, "status": "script_not_found", "panels": []}
    
    print(f"\n{'='*70}")
    print(f"Processing: {script_rel}")
    print(f"Panels dir: {panels_dir}")
    print(f"{'='*70}")
    
    # Create patched copy
    copy_path = make_patched_copy(script_path, panels_dir)
    
    # Count panels before
    existing_before = set(panels_dir.glob("*.png")) if panels_dir.exists() else set()
    
    # Run the copy
    cmd = [PYTHON, str(copy_path)] + argv
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout
            env={**os.environ, "MPLBACKEND": "Agg"}
        )
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        if result.returncode != 0:
            print(f"STDERR: {result.stderr[-1000:]}")
            return {"script": script_rel, "status": "failed", "error": result.stderr[-500:], "panels": []}
    except subprocess.TimeoutExpired:
        return {"script": script_rel, "status": "timeout", "panels": []}
    except Exception as e:
        return {"script": script_rel, "status": "error", "error": str(e), "panels": []}
    
    # Count panels after
    existing_after = set(panels_dir.glob("*.png")) if panels_dir.exists() else set()
    new_panels = existing_after - existing_before
    
    return {
        "script": script_rel,
        "status": "success",
        "panels": sorted([p.name for p in new_panels]),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("PANEL DECOMPOSER — Multi-Panel Figure Decomposition")
    print(f"Root: {ROOT_DIR}")
    print(f"Scratch: {SCRATCH_DIR}")
    print("=" * 70)
    
    all_results = []
    
    for entry in REGISTRY:
        result = run_entry(entry)
        all_results.append(result)
    
    # Summary
    print("\n\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    total_panels = 0
    for r in all_results:
        status_icon = "✓" if r["status"] == "success" and r["panels"] else "✗" if r["status"] != "success" else "○"
        n = len(r["panels"])
        total_panels += n
        print(f"  {status_icon} {r['script']}: {r['status']} ({n} panels)")
        if r.get("error"):
            print(f"      Error: {r['error'][:200]}")
        for p in r["panels"]:
            print(f"      → {p}")
    
    print(f"\nTotal: {total_panels} panels extracted.")
    print("No original files were modified, moved, renamed, or deleted.")
