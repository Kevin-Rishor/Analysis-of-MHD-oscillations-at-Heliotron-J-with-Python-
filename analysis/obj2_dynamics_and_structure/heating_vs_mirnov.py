import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------
# SHOT CONFIGURATION (Change this to plot different discharges)
# -------------------------------------------------------------
SHOT = 88653

# Add jpack library to python path for turnelib
jpack_path = str(Path("jpack").resolve())
if jpack_path not in sys.path:
    sys.path.append(jpack_path)

import turnelib as TE

# 1. High-Priority Data Searches
print(f"--- HIGH-PRIORITY DATA SEARCHES FOR SHOT #{SHOT} ---")
base_dir = Path(f"data/hj{SHOT}")

# Locate known files
wp_file = base_dir / f"Wp@{SHOT}.edf"
mp1_file = base_dir / f"MP1@{SHOT}.edf"

print(f"Wp file: {wp_file} (Exists: {wp_file.exists()})")
print(f"MP1 file: {mp1_file} (Exists: {mp1_file.exists()})")

# Locate associated heating files
ech_file = base_dir / f"ECHRG500@{SHOT}.edf"
nbi_files = sorted(list(base_dir.glob(f"NBIS*@{SHOT}.edf")))

print(f"ECH file: {ech_file} (Exists: {ech_file.exists()})")
print(f"NBI files found: {[f.name for f in nbi_files]}")

# Load NIRMON file for Pellet Injection (time vs adimensional, 0 except for the launch)
nir_file = base_dir / f"NIRMON@{SHOT}.edf"
print(f"NIRMON (Pellet Injection) file: {nir_file} (Exists: {nir_file.exists()})")

# 2. Loading data
print("\n--- LOADING DATA ---")

# Load Wp
edf_wp = TE.edf()
dat_wp = edf_wp.load(str(wp_file))
t_wp = dat_wp[:, 0]
y_wp = dat_wp[:, 1]

# Load MP1
edf_mp1 = TE.edf()
dat_mp1 = edf_mp1.load(str(mp1_file))
t_mp1 = dat_mp1[:, 0]
y_mp1 = dat_mp1[:, 1]

# Load ECH
edf_ech = TE.edf()
dat_ech = edf_ech.load(str(ech_file))
t_ech = dat_ech[:, 0]
y_ech = dat_ech[:, 1]

# Load NBI and sum them
total_nbi = None
t_nbi = None
for fpath in nbi_files:
    edf_nbi = TE.edf()
    dat_nbi = edf_nbi.load(str(fpath))
    if t_nbi is None:
        t_nbi = dat_nbi[:, 0]
        total_nbi = np.zeros_like(dat_nbi[:, 1])
    total_nbi += dat_nbi[:, 1]

# Load and process NIRMON to get the Pellet Injection signal (0 everywhere except for the launch, dimensionless)
edf_nir = TE.edf()
dat_nir = edf_nir.load(str(nir_file))
t_nir = dat_nir[:, 0]
y_nir = dat_nir[:, 1]

# Threshold to keep only the launch/spike (dimensionless, 0 everywhere else)
y_pellet = np.where(y_nir >= 0.15, y_nir, 0.0)
y_pellet = y_pellet / y_pellet.max() # Normalize to make it dimensionless (adimensional, 0 to 1)

# 3. Figure Plotting Specifications
print("\n--- PLOTTING FIGURE ---")
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6.5))

# Top Subplot: Heating & Plasma
# Plot Wp (Blue smooth curve)
ax1.plot(t_wp, y_wp, color='blue', linewidth=2.0, label='Wp(kJ)')

# Plot Pellet Injection (Orange sharp spike)
ax1.plot(t_nir, y_pellet * 0.5, color='orange', linewidth=1.5, label='Pellet Injection')

# Plot ECH (Green step function)
ax1.plot(t_ech, y_ech, color='green', linewidth=1.8, label='ECH(V)')

# Plot NBI (Red step function)
if total_nbi is not None:
    ax1.plot(t_nbi, total_nbi, color='red', linewidth=1.8, label='NBI(V)')

# Title and axis styling for top plot (including Shot #{SHOT} in the title)
ax1.set_title(f"Plasma Heating & Confinement / Mirnov Coil Comparison (#{SHOT})", fontsize=12, fontweight='bold')
ax1.set_ylabel("Heating & Plasma", fontsize=10)
ax1.set_ylim(-0.1, 5.0)
ax1.grid(True, linestyle=':', alpha=0.6)

# Legend in the top right
ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, edgecolor='gray')

# Bottom Subplot: Mirnov Coil
ax2.plot(t_mp1, y_mp1, color='blue', linewidth=0.5, alpha=0.8, label='MP1')
ax2.set_ylabel("Mirnov(V)", fontsize=10)
ax2.set_xlabel("Time (ms)", fontsize=10)
ax2.set_ylim(-1.5, 1.5)
ax2.grid(True, linestyle=':', alpha=0.6)

# Shared time axis limit (150 ms to 350 ms)
plt.xlim(150, 350)

plt.tight_layout()

# Save image
output_path = f"heating_vs_mirnov (Shot_{SHOT}).png"
plt.savefig(output_path, dpi=200)
print(f"Recreated figure successfully saved to: {output_path}")
