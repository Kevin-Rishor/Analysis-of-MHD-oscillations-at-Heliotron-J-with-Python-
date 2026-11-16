import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
IMG = ROOT / "Images" / "obj3_nonlinear_coupling"
IMG.mkdir(parents=True, exist_ok=True)

with open(ROOT / "analysis" / "obj_ech" / "gap_inventory" / "asymp_k_search_top15.json") as f:
    data = json.load(f)

with open(ROOT / "analysis" / "obj_ech" / "gap_inventory" / "t41_convergence_500surr.json") as f:
    t41 = json.load(f)

plt.rcParams.update({
    "mathtext.fontset": "stixsans",
    "font.size": 14.0,
    "axes.labelsize": 15.5,
    "axes.titlesize": 15.5,
    "xtick.labelsize": 14.0,
    "ytick.labelsize": 14.0,
    "legend.fontsize": 10.8,
})

fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.65), sharey=True)

# -----------------------------------------------------------------
# PANEL 1: SHOT #88653 (Clean, No Shaded Background Bands)
# -----------------------------------------------------------------
ax1 = axes[0]
Ks_53 = np.array(data["88653"]["Ks"])
x_53 = 1.0 / Ks_53
xx_53 = np.linspace(0, 0.06, 50)

ax1.plot(xx_53, xx_53, "--", color="0.65", lw=0.9, label=r"Theoretical $1/K$ (indep.)")
ax1.plot(xx_53, 2 * xx_53, ":", color="0.65", lw=0.9, label=r"Theoretical $2/K$ ($K_{\mathrm{eff}}=K/2$)")
ax1.axhline(0, color="0.8", lw=0.6)

# 1. Sum Harm (88+88): Rank 1 sum
p1_53 = data["88653"]["sum"][0]
# 2. Diff Triad (115-33): Rank 4 diff
p2_53 = data["88653"]["diff"][3]
# 3. Cross-Mode Interaction (47+80): Rank 12 sum
p3_53 = data["88653"]["sum"][11]
# 4. Noise Baseline (33+80): Rank 2 sum
pn_53 = data["88653"]["sum"][1]

items_53 = [
    (p1_53, "C0", "o", f"Sum Harm (88+88): b0={p1_53['b0']:+.3f}, C={p1_53['C']:.2f}, p={p1_53['p_surr']:.3f}"),
    (p2_53, "C1", "s", f"Diff Triad (115-33): b0={p2_53['b0']:+.3f}, C={p2_53['C']:.2f}, p={p2_53['p_surr']:.3f}"),
    (p3_53, "C2", "^", f"Cross-Mode (47+80): b0={p3_53['b0']:+.3f}, C={p3_53['C']:.2f}, p={p3_53['p_surr']:.3f}"),
    (pn_53, "C3", "d", f"Noise Ctrl (33+80): b0={pn_53['b0']:+.3f}, C={pn_53['C']:.2f}, p={pn_53['p_surr']:.3f}")
]

for p, col, marker, lbl in items_53:
    y = np.array(p["y"])
    # Clean plot without background shaded bands
    ax1.plot(x_53, y, marker=marker, color=col, ms=5, label=lbl, ls="none")
    ax1.plot(xx_53, p["b0"] + p["C"] * xx_53, "-", color=col, lw=1.3)
    ax1.axhline(p["surr_95"], color=col, ls=":", lw=1.0, alpha=0.7)

ax1.set_title(r"#88653 (ECH+NBI 32.1 ms, $1/K \leq 0.055$)", fontsize=15.0, fontweight="bold")
ax1.set_xlabel(r"Inverse Sub-Ensemble Count $1/K$ [-]", fontsize=14.0)
ax1.set_ylabel(r"Squared Auto-Bicoherence $\hat{b}^2(K)$ [-]", fontsize=13.5)
ax1.set_xlim(0, 0.06)
ax1.set_ylim(0, 1.15)
ax1.tick_params(labelsize=14.0)
ax1.legend(fontsize=10.3, loc="upper left", framealpha=0.94, handlelength=1.1, handletextpad=0.30, borderpad=0.25, borderaxespad=0.30, labelspacing=0.26)
ax1.grid(True, ls=":", alpha=0.4)

# -----------------------------------------------------------------
# PANEL 2: SHOT #88654 (Clean, No Shaded Background Bands)
# -----------------------------------------------------------------
ax2 = axes[1]
Ks_54 = np.array(data["88654"]["Ks"])
x_54 = 1.0 / Ks_54
xx_54 = np.linspace(0, 0.06, 50)

ax2.plot(xx_54, xx_54, "--", color="0.65", lw=0.9, label=r"Theoretical $1/K$ (indep.)")
ax2.plot(xx_54, 2 * xx_54, ":", color="0.65", lw=0.9, label=r"Theoretical $2/K$ ($K_{\mathrm{eff}}=K/2$)")
ax2.axhline(0, color="0.8", lw=0.6)

# 1. Diff Harm (180-90): Rank 4 diff
p1_54 = data["88654"]["diff"][3]

# 2. Sum Harm (88+88) from cross-shot evaluation
cross_pk1 = t41["cross"]["88653"][0]
Ks_c = np.array(cross_pk1["K"])
xc = 1.0 / Ks_c
mask_c = xc <= 0.055
xc_sub = xc[mask_c]
yc_sub = np.array(cross_pk1["y_subset"])[mask_c]
C_c, b0_c = np.polyfit(xc_sub, yc_sub, 1)

# 3. Sub-harmonic triad (39-20): Rank 15 diff
p3_54 = data["88654"]["diff"][14]

# 4. Noise Baseline (72+76): Rank 1 sum
pn_54 = data["88654"]["sum"][0]

# Plot Diff Harm 88654
y1_54 = np.array(p1_54["y"])
ax2.plot(x_54, y1_54, "s", color="C1", ms=5, label=f"Diff Harm (180-90): b0={p1_54['b0']:+.3f}, C={p1_54['C']:.2f}, p={p1_54['p_surr']:.3f}", ls="none")
ax2.plot(xx_54, p1_54["b0"] + p1_54["C"] * xx_54, "-", color="C1", lw=1.3)
ax2.axhline(p1_54["surr_95"], color="C1", ls=":", lw=1.0, alpha=0.7)

# Plot Sum Harm 88654
ax2.plot(xc_sub, yc_sub, "o", color="C0", ms=5, label=f"Sum Harm (88+88, K≤112): b0={b0_c:+.3f}, C={C_c:.2f}, p={cross_pk1['p_surr']:.3f}", ls="none")
ax2.plot(xx_54, b0_c + C_c * xx_54, "-", color="C0", lw=1.3)
ax2.axhline(cross_pk1["surr_95"], color="C0", ls=":", lw=1.0, alpha=0.7)

# Plot Sub-Harmonic Triad 88654
y3_54 = np.array(p3_54["y"])
ax2.plot(x_54, y3_54, "^", color="C2", ms=5, label=f"Sub-Harm (39-20): b0={p3_54['b0']:+.3f}, C={p3_54['C']:.2f}, p={p3_54['p_surr']:.3f}", ls="none")
ax2.plot(xx_54, p3_54["b0"] + p3_54["C"] * xx_54, "-", color="C2", lw=1.3)
ax2.axhline(p3_54["surr_95"], color="C2", ls=":", lw=1.0, alpha=0.7)

# Plot Noise Baseline 88654
yn_54 = np.array(pn_54["y"])
ax2.plot(x_54, yn_54, "d", color="C3", ms=5, label=f"Noise Ctrl (72+76): b0={pn_54['b0']:+.3f}, C={pn_54['C']:.2f}, p={pn_54['p_surr']:.3f}", ls="none")
ax2.plot(xx_54, pn_54["b0"] + pn_54["C"] * xx_54, "-", color="C3", lw=1.3)
ax2.axhline(pn_54["surr_95"], color="C3", ls=":", lw=1.0, alpha=0.7)

ax2.set_title(r"#88654 Replicate ($1/K \leq 0.055$)", fontsize=15.0, fontweight="bold")
ax2.set_xlabel(r"Inverse Sub-Ensemble Count $1/K$ [-]", fontsize=14.0)
ax2.set_xlim(0, 0.06)
ax2.tick_params(labelsize=14.0)
ax2.legend(fontsize=10.0, loc="upper left", framealpha=0.94, handlelength=1.1, handletextpad=0.30, borderpad=0.24, borderaxespad=0.30, labelspacing=0.26)
ax2.grid(True, ls=":", alpha=0.4)

plt.tight_layout()

# Save to both standard names so all documents and presentations pick it up seamlessly
out_main = IMG / "fig_convergence_cumulative_v3_general.png"
out_both = IMG / "fig_convergence_cumulative_v3_general_both_shots.png"
plt.savefig(out_main, dpi=300, bbox_inches="tight")
plt.savefig(out_both, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[OK] Generated clean {out_main} and {out_both}")
