# plot_control_windows.py -- reads analysis/obj_ech/bicoherence_maps_and_controls_results.json
# Window length 7.936 ms (7936 samples, compute script).
# Event times typed (setup): ECH on 171.2, NBI on 222.4, ECH off 291.32, NBI off 300.4 ms.
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]
d = json.loads((ROOT_DIR / "analysis" / "obj_ech" / "bicoherence_maps_and_controls_results.json").read_text(encoding="utf-8"))
WL = 7.936
ECH_ON, NBI_ON, ECH_OFF, NBI_OFF = 171.2, 222.4, 291.32, 300.4
SEG = [(-1e9, ECH_ON, "before ECH on"), (ECH_ON, NBI_ON, "ECH only"), (NBI_ON, ECH_OFF, "ECH+NBI"),
       (ECH_OFF, NBI_OFF, "NBI only"), (NBI_OFF, 1e9, "after NBI off")]
STYLE = {"before ECH on": ("o", "#1f77b4"), "ECH only": ("v", "#2ca02c"), "ECH+NBI": ("s", "#d62728"),
         "NBI only": ("^", "#ff7f0e"), "after NBI off": ("D", "#9467bd"), "straddles boundary": ("x", "#555555")}


def state(s):
    for a, b, name in SEG:
        if s >= a and s + WL <= b:
            return name
    return "straddles boundary"


plt.rcParams.update({
    "font.size": 12.0,
    "axes.labelsize": 13.0,
    "axes.titlesize": 13.5,
    "xtick.labelsize": 12.0,
    "ytick.labelsize": 12.0,
    "legend.fontsize": 10.2,
})

fig, axes = plt.subplots(2, 1, figsize=(7.6, 6.4), sharex=True)
for ax, sh in zip(axes, ("88653", "88654")):
    cw = d[sh]["control_windows"]
    ws = np.array(cw["window_starts_ms"], dtype=float)
    mx = np.array(cw["max_b2_values"], dtype=float)
    sts = [state(s) for s in ws]
    print(f"[{sh}] n windows:", len(ws))
    print(f"[{sh}] window starts (ms):", [float(x) for x in ws])
    print(f"[{sh}] max b2:", [float(x) for x in mx])
    print(f"[{sh}] states:", sts)
    e_max = d[sh]["ech_nbi"]["max_b2"]; n_max = d[sh]["nbi_only"]["max_b2"]
    print(f"[{sh}] heated map maxima: ech_nbi={e_max} nbi_only={n_max}")
    for name in STYLE:
        idx = [i for i, s in enumerate(sts) if s == name]
        print(f"[{sh}] {name}: n={len(idx)}, above ech_nbi max={int(np.sum(mx[idx] > e_max))}, above nbi_only max={int(np.sum(mx[idx] > n_max))}")
    ax.axvspan(ECH_ON, ECH_OFF, color="#fee0b6", alpha=0.6, label="ECH on 171.2 to off 291.32 ms")
    ax.axvspan(NBI_ON, NBI_OFF, color="#d9f0d3", alpha=0.5, label="NBI on 222.4 to off 300.4 ms")
    ax.axhline(e_max, color="#d62728", ls="--", lw=1.4, label=f"ECH+NBI map max (259.1–267.0 ms) {e_max:.3f}")
    ax.axhline(n_max, color="#1f77b4", ls="--", lw=1.4, label=f"NBI-only map max (291.3–299.2 ms) {n_max:.3f}")
    ax.plot(ws + WL / 2, mx, color="#7f7f7f", ls=":", lw=1.0)
    done = set()
    for x, y, st in zip(ws + WL / 2, mx, sts):
        m, c = STYLE[st]
        ax.plot(x, y, marker=m, color=c, ls="None", ms=6.5, mec="black", mew=0.6, label=None if st in done else f"Reference: {st}")
        done.add(st)
    ax.set_ylabel("Max Bicoherence\n" r"$\max\,b^2$ [-]", fontsize=12.5)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"#{sh} (33 contiguous 7.936 ms windows)", fontsize=13.5, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.4)
axes[1].set_xlabel(r"Window Center Time $t_c$ [ms]", fontsize=13.0)
h, l = axes[0].get_legend_handles_labels()
h2, l2 = axes[1].get_legend_handles_labels()
seen = dict(zip(l, h)); seen.update(dict(zip(l2, h2)))
fig.legend(list(seen.values()), list(seen.keys()), loc="lower center", ncol=2, fontsize=10.0, framealpha=0.95)
fig.tight_layout(rect=(0, 0.21, 1, 1))
out = ROOT_DIR / "Images" / "obj3_nonlinear_coupling" / "fig_control_windows_v2.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches="tight")
print("written:", out)
