# plot_ech_figures.py -- reads cache npz + JSON to produce Thesis ECH transition panels.
# Fit and window definitions match analyze_amplitude_transients.py
# (t_off=291.2, pre=[t_off-dt,t_off], post=[t_off,t_off+dt], slope in kJ/s)
# and analyze_task4g_reconcile_and_repair.py (calc_transition_stats: pre=[t0-2,t0], post=[t0+1.5,t0+3.5],
# secondary band 30-55 kHz, power-weighted centroid and argmax, t0=291.20, STFT = psd_mp1_std).
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]
R = ROOT_DIR / "analysis" / "obj_ech"
CACHE = ROOT_DIR / "cache"
OUT = ROOT_DIR / "Images" / "obj2_dynamics_and_structure" / "panels"
OUT.mkdir(parents=True, exist_ok=True)

amp = json.loads((R / "amplitude_transients_results.json").read_text(encoding="utf-8"))
t4g = json.loads((R / "task4g_results.json").read_text(encoding="utf-8"))
dfr = json.loads((R / "density_fit_ranges.json").read_text(encoding="utf-8"))

T_OFF = 291.2      # t_off in amplitude script; t0 in task4g
T_ECHRG = 291.322  # typed in analyze_task4g_reconcile_and_repair.py line 564, not read from data
T_NBI_OFF = 300.4  # stated in the setup, typed in other scripts
written = []

plt.rcParams.update({
    "mathtext.fontset": "dejavusans",
    "font.size": 13.0,
    "axes.labelsize": 14.5,
    "axes.titlesize": 15.0,
    "xtick.labelsize": 13.0,
    "ytick.labelsize": 13.0,
    "legend.fontsize": 10.5,
})


def wp_fit(t, w, dt):
    m_pre = (t >= T_OFF - dt) & (t <= T_OFF)
    m_post = (t >= T_OFF) & (t <= T_OFF + dt)
    sp = np.polyfit((t[m_pre] - T_OFF) * 1e-3, w[m_pre], 1)
    sq = np.polyfit((t[m_post] - T_OFF) * 1e-3, w[m_post], 1)
    return sp, sq


for sh in ("88653", "88654"):
    c = np.load(CACHE / f"shot_{sh}_full_record_cache.npz")
    t = c["t_wp_ms"].astype(float)
    w = c["wp"].astype(float)
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    m = (t >= 284.0) & (t <= T_NBI_OFF)
    ax.plot(t[m], w[m], color="k", lw=1.2, label="Wp")
    for key, dt, col in (("window_5ms", 5.0, "tab:red"), ("window_2ms", 2.0, "tab:blue")):
        sp, sq = wp_fit(t, w, dt)
        j = amp[sh]["wp_discontinuity"][key]
        print(f"[{sh}] {key}: recomputed slope_pre={sp[0]:.6f} slope_post={sq[0]:.6f} delta={sq[0]-sp[0]:.6f} kJ/s")
        print(f"[{sh}] {key}: json       slope_pre={j['slope_pre_kj_s']:.6f} slope_post={j['slope_post_kj_s']:.6f} delta={j['delta_slope_kj_s']:.6f} ci={j['delta_slope_ci']}")
        tp = np.linspace(T_OFF - dt, T_OFF, 20)
        tq = np.linspace(T_OFF, T_OFF + dt, 20)
        ax.plot(tp, np.polyval(sp, (tp - T_OFF) * 1e-3), color=col, lw=2.2,
                label=f"{dt:.0f} ms: {j['delta_slope_kj_s']:.1f} [{j['delta_slope_ci'][0]:.1f}, {j['delta_slope_ci'][1]:.1f}] kW")
        ax.plot(tq, np.polyval(sq, (tq - T_OFF) * 1e-3), color=col, lw=2.2)
    ax.axvline(T_OFF, color="gray", ls="--", lw=1.2, label="Fit boundary 291.2 ms")
    ax.axvline(T_ECHRG, color="green", ls=":", lw=1.4, label="ECHRG500 drop 291.322 ms")
    ax.axvline(T_NBI_OFF, color="magenta", ls="-.", lw=1.4, label="NBI off 300.4 ms")
    ax.set_xlim(284.0, 300.6)
    ax.set_xlabel(r"Time $t$ [ms]")
    ax.set_ylabel(r"Stored Energy $W_p$ [kJ]")
    ax.set_title(f"#{sh}", fontweight="bold")
    ax.legend(fontsize=9.2, loc="lower left", framealpha=0.94)
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    p = OUT / f"fig_ech_wp_step_{sh}.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    written.append(p)


def est_win(f, psd, tt, a, b):
    mt = (tt >= a) & (tt <= b)
    mf = (f >= 30.0) & (f <= 55.0)
    fs_ = f[mf]
    ps = np.mean(psd[mf, :][:, mt], axis=1)
    return float(np.sum(fs_ * ps) / np.sum(ps)), float(fs_[np.argmax(ps)])


def json_delta(sh, k):
    if sh == "88653":
        return t4g["88653"]["section_B_transition_matched_null"][k]["observed_at_t0"]
    return t4g["88654"]["section_B_direction_check"][k]["val_88654"]


for sh in ("88653", "88654"):
    c = np.load(CACHE / f"shot_{sh}_full_record_cache.npz")
    tt = c["t_stft_std"].astype(float)
    f = c["f_stft_std"].astype(float) / 1e3
    psd = c["psd_mp1_std"].astype(float)
    te = np.arange(270.0, 300.01, 0.5)
    fc = np.zeros(len(te))
    fd = np.zeros(len(te))
    for i, x in enumerate(te):
        fc[i], fd[i] = est_win(f, psd, tt, x - 2.0, x)
    print(f"[{sh}] track window end (ms):", [float(x) for x in te])
    print(f"[{sh}] secondary centroid (kHz):", [float(x) for x in fc])
    print(f"[{sh}] secondary dominant (kHz):", [float(x) for x in fd])
    c_pre, d_pre = est_win(f, psd, tt, T_OFF - 2.0, T_OFF)
    c_post, d_post = est_win(f, psd, tt, T_OFF + 1.5, T_OFF + 3.5)
    print(f"[{sh}] recomputed delta centroid={c_post-c_pre:.6f} dominant={d_post-d_pre:.6f}")
    print(f"[{sh}] json       delta centroid={json_delta(sh,'delta_fc_sec'):.6f} dominant={json_delta(sh,'delta_fdom_sec'):.6f}")
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    ax.plot(te, fc, color="tab:blue", lw=1.8, label="Centroid (30–55 kHz)")
    ax.plot(te, fd, color="tab:orange", lw=1.2, marker="o", ms=3.0, label="Dominant (30–55 kHz)")
    ax.axvspan(T_OFF - 2.0, T_OFF, color="tab:blue", alpha=0.12, label="Pre 2.0 ms")
    ax.axvspan(T_OFF, T_OFF + 1.5, color="gray", alpha=0.12, label="Gap 1.5 ms")
    ax.axvspan(T_OFF + 1.5, T_OFF + 3.5, color="tab:red", alpha=0.12, label="Post 2.0 ms")
    ax.axvline(T_OFF, color="gray", ls="--", lw=1.2)
    ax.axvline(T_ECHRG, color="green", ls=":", lw=1.4, label="ECHRG500 291.322 ms")
    ax.axvline(T_NBI_OFF, color="magenta", ls="-.", lw=1.4, label="NBI off 300.4 ms")
    ax.set_xlim(270.0, 300.6)
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + 0.36 * (y1 - y0))
    ax.vlines(271.0, y0, y0 + 0.977, color="k", lw=3)
    ax.text(271.5, y0 + 0.15, "0.977 kHz bin", fontsize=11.0, fontweight="bold")
    ax.set_xlabel(r"Window End Time $t_{\mathrm{end}}$ [ms]")
    ax.set_ylabel(r"Frequency $f$ [kHz]")
    ax.set_title(f"#{sh}", fontweight="bold")
    ax.legend(fontsize=9.2, ncol=2, loc="upper center", framealpha=0.94)
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    p = OUT / f"fig_ech_secondary_freq_{sh}.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

keys = list(dfr["fit_intervals"].keys())
vals = [dfr["fit_intervals"][k]["delta_f_pred_khz"] for k in keys]
labs = [dfr["fit_intervals"][k]["fraction_pct_str"] for k in keys]
obs = dfr["observed_delta_fc_sec_khz"]
print("density fit intervals:", keys)
print("delta_f_pred_khz:", vals)
print("fraction_pct_str:", labs)
print("observed_delta_fc_sec_khz:", obs)
fig, ax = plt.subplots(figsize=(5.2, 4.0))
x = np.arange(len(keys))
ax.bar(x, vals, color="tab:blue", width=0.55)
for xi, v, lb in zip(x, vals, labs):
    ax.text(xi, v - 0.35, lb, ha="center", va="top", fontsize=8)
ax.axhline(obs, color="tab:red", ls="--", lw=1.4, label=f"observed centroid shift {obs:.2f} kHz")
ax.set_xticks(x)
ax.set_xticklabels([f"{k} ms" for k in keys])
ax.set_xlabel(r"Linear Density Fit Window $\Delta t_{\mathrm{fit}}$ [ms]")
ax.set_ylabel(r"Predicted Frequency Shift $\Delta f$ [kHz]")
ax.set_title("#88653: Density-Following Check", fontweight="bold")
ax.legend(fontsize=9.2, loc="lower right")
ax.set_ylim(min(obs, min(vals)) - 1.6, 0.5)
ax.grid(True, ls=":", alpha=0.4, axis="y")
fig.tight_layout()
p = OUT / "fig_ech_density_following_88653.png"
fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.08)
plt.close(fig)
written.append(p)
for p in written:
    print("written:", p)
