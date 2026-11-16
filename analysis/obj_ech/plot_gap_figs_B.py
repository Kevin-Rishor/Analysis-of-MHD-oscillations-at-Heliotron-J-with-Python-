# plot_gap_figs_B.py - F1 heating timeline, F2 RMS vs time, F3 spectrograms (both shots).
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding="utf-8")
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR / "jpack"))
CACHE = ROOT_DIR / "cache"; R = ROOT_DIR / "analysis" / "obj_ech"
PAN = ROOT_DIR / "Images" / "obj2_dynamics_and_structure" / "panels"
PAN.mkdir(parents=True, exist_ok=True)
SHOTS = ("88653", "88654")
BLK = 1000   # samples per RMS block (1 ms at 1 MHz)
t4g = json.loads((R / "task4g_results.json").read_text(encoding="utf-8"))
C = {s: np.load(CACHE / ("shot_%s_full_record_cache.npz" % s)) for s in SHOTS}
def summ(name, a):
    a = np.asarray(a, float)
    print("  %s n=%d first3=%s last3=%s min=%r max=%r mean=%r" % (name, a.size, a[:3].tolist(), a[-3:].tolist(), float(a.min()), float(a.max()), float(a.mean())))
def cross(t, e, t0, t1, level, rising):
    m = np.where((t >= t0) & (t <= t1))[0]
    for i in range(m[0] + 1, m[-1] + 1):
        a, b = e[i - 1], e[i]
        if (rising and a < level <= b) or ((not rising) and a >= level > b):
            return float(t[i - 1] + (level - a) * (t[i] - t[i - 1]) / (b - a))
    return None
EV = {}
print("== EVENTS (half-plateau crossings of stored traces; plateau = median over the stated window)")
for s in SHOTS:
    c = C[s]; ev = {}
    for key, tk, w_pl, w_on, w_off in (("ech", "t_ech_ms", (260, 280), (150, 250), (285, 330)),
                                       ("nbis9", "t_nbis9_ms", (260, 290), (150, 260), (285, 340)),
                                       ("nbis10", "t_nbis10_ms", (260, 290), (150, 260), (285, 340))):
        t = c[tk].astype(float); e = c[key].astype(float)
        pl = float(np.median(e[(t >= w_pl[0]) & (t <= w_pl[1])])); lv = 0.5 * pl
        ev[key] = (cross(t, e, w_on[0], w_on[1], lv, True), cross(t, e, w_off[0], w_off[1], lv, False))
        print("  shot %s %s plateau(%s)=%r half-level=%r on=%r off=%r" % (s, key, w_pl, pl, lv, ev[key][0], ev[key][1]))
    EV[s] = ev
def load_ip(s):
    import turnelib as TE
    f = ROOT_DIR / "data" / f"hj{s}" / f"Ip15@{s}.edf"
    e = TE.edf(); d = e.load(str(f)); t = np.array(d[:, 0], float)
    if e.DimUnit[0] == "s": t = t * 1000.0
    return t, np.array(d[:, 1], float), getattr(e, "ValUnit", ["?"])[0], e.DimUnit[0]
IP = {}
print("== Ip15 (EDF via turnelib)")
for s in SHOTS:
    try:
        IP[s] = load_ip(s); t, y, u, du = IP[s]
        print("  shot %s ValUnit=%r DimUnit0=%r" % (s, u, du)); summ("Ip t_ms", t); summ("Ip y", y)
    except Exception as ex:
        IP[s] = None; print("  shot %s Ip UNAVAILABLE: %r" % (s, ex))

plt.rcParams.update({
    "font.size": 13.0,
    "axes.labelsize": 14.0,
    "axes.titlesize": 15.0,
    "xtick.labelsize": 12.5,
    "ytick.labelsize": 12.5,
    "legend.fontsize": 12.0,
})

# ---------------- F1
XL = (150.0, 370.0)
fig, axs = plt.subplots(5, 2, figsize=(9.6, 8.8), sharex=True)
for j, s in enumerate(SHOTS):
    c = C[s]; print("== F1 shot", s)
    rows = [(r"ECH Monitor" "\n" r"$V_{\mathrm{ECH}}$ [V]", [("t_ech_ms", "ech", "C3", "ECH")]),
            (r"NBI Monitors" "\n" r"$V_{\mathrm{NBI}}$ [V]", [("t_nbis9_ms", "nbis9", "C0", "NBIS9"), ("t_nbis10_ms", "nbis10", "C1", "NBIS10")]),
            (r"Stored Energy" "\n" r"$W_p$ [kJ]", [("t_wp_ms", "wp", "k", "Wp")]),
            (r"Line Density $\bar{n}_e$" "\n" r"[$10^{19}\ \mathrm{m^{-3}}$]", [("t_nave_ms", "nave", "C2", "nave")])]
    for i, (yl, trs) in enumerate(rows):
        ax = axs[i, j]
        for tk, k, col, lab in trs:
            t = c[tk].astype(float); y = c[k].astype(float); summ("%s %s" % (k, "plotted raw" if t.size < 30000 else "plotted every 20th"), y)
            st = 1 if t.size < 30000 else 20
            ax.plot(t[::st], y[::st], color=col, lw=1.1, label=lab)
        if len(trs) > 1: ax.legend(fontsize=11.5, loc="upper left", framealpha=0.92)
        if j == 0: ax.set_ylabel(yl, fontsize=11.5)
    ax = axs[4, j]
    if IP[s] is not None:
        t, y, u, du = IP[s]; ax.plot(t, y, color="purple", lw=1.1)
        if j == 0: ax.set_ylabel(r"Plasma Current" "\n" r"$I_p$ [%s]" % u, fontsize=11.5)
    else:
        ax.text(0.5, 0.5, "Ip unavailable", transform=ax.transAxes, ha="center")
    for i in range(5):
        for k, col, ls in (("ech", "C3", "--"), ("nbis10", "C1", ":")):
            for v in EV[s][k]:
                if v is not None: axs[i, j].axvline(v, color=col, ls=ls, lw=1.0)
    axs[0, j].set_title("#%s" % s, fontsize=15.0, fontweight="bold"); axs[4, j].set_xlabel(r"Time $t$ [ms]", fontsize=14.0)
    axs[4, j].set_xlim(*XL)
plt.tight_layout(); o1 = PAN / "fig_heating_timeline_both_shots.png"; plt.savefig(o1, dpi=300, bbox_inches="tight"); plt.close(fig)
print("saved", o1, o1.stat().st_size, "bytes")

# ---------------- F2
print("== F2 block RMS (block = %d samples), values rounded to 6 decimals for printing only" % BLK)
fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.1), sharey=True)
for j, s in enumerate(SHOTS):
    c = C[s]; t = c["t_ms"].astype(float); n = (t.size // BLK) * BLK
    tb = t[:n].reshape(-1, BLK).mean(axis=1)
    print("shot", s, "n_blocks =", tb.size, "| t_block =", [round(v, 3) for v in tb.tolist()])
    for k, lab, col in (("mp1", "MP1 broadband", "k"), ("flt_pri", "Primary (75–110 kHz)", "C3"), ("flt_sec", "Secondary (30–65 kHz)", "C0")):
        x = c[k].astype(float)[:n].reshape(-1, BLK); r = np.sqrt((x ** 2).mean(axis=1))
        print("  rms %s =" % k, [round(v, 6) for v in r.tolist()])
        axs[j].semilogy(tb, r, color=col, lw=1.2, label=lab)
    for k, col, ls in (("ech", "C3", "--"), ("nbis10", "C1", ":")):
        for v in EV[s][k]:
            if v is not None: axs[j].axvline(v, color=col, ls=ls, lw=1.0)
    axs[j].set_title("#%s" % s, fontsize=15.0, fontweight="bold"); axs[j].set_xlabel(r"Time $t$ [ms]", fontsize=14.0)
axs[0].set_ylabel(r"1-ms Block RMS $B_{\mathrm{rms}}$ [V]", fontsize=14.0); axs[0].legend(fontsize=10.8, loc="upper right", framealpha=0.94)
plt.tight_layout(); o2 = PAN / "fig_rms_vs_time_both_shots.png"; plt.savefig(o2, dpi=300, bbox_inches="tight"); plt.close(fig)
print("saved", o2, o2.stat().st_size, "bytes")
print("== F2 recomputed-vs-JSON (task4g reconciliation/section_A_iii_rms_audit, shot 88653 only)")
aud = t4g["reconciliation"]["section_A_iii_rms_audit"]; c = C["88653"]; t = c["t_ms"].astype(float)
for name, d in aud.items():
    w0, w1 = d["window_ms"]; m = (t >= w0) & (t <= w1)
    for k, jk in (("mp1", "broadband_mp1_rms_v"), ("flt_pri", "primary_75_110khz_rms_v"), ("flt_sec", "secondary_30_65khz_rms_v")):
        rc = float(np.sqrt(np.mean(c[k].astype(float)[m] ** 2)))
        print("  %s | %s: recomputed=%r json=%r diff=%r" % (name, k, rc, d.get(jk), None if d.get(jk) is None else rc - d[jk]))

# ---------------- F3
print("== F3 spectrograms from psd_mp1_std")
Z = {}
for s in SHOTS:
    c = C[s]; f = c["f_stft_std"].astype(float) / 1e3; t = c["t_stft_std"].astype(float); P = c["psd_mp1_std"].astype(float)
    print("  shot %s df_khz=%r dt_ms=%r P.shape=%s" % (s, float(f[1] - f[0]), float(t[1] - t[0]), P.shape))
    fm = f <= 150.0; Z[s] = (t, f[fm], 10 * np.log10(P[fm] + np.finfo(float).tiny))
allz = np.concatenate([Z[s][2].ravel() for s in SHOTS]); vmin, vmax = np.percentile(allz, 1.0), np.percentile(allz, 99.5)
print("  color limits: 1st and 99.5th percentile of 10log10(PSD) over both shots =", vmin, vmax, "(PSD units as stored, uncalibrated)")
fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for j, s in enumerate(SHOTS):
    t, f, z = Z[s]
    im = axs[j].imshow(z, origin="lower", aspect="auto", extent=[t[0], t[-1], f[0], f[-1]], vmin=vmin, vmax=vmax, cmap="viridis")
    for v in EV[s]["ech"]:
        if v is not None: axs[j].axvline(v, color="w", ls="--", lw=0.7)
    axs[j].set_title("#%s MP1 STFT (1024 samples)" % s); axs[j].set_xlabel(r"Time $t$ [ms]")
axs[0].set_ylabel(r"Frequency $f$ [kHz]")
fig.colorbar(im, ax=axs, label=r"Power Spectral Density $10\log_{10} P_{xx}$ [$\mathrm{V}^2/\mathrm{Hz}$]", shrink=0.9)
o3 = PAN / "fig_spectrogram_both_shots.png"; plt.savefig(o3, dpi=300, bbox_inches="tight", pad_inches=0.08); plt.close(fig)
print("saved", o3, o3.stat().st_size, "bytes")
