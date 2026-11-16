# plot_gap_figs_C.py - F4 coherence spectra by phase, F5 Morlet-type scalogram (both shots). Cache + JSON only.
import json, sys
from pathlib import Path
import numpy as np
import scipy.signal as dsp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding="utf-8")
ROOT_DIR = Path(__file__).resolve().parents[2]
CACHE = ROOT_DIR / "cache"; R = ROOT_DIR / "analysis" / "obj_ech"
OUT_IMG = ROOT_DIR / "Images" / "obj1_mode_identification"
OUT_IMG.mkdir(parents=True, exist_ok=True)
SHOTS = ("88653", "88654"); FS = 1.0e6; NPS = 1024; NOV = 512; W0 = 6.0
C = {s: np.load(CACHE / ("shot_%s_full_record_cache.npz" % s)) for s in SHOTS}
t1 = json.loads((R / "task1_ech_results.json").read_text(encoding="utf-8"))
ph = t1["88653"]["whole_discharge_phase_modes"]
NAMES = ["Pre-plasma", "ECH only", "Early ECH+NBI", "Late ECH+NBI", "NBI only", "Post-heating decay"]
IDX = [1, 2, 3, 4]
print("== phase windows from task1_ech_results.json 88653/whole_discharge_phase_modes/<i>/t_range_ms (applied to both shots)")
for i in IDX: print("  ", i, NAMES[i], ph[i]["t_range_ms"])
def cross(t, e, t0, t1_, level, rising):
    m = np.where((t >= t0) & (t <= t1_))[0]
    for i in range(m[0] + 1, m[-1] + 1):
        a, b = e[i - 1], e[i]
        if (rising and a < level <= b) or ((not rising) and a >= level > b):
            return float(t[i - 1] + (level - a) * (t[i] - t[i - 1]) / (b - a))
    return None
ECHOFF = {}
for s in SHOTS:
    t = C[s]["t_ech_ms"].astype(float); e = C[s]["ech"].astype(float)
    pl = float(np.median(e[(t >= 260) & (t <= 280)]))
    ECHOFF[s] = cross(t, e, 285, 330, 0.5 * pl, False)
    print("  shot %s ECH off (half-level crossing, plateau median 260-280 ms = %r) = %r ms" % (s, pl, ECHOFF[s]))

# ---------------- F4
print("== F4 coherence: scipy.signal.coherence, hann, nperseg=%d, noverlap=%d (%.4f kHz bins); curves rounded to 3 decimals for printing only" % (NPS, NOV, FS / NPS / 1e3))
plt.rcParams.update({
    "font.size": 12.5,
    "axes.labelsize": 14.0,
    "axes.titlesize": 13.5,
    "xtick.labelsize": 12.5,
    "ytick.labelsize": 12.5,
    "legend.fontsize": 12.0,
})
fig, axs = plt.subplots(len(IDX), 2, figsize=(9.8, 8.2), sharex=True, sharey=False)
for j, s in enumerate(SHOTS):
    c = C[s]; t = c["t_ms"].astype(float)
    for i, idx in enumerate(IDX):
        a, b = ph[idx]["t_range_ms"]; m = (t >= a) & (t <= b)
        x1 = c["mp1"].astype(float)[m]; n = int(m.sum()); K = n // NPS; thr = 1.0 - 0.05 ** (1.0 / (K - 1))
        print("-- shot %s | %s | window %s | n=%d samples | K=n//nperseg=%d | independent-segment 95%% level 1-0.05^(1/(K-1)) = %r" % (s, NAMES[idx], [a, b], n, K, thr))
        ax = axs[i, j]
        for k, lab, col in (("mp3", "MP1-MP3", "C0"), ("mp4", "MP1-MP4", "C3")):
            f, g = dsp.coherence(x1, c[k].astype(float)[m], fs=FS, window="hann", nperseg=NPS, noverlap=NOV)
            sel = f / 1e3 <= 150.0
            if k == "mp3": print("   f_khz n=%d first=%r last=%r" % (int(sel.sum()), float(f[sel][0] / 1e3), float(f[sel][-1] / 1e3)))
            print("   gamma2 %s =" % lab, [round(v, 3) for v in g[sel].tolist()])
            ax.plot(f[sel] / 1e3, g[sel], color=col, lw=1.1, label=lab)
        ax.axhline(thr, color="k", ls=":", lw=1.0)
        ax.set_title("#%s  %s  (%.1f–%.1f ms)" % (s, NAMES[idx], a, b), fontsize=13.0, fontweight="bold"); ax.set_ylim(0, 1)
        if i == 0 and j == 0: ax.legend(fontsize=11.5, loc="upper right", framealpha=0.92)
        if j == 0: ax.set_ylabel(r"Coherence $\gamma^2$ [-]", fontsize=14.0)
for j in range(2): axs[-1, j].set_xlabel(r"Frequency $f$ [kHz]", fontsize=14.0)
plt.tight_layout(); o4 = OUT_IMG / "fig_coherence_spectra_both_shots.png"; plt.savefig(o4, dpi=300, bbox_inches="tight"); plt.close(fig)
print("saved", o4, o4.stat().st_size, "bytes")

# ---------------- F5
T0, T1 = 250.0, 305.0; TP0, TP1 = 256.0, 300.0; SUB = 10
FQ = np.linspace(20e3, 130e3, 111)
def amp_bank(x):
    X = np.fft.fft(x - x.mean()); nu = np.fft.fftfreq(x.size, 1.0 / FS); pos = nu > 0
    out = np.empty((FQ.size, x.size))
    for k, f in enumerate(FQ):
        H = np.zeros(x.size); H[pos] = 2.0 * np.exp(-0.5 * ((nu[pos] - f) / (f / W0)) ** 2)
        out[k] = np.abs(np.fft.ifft(X * H))
    return out
print("== F5 analytic Gaussian filter bank, relative bandwidth sigma_f = f/%.1f, amplitude units V (filter peak gain 1 on a real sinusoid)" % W0)
print("  freq axis kHz: n=%d first=%r last=%r step=%r" % (FQ.size, FQ[0] / 1e3, FQ[-1] / 1e3, (FQ[1] - FQ[0]) / 1e3))
tt = np.arange(int(round((T1 - T0) * 1e3))) / FS * 1e3 + T0
syn = np.cos(2 * np.pi * 90e3 * tt * 1e-3)
sa = amp_bank(syn); mid = (tt >= 270) & (tt <= 285); k90 = int(np.argmin(np.abs(FQ - 90e3))); k60 = int(np.argmin(np.abs(FQ - 60e3)))
print("  synthetic unit cosine at 90 kHz: recovered amplitude at %r kHz = %r (mean over 270-285 ms), at %r kHz = %r" % (FQ[k90] / 1e3, float(sa[k90][mid].mean()), FQ[k60] / 1e3, float(sa[k60][mid].mean())))
Z = {}
for s in SHOTS:
    c = C[s]; t = c["t_ms"].astype(float); m = (t >= T0) & (t < T0 + len(tt) / 1e3 - 1e-9)
    x = c["mp1"].astype(float)[m]; print("  shot %s samples used=%d (expected %d)" % (s, x.size, tt.size))
    A = amp_bank(x); tp = t[m]; mp = (tp >= TP0) & (tp <= TP1)
    Z[s] = (tp[mp][::SUB], 20 * np.log10(A[:, mp][:, ::SUB] + 1e-12))
    print("  shot %s plotted: n_t=%d t first=%r last=%r | amplitude(V) min=%r max=%r mean=%r" % (s, Z[s][0].size, float(Z[s][0][0]), float(Z[s][0][-1]), float(A[:, mp].min()), float(A[:, mp].max()), float(A[:, mp].mean())))
allz = np.concatenate([Z[s][1].ravel() for s in SHOTS]); vmin, vmax = np.percentile(allz, 1.0), np.percentile(allz, 99.5)
print("  colour limits (1st/99.5th percentile of 20log10(amplitude/1 V) over both shots):", vmin, vmax)
fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for j, s in enumerate(SHOTS):
    tp, z = Z[s]
    im = axs[j].imshow(z, origin="lower", aspect="auto", extent=[tp[0], tp[-1], FQ[0] / 1e3, FQ[-1] / 1e3], vmin=vmin, vmax=vmax, cmap="viridis")
    if ECHOFF[s] is not None: axs[j].axvline(ECHOFF[s], color="w", ls="--", lw=0.8)
    axs[j].set_title("#%s MP1 Morlet-type amplitude" % s); axs[j].set_xlabel(r"Time $t$ [ms]")
axs[0].set_ylabel(r"Frequency $f$ [kHz]")
fig.colorbar(im, ax=axs, label=r"Wavelet Amplitude $20\log_{10}(|A|/1\ \mathrm{V})$ [dBV]", shrink=0.9)
o5 = OUT_IMG / "fig_morlet_both_shots.png"; plt.savefig(o5, dpi=300, bbox_inches="tight", pad_inches=0.08); plt.close(fig)
print("saved", o5, o5.stat().st_size, "bytes")
