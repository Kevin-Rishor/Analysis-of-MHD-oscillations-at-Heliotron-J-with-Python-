import sys, json, textwrap
from pathlib import Path
import numpy as np
import scipy.signal as dsp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
OUTD = ROOT / "analysis" / "obj_ech" / "gap_inventory"
OUTD.mkdir(parents=True, exist_ok=True)
IMG = ROOT / "Images" / "obj3_nonlinear_coupling"
IMG.mkdir(parents=True, exist_ok=True)
TABLE_DIR = ROOT / "analysis" / "obj_ech" / "results_tables_v2"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

FS = 1.0e6
NFFT = 512
NOV = 256
SEED = 20261008
DF = FS / NFFT / 1e3
B = 4
NSUB = 500
NSURR = 500
SHOTS = ("88653", "88654")

f = np.arange(256) * DF
LO = int(np.ceil(10.0 / DF))
HI = int(np.floor(100.0 / DF))
ga, gb = [a.ravel() for a in np.meshgrid(np.arange(LO, HI + 1), np.arange(LO, HI + 1), indexing="ij")]
up = ga <= gb
ga = ga[up]
gb = gb[up]

def spec_of(sig):
    _, _, sp = dsp.spectrogram(sig, fs=FS, window="hann", nperseg=NFFT, noverlap=NOV, nfft=NFFT, detrend=False, return_onesided=False, scaling="density", mode="complex")
    return sp.T

def phase_randomise_ts(x, rng):
    F = np.fft.rfft(x)
    th = rng.uniform(0, 2 * np.pi, size=F.shape)
    th[0] = 0.0
    if x.size % 2 == 0:
        th[-1] = 0.0
    return np.fft.irfft(F * np.exp(1j * th), n=x.size)

def b2cells(sp, ia, ib):
    Xa = sp[:, ia]
    Xb = sp[:, ib]
    Xc = sp[:, (ia + ib) % sp.shape[1]]
    Xab = Xa * Xb
    num = np.abs(np.mean(Xab * np.conj(Xc), axis=0)) ** 2
    den = np.mean(np.abs(Xab) ** 2, axis=0) * np.mean(np.abs(Xc) ** 2, axis=0)
    return np.where(den == 0, 0.0, num / den)

def Kgrid(n, kmin, frac=0.9):
    return np.unique(np.round(np.linspace(kmin, frac * n, 8)).astype(int))

def fit(Ks, y):
    x = 1.0 / np.asarray(Ks, float)
    C, b0 = np.polyfit(x, y, 1)
    res = np.asarray(y) - (b0 + C * x)
    ss = float(np.sum((np.asarray(y) - np.mean(y)) ** 2))
    r2 = (1.0 - float(np.sum(res ** 2)) / ss) if ss > 0 else float("nan")
    return float(b0), float(C), float(r2)

def cum_curve(sp, order, Ks, ia, ib):
    return np.array([b2cells(sp[order[:K]], ia, ib).mean() for K in Ks])

def subset_curve(sp, pool, Ks, ia, ib, nd, rng):
    out = np.empty((len(Ks), nd))
    for i, K in enumerate(Ks):
        for d in range(nd):
            idx = rng.choice(pool, size=K, replace=False)
            out[i, d] = b2cells(sp[idx], ia, ib).mean()
    y = out.mean(axis=1)
    b0, C, r2 = fit(Ks, y)
    b0_draws = np.array([fit(Ks, out[:, d])[0] for d in range(nd)])
    lo = float(np.percentile(b0_draws, 2.5))
    hi = float(np.percentile(b0_draws, 97.5))
    band_lo = np.percentile(out, 2.5, axis=1)
    band_hi = np.percentile(out, 97.5, axis=1)
    return y, b0, C, r2, lo, hi, band_lo, band_hi

def units_of(N):
    return [np.arange(i, i + B) for i in range(0, N - B + 1, B)]

def r5(a):
    return [round(float(v), 5) for v in np.asarray(a).ravel()]

def run(write_table=False):
    print("=" * 80)
    print("RUNNING 500-SUBSET & 500-SURROGATE CONVERGENCE PIPELINE")
    print("=" * 80)

    # 1. Load data and setup split-sample peaks
    records = {}
    specs = {}
    for s in SHOTS:
        c = np.load(ROOT / "cache" / f"shot_{s}_full_record_cache.npz")
        t = np.round(c["t_ms"].astype(float), 4)
        sig = c["mp1"][(t >= 259.1) & (t <= 291.2)]
        records[s] = sig
        specs[s] = spec_of(sig)

    peaks_dict = {}
    own_results = {}
    surr_specs = {}

    # Pre-generate 500 Fourier phase-randomized surrogate spectrograms for each shot
    for s in SHOTS:
        print(f"Generating {NSURR} Fourier phase-randomized surrogates for Shot #{s}...")
        rng_surr = np.random.default_rng(SEED + 5000 + int(s))
        surr_specs[s] = [spec_of(phase_randomise_ts(records[s], rng_surr)) for _ in range(NSURR)]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig_88653, ax_88653 = plt.subplots(figsize=(6.5, 4.0))

    for i, s in enumerate(SHOTS):
        sp = specs[s]
        N = sp.shape[0]
        pos = np.arange(N)
        keep = (pos % 6 >= 1) & (pos % 6 <= 4)
        sel = pos[keep & ((pos // 6) % 2 == 0)]
        EU = [pos[keep & (pos // 6 == q)] for q in range(N // 6 + 1) if q % 2 == 1]
        EU = [u for u in EU if len(u) == 4]
        pool = np.concatenate(EU)
        Ks = Kgrid(len(pool), 4)

        order = np.argsort(-b2cells(sp[sel], ga, gb))
        peaks = []
        for k in order:
            if all(max(abs(ga[k] - ga[p]), abs(gb[k] - gb[p])) >= 3 for p in peaks):
                peaks.append(k)
            if len(peaks) == 3:
                break
        peaks_dict[s] = peaks

        print(f"\n--- Shot #{s} Own Estimation (Pool={len(pool)} segs, K={Ks.tolist()}) ---")
        ax = axs[i]
        xx = np.linspace(0, (1.0 / Ks).max() * 1.05, 30)
        ax.plot(xx, xx, "--", color="0.6", lw=0.8, label="1/K (indep.)")
        ax.plot(xx, 2 * xx, ":", color="0.6", lw=0.8, label="2/K (Keff=K/2)")
        ax.axhline(0, color="0.7", lw=0.6)

        if s == "88653":
            ax_88653.plot(xx, xx, "--", color="0.6", lw=0.9, label="1/K (indep.)")
            ax_88653.plot(xx, 2 * xx, ":", color="0.6", lw=0.9, label="2/K (Keff=K/2)")
            ax_88653.axhline(0, color="0.7", lw=0.6)

        own_results[s] = []
        for n_, k in enumerate(peaks):
            ia = ga[k:k+1]
            ib = gb[k:k+1]
            f1 = float(f[ga[k]])
            f2 = float(f[gb[k]])
            fsum = f1 + f2

            rng = np.random.default_rng(SEED + 77 + 10 * i + n_)
            y_sub, b0_sub, C_sub, r2_sub, lo_sub, hi_sub, band_lo, band_hi = subset_curve(sp, pool, Ks, ia, ib, NSUB, rng)
            yf = cum_curve(sp, pool, Ks, ia, ib)
            yr = cum_curve(sp, pool[::-1], Ks, ia, ib)
            bf, Cf, _ = fit(Ks, yf)
            br, Cr, _ = fit(Ks, yr)

            # Evaluate surrogate null on own estimation pool
            b0_surr = np.array([fit(Ks, [b2cells(sps[pool[:K]], ia, ib).mean() for K in Ks])[0] for sps in surr_specs[s]])
            surr_med = float(np.median(b0_surr))
            surr_95 = float(np.percentile(b0_surr, 95))
            surr_max = float(np.max(b0_surr))
            p_surr = float((1 + np.sum(b0_surr >= b0_sub)) / (NSURR + 1))
            excl_zero = bool(lo_sub > 0)
            sig_validated = bool(excl_zero and (p_surr < 0.05))

            res = {
                "peak": n_ + 1,
                "f1": f1,
                "f2": f2,
                "fsum": fsum,
                "b0_sub": b0_sub,
                "C_sub": C_sub,
                "r2_sub": r2_sub,
                "ci_sub": [lo_sub, hi_sub],
                "excl_zero": excl_zero,
                "b0_forward": bf,
                "C_forward": Cf,
                "b0_reverse": br,
                "C_reverse": Cr,
                "surr_med": surr_med,
                "surr_95": surr_95,
                "surr_max": surr_max,
                "p_surr": p_surr,
                "significant": sig_validated,
                "K": Ks.tolist(),
                "y_subset": r5(y_sub),
                "y_forward": r5(yf),
                "y_reverse": r5(yr)
            }
            own_results[s].append(res)

            print(f"  Peak {n_+1} ({f1:.1f}, {f2:.1f} -> {fsum:.1f} kHz):")
            print(f"    b0_sub = {b0_sub:+.4f} (C={C_sub:+.3f}, R2={r2_sub:.3f}) | 500-draw CI: [{lo_sub:+.4f}, {hi_sub:+.4f}] (Excl. 0: {excl_zero})")
            print(f"    Surrogate null: med={surr_med:+.4f}, 95%={surr_95:+.4f}, p_surr={p_surr:.4f} | Validated: {sig_validated}")
            print(f"    Forward: b0={bf:+.4f} | Reverse: b0={br:+.4f}")

            # Plotting 2-panel
            x = 1.0 / Ks
            ax.fill_between(x, band_lo, band_hi, alpha=0.18, color=f"C{n_}")
            ax.plot(x, y_sub, "o", color=f"C{n_}", label=f"Pk {n_+1} ({f1:.0f}+{f2:.0f}): b0={b0_sub:+.3f} [{lo_sub:+.3f}, {hi_sub:+.3f}], p={p_surr:.3f}")
            ax.plot(xx, b0_sub + C_sub * xx, "-", color=f"C{n_}", lw=1.2)
            ax.axhline(surr_95, color=f"C{n_}", ls=":", alpha=0.7, lw=0.9)

            # Plotting single-panel for Shot 88653
            if s == "88653":
                ax_88653.fill_between(x, band_lo, band_hi, alpha=0.18, color=f"C{n_}")
                ax_88653.plot(x, y_sub, "o", color=f"C{n_}", label=f"Pk {n_+1} ({f1:.0f}+{f2:.0f}): b0={b0_sub:+.3f} [{lo_sub:+.3f}, {hi_sub:+.3f}], p={p_surr:.3f}")
                ax_88653.plot(xx, b0_sub + C_sub * xx, "-", color=f"C{n_}", lw=1.3)
                ax_88653.axhline(surr_95, color=f"C{n_}", ls=":", alpha=0.7, lw=1.0)

        ax.set_title(f"Shot #{s} General Search Convergence (ECH+NBI 32.1 ms)", fontsize=9, fontweight="bold")
        ax.set_xlabel("1/K (Estimation Segments)", fontsize=8)
        ax.set_ylabel(r"Squared Auto-Bicoherence $\hat{b}^2(K)$", fontsize=8)
        ax.set_xlim(0, xx.max())
        ax.set_ylim(-0.05, 0.45)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, ls=":", alpha=0.4)

        if s == "88653":
            ax_88653.set_title(r"Shot #88653 General Search Convergence ($\mathrm{ECH}+\mathrm{NBI}$ 32.1 ms)", fontsize=10, fontweight="bold")
            ax_88653.set_xlabel("1/K (Estimation Segments)", fontsize=9)
            ax_88653.set_ylabel(r"Squared Auto-Bicoherence $\hat{b}^2(K)$", fontsize=9)
            ax_88653.set_xlim(0, xx.max())
            ax_88653.set_ylim(-0.05, 0.45)
            ax_88653.legend(fontsize=8, loc="upper left", framealpha=0.9)
            ax_88653.grid(True, ls=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(IMG / "fig_convergence_cumulative_v3_general_both_shots.png", dpi=300)
    plt.close(fig)

    fig_88653.tight_layout()
    fig_88653.savefig(IMG / "fig_convergence_cumulative_v3_general_88653.png", dpi=300)
    plt.close(fig_88653)

    # Generate approved 2-panel figure with top 3 candidates and noise baseline for both shots
    import subprocess
    cmd = [sys.executable, str(Path(__file__).resolve().parent / "generate_two_panel_convergence_fig.py")]
    subprocess.run(cmd, check=True)
    fig_path = IMG / "fig_convergence_cumulative_v3_general.png"
    print(f"\n[OK] Updated 2-panel figure saved to: {fig_path}")

    # 2. Cross-Shot Evaluation
    cross_results = {}
    for src, dst in (("88653", "88654"), ("88654", "88653")):
        sp = specs[dst]
        N = sp.shape[0]
        pool = np.arange((N // B) * B)
        Ks = Kgrid(len(pool), 8)
        cross_results[src] = []

        print(f"\n--- Cross-Shot #{src} -> #{dst} (Pool={len(pool)} segs, K={Ks.tolist()}) ---")
        for n_, k in enumerate(peaks_dict[src]):
            f1 = float(f[ga[k]])
            f2 = float(f[gb[k]])
            ia = np.array([int(round(f1 / DF))])
            ib = np.array([int(round(f2 / DF))])

            rng = np.random.default_rng(SEED + 900 + 10 * (src == "88654") + n_)
            y_sub, b0_sub, C_sub, r2_sub, lo_sub, hi_sub, _, _ = subset_curve(sp, pool, Ks, ia, ib, NSUB, rng)
            yf = cum_curve(sp, pool, Ks, ia, ib)
            yr = cum_curve(sp, pool[::-1], Ks, ia, ib)
            bf, Cf, _ = fit(Ks, yf)
            br, Cr, _ = fit(Ks, yr)

            b0_surr = np.array([fit(Ks, [b2cells(sps[pool[:K]], ia, ib).mean() for K in Ks])[0] for sps in surr_specs[dst]])
            surr_med = float(np.median(b0_surr))
            surr_95 = float(np.percentile(b0_surr, 95))
            surr_max = float(np.max(b0_surr))
            p_surr = float((1 + np.sum(b0_surr >= b0_sub)) / (NSURR + 1))
            excl_zero = bool(lo_sub > 0)
            sig_validated = bool(excl_zero and (p_surr < 0.05))

            res = {
                "dst": dst,
                "peak": n_ + 1,
                "f1": f1,
                "f2": f2,
                "fsum": f1 + f2,
                "b0_sub": b0_sub,
                "C_sub": C_sub,
                "r2_sub": r2_sub,
                "ci_sub": [lo_sub, hi_sub],
                "excl_zero": excl_zero,
                "b0_forward": bf,
                "C_forward": Cf,
                "b0_reverse": br,
                "C_reverse": Cr,
                "surr_med": surr_med,
                "surr_95": surr_95,
                "surr_max": surr_max,
                "p_surr": p_surr,
                "significant": sig_validated,
                "K": Ks.tolist(),
                "y_subset": r5(y_sub)
            }
            cross_results[src].append(res)

            print(f"  Peak {n_+1} ({f1:.1f}, {f2:.1f} -> {f1+f2:.1f} kHz in #{dst}):")
            print(f"    b0_sub = {b0_sub:+.4f} (C={C_sub:+.3f}, R2={r2_sub:.3f}) | 500-draw CI: [{lo_sub:+.4f}, {hi_sub:+.4f}] (Excl. 0: {excl_zero})")
            print(f"    Surrogate null: med={surr_med:+.4f}, 95%={surr_95:+.4f}, p_surr={p_surr:.4f} | Validated: {sig_validated}")
            print(f"    Forward: b0={bf:+.4f} | Reverse: b0={br:+.4f}")

    # 3. Generate Updated LaTeX Table: table_convergence.tex
    def f3(x): return ("%.3f" % x).replace("-", "$-$")
    def iv(lo, hi): return "[" + f3(lo) + ", " + f3(hi) + "]"
    def yn(b): return "yes" if b else "no"

    L = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{Convergence of the three strongest cells from the general search in the $\mathrm{ECH}+\mathrm{NBI}$ window ($259.1$--$291.2$~ms): asymptotic intercept $b_0$, $500$-draw subsampling confidence interval, and $500$-surrogate phase-randomized null significance for the selecting shot (own) and replicate shot (other).}",
        r"\label{tab:convergence_general}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccccccc}",
        r"\toprule",
        r"Cell (selected in $\to$ tested in) & $f_1$, $f_2$ (kHz) & Own $b_0$ & Own $95~\%$ CI & Own $p_{\rm surr}$ & Other $b_0$ & Other $95~\%$ CI & Other $p_{\rm surr}$ & Other rev. $b_0$ & Other fwd. $b_0$ & Excl. $0$ (own; other) \\",
        r"\midrule"
    ]

    for src, dst in (("88653", "88654"), ("88654", "88653")):
        for n_ in range(3):
            own = own_results[src][n_]
            oth = cross_results[src][n_]
            excl_str = f"{yn(own['excl_zero'])}; {yn(oth['excl_zero'])}"
            row_str = (
                f"\\#{src}$\\to$\\#{dst}, peak {n_ + 1} & "
                f"{own['f1']:.1f}, {own['f2']:.1f} & "
                f"{f3(own['b0_sub'])} & "
                f"{iv(own['ci_sub'][0], own['ci_sub'][1])} & "
                f"{own['p_surr']:.3f} & "
                f"{f3(oth['b0_sub'])} & "
                f"{iv(oth['ci_sub'][0], oth['ci_sub'][1])} & "
                f"{oth['p_surr']:.3f} & "
                f"{f3(oth['b0_reverse'])} & "
                f"{f3(oth['b0_forward'])} & "
                f"{excl_str} \\\\"
            )
            L.append(row_str)

    L += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\par\smallskip{\footnotesize Mask $11.7$--$99.6$~kHz, $f_1\le f_2$, no detrending. Cells were chosen on selection segments ($43$ segments) and fitted on separate estimation segments ($40$ segments) of the selecting shot (blocks of $4$ segments); in the other shot the same cell is evaluated on all $124$ segments of the same window, without pooling. Intercept $b_0$: linear fit $b^2(K)=b_0+C/K$ across $8$ values of $K$ to the mean of $500$ random subsets of $K$ segments chosen without replacement; $95~\%$ CI: $2.5$th--$97.5$th percentile of the $500$ individual trajectory intercepts; $p_{\rm surr}$: empirical $p$-value $(1 + \#\{b_{0,\rm surr} \ge b_0\}) / 501$ evaluated against $500$ time-domain Fourier phase-randomized surrogates sharing the identical window power spectrum and $50~\%$ Hann overlap. Peak~1 of \#88653 ($(87.9, 87.9)\to 175.8$~kHz) represents diagonal second-harmonic generation of the primary mode and strictly excludes zero in both shots ($p_{\rm surr} = 0.002$). All off-diagonal candidate triads extrapolate to zero ($p_{\rm surr} \ge 0.080$). The brief $\mathrm{NBI}$-only window has too few estimation blocks and is untestable.}",
        r"\end{table}"
    ]

    table_tex = "\n".join(L) + "\n"
    table_path = TABLE_DIR / "table_convergence.tex"
    # Thesis.pdf uses the 7-column table_convergence.tex built from asymp_k_search_top15.json;
    # keep write_table=False by default so regenerating JSONs never overwrites it accidentally.
    if write_table:
        table_path.write_text(table_tex, encoding="utf-8")
        print(f"\n[OK] Updated legacy 10-col table saved to: {table_path}")
    else:
        print(f"\n[Skip] Left Thesis 7-column {table_path.name} untouched (pass --write-table to overwrite).")

    # 4. Save structured JSON files
    json_out = {
        "settings": {
            "NFFT": NFFT,
            "NOV": NOV,
            "seed": SEED,
            "n_subsets": NSUB,
            "n_surrogates": NSURR,
            "window": "ECH+NBI 32.1 ms (259.1--291.2 ms)"
        },
        "own": own_results,
        "cross": cross_results
    }
    json_path = OUTD / "t41_convergence_500surr.json"
    json_path.write_text(json.dumps(json_out, indent=2), encoding="utf-8")
    print(f"[OK] Stored JSON results in: {json_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="500-subset and 500-surrogate asymptotic 1/K convergence test")
    parser.add_argument("--write-table", action="store_true",
                        help="Overwrite analysis/obj_ech/results_tables_v2/table_convergence.tex (disabled by default)")
    args = parser.parse_args()
    run(write_table=args.write_table)
