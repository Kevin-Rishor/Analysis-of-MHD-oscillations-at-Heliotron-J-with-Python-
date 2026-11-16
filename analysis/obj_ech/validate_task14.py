"""
Task 14 validation on synthetic data (run BEFORE any real-data direction is reported).

Cases
  case1   spec case: three-wave energy-conserving coupling, n=(-1,-1,-1) at probes
          phases 0, pi, 3pi/2; sub-cases 1a (A3 sink, true T>0) / 1b (A3 pump, true T<0),
          each with Q>0 and Q<0.
  case1S  same with selection-consistent n=(-1,-2,-3).
  null_a  independent modes, f3=f1+f2 exactly, no coupling
  null_b  rigid phase-locked components + common amplitude modulation (no exchange)
  null_c  real #88653 MP1/MP3/MP4 window, same random phase applied to all probes
Truth of direction = latent energy accounting: sign of mean 2Re[A3* Q3 A1 A2].
Stop rule: lower Wilson CI of case-1 sign accuracy < 0.8 (worst probe) -> UNRESOLVED.
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import scipy.signal as dsp
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_task14_multiprobe_energy_transfer as L

HERE = L.HERE
FS = L.FS
PHI = {"mp1": 0.0, "mp3": np.pi, "mp4": 1.5 * np.pi}
GAMMA = 2.0e-3          # 1/us amplitude damping (2 /ms); line FWHM ~ |gamma|/pi
DT_ENV = 8.0            # us
WINDOWS = {"w8p7": (282.5, 291.2), "w32p1": (259.1, 291.2)}
CFG = {}


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


# ------------------------------------------------- parameters from real data
def real_params(trip):
    rec, _ = L.load_window("88653", *WINDOWS["w8p7"])
    x = rec["mp1"]
    f, pxx = dsp.welch(x, fs=FS, nperseg=1024)
    m = (f >= 115e3) & (f <= 145e3)
    noise_var_per_hz = float(np.median(pxx[m]))   # one-sided PSD, V^2/Hz
    sigma = float(np.sqrt(noise_var_per_hz * FS / 2.0))
    X = np.fft.rfft(x * np.hanning(len(x)))
    amps = []
    for b in trip:
        fb = b * L.DF_KHZ * 1e3
        mask = np.abs(np.fft.rfftfreq(len(x), 1 / FS) - fb) <= 2.5 * L.DF_KHZ * 1e3
        y = np.fft.irfft(np.where(mask, np.fft.rfft(x), 0), n=len(x))
        v = float(np.var(y)) - sigma ** 2 * (2 * 2.5 * L.DF_KHZ * 1e3) / (FS / 2)
        amps.append(float(np.sqrt(max(2 * v, 1e-30))))   # rms of |A|
    return sigma, amps


# --------------------------------------------------------------- generators
def env_sde(N, a_rms, r, qs, sub, sign, rng):
    """Return complex envelopes (3,N) at 1 us, and latent truth."""
    ns = int(np.ceil(N / DT_ENV)) + 2
    burn = 600
    g = GAMMA
    a1, a2, a3 = a_rms
    qa = qs * sign
    Q1 = Q2 = qa
    Q3 = -(Q1 + Q2)            # real q: Q3 = -conj(Q1+Q2) = -2 qa
    if sub == "1a":
        s = [np.sqrt(2 * g) * a1, np.sqrt(2 * g) * a2, np.sqrt(2 * g * (1 - r)) * a3]
    else:  # 1b: A3 forced, A1/A2 parametrically driven
        s = [np.sqrt(2 * g * (1 - r)) * a1, np.sqrt(2 * g * (1 - r)) * a2, np.sqrt(2 * g) * a3]
    A = np.zeros((3, ns + burn), complex)
    A[:, 0] = [a1, a2, a3] * (rng.standard_normal(3) + 1j * rng.standard_normal(3)) / np.sqrt(2)
    sq = np.sqrt(DT_ENV / 2)
    W = (rng.standard_normal((3, ns + burn)) + 1j * rng.standard_normal((3, ns + burn))) * sq
    for k in range(ns + burn - 1):
        A1, A2, A3 = A[:, k]
        d1 = -g * A1 + Q1 * A3 * np.conj(A2)
        d2 = -g * A2 + Q2 * A3 * np.conj(A1)
        d3 = -g * A3 + Q3 * A1 * A2
        A[0, k + 1] = A1 + d1 * DT_ENV + s[0] * W[0, k]
        A[1, k + 1] = A2 + d2 * DT_ENV + s[1] * W[1, k]
        A[2, k + 1] = A3 + d3 * DT_ENV + s[2] * W[2, k]
    A = A[:, burn:]
    tt = np.arange(ns) * DT_ENV
    t1 = np.arange(N) + 0.0
    Af = np.stack([np.interp(t1, tt, A[m].real) + 1j * np.interp(t1, tt, A[m].imag) for m in range(3)])
    # truth from the coarse latent amplitudes
    flow = np.mean(2 * np.real(np.conj(A[2]) * Q3 * A[0] * A[1]))
    return Af, float(flow)


def make_probes(Af, bins, nvec, offs, sigma, N, rng, noise=True):
    t = np.arange(N) / FS
    f1 = (bins[0] + offs[0]) * L.DF_KHZ * 1e3
    f2 = (bins[1] + offs[1]) * L.DF_KHZ * 1e3
    fr = [f1, f2, f1 + f2]
    rec = {}
    for pr in L.PROBES:
        x = np.zeros(N)
        for m in range(3):
            x += np.real(Af[m] * np.exp(1j * (2 * np.pi * fr[m] * t + nvec[m] * PHI[pr])))
        if noise:
            x = x + sigma * rng.standard_normal(N)
        rec[pr] = x
    return rec


def latent_b2(Af, bins, nvec, offs, sigma, N, rng):
    rec = make_probes(Af, bins, nvec, offs, sigma, N, rng, noise=False)
    X = L.seg_spectra(rec["mp1"])
    return float(L.b2_pairs(X, np.array([bins[0]]), np.array([bins[1]]))[0])


def gen_case(case, params, N, rng):
    """case: dict(kind, sub, sign, r, qs, nvec)."""
    sigma, amps, bins = params["sigma"], params["amps"], params["bins"]
    offs = rng.uniform(-0.4, 0.4, size=2)
    kind = case["kind"]
    if kind in ("case1", "case1S", "null_a"):
        qs = 0.0 if kind == "null_a" else case["qs"]
        sub = case["sub"] if kind != "null_a" else "1a"
        sign = 1 if kind == "null_a" else case["sign"]
        Af, flow = env_sde(N, amps, case["r"], qs, sub, sign, rng)
        nvec = case["nvec"]
        rec = make_probes(Af, bins, nvec, offs, sigma, N, rng)
        truth = 0 if kind == "null_a" else int(np.sign(flow))
        return rec, truth
    if kind == "null_b":
        t = np.arange(N) / FS
        ph1, ph2 = rng.uniform(0, 2 * np.pi, 2)
        mod = 1 + 0.3 * np.sin(2 * np.pi * rng.uniform(0.5, 3) * 1e3 * t + rng.uniform(0, 6.28))
        fr = [bins[0] * L.DF_KHZ * 1e3, bins[1] * L.DF_KHZ * 1e3, (bins[0] + bins[1]) * L.DF_KHZ * 1e3]
        phs = [ph1, ph2, ph1 + ph2]
        rec = {}
        for pr in L.PROBES:
            x = np.zeros(N)
            for m in range(3):
                x += mod * amps[m] / np.sqrt(2) * np.cos(2 * np.pi * fr[m] * t + phs[m] + case["nvec"][m] * PHI[pr])
            rec[pr] = x + sigma * rng.standard_normal(N)
        return rec, 0
    raise ValueError(kind)


# ------------------------------------------------------------------ trials
def init_worker(cfg):
    global CFG
    CFG = cfg


def run_trial(args):
    seed, case, wname = args
    rng = np.random.default_rng(seed)
    t0, t1 = WINDOWS[wname]
    N = int(round((t1 - t0) * 1e3)) + 1
    params = CFG["params"]
    trip = tuple(params["bins"]) + (params["bins"][0] + params["bins"][1],)
    if case["kind"] == "null_c":
        rec = CFG["real"][wname]
        nb = N // 2 + 1
        theta = rng.uniform(0, 2 * np.pi, size=nb)
        rec = {pr: L.phase_randomise(rec[pr], rng, theta=theta) for pr in L.PROBES}
        truth = 0
    else:
        rec, truth = gen_case(case, params, N, rng)
    r = L.analyze_triad(rec, trip, CFG["S"], CFG["B"], rng)
    P = r["probes"]
    return dict(
        case=case["kind"], sub=case.get("sub"), sign=case.get("sign"), r=case.get("r"),
        wname=wname, truth=truth,
        gate=[P[p]["gate"] for p in L.PROBES], p_fw=[P[p]["p_fw"] for p in L.PROBES],
        b2=[P[p]["b2"] for p in L.PROBES],
        Tsign=[P[p]["T_sign"] for p in L.PROBES], Tci0=[P[p]["T_ci_excludes0"] for p in L.PROBES],
        Tci=[list(P[p]["kp"]["T_ci"]) for p in L.PROBES], T=[P[p]["kp"]["T"] for p in L.PROBES],
        rule=r["toroidal"]["rule"], rule_n=r["toroidal"]["rule_n"],
        nmod=r["toroidal"]["n_mod4"], crit=r["criteria"], joint=r["joint"], jsign=r["joint_sign"])


def calibrate(kind, sub, sign, r_target, params, N, rng, nvec, iters=7, pil=8):
    """Find coupling scale qs so that latent b2 ~ r_target (log-bisection)."""
    sigma, amps, bins = params["sigma"], params["amps"], params["bins"]
    lo, hi = 1e-7, 3e-3 if sub == "1a" else 0.9 * GAMMA / max(amps[2], 1e-30)
    hi = min(hi, 3e-3 / max(1e-30, 1.0)) if sub == "1a" else hi
    best = None
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        v = []
        for _ in range(pil):
            Af, _f = env_sde(N, amps, 0.0 if False else r_target, mid, sub, sign, rng)
            v.append(latent_b2(Af, bins, nvec, rng.uniform(-0.4, 0.4, 2), sigma, N, rng))
        m = float(np.mean(v))
        best = (mid, m)
        if m < r_target:
            lo = mid
        else:
            hi = mid
    return best


def summarize(rows, label):
    n = len(rows)
    out = dict(n=n)
    dirs = [r for r in rows if r["truth"] != 0]
    if dirs:
        out["sign_accuracy"] = {}
        for k, pr in enumerate(L.PROBES):
            c = sum(1 for r in dirs if r["Tsign"][k] == r["truth"])
            out["sign_accuracy"][pr] = dict(k=c, n=len(dirs), acc=c / len(dirs), wilson=wilson(c, len(dirs)))
        out["sign_accuracy_by_subcase"] = {}
        for sub in sorted({(r["sub"], r["sign"]) for r in dirs}, key=str):
            rr = [r for r in dirs if (r["sub"], r["sign"]) == sub]
            out["sign_accuracy_by_subcase"][f"{sub[0]}_Q{'+' if sub[1] > 0 else '-'}"] = {
                pr: dict(k=sum(1 for r in rr if r["Tsign"][k] == r["truth"]), n=len(rr))
                for k, pr in enumerate(L.PROBES)}
        out["sign_accuracy_by_truth"] = {}
        for tv in (1, -1):
            rr = [r for r in dirs if r["truth"] == tv]
            out["sign_accuracy_by_truth"][str(tv)] = {
                pr: dict(k=sum(1 for r in rr if r["Tsign"][k] == tv), n=len(rr))
                for k, pr in enumerate(L.PROBES)}
        out["truth_counts"] = {str(tv): sum(1 for r in dirs if r["truth"] == tv) for tv in (1, -1)}
        out["sign_acc_given_gate"] = {}
        out["CI_excl0_and_correct"] = {}
        for k, pr in enumerate(L.PROBES):
            g = [r for r in dirs if r["gate"][k]]
            out["sign_acc_given_gate"][pr] = dict(
                k=sum(1 for r in g if r["Tsign"][k] == r["truth"]), n=len(g))
            c = sum(1 for r in dirs if r["Tci0"][k] and r["Tsign"][k] == r["truth"])
            out["CI_excl0_and_correct"][pr] = dict(k=c, n=len(dirs), wilson=wilson(c, len(dirs)))
    out["gate_rate"] = {}
    for k, pr in enumerate(L.PROBES):
        c = sum(1 for r in rows if r["gate"][k])
        out["gate_rate"][pr] = dict(k=c, n=n, rate=c / n, wilson=wilson(c, n))
    c = sum(1 for r in rows if all(r["gate"]))
    out["gate_all3"] = dict(k=c, n=n, rate=c / n, wilson=wilson(c, n))
    c = sum(1 for r in rows if any(r["gate"]))
    out["gate_any"] = dict(k=c, n=n, rate=c / n, wilson=wilson(c, n))
    c = sum(1 for r in rows if r["rule"])
    out["rule_satisfied"] = dict(k=c, n=n, rate=c / n, wilson=wilson(c, n))
    c = sum(1 for r in rows if r["joint"])
    out["joint"] = dict(k=c, n=n, rate=c / n, wilson=wilson(c, n))
    if dirs:
        c = sum(1 for r in rows if r["joint"] and r["jsign"] == r["truth"])
        out["joint_and_correct_sign"] = dict(k=c, n=n, wilson=wilson(c, n))
    return out


def pretty(label, s):
    print(f"\n=== {label} (n={s['n']}) ===")
    for k, v in s.items():
        if k == "n":
            continue
        print(f"  {k}: {json.dumps(v)}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--S", type=int, default=200)
    ap.add_argument("--B", type=int, default=300)
    ap.add_argument("--ntrial", type=int, default=60, help="per sub-case (case1)")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--skip32", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.S, a.B, a.ntrial = 40, 60, 3
    ph, fk = L.triad_bins()
    print("task1 Late NBI+ECH window-1 discrete modes:", ph["discrete_modes"])
    f1, f3 = fk[0], fk[-1]
    f2 = f3 - f1
    bins = [int(round(f / L.DF_KHZ)) for f in (f1, f2)]
    print(f"Delta f = {L.DF_KHZ} kHz; f=({f1},{f2},{f3}) kHz; bins={bins}+{bins[0]+bins[1]}")
    sigma, amps = real_params((bins[0], bins[1], bins[0] + bins[1]))
    params = dict(sigma=sigma, amps=amps, bins=bins)
    print("synthetic noise sigma (from 115-145 kHz median PSD):", sigma, " mode rms |A| at 3 lines:", amps)
    real = {w: L.load_window("88653", *WINDOWS[w])[0] for w in WINDOWS}
    rng0 = np.random.default_rng(2026)
    ncs = {"case1": (-1, -1, -1), "case1S": (-1, -2, -3)}
    cal = {}
    N8 = int(round((WINDOWS["w8p7"][1] - WINDOWS["w8p7"][0]) * 1e3)) + 1
    r_main = 0.5
    r_sens = (0.2, 0.9)
    t0 = time.time()
    for r_t in (r_main,) + r_sens:
        for sub in ("1a", "1b"):
            q, m = calibrate("case1", sub, +1, r_t, params, N8, rng0, ncs["case1"],
                             iters=(3 if a.quick else 7), pil=(3 if a.quick else 8))
            cal[(r_t, sub)] = q
            print(f"calibration r_target={r_t} sub={sub}: qs={q:.4g} /us, latent b2 at triad={m:.3f}")
    print(f"calibration time {time.time()-t0:.0f}s")
    CF = dict(params=params, real=real, S=a.S, B=a.B)
    jobs = []
    seed_ctr = [10_000]

    def add(kind, sub, sign, r_t, nvec, w, n, tag):
        for _ in range(n):
            seed_ctr[0] += 1
            case = dict(kind=kind, sub=sub, sign=sign, r=r_t, nvec=nvec, tag=tag,
                        qs=cal.get((r_t, sub), 0.0))
            jobs.append((seed_ctr[0], case, w))

    groups = []
    def grp(name, kind, w, r_t=r_main, n_sub=None):
        ns = n_sub or a.ntrial
        if kind in ("case1", "case1S"):
            for sub in ("1a", "1b"):
                for sign in (+1, -1):
                    add(kind, sub, sign, r_t, ncs[kind], w, ns, name)
        else:
            add(kind, None, None, r_main, (-1, -1, -1), w, max(a.ntrial * 4 if not a.quick else 6, 200 if not a.quick else 6), name)
        groups.append(name)

    grp("case1|w8p7|r0.5", "case1", "w8p7")
    grp("case1S|w8p7|r0.5", "case1S", "w8p7", n_sub=max(50, a.ntrial) if not a.quick else 3)
    grp("null_a|w8p7", "null_a", "w8p7")
    grp("null_b|w8p7", "null_b", "w8p7")
    grp("null_c|w8p7", "null_c", "w8p7")
    for r_t in r_sens:
        grp(f"case1|w8p7|r{r_t}", "case1", "w8p7", r_t=r_t, n_sub=50 if not a.quick else 3)
    if not a.skip32:
        grp("case1|w32p1|r0.5", "case1", "w32p1", n_sub=50 if not a.quick else 3)
        grp("null_a|w32p1", "null_a", "w32p1")
        grp("null_c|w32p1", "null_c", "w32p1")
    # tag by group for r (names must match job tags); rebuild tags
    for i, (sd, case, w) in enumerate(jobs):
        kind = case["kind"]
        if kind in ("case1", "case1S"):
            case["tag"] = f"{kind}|{w}|r{case['r']}"
        else:
            case["tag"] = f"{kind}|{w}"
    print("total trials:", len(jobs), " S=", a.S, " B=", a.B)
    t0 = time.time()
    results = []
    with Pool(a.procs, initializer=init_worker, initargs=(CF,)) as pool:
        for i, r in enumerate(pool.imap_unordered(run_trial, jobs, chunksize=2)):
            results.append(r)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(jobs)} done, {time.time()-t0:.0f}s", flush=True)
    print(f"trials done in {time.time()-t0:.0f}s")
    # group
    by = {}
    for (sd, case, w), r in zip(jobs, results):
        pass
    for r in results:
        k = (f"{r['case']}|{r['wname']}|r{r['r']}" if r["case"] in ("case1", "case1S")
             else f"{r['case']}|{r['wname']}")
        by.setdefault(k, []).append(r)
    summ = {}
    for k, rows in by.items():
        summ[k] = summarize(rows, k)
        pretty(k, summ[k])
    # decisions
    def worst_low(key):
        s = summ[key]["sign_accuracy"]
        return min(s[pr]["wilson"][0] for pr in L.PROBES), {pr: s[pr]["acc"] for pr in L.PROBES}
    dec = {}
    status = "SUPPORTED_ESTIMATOR"
    for key in ("case1|w8p7|r0.5", "case1|w32p1|r0.5"):
        if key in summ:
            lo, acc = worst_low(key)
            dec[key] = dict(worst_probe_lower_wilson=lo, acc=acc, resolved=bool(lo >= 0.8))
            if lo < 0.8:
                status = "UNRESOLVED"
    nulls_ok = True
    for key in summ:
        if key.startswith("null_"):
            up = summ[key]["joint"]["wilson"][1]
            dec[key] = dict(joint_fpr=summ[key]["joint"]["rate"], upper_wilson=up, ok=bool(up <= 0.10),
                            gate_all3_fpr=summ[key]["gate_all3"]["rate"],
                            gate_any_fpr=summ[key]["gate_any"]["rate"])
            nulls_ok &= bool(up <= 0.10)
    if status != "UNRESOLVED" and not nulls_ok:
        status = "NOT_VALIDATED_AGAINST_NULLS"
    print("\nDECISIONS:", json.dumps(dec, indent=1))
    print("ESTIMATOR STATUS:", status)
    out = dict(status=status, decisions=dec, summaries=summ, calibration={f"{k[0]}_{k[1]}": v for k, v in cal.items()},
               params=dict(sigma=sigma, amps=amps, bins=bins, S=a.S, B=a.B, quick=a.quick, gamma_per_us=GAMMA,
                           df_khz=L.DF_KHZ))
    fn = "task14_validation_quick.json" if a.quick else "task14_validation.json"
    json.dump(out, open(os.path.join(HERE, fn), "w"), indent=1, default=float)
    print("wrote", fn)


if __name__ == "__main__":
    main()
