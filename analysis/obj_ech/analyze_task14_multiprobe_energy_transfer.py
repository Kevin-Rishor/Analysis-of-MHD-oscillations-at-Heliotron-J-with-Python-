"""
Task 14 -- joint multi-probe bicoherence and energy-transfer direction.

Library (used by validate_task14.py) + real-data driver.

Estimator (grid: nfft=1024, hop=512, Hann, no detrend):
  b2(f1,f2)  segment-averaged squared bicoherence per probe
  surrogate  Fourier phase randomisation (independent per probe), family-wise
             p from the max of b2 over the search box
  Kim-Powers X3[k+1] = L*X3[k] + Q*P[k], P = X1*X2 (complex least squares on
             segment spectra); B = mean(P conj X3); T = Re[conj(L) Q B]
             (cross term of the |X3|^2 budget; T>0 = net transfer INTO f3);
             circular block bootstrap, block length from autocorrelation
  toroidal   cross-phases MP1->MP3, MP1->MP4 at f1,f2,f3 -> n mod 4,
             test n1+n2 = n3 (mod 4) and phase additivity
Real data are analysed only if the validation file allows it.
"""
import os
import sys
import json
import numpy as np
import scipy.signal as dsp

sys.path.insert(0, "c:/TFG/analysis/common")

CACHE = "c:/TFG/cache"
HERE = os.path.dirname(os.path.abspath(__file__))
FS = 1.0e6
NFFT = 1024
HOP = 512
DF_KHZ = FS / NFFT / 1e3
BOX_KHZ = (10.0, 100.0)
PROBES = ("mp1", "mp3", "mp4")
WIN = np.hanning(NFFT + 1)[:-1]  # periodic Hann


# ----------------------------------------------------------------- basics
def seg_spectra(x, nfft=NFFT, hop=HOP):
    """x (..., N) -> complex (..., K, nfft//2+1), Hann windowed."""
    x = np.asarray(x)
    N = x.shape[-1]
    K = 1 + (N - nfft) // hop
    idx = np.arange(nfft)[None, :] + hop * np.arange(K)[:, None]
    seg = x[..., idx] * WIN
    return np.fft.rfft(seg, axis=-1)


def box_pairs(lo=BOX_KHZ[0], hi=BOX_KHZ[1]):
    b0 = int(np.ceil(lo / DF_KHZ))
    b1 = int(np.floor(hi / DF_KHZ))
    ii, jj = np.meshgrid(np.arange(b0, b1 + 1), np.arange(b0, b1 + 1), indexing="ij")
    m = ii <= jj
    return ii[m], jj[m]


I_BOX, J_BOX = box_pairs()
K_BOX = I_BOX + J_BOX


def b2_pairs(X, I, J):
    """X (..., K, nb) -> b2 (..., npairs)."""
    P = X[..., I] * X[..., J]
    X3 = X[..., I + J]
    num = np.abs(np.mean(P * np.conj(X3), axis=-2)) ** 2
    den = np.mean(np.abs(P) ** 2, axis=-2) * np.mean(np.abs(X3) ** 2, axis=-2)
    return num / den


def phase_randomise(x, rng, theta=None):
    """Fourier phase randomisation of the window record (preserves |FFT|)."""
    F = np.fft.rfft(x, axis=-1)
    nb = F.shape[-1]
    if theta is None:
        theta = rng.uniform(0, 2 * np.pi, size=F.shape)
    theta = theta.copy()
    theta[..., 0] = 0.0
    if x.shape[-1] % 2 == 0:
        theta[..., -1] = 0.0
    return np.fft.irfft(F * np.exp(1j * theta), n=x.shape[-1], axis=-1)


def surrogate_stats(x, trip, S, rng, theta_bank=None, chunk=50):
    """Surrogate b2 at the triad (S,) and family-wise max over box (S,)."""
    i1, i2, i3 = trip
    at_t = np.empty(S)
    mx = np.empty(S)
    for s0 in range(0, S, chunk):
        s1 = min(S, s0 + chunk)
        xs = np.stack([phase_randomise(x, rng) for _ in range(s1 - s0)])
        Xs = seg_spectra(xs)
        bb = b2_pairs(Xs, I_BOX, J_BOX)
        mx[s0:s1] = bb.max(axis=-1)
        at_t[s0:s1] = b2_pairs(Xs, np.array([i1]), np.array([i2]))[:, 0]
    return at_t, mx


# --------------------------------------------------------- block bootstrap
def circ_blocks(n, L, B, rng):
    nb = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]) % n
    return idx.reshape(B, -1)[:, :n]


def acf_block_length(series_list, K):
    """First lag where |rho| < 2/sqrt(K), for each series; return max (>=2, <=K/3)."""
    Ls = []
    for s in series_list:
        s = np.asarray(s)
        s = s - s.mean()
        v = np.vdot(s, s).real
        thr = 2.0 / np.sqrt(len(s))
        L = 1
        for lag in range(1, max(2, len(s) // 3)):
            rho = np.vdot(s[:-lag], s[lag:]) / v
            if abs(rho) < thr:
                break
            L = lag + 1
        Ls.append(L)
    L = max(Ls)
    return int(min(max(L, 2), max(2, K // 3))), Ls


def _solve_kp(x3, p, y, idx):
    """Normal equations for y = L x3 + Q p over index sets idx (B,n)."""
    a = x3[idx]
    b = p[idx]
    c = y[idx]
    S11 = np.sum(np.abs(a) ** 2, axis=-1)
    S22 = np.sum(np.abs(b) ** 2, axis=-1)
    S12 = np.sum(np.conj(a) * b, axis=-1)
    r1 = np.sum(np.conj(a) * c, axis=-1)
    r2 = np.sum(np.conj(b) * c, axis=-1)
    det = S11 * S22 - np.abs(S12) ** 2
    Lam = (S22 * r1 - S12 * r2) / det
    Q = (S11 * r2 - np.conj(S12) * r1) / det
    return Lam, Q, det, S11, S22, S12


def kim_powers(X, trip, B, rng):
    """X (K, nb) spectra of one probe. Returns dict with arrays."""
    i1, i2, i3 = trip
    K = X.shape[0]
    x3 = X[:, i3]
    p = X[:, i1] * X[:, i2]
    dt_seg = HOP / FS
    xa, pa = x3[:-1], p[:-1]
    ya = (x3[1:] - x3[:-1]) / dt_seg
    n = K - 1
    one = np.arange(n)[None, :]
    Lam, Q, det, S11, S22, S12 = _solve_kp(xa, pa, ya, one)
    Lam, Q = Lam[0], Q[0]
    Bm = np.mean(p * np.conj(x3))
    T = float(np.real(Q * Bm))
    pw3 = float(np.mean(np.abs(x3) ** 2))
    resid = ya - Lam * xa - Q * pa
    Lb, L_list = acf_block_length([np.abs(x3), np.abs(p), resid], n)
    idx = circ_blocks(n, Lb, B, rng)
    LamB, QB, detB, _, _, _ = _solve_kp(xa, pa, ya, idx)
    # B for resample uses the corresponding segment indices (pairs k -> k+1: use k)
    BB = np.mean((p[:-1] * np.conj(x3[:-1]))[idx], axis=-1)
    TB = np.real(QB * BB)
    ok = np.isfinite(TB)
    lo, hi = np.percentile(TB[ok], [2.5, 97.5])
    cond = float(np.abs(S12[0]) / np.sqrt(S11[0] * S22[0]))
    return dict(Lam=Lam, Q=Q, B=Bm, T=T, T_norm=T / pw3, T_ci=(float(lo), float(hi)),
                block_len=Lb, acf_lens=L_list, collinearity=cond,
                T_boot=TB)


# --------------------------------------------------------------- toroidal
def circ_dist(a, b):
    return np.abs(np.angle(np.exp(1j * (a - b))))


def n_mod4_from_phases(phi13, phi14):
    """phi13 = n*180deg, phi14 = n*270deg (rel. MP1). n in 0..3 minimising error."""
    best, besterr = None, np.inf
    for n in range(4):
        e = circ_dist(phi13, n * np.pi) + circ_dist(phi14, n * 1.5 * np.pi)
        if e < besterr:
            best, besterr = n, e
    return best, float(besterr)


def toroidal(Xs, trip, B, rng, Lb):
    """Xs: dict probe-> (K,nb). Phase = angle(mean(conj(X_mp1) X_probe))."""
    K = Xs["mp1"].shape[0]
    bins = list(trip)
    def phases(idx):
        out = []
        for pr in ("mp3", "mp4"):
            c = np.mean(np.conj(Xs["mp1"][idx][:, bins]) * Xs[pr][idx][:, bins], axis=0)
            out.append(np.angle(c))
        return np.array(out)  # (2, 3)
    ph = phases(np.arange(K))
    idxB = circ_blocks(K, max(2, Lb), B, rng)
    phB = np.array([phases(ix) for ix in idxB])  # (B,2,3)
    nmod = [n_mod4_from_phases(ph[0, f], ph[1, f])[0] for f in range(3)]
    add = ph[:, 2] - ph[:, 0] - ph[:, 1]
    addB = phB[:, :, 2] - phB[:, :, 0] - phB[:, :, 1]
    d = np.angle(np.exp(1j * add))
    dB = np.angle(np.exp(1j * (addB - add[None, :])))  # centred bootstrap deviations
    half = np.percentile(np.abs(dB), 95, axis=0)       # 95 % half-width
    includes0 = bool(np.all(np.abs(d) <= half))
    rule_n = ((nmod[0] + nmod[1] - nmod[2]) % 4) == 0
    return dict(phases_rad=ph, n_mod4=nmod, additivity_rad=d, additivity_half95=half,
                additivity_includes_zero=includes0, rule_n=bool(rule_n),
                rule=bool(rule_n and includes0))


# ---------------------------------------------------------------- pipeline
def analyze_triad(rec, trip, S, B, rng, gate_alpha=0.05, record_boot=False):
    """rec: dict probe -> 1D window record. trip = (i1, i2, i3) bins."""
    out = dict(probes={})
    Xs = {pr: seg_spectra(rec[pr]) for pr in PROBES}
    K = Xs["mp1"].shape[0]
    i1, i2, i3 = trip
    Lb_all = []
    for pr in PROBES:
        x = rec[pr]
        bb_all = b2_pairs(Xs[pr], I_BOX, J_BOX)
        b2_t = float(b2_pairs(Xs[pr], np.array([i1]), np.array([i2]))[0])
        sur_t, sur_max = surrogate_stats(x, trip, S, rng)
        p_unadj = float((1 + np.sum(sur_t >= b2_t)) / (1 + S))
        p_fw = float((1 + np.sum(sur_max >= b2_t)) / (1 + S))
        rank = int(1 + np.sum(bb_all > b2_t))
        kp = kim_powers(Xs[pr], trip, B, rng)
        Lb_all.append(kp["block_len"])
        if not record_boot:
            kp = {k: v for k, v in kp.items() if k != "T_boot"}
        sign_ok = (kp["T_ci"][0] > 0) or (kp["T_ci"][1] < 0)
        out["probes"][pr] = dict(
            b2=b2_t, p_unadj=p_unadj, p_fw=p_fw, gate=bool(p_fw < gate_alpha),
            rank=rank, n_pairs=int(len(I_BOX)), floor_1_over_N=float(1.0 / K),
            surrogate_max_p95=float(np.percentile(sur_max, 95)),
            kp=kp, T_sign=int(np.sign(kp["T"])), T_ci_excludes0=bool(sign_ok))
    tor = toroidal(Xs, trip, B, rng, int(max(Lb_all)))
    out["toroidal"] = tor
    out["N_segments"] = int(K)
    P = out["probes"]
    c_i = all(P[pr]["gate"] for pr in PROBES)
    signs = [P[pr]["T_sign"] for pr in PROBES]
    c_iii = bool(all(P[pr]["T_ci_excludes0"] for pr in PROBES) and len(set(signs)) == 1)
    out["criteria"] = dict(i_gate_all=bool(c_i), ii_rule=tor["rule"], iii_sign_agree=c_iii)
    out["joint"] = bool(c_i and tor["rule"] and c_iii)
    out["joint_sign"] = int(signs[0]) if c_iii else 0
    return out


# ------------------------------------------------------------ data helpers
def triad_bins():
    d = json.load(open(os.path.join(HERE, "task1_ech_results.json")))
    ph = [p for p in d["88653"]["whole_discharge_phase_modes"]
          if p["phase"] == "Late NBI+ECH (Window 1)"][0]
    fk = sorted(m["f_khz"] if isinstance(m, dict) else m for m in ph["discrete_modes"])
    return ph, fk


def load_window(shot, t0, t1):
    c = np.load(os.path.join(CACHE, f"shot_{shot}_full_record_cache.npz"))
    t = c["t_ms"]
    m = (t >= t0) & (t <= t1)
    return {pr: np.asarray(c[pr][m], dtype=float) for pr in PROBES}, t[m]


def task5_window():
    src = open(os.path.join(HERE, "analyze_task5_nonlinear.py")).read()
    return src


def control_pairs(rec_mp1):
    """Largest-b2 pairs with the Task 5 estimator (nfft=512), mapped to the 1024 grid."""
    from abicoh2_corrected import abicoh2_corrected
    _, b2 = abicoh2_corrected(rec_mp1, fs=FS, nfft=512, noverlap=256)
    df5 = FS / 512 / 1e3
    lo, hi = int(np.ceil(BOX_KHZ[0] / df5)), int(np.floor(BOX_KHZ[1] / df5))
    cand = []
    for i in range(lo, hi + 1):
        for j in range(i, hi + 1):
            if j < b2.shape[1]:
                cand.append((float(b2[i, j]), i, j))
    cand.sort(reverse=True)
    return [(v, 2 * i, 2 * j, i * df5, j * df5) for v, i, j in cand[:2]]


def ci_str(lo, hi):
    return f"[{lo:.3g}, {hi:.3g}]"


def main(S=2000, B=1000, seed=14):
    sys.stdout.reconfigure(encoding="utf-8")
    vpath = os.path.join(HERE, "task14_validation.json")
    if not os.path.exists(vpath):
        print("task14_validation.json missing: run validate_task14.py first. STOP.")
        return
    val = json.load(open(vpath))
    print("VALIDATION STATUS:", val["status"])
    estimator_validated = (val["status"] == "SUPPORTED_ESTIMATOR")
    if not estimator_validated:
        print("NOTICE: Estimator labelled UNRESOLVED by validation stop rule (lower Wilson CI < 0.8).")
        print("Real-data transfer directions are NOT validated and cannot be claimed.")
    rng = np.random.default_rng(seed)
    ph, fk = triad_bins()
    print("task1 modes (kHz), Late NBI+ECH window 1:", fk)
    res = dict(estimator_label=val["status"], validation=val, shots={})
    dfk = DF_KHZ
    print(f"Delta f = {dfk} kHz (nfft={NFFT}, hop={HOP}); box {BOX_KHZ} kHz; pairs={len(I_BOX)}")
    f1, f3 = fk[0], fk[-1]
    f2 = f3 - f1
    trip_t = tuple(int(round(f / dfk)) for f in (f1, f2, f3))
    print("targeted triad kHz:", (f1, f2, f3), "bins:", trip_t)
    for shot in ("88653", "88654"):
        res["shots"][shot] = {}
        for wname, (t0, t1) in dict(w8p7=(282.5, 291.2), w32p1=(259.1, 291.2)).items():
            rec, t = load_window(shot, t0, t1)
            print(f"\n#{shot} {wname}: t=[{t[0]:.3f},{t[-1]:.3f}] ms n={len(t)}")
            triads = {"targeted": trip_t}
            cp = control_pairs(rec["mp1"])
            for k, (v, a, b, fa, fb) in enumerate(cp):
                print(f"  control {k+1}: Task5 estimator b2={v:.4f} at f1={fa:.3f}, f2={fb:.3f} kHz -> bins {a},{b},{a+b}")
                triads[f"control{k+1}"] = (a, b, a + b)
            res["shots"][shot][wname] = {}
            for name, trip in triads.items():
                r = analyze_triad(rec, trip, S, B, rng, record_boot=False)
                res["shots"][shot][wname][name] = r
                fr = tuple(round(b * dfk, 4) for b in trip)
                print(f"  --- {name} bins {trip} f(kHz)={fr}, N_seg={r['N_segments']}, "
                      f"1/N floor={1.0 / r['N_segments']:.4f}")
                for pr in PROBES:
                    P = r["probes"][pr]
                    kp = P["kp"]
                    print(f"    {pr}: b2={P['b2']:.4f} p={P['p_unadj']:.4f} p_fw={P['p_fw']:.4f} "
                          f"rank={P['rank']}/{P['n_pairs']} T={kp['T']:.4g} CI={ci_str(*kp['T_ci'])} "
                          f"Tnorm={kp['T_norm']:.3g} L={kp['block_len']} acf={kp['acf_lens']} "
                          f"collin={kp['collinearity']:.3f}")
                    print(f"        Lambda={kp['Lam']}  Q={kp['Q']}  B={kp['B']}")
                tor = r["toroidal"]
                print("    n mod 4 (f1,f2,f3):", tor["n_mod4"], "| rule_n:", tor["rule_n"],
                      "| additivity (rad):", tor["additivity_rad"], "half95:", tor["additivity_half95"],
                      "| rule:", tor["rule"])
                print("    phases rad (rows mp3, mp4; cols f1,f2,f3):\n", tor["phases_rad"])
                print("    criteria:", r["criteria"], "joint:", r["joint"])
    # labels: #88653 only decides; #88654 is a same-direction check
    res["labels"] = {}
    for wname in ("w8p7", "w32p1"):
        for name in res["shots"]["88653"][wname]:
            a = res["shots"]["88653"][wname][name]
            b = res["shots"]["88654"][wname][name]
            gate_any = any(a["probes"][pr]["gate"] for pr in PROBES)
            if not gate_any:
                lab = "NULL"
            elif a["joint"] and val["status"] == "SUPPORTED_ESTIMATOR" \
                    and b["joint_sign"] == a["joint_sign"] and b["joint"]:
                lab = "SUPPORTED"
            else:
                lab = "INCONCLUSIVE"
            res["labels"][f"{wname}/{name}"] = dict(
                label=lab, joint_88653=a["joint"], criteria_88653=a["criteria"],
                joint_88654=b["joint"], sign_88653=a["joint_sign"], sign_88654=b["joint_sign"],
                gate_88654=[b["probes"][pr]["gate"] for pr in PROBES])
            print(f"LABEL {wname}/{name}: {lab}  (criteria #88653 {a['criteria']}; #88654 joint={b['joint']})")

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        if isinstance(o, complex) or isinstance(o, np.complexfloating):
            return [float(o.real), float(o.imag)]
        if isinstance(o, np.generic):
            return o.item()
        return o
    json.dump(clean(res), open(os.path.join(HERE, "task14_results.json"), "w"), indent=1)
    print("wrote task14_results.json")


if __name__ == "__main__":
    main()
