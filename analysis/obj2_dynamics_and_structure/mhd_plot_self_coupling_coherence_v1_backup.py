import sys
from pathlib import Path
import numpy as np
import scipy.signal as dsp
import matplotlib.pyplot as plt

current = Path(__file__).resolve().parent
root_dir = None
for p in [current] + list(current.parents):
    if (p / "jpack").exists():
        root_dir = p
        break
if root_dir is None:
    root_dir = Path("c:/TFG")

for p_add in [root_dir / "jpack", root_dir / "analysis", root_dir / "analysis" / "common"]:
    if str(p_add) not in sys.path:
        sys.path.append(str(p_add))

import turnelib as TE
from mhd_common import extract_instantaneous_frequency


def generate_self_coupling_plot(shot=88653, out_dir=None):
    if out_dir is None:
        out_dir = root_dir
    else:
        out_dir = Path(out_dir)

    data_dir = root_dir / "data" / f"hj{shot}"
    probes = ["MP1", "MP3", "MP4"]
    colors = {"MP1": "tab:blue", "MP3": "tab:purple", "MP4": "tab:green"}
    styles = {"MP1": "-", "MP3": "-.", "MP4": "--"}

    t_start, t_end = 259.1, 275.0
    edf = TE.edf()
    results = {}

    for p in probes:
        file_path = data_dir / f"{p}@{shot}.edf"
        if not file_path.exists():
            continue
        dat = edf.load(str(file_path))
        t_raw = dat[:, 0]
        t_ms = t_raw if edf.DimUnit[0] == "ms" else t_raw * 1000.0
        dt = (t_ms[1] - t_ms[0]) / 1000.0
        fs = 1.0 / dt
        ys = dat[:, 1]

        envelope, _, _, _ = extract_instantaneous_frequency(ys, fs, 80000, 120000, 4, 325)

        idx_win = np.where((t_ms >= t_start) & (t_ms <= t_end))[0]
        ys_win = ys[idx_win]
        env_win = envelope[idx_win]

        nperseg = 2048
        noverlap = 1536
        nfft = 4096
        f, Pxy = dsp.csd(ys_win, env_win, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, nfft=nfft)
        f, coh2 = dsp.coherence(ys_win, env_win, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, nfft=nfft)

        f_khz = f / 1000.0
        phase_deg = np.degrees(np.angle(Pxy))

        results[p] = {
            "f_khz": f_khz,
            "coh2": coh2,
            "phase_deg": phase_deg,
            "n_seg": int(np.floor((len(ys_win) - nperseg) / (nperseg - noverlap))) + 1
        }

    n_seg = list(results.values())[0]["n_seg"]
    sig_floor = 1.0 - (0.05) ** (1.0 / max(1, n_seg - 1))

    plt.rcdefaults()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.2), sharex=True, facecolor="white")
    fig.suptitle(
        f"Heliotron J #{shot} — Self-Coupling Coherence: $\\tilde{{B}}$ vs. Mode Envelope\n"
        f"($f_0 \\approx 89.0$ kHz Mode, Active Window: {t_start:.1f} – {t_end:.1f} ms)",
        fontsize=12, fontweight="bold", y=0.97
    )

    for p in probes:
        if p in results:
            r = results[p]
            ax1.plot(
                r["f_khz"], r["coh2"],
                color=colors[p], linestyle=styles[p], lw=1.7,
                label=f"{p} (Self: $\\tilde{{B}}_{{{p}}}$ vs. $\\mathrm{{Env}}_{{{p}}}$)"
            )

    ax1.axhline(sig_floor, color="dimgray", ls=":", lw=1.4, label=rf"95% Confidence Noise Floor ($\gamma^2 = {sig_floor:.2f}$)")
    ax1.text(4.92, sig_floor + 0.008, f"95% Floor ({sig_floor:.2f})", color="dimgray", fontsize=8.5, fontweight="bold", ha="right", va="bottom")

    ax1.axvspan(0.8, 2.5, color="gray", alpha=0.08, label="Coupling band ($0.8 - 2.5$ kHz)")
    ax1.axvline(1.22, color="tab:blue", ls="--", lw=1.0, alpha=0.6)
    ax1.axvline(2.20, color="tab:purple", ls="--", lw=1.0, alpha=0.6)

    ax1.scatter([1.22], [0.182], color="tab:blue", s=50, zorder=5, edgecolors="white", lw=1.0)
    ax1.annotate(
        "1.22 kHz (MP1, MP4)", xy=(1.22, 0.182), xytext=(1.22, 0.235),
        ha="center", fontsize=8.5, fontweight="bold", color="tab:blue",
        arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.2)
    )

    ax1.scatter([2.20], [0.176], color="tab:purple", s=50, zorder=5, edgecolors="white", lw=1.0)
    ax1.annotate(
        "2.20 kHz (MP3)", xy=(2.20, 0.176), xytext=(2.20, 0.225),
        ha="center", fontsize=8.5, fontweight="bold", color="tab:purple",
        arrowprops=dict(arrowstyle="->", color="tab:purple", lw=1.2)
    )

    ax1.set_xlim(0.0, 5.0)
    ax1.set_ylim(0.0, 0.35)
    ax1.set_ylabel(r"Coherence $\gamma^2$", fontsize=10.5)
    ax1.set_title(r"(a) Self-Coherence $\gamma^2(\tilde{B}, \mathrm{Envelope})$", fontsize=11, fontweight="bold", pad=6)
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.95)

    for p in probes:
        if p in results:
            r = results[p]
            ax2.plot(
                r["f_khz"], r["phase_deg"],
                color=colors[p], linestyle=styles[p], lw=1.7,
                label=f"{p}"
            )

    ax2.axvspan(0.8, 2.5, color="gray", alpha=0.08)
    ax2.axvline(1.22, color="tab:blue", ls="--", lw=1.0, alpha=0.6)
    ax2.axvline(2.20, color="tab:purple", ls="--", lw=1.0, alpha=0.6)

    ax2.set_xlim(0.0, 5.0)
    ax2.set_ylim(-185.0, 185.0)
    ax2.set_yticks([-180, -90, 0, 90, 180])
    ax2.set_xlabel("Modulation Frequency (kHz)", fontsize=10.5)
    ax2.set_ylabel("Phase (deg)", fontsize=10.5)
    ax2.set_title(r"(b) Cross-Spectral Phase $\Delta\phi$", fontsize=11, fontweight="bold", pad=6)
    ax2.grid(True, ls=":", alpha=0.4)
    ax2.legend(loc="lower right", fontsize=8.5, framealpha=0.95)

    plt.tight_layout()
    out_png = out_dir / f"mhd_self_coupling_coherence_shot_{shot}.png"
    plt.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Figure saved to: '{out_png}'")
    return str(out_png)


if __name__ == "__main__":
    generate_self_coupling_plot()

