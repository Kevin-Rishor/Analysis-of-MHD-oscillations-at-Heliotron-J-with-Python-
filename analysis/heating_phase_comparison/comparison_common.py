import sys
from pathlib import Path
import numpy as np

current = Path(__file__).resolve().parent
root_dir = None
for p in [current] + list(current.parents):
    if (p / "jpack").exists():
        root_dir = p
        break
if root_dir is None:
    root_dir = current.parents[1]

for p_add in [root_dir / "jpack", root_dir / "analysis", root_dir / "analysis" / "common"]:
    if str(p_add) not in sys.path:
        sys.path.append(str(p_add))

import turnelib as TE

DEFAULT_SHOT = 88653
DEFAULT_OUT_DIR = root_dir / "Images" / "heating_phase_comparison"

# Heating Phase Windows
T_NOISE_START = 150.0  # ms
T_NOISE_END = 175.0    # ms
T_W1_START = 259.1     # ms (ECH + NBI Phase)
T_W1_END = 291.3       # ms (Measured ECH disconnection)
T_W2_START = 291.3     # ms (Pure NBI Phase)
T_W2_END = 300.0       # ms

PROBE_ANGLES_TOR = {"MP1": 33.3, "MP3": 213.3, "MP4": 303.3}
PMP_RAW_ANGLE_LABELS_DEG = [0., 10., 20., 30., 40., 50., 60., 80., 90., 100., 110., 120., 150., 180.]
PMP_ANGLES_DEG = [(360.0 - a) % 360.0 for a in PMP_RAW_ANGLE_LABELS_DEG]
PMP_INVERT = ("PMP1", "PMP2", "PMP3", "PMP4")


def safe_savefig(fig, out_path, dpi=150):
    out_path = Path(out_path)
    try:
        if out_path.exists():
            try:
                out_path.unlink()
            except Exception:
                pass
        fig.savefig(str(out_path), dpi=dpi)
    except Exception:
        import shutil
        tmp_file = out_path.parent / f"tmp_{out_path.name}"
        fig.savefig(str(tmp_file), dpi=dpi)
        try:
            if out_path.exists():
                try:
                    out_path.unlink()
                except Exception:
                    pass
            shutil.move(str(tmp_file), str(out_path))
        except Exception:
            pass


def load_channel(shot, channel_name, data_dir=None):
    if data_dir is None:
        data_dir = root_dir / "data" / f"hj{shot}"
    else:
        data_dir = Path(data_dir)

    fname = f"{channel_name}@{shot}.edf" if "@" not in channel_name else channel_name
    fpath = data_dir / fname
    if not fpath.exists():
        return None, None, None, None

    edf = TE.edf()
    dat = edf.load(str(fpath))
    t_raw = dat[:, 0]
    t_ms = t_raw if edf.DimUnit[0] == "ms" else t_raw * 1000.0
    ys = dat[:, 1]
    dt_s = (t_ms[1] - t_ms[0]) / 1000.0
    fs = 1.0 / dt_s
    return t_ms, ys, dt_s, fs


def load_signals(shot=DEFAULT_SHOT, data_dir=None):
    signals_dict = {}
    time_ms = None
    dt_s = None
    fs = None
    for p in ["MP1", "MP3", "MP4"]:
        t, ys, dt, f = load_channel(shot, p, data_dir=data_dir)
        if ys is not None:
            signals_dict[p] = ys
            if time_ms is None:
                time_ms = t
                dt_s = dt
                fs = f

    pmp_signals = {}
    for i in range(1, 15):
        ch = f"PMP{i}"
        t, ys, _, _ = load_channel(shot, ch, data_dir=data_dir)
        if ys is not None:
            if ch in PMP_INVERT:
                ys = -ys
            pmp_signals[ch] = ys

    return time_ms, dt_s, fs, signals_dict, pmp_signals


def plot_toroidal_phase_panel(ax, measured, fits, title=None):
    """Draw the Measured Phase vs. Theoretical Toroidal Phase panel with consistent layout."""
    d_angles_plot = np.linspace(-180, 270, 300)
    for n_test, col, ls in [(-1, "crimson", "-"), (+3, "navy", "--"), (+1, "gray", ":"), (+2, "darkorange", "-.")]:
        theo_line = [(n_test * np.radians(a) + np.pi) % (2 * np.pi) - np.pi for a in d_angles_plot]
        rmse_val = fits[n_test]["rmse_deg"] if isinstance(fits[n_test], dict) else fits[n_test]
        ax.plot(d_angles_plot, np.degrees(theo_line), color=col, ls=ls, lw=1.5,
                label=f"Theory n = {n_test:+d} (RMSE={rmse_val:.1f} deg)")

    pair_labels = {
        ("MP1", "MP3"): "MP1-MP3 (+180 deg)",
        ("MP1", "MP4"): "MP1-MP4 (+270 deg)",
        ("MP3", "MP4"): "MP3-MP4 (+90 deg)",
    }
    for pair_key, lbl in pair_labels.items():
        pdict = measured[pair_key]
        a = pdict["d_tor_deg"]
        ph = pdict["phase_deg"]
        ax.scatter([a], [ph], color="blue", s=90, zorder=5, edgecolors="black", lw=1.5)
        # Avoid title overlap when phase is near top (>100 deg) and legend overlap near bottom
        if ph > 100:
            xytext = (0, -28)
            ha = "center"
        elif ph < -100 and a > 120:
            xytext = (-10, 10)
            ha = "right"
        else:
            xytext = (0, 10)
            ha = "center"
        ax.annotate(f"{lbl}\n({ph:+.1f} deg)", (a, ph), textcoords="offset points",
                    xytext=xytext, ha=ha, fontsize=9.0, fontweight="bold")

    ax.set_xlim(-10, 290)
    ax.set_ylim(-195, 195)
    ax.set_xlabel(r"Toroidal Probe Separation $\Delta\varphi$ (deg)", fontsize=10)
    ax.set_ylabel(r"Cross-Spectral Phase $\Delta\phi$ (deg)", fontsize=10)
    if title is None:
        title = r"Measured Phase vs. Theoretical Toroidal Phase $\Delta\phi = n \Delta\varphi$"
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower right", fontsize=8.0)
    ax.grid(True, alpha=0.3)


def extract_panel_axes(fig, ax, out_path, dpi=150, pad=0.08, facecolor=None, edgecolor='none'):
    """Extract a single panel axes from a figure using tightbbox and specified padding."""
    import matplotlib.transforms as mtransforms
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox_display = ax.get_tightbbox(renderer)
    if bbox_display is None:
        bbox_display = ax.get_window_extent(renderer)
    bbox_inches = bbox_display.transformed(fig.dpi_scale_trans.inverted())
    bbox_padded = mtransforms.Bbox.from_extents(
        bbox_inches.x0 - pad, bbox_inches.y0 - pad,
        bbox_inches.x1 + pad, bbox_inches.y1 + pad
    )
    fc = facecolor if facecolor is not None else fig.get_facecolor()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches=bbox_padded,
                facecolor=fc, edgecolor=edgecolor, pad_inches=0)

