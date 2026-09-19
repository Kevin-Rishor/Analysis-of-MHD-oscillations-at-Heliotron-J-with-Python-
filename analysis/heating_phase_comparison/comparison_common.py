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
    root_dir = Path("c:/TFG")

for p_add in [root_dir / "jpack", root_dir / "analysis", root_dir / "analysis" / "common"]:
    if str(p_add) not in sys.path:
        sys.path.append(str(p_add))

import turnelib as TE

DEFAULT_SHOT = 88653
DEFAULT_OUT_DIR = root_dir / "NBI vs NBI and ECH"

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
