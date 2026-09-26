import sys
import numpy as np
from pathlib import Path

root_dir = Path("c:/TFG")
sys.path.append(str(root_dir / "analysis" / "common"))
sys.path.append(str(root_dir / "jpack"))
import turnelib as TE

def check_zeros():
    data_dir = root_dir / "data" / "hj88653"
    edf = TE.edf()
    for i in range(1, 17):
        f_ece = data_dir / f"ECE{i}FAST@88653.edf"
        dat = edf.load(str(f_ece))
        y = dat[:, 1]
        zeros = np.sum(y == 0.0)
        frac = zeros / len(y)
        print(f"ECE{i}FAST: {frac:.2%} exactly zero (max: {np.max(np.abs(y)):.2e})")

if __name__ == "__main__":
    check_zeros()
