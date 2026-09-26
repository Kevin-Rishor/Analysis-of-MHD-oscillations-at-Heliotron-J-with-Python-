import sys
from pathlib import Path

root_dir = Path("c:/TFG")
sys.path.append(str(root_dir / "jpack"))
import turnelib as TE

def run():
    edf = TE.edf()
    for i in range(1, 17):
        f = f"c:/TFG/data/hj88653/ECE{i}FAST@88653.edf"
        try:
            dat = edf.load(f)
            props = [c for c in edf.comments if "property" in c]
            print(f"ECE{i}FAST: {props}")
        except Exception as e:
            print(f"ECE{i}FAST: Error {e}")

if __name__ == "__main__":
    run()
