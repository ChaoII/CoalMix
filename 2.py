import numpy as np
from pathlib import Path

root_dir = Path("test_data/purchase_data")

for item in root_dir.iterdir():
    if item.is_file() and item.suffix == ".csv":
        s = np.loadtxt(item, delimiter=",", encoding="utf8")
        np.save((root_dir / "npy" / (item.stem + ".npy")).resolve().as_posix(), s)
