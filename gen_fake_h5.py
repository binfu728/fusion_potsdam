#!/usr/bin/env python3
"""Generate a fake H5 file for tensor-shape verification.

Format follows the H5OlmoEarthDataset load convention:
- HR: (480, 480, 3)  the high-res image that feeds the DINOv3 ViT.
- sentinel2_l2a (= MS): (24, 24, 12, 3)  (H,W,T,C).
- sentinel1 (= SAR):    (24, 24, 12, 2).
- timestamps, latlon, missing_timesteps_masks: metadata (required by
  the real loader; dummy here since we use a stub OlmoEarth).

Usage:  python gen_fake_h5.py              # writes fake_sample.h5 in cwd
"""

import h5py
import numpy as np
from pathlib import Path

HR_SIZE = 480
HR_RATIO = 20          # hr_h5_resolution_ratio
H5_SIZE = HR_SIZE // HR_RATIO   # 24
MAX_T = 12              # max_sequence_length

OUT = Path(__file__).resolve().parent / "fake_sample.h5"

rng = np.random.default_rng(42)

with h5py.File(OUT, "w") as f:
    f.create_dataset("HR", data=rng.standard_normal((HR_SIZE, HR_SIZE, 3)).astype(np.float32))

    f.create_dataset("sentinel2_l2a", data=rng.standard_normal((H5_SIZE, H5_SIZE, MAX_T, 3)).astype(np.float32))
    f.create_dataset("sentinel1",     data=rng.standard_normal((H5_SIZE, H5_SIZE, MAX_T, 2)).astype(np.float32))

    f.create_dataset("timestamps", data=np.arange(MAX_T, dtype=np.int64).reshape(-1, 1))
    f.create_dataset("latlon",     data=rng.standard_normal((H5_SIZE, H5_SIZE)).astype(np.float32))

    grp = f.create_group("missing_timesteps_masks")
    grp.create_dataset("sentinel2_l2a", data=np.ones(MAX_T, dtype=bool))
    grp.create_dataset("sentinel1",     data=np.ones(MAX_T, dtype=bool))

print(f"Fake H5 written to: {OUT}")
print(f"  HR:              {HR_SIZE}x{HR_SIZE}x3")
print(f"  sentinel2_l2a:   {H5_SIZE}x{H5_SIZE}x{MAX_T}x3 (MS)")
print(f"  sentinel1:       {H5_SIZE}x{H5_SIZE}x{MAX_T}x2 (SAR)")
print(f"  + timestamps, latlon, missing_timesteps_masks")
