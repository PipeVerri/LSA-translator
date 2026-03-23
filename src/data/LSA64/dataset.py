import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

class LSA64Dataset(Dataset):
    def __init__(self, root, hand_label=False, filter_handedness=False, no_sign_count=150):
        self.root = root
        self.hand_label = hand_label
        self.meta = pd.read_csv(os.path.join(root, "./meta.csv"))

        if filter_handedness:
            def evaluator(f):
                if not f.endswith(".npy"):
                    return False
                id = re.search(r"(\d+)_\d+\.npy", str(f)).group(1)
                return self.meta.iloc[int(id) - 1]["H"] == filter_handedness
        else:
            evaluator = lambda f: f.endswith(".npy")

        self.files = [f for f in os.listdir(root) if evaluator(f)] + [f for f in os.listdir(os.path.join(root, "../TED/landmarks")) if f.endswith(".npy")][:no_sign_count]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        label = int(filename.split("_")[0]) - 1  # -1 para que los labels empiezen en 0

        arr = np.load(os.path.join(self.root, filename) if label != 64 else os.path.join(self.root, "../TED/landmarks", filename))

        if arr.shape[0] == 0:
            raise ValueError(f"Empty array at {filename}")
        if arr.shape[1] != 144:
            raise ValueError(f"Wrong array shape at {filename}: {arr.shape}")

        tensor = torch.from_numpy(arr).float() # Numpy trabaja con float64 pero torch con float32
        if self.hand_label:
            hand_label = self.meta.iloc[label]["H"]
            hand_label = 0 if hand_label == "R" else 1
            return tensor, hand_label
        else:
            return tensor, label