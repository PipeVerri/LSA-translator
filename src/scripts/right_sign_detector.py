from src.data import LSA64Dataset
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
dataset = LSA64Dataset(root_dir / "data" / "LSA64" / "landmarks", filter_handedness="R")

print(len(dataset))
