from src.data.utils.dataloader import generate_collated_dataloader
from src.data.utils.train_test_split import train_test_split_dataset
from src.data import LSA64Dataset
from src.models.HandDetector import LitHandDetector
import torch
from pathlib import Path
import lightning as L
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint

root_path = Path(__file__).parent.parent
dataset = LSA64Dataset(root_path / "data" / "LSA64" / "landmarks", hand_label=True)
train_ds, test_ds = train_test_split_dataset(dataset)
torch.manual_seed(42)

all_labels = []
for _, y in train_ds:
    all_labels.append(int(y))
labels = torch.tensor(all_labels)
num_pos = (labels == 1).sum()
num_neg = (labels == 0).sum()
pos_weight = (num_neg / num_pos).unsqueeze(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

train_loader = generate_collated_dataloader(train_ds)
test_loader = generate_collated_dataloader(test_ds, shuffle=False)

checkpoint_callback = ModelCheckpoint(
    monitor="val_f1",
    mode="max",
    save_top_k=1,
    dirpath=root_path / "checkpoints" / "hand_detector",
    filename="best_params",
    save_last=False
)

trainer = L.Trainer(max_epochs=25, callbacks=[RichProgressBar(), checkpoint_callback], logger=False)
model = LitHandDetector(pos_weight=pos_weight)
trainer.fit(model, train_loader, val_dataloaders=test_loader)
