from src.models.SimpleRNN import SimpleRNN
import torch.nn as nn
from torchmetrics import F1Score
import lightning as L
import torch

class LitHandDetector(L.LightningModule):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.model = SimpleRNN(input_dim=144, hidden_dim=144, hidden_layers=3, output_dim=1)
        self.train_f1 = F1Score(num_classes=2, average="macro", task="binary")
        self.validation_f1 = F1Score(num_classes=2, average="macro", task="binary")
        self.pos_weight = pos_weight

    def training_step(self, batch, batch_idx):
        x, lengths, y = batch
        y_pred = self.model(x, lengths)
        loss = nn.functional.binary_cross_entropy_with_logits(y_pred, y, pos_weight=self.pos_weight)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.train_f1.update(y_pred, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, lengths, y = batch
        y_pred = self.model(x, lengths)
        self.validation_f1.update(y_pred, y)

    def on_validation_epoch_end(self):
        self.log("val_f1", self.validation_f1.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.validation_f1.reset()

    def on_train_epoch_end(self):
        self.log("train_f1", self.train_f1.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.train_f1.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        return optimizer
