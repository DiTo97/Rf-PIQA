import torch
import torch.nn as nn
import pytorch_lightning as pl

from Rf_PIQA.nn.losses.predintervals import PIVEN
from Rf_PIQA.nn.teacher.model import ReferencePIQAModel


class ReferencePIQATrainer(pl.LightningModule):
    """
    LightningModule wrapper for training the reference (teacher) PIQA model.
    """
    def __init__(self, model_config, training_config):
        super().__init__()
        self.model = ReferencePIQAModel(model_config)
        head_type = model_config.get("head_type", "point")
        if head_type == "PIVEN":
            head_config = model_config.get(head_type, {})

            self.loss_fn = PIVEN(
                lambda_=head_config.get("lambda", 15.0),
                soft=head_config.get("soft", 160.0),
                alpha=head_config.get("alpha", 0.05),
                beta=head_config.get("beta", 0.5),
                eps=head_config.get("eps", 1e-6)
            )
        else:
            self.loss_fn = nn.MSELoss()

        self.learning_rate = training_config.get("learning_rate", 1e-4)
    
    def forward(self, panoramas, constituents_batch, padding_mask):
        return self.model(panoramas, constituents_batch, padding_mask)
    
    def training_step(self, batch, batch_idx):
        panoramas, constituents_batch, padding_mask = batch["record"]
        scores = batch["score"]
        outputs = self.model(panoramas, constituents_batch, padding_mask)
        loss = self.loss_fn(outputs, scores)
        self.log("training-loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        panoramas, constituents_batch, padding_mask = batch["record"]
        scores = batch["score"]
        outputs = self.model(panoramas, constituents_batch, padding_mask)
        loss = self.loss_fn(outputs, scores)
        self.log("validation-loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer
