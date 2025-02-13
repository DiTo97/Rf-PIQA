import torch
import torch.nn as nn
import pytorch_lightning as pl

from Rf_PIQA.nn.losses.predintervals import PIVEN


class ReferencefreePIQATrainer(pl.LightningModule):
    """
    PyTorch Lightning module that performs teacher-student distillation.
    The teacher (a reference model) is frozen and produces soft scores.
    The student (reference-less model) is trained with a combination of its own loss
    and a distillation loss (MSE between teacher and student outputs).
    """
    def __init__(self, teacher, student, head_type="point", distillation_config=None, loss_config=None, learning_rate=1e-4):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.learning_rate = learning_rate
        self.distillation_config = distillation_config or {}
        self.alpha = self.distillation_config.get("alpha", 0.5)  # weight for student's own loss
        self.temperature = self.distillation_config.get("temperature", 2.0)

        loss_config = loss_config or {}
        
        if head_type == "PIVEN":
            self.student_loss_fn = PIVEN(
                lambda_=loss_config.get("lambda", 15.0),
                soft=loss_config.get("soft", 160.0),
                alpha=loss_config.get("alpha", 0.05),
                beta=loss_config.get("beta", 0.5),
                eps=loss_config.get("eps", 1e-6)
            )
        else:
            self.student_loss_fn = nn.MSELoss()
        
        # Distillation loss is defined as MSE between teacher and student outputs.
        self.distillation_loss_fn = nn.MSELoss()
        
        # Freeze teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
    
    def training_step(self, batch, batch_idx):
        # Unpack the batch: teacher gets full inputs; student only high-res images.
        panoramas, constituents_batch, padding_mask, scores = batch
        with torch.no_grad():
            teacher_output = self.teacher(panoramas, constituents_batch, padding_mask)
        student_output = self.student(panoramas)
        
        student_loss = self.student_loss_fn(student_output, scores)
        distill_loss = self.distillation_loss_fn(student_output, teacher_output)
        loss = self.alpha * student_loss + (1 - self.alpha) * distill_loss
        self.log("training-loss", loss)
        self.log("student-loss", student_loss)
        self.log("distill-loss", distill_loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        panoramas, constituents_batch, padding_mask, scores = batch
        teacher_output = self.teacher(panoramas, constituents_batch, padding_mask)
        student_output = self.student(panoramas)
        student_loss = self.student_loss_fn(student_output, scores)
        distill_loss = self.distillation_loss_fn(student_output, teacher_output)
        loss = self.alpha * student_loss + (1 - self.alpha) * distill_loss
        self.log("validation-loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.student.parameters(), lr=self.learning_rate)
        return optimizer
