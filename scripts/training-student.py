import sys

import torch
import pytorch_lightning as pl
import yaml
from torch.utils.data import DataLoader

from Rf_PIQA.datasets.common import load_dataset
from Rf_PIQA.nn.student.model import ReferencefreePIQAModel
from Rf_PIQA.nn.teacher.collation import collate
from Rf_PIQA.nn.teacher.model import ReferencePIQAModel
from Rf_PIQA.nn.teacher.processing import get_panorama_processor, get_constituents_processor
from Rf_PIQA.training.student import ReferencefreePIQATrainer


def collate_batch(batch):
    # Use the provided collate_fn to process images.
    panoramas, constituents_batch, padding_mask = collate(batch, panorama_processor, constituents_processor)
    scores = torch.stack([item["score"] for item in batch])
    return panoramas, constituents_batch, padding_mask, scores


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python training-student.py <config-file>"

    with open(sys.argv[1], "r") as f:
        config = yaml.safe_load(f)

    teacher_config = config["teacher"]
    student_config = config["student"]
    training_config = config["training"]
    
    global panorama_processor, constituents_processor
    panorama_processor = get_panorama_processor(teacher_config["panorama_model_name"],
                                                teacher_config["image_size_panorama"])
    constituents_processor = get_constituents_processor(teacher_config["constituents_model_name"],
                                              teacher_config["image_size_constituents"])
    
    # Initialize the teacher model and assume it is pretrained (or load a checkpoint)
    teacher_model = ReferencePIQAModel(teacher_config)
    
    student_model = ReferencefreePIQAModel(student_config)
    
    distillation_config = training_config.get("distillation", {})
    
    student_head_type = student_config.get("head_type", "point")

    model = ReferencefreePIQATrainer(
        teacher=teacher_model,
        student=student_model,
        head_type=student_head_type,
        distillation_config=distillation_config,
        loss_config=student_config.get(student_head_type, {}),
        learning_rate=training_config.get("learning_rate", 1e-4)
    )
    
    train_dataset, val_dataset = load_dataset(config["dataset"])
    
    train_loader = DataLoader(train_dataset, batch_size=training_config.get("batch_size", 16),
                              collate_fn=collate_batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_config.get("batch_size", 16),
                            collate_fn=collate_batch)
    
    # TODO: training job logging
    
    trainer = pl.Trainer(max_epochs=training_config.get("num_epochs", 10))
    trainer.fit(model, train_loader, val_loader)
