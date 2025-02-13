import sys

import torch
import pytorch_lightning as pl
import yaml
from torch.utils.data import DataLoader

from Rf_PIQA.datasets.common import load_dataset
from Rf_PIQA.nn.teacher.collation import collate
from Rf_PIQA.nn.teacher.processing import get_panorama_processor, get_constituents_processor
from Rf_PIQA.training.teacher import ReferencePIQATrainer


def collate_batch(batch):
    # Use the provided collate_fn to process images.
    panoramas, constituents_batch, padding_mask = collate(batch, panorama_processor, constituents_processor)
    scores = torch.stack([item["score"] for item in batch])
    return {"record": (panoramas, constituents_batch, padding_mask), "score": scores}


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python training-teacher.py <config-file>"

    with open(sys.argv[1], "r") as f:
        config = yaml.safe_load(f)

    teacher_config = config["teacher"]
    training_config = config["training"]
    
    # Create image processors
    global panorama_processor, constituents_processor
    panorama_processor = get_panorama_processor(teacher_config["panorama_model_name"],
                                                teacher_config["image_size_panorama"])
    constituents_processor = get_constituents_processor(teacher_config["constituents_model_name"],
                                              teacher_config["image_size_constituents"])
    
    train_dataset, val_dataset = load_dataset(config["dataset"])

    train_loader = DataLoader(train_dataset, batch_size=training_config.get("batch_size", 16),
                              collate_fn=collate_batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_config.get("batch_size", 16),
                            collate_fn=collate_batch)
    
    model = ReferencePIQATrainer(teacher_config, training_config)

    # TODO: training job logging
    
    trainer = pl.Trainer(max_epochs=training_config.get("num_epochs", 10))
    trainer.fit(model, train_loader, val_loader)
