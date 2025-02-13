import sys

import torch
import yaml
from torch.utils.data import DataLoader

from Rf_PIQA.datasets.common import load_dataset
from Rf_PIQA.nn.student.model import ReferencefreePIQAModel
from Rf_PIQA.nn.teacher.processing import get_panorama_processor


def collate_batch(batch):
    panoramas = [item["panorama"] for item in batch]
    scores = torch.stack([item["score"] for item in batch])
    panoramas = processor(panoramas, return_tensors="pt").pixel_values
    return panoramas, scores


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python evaluation.py <config-file>"

    with open(sys.argv[1], "r") as f:
        config = yaml.safe_load(f)

    student_config = config["student"]
    
    global processor
    processor = get_panorama_processor(student_config["model_name"], student_config["image_size"])
    
    model = ReferencefreePIQAModel(student_config)

    dataset, _ = load_dataset(config["dataset"], False)

    dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_batch)
    
    model.eval()
    prediction = []
    golden = []

    with torch.no_grad():
        for panoramas, scores in dataloader:
            prediction.append(model(panoramas))
            golden.append(scores)

    prediction = torch.cat(prediction, dim=0)
    golden = torch.cat(golden, dim=0)
    mse = torch.nn.functional.mse_loss(prediction, golden)
    print("Evaluation MSE:", mse.item())
