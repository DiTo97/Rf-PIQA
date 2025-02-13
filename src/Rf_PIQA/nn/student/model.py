import torch
import torch.nn as nn
from transformers import Swinv2Model

from Rf_PIQA.nn.heads.point import Point
from Rf_PIQA.nn.heads.predintervals import PIVEN


class ReferencefreePIQAModel(nn.Module):
    """The reference-free (student) PIQA model only processing a panorama (high-res image)
    
    The features are combined via a vision encoder, and passed through a configurable head (point-estimate or PIVEN).
    """
    def __init__(self, config: dict):
        super().__init__()
        self.model = Swinv2Model.from_pretrained(config["model_name"])
        self.hidden_size = config.get("hidden_size", self.model.config.hidden_size)
        head_type = config.get("head_type", "point")
        
        if head_type == "point":
            self.head = Point(self.hidden_size)
        elif head_type == "PIVEN":
            self.head = PIVEN(self.hidden_size)
        else:
            raise ValueError("Unknown head type: {}".format(head_type))
        
        self.to(config.get("device", "cpu"))
        
        if "checkpoint" in config:
            self.load_state_dict(torch.load(config["checkpoint"]))
    
    def forward(self, images):
        features = self.model(images).last_hidden_state.mean(dim=1)
        output = self.head(features)
        return output
