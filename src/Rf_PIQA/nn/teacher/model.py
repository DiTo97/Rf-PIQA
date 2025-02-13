import torch
import torch.nn as nn
from transformers import Swinv2Model

from Rf_PIQA.nn.heads.point import Point
from Rf_PIQA.nn.heads.predintervals import PIVEN


class ReferencePIQAModel(nn.Module):
    """The reference panorama IQA model processing a high-resolution panorama and a batch of low-resolution constituents.
    
    The features are combined via subtraction, and passed through a configurable head (point-estimate or PIVEN).
    """
    def __init__(self, config):
        super().__init__()
        self.high_res_model = Swinv2Model.from_pretrained(config["high_res_model_name"])
        self.low_res_model = Swinv2Model.from_pretrained(config["low_res_model_name"])
        self.hidden_size = config.get("hidden_size", self.high_res_model.config.hidden_size)
        head_type = config.get("head_type", "point")
        
        if head_type == "point":
            self.head = Point(self.hidden_size)
        elif head_type == "PIVEN":
            self.head = PIVEN(self.hidden_size)
        else:
            raise ValueError("Unknown head_type: {}".format(head_type))
        
        self.to(config.get("device", "cpu"))
        
        if "checkpoint" in config:
            self.load_state_dict(torch.load(config["checkpoint"]))

    def forward(self, high_res_images, low_res_images_batch, padding_mask):
        # Process high-res images
        high_res_features = self.high_res_model(high_res_images).last_hidden_state.mean(dim=1)
        # Process low-res images
        batch_size, max_num_low_res, C, H, W = low_res_images_batch.shape
        low_res_images_flattened = low_res_images_batch.view(-1, C, H, W)
        low_res_features_flattened = self.low_res_model(low_res_images_flattened).last_hidden_state.mean(dim=1)
        low_res_features = low_res_features_flattened.view(batch_size, max_num_low_res, -1)
        # Apply mask to ignore padded entries
        low_res_features_masked = low_res_features * padding_mask.unsqueeze(-1)
        low_res_features_mean = low_res_features_masked.sum(dim=1) / padding_mask.sum(dim=1, keepdim=True)
        # Combine features (using a “diff” feature)
        combined_features = high_res_features - low_res_features_mean
        # Pass through the head
        output = self.head(combined_features)
        return output
