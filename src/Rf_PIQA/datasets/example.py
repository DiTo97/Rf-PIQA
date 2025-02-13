import torch
from torch.utils.data import Dataset


class ExamplePIQADataset(Dataset):
    """
    A dummy dataset that returns a high-res image, a list of low-res images,
    and a target quality score.
    """
    def __init__(self, num_samples=100, constituents=True):
        self.num_samples = num_samples
        self.constituents = constituents
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        panorama = torch.randint(0, 256, (3, 1024, 1024), dtype=torch.uint8)
        score = torch.rand(1)  # a random quality score in [0,1]

        record = {"panorama": panorama, "score": score}

        if self.constituents:
            num_constituents = torch.randint(3, 7, (1,)).item()
            constituents = [torch.randint(0, 256, (3, 500, 500), dtype=torch.uint8) for _ in range(num_constituents)]
            record["constituents"] = constituents
        
        return record
