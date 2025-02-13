import torch.nn as nn


class Point(nn.Module):
    """
    A simple regression head that maps features to a single quality score.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        return self.fc(x)
