import torch
from torch import nn


class Surrogate(nn.Module):
    def __init__(self, cells=256, width=128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(4, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, cells),
        )

    def forward(self, inputs):
        scores = self.network(inputs)

        return torch.log_softmax(scores, dim=-1)