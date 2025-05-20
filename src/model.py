
import torch.nn as nn
   
class EvolvedNN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        # First, flatten any image tensor (N,C,H,W) → (N, C·H·W)
        # Then apply the sequence of MLP layers
        self.model = nn.Sequential(
            nn.Flatten(),
            *layers
        )

    def forward(self, x):
        return self.model(x)