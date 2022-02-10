import torch.nn as nn

class MLPModel(nn.Module):
    """
    Take float115_v2 (115 dimension vector) as input
    """
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(115, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.actor = nn.Linear(256, 19)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = self.mlp(x)
        logits = self.actor(x)
        return logits