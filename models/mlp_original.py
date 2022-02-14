import torch.nn as nn

class MLPModel(nn.Module):
    """
    Take float115_v2 (115 dimension vector) as input
    """
    def __init__(self, hidden_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(115, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
        )
        self.actor = nn.Linear(hidden_size, 19)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = self.mlp(x)
        logits = self.actor(x)
        return logits