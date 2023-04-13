
import torch
import torch.nn as nn

# %%

config = {}

config['dim'] = 10_000    # N in paper
config['pct'] = 0.999     # S in paper
config['examples'] = 1000 # T in paper (unknown)
# config.hdim = int(config.dim * (1-config.pct)) # (unknown)
config['hdim'] = 2

dim, pct, examples, hdim = config['dim'], config['pct'], config['examples'], config['hdim']

# %%

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.W = nn.Parameter(torch.randn(hdim, dim))
        self.b = nn.Parameter(torch.randn(dim))

        # Xavier initialization
        # reasoning: TODO
        nn.init.xavier_uniform_(self.W)
        
    def forward(self, x):
        h = torch.einsum('ij,kj->ki', self.W, x) # [batch, hdim]
        x = torch.einsum('ij,ki->kj', self.W, h) # [batch, dim]
        x = torch.relu(x + self.b)
        return x