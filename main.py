# %%

import torch
import torch.nn as nn
import wandb

wandb.init(project='toy-double-descent')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
# Determinism

torch.manual_seed(0);

# %%
# Hyperparameters and dataset creation. (unkown = unknown in paper)

config = wandb.config
config.dim = 10_000    # N in paper
config.pct = 0.999     # S in paper
config.examples = 1000 # T in paper (unknown)
config.hdim = int(config.dim * (1-config.pct)) # (unknown)

dim, pct, examples, hdim = config.dim, config.pct, config.examples, config.hdim

# Create dataset of sparse vectors
X = torch.rand(examples, dim)
X[torch.rand(examples, dim) < pct] = 0
X = X / torch.norm(X, dim=1, keepdim=True)
X = X.to(device)

# %%
# Create model. essentially an autoencoder where W.T is the unembedding

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.W = nn.Parameter(torch.randn(hdim dim))
        self.b = nn.Parameter(torch.randn(dim))

        # Xavier initialization
        # reasoning: TODO
        nn.init.xavier_uniform_(self.W)
        
    def forward(self, x):
        h = torch.einsum('ij,kj->ki', self.W, x) # [batch, hdim]
        x = torch.einsum('ij,ki->kj', self.W, h) # [batch, dim]
        x = torch.relu(x + self.b)
        return x


def loss_fn(x, y):
    return torch.mean((x - y)**2)


model = Model()
model.to(device)

print(f'num params = {sum(p.numel() for p in model.parameters()) / 1e3:.0f}k')
print(f'loss = {loss_fn(model(X), X):.3f}')
# %%
# Train the model!

optim = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10000):
    optim.zero_grad()
    y = model(X)
    loss = loss_fn(y, X)
    loss.backward()
    optim.step()
    
    if epoch % 10 == 0:
        print(f'epoch {epoch} loss = {loss}')
        U, S, V = torch.svd(model.W)
        wandb.log({
            'loss': loss,
            'norm': torch.norm(model.W, dim=1).mean(),
            'std': model.W.std(),
            'singular_values': wandb.Histogram(S.cpu().detach().numpy()),
        })

# %%
# Save model

torch.save(model.state_dict(), 'model.pt')
artifact = wandb.Artifact('model', type='model')
artifact.add_file('model.pt')
wandb.run.log_artifact(artifact)


