# %%

import torch
import wandb
from model import Model, config, get_dataset, loss_fn
from torch.optim.lr_scheduler import LambdaLR

wandb.init(project='toy-double-descent')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for key in config:
    wandb.config[key] = config[key]

X = get_dataset(dim=config['dim'], pct=config['pct'], examples=config['examples'])
X = X.to(device)

model = Model()
model.to(device)

print(f'num params = {sum(p.numel() for p in model.parameters()) / 1e3:.0f}k')
print(f'loss = {loss_fn(model(X), X):.3f}')

# %%
# Train the model!

optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

for epoch in range(config['steps']):
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
