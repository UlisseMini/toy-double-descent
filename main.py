# %%

import torch
import wandb
from model import Model, config, get_dataset, loss_fn
from torch.optim.lr_scheduler import LambdaLR

wandb.init(project='toy-double-descent')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for key in config:
    wandb.config[key] = config[key]

X = get_dataset()
X = X.to(device)

model = Model()
model.to(device)

print(f'num params = {sum(p.numel() for p in model.parameters()) / 1e3:.0f}k')
print(f'loss = {loss_fn(model(X), X):.3f}')

# %%
# Train the model!
# TODO: Reproduce double descent

# "Our learning rate schedule includes a 2,500 step linear-warmup to 1e-3, followed by a cosine-decay to zero."
lr = 1e-3
optim = torch.optim.AdamW(model.parameters(), lr=lr)

# Custom learning rate scheduler
def custom_lr_scheduler(optimizer, step, warmup_steps=2500, lr_start=1e-3, final_lr=0):
    if step < warmup_steps:
        lr = lr_start * (step / warmup_steps)
    else:
        lr = final_lr + (lr_start - final_lr) * 0.5 * (1 + torch.cos((step - warmup_steps) / (config['steps'] - warmup_steps) * 3.14159))


    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


lr_scheduler = LambdaLR(optim, lambda step: custom_lr_scheduler(optim, step))


for epoch in range(config['steps']):
    optim.zero_grad()
    y = model(X)
    loss = loss_fn(y, X)
    loss.backward()
    optim.step()
    lr_scheduler.step()
    
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
