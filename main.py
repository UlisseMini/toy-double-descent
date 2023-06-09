# %%

import torch
import wandb
from model import Model, config, get_dataset, loss_fn
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
from tqdm import tqdm

# %%

wandb.init(project='toy-double-descent')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.has_mps:
    device = 'mps'

for key in config:
    wandb.config[key] = config[key]

X = get_dataset(dim=config['dim'], pct=config['pct'], examples=config['examples'])
X = X.to(device)

model = Model(dim=config['dim'], hdim=config['hdim'])
model.to(device)

print(f'num params = {sum(p.numel() for p in model.parameters()) / 1e3:.0f}k')
print(f'loss = {loss_fn(model(X), X):.3f}')

optim = torch.optim.AdamW(model.parameters(), lr=3e-2, weight_decay=5e-2) # paper uses decay of 1e-2

# %%
# Train the model!

pbar = tqdm(range(1, config['steps']+1))
for epoch in pbar:
    optim.zero_grad()
    y = model(X)
    loss = loss_fn(y, X)
    loss.backward()
    optim.step()

    
    if epoch % 100 == 0:
        pbar.set_description(f'epoch {epoch} loss {loss:.3e}')
        U, S, V = torch.svd(model.W.cpu())
        # TODO: better solution, this is ugly
        h = torch.einsum('ij,kj->ki', model.W, X).cpu().detach().numpy()
        W = model.W.cpu().detach().numpy()

        # Plot weights and hidden vectors in two columns
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f'epoch {epoch} loss {loss:.2e}')

        # Plot features (weights)
        ax[0].set_xlim(-2, 2); ax[0].set_ylim(-2, 2); ax[0].set_aspect('equal')
        ax[0].scatter(W[0, :], W[1, :])
        ax[0].set_title('Features')

        # Plot datapoints (hidden vectors)
        ax[1].set_xlim(-4, 4); ax[1].set_ylim(-4, 4); ax[1].set_aspect('equal')
        ax[1].scatter(h[:, 0], h[:, 1])
        ax[1].set_title('Datapoints')

        wandb.log({
            'loss': loss,
            'norm': torch.norm(model.W, dim=1).mean(),
            'std': model.W.std(),
            'singular_values': wandb.Histogram(S.cpu().detach().numpy()),
            'plots': wandb.Image(fig),
        })
        plt.close(fig)


# %%
# Save model

torch.save(model.state_dict(), 'model.pt')
artifact = wandb.Artifact('model', type='model')
artifact.add_file('model.pt')
wandb.run.log_artifact(artifact)

# %%
# Generate a gif from saved wandb images

from glob import glob

image_files = glob('wandb/latest-run/files/media/images/plots_*')
image_files.sort(key=lambda x: int(x.split('_')[1])) # sort by step

# Create gif
import imageio
images = [imageio.imread(f) for f in image_files]

imageio.mimsave('features.gif', images, duration=1/30)

# Embed gif in jupyter
from IPython.display import Image
Image(filename='features.gif')


# %%
