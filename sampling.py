from denoising_diffusion_PyTorch import Unet, GaussianDiffusion, Trainer
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import denoising_diffusion_PyTorch

device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
).to(device)

diffusion = denoising_diffusion_PyTorch.GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,           # number of steps
).to(device)

model_path = './results'
model_path = Path(model_path)

data = torch.load((str(model_path / f'my_model-15.pt')), map_location=device)
diffusion.load_state_dict(data['model'])

trainer = Trainer(
    diffusion,
    '/home/cvintern3/Desktop/DDPM_project/images',
    train_batch_size = 32,
    train_lr = 8e-3,
    train_num_steps = 7000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
)

trainer.sampling(1211111)