import os

import PIL
import click
import torch
import numpy as np
import torchvision

import dnnlib
import legacy
from training.gan_diffuse import DiffusionGenerator
from training.gaussian_diffusion import GaussianDiffusion, get_named_beta_schedule, ModelMeanType, ModelVarType, \
    LossType


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
@click.option('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
def run_projection(
    network_pkl: str,
    outdir: str,
    seed: int,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        pkl = legacy.load_network_pkl(fp)
        G = DiffusionGenerator(**pkl['G'].init_kwargs)
        G.load_state_dict(pkl['G_ema'].state_dict())
        G = G.requires_grad_(False).to(device) # type: ignore

    # Build the diffuser
    diffuser = GaussianDiffusion(betas=get_named_beta_schedule('cosine', 2000),
                                      model_mean_type=ModelMeanType.EPSILON,
                                      model_var_type=ModelVarType.FIXED_LARGE,
                                      loss_type=LossType.MSE)

    # Sample
    z = torch.randn(8, 512).to(device)
    sample = diffuser.p_sample_loop(G, (8,3,48,64), model_kwargs={'z': z})
    os.makedirs(outdir, exist_ok=True)
    torchvision.utils.save_image(sample, os.path.join(outdir, 'result.png'))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_projection() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------