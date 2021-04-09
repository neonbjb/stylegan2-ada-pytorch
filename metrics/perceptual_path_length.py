﻿# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Perceptual Path Length (PPL) from the paper "A Style-Based Generator
Architecture for Generative Adversarial Networks". Matches the original
implementation by Karras et al. at
https://github.com/NVlabs/stylegan/blob/master/metrics/perceptual_path_length.py"""

import copy
import numpy as np
import torch
from torch.utils.data import DataLoader

import dnnlib
from . import metric_utils

#----------------------------------------------------------------------------

# Spherical interpolation of a batch of vectors.
def slerp(a, b, t):
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    d = (a * b).sum(dim=-1, keepdim=True)
    p = t * torch.acos(d)
    c = b - d * a
    c = c / c.norm(dim=-1, keepdim=True)
    d = a * torch.cos(p) + c * torch.sin(p)
    d = d / d.norm(dim=-1, keepdim=True)
    return d

#----------------------------------------------------------------------------

class PPLSampler(torch.nn.Module):
    def __init__(self, G, G_kwargs, latent_encoder, epsilon, space, sampling, crop, vgg16, dataset_kwargs):
        assert space in ['z', 'w']
        assert sampling in ['full', 'end']
        super().__init__()
        self.G = copy.deepcopy(G)
        self.latent_encoder = latent_encoder
        self.G_kwargs = G_kwargs
        self.from_layer = G.enc_input_resolution_log2+1
        self.epsilon = epsilon
        self.space = space
        self.sampling = sampling
        self.crop = crop
        self.vgg16 = copy.deepcopy(vgg16)
        dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs)
        # TODO: specify batch_size properly.
        self.dataloader = DataLoader(dataset=dataset, batch_size=2, pin_memory=True, num_workers=2, prefetch_factor=2)
        self.dliter = iter(self.dataloader)

    def forward(self, c):
        # Fetch bs lq images
        bs = c.shape[0]  # Note: c is otherwise unused.
        lq = torch.empty((0,))
        while lq.shape[0] < bs:
            ims, lqs, lbls = next(self.dliter)
            lq = torch.cat([lqs, lq], dim=0).to(c.device)
        enc_conv = self.G.do_encoder(lq, force_fp32=True, **self.G_kwargs)
        enc_latent = self.latent_encoder(torch.nn.functional.interpolate(lq, size=(224,224), mode='bilinear', align_corners=False))

        # Generate random latents and interpolation t-values.
        t = torch.rand([c.shape[0]], device=c.device) * (1 if self.sampling == 'full' else 0)
        z0, z1 = torch.randn([c.shape[0] * 2, self.G.gen_bank.z_dim], device=c.device).chunk(2)

        # Interpolate in W or Z.
        if self.space == 'w':
            w0 = self.G.do_latent_mapping_for_single(z0, enc_latent)
            w1 = self.G.do_latent_mapping_for_single(z1, enc_latent)
            wt0 = w0.lerp(w1, t.unsqueeze(1).unsqueeze(2))
            wt1 = w0.lerp(w1, t.unsqueeze(1).unsqueeze(2) + self.epsilon)
        else: # space == 'z'
            zt0 = slerp(z0, z1, t.unsqueeze(1))
            zt1 = slerp(z0, z1, t.unsqueeze(1) + self.epsilon)
            wt0 = self.G.do_latent_mapping_for_single(zt0, enc_latent)
            wt1 = self.G.do_latent_mapping_for_single(zt1, enc_latent)

        # Randomize noise buffers.
        for name, buf in self.G.named_buffers():
            if name.endswith('.noise_const'):
                buf.copy_(torch.randn_like(buf))

        # Generate images.
        enc_conv = {k:v.repeat(2,1,1,1) for k,v in enc_conv.items()}

        img = self.G(ws=torch.cat([wt0,wt1]), enc_conv_outs=enc_conv, force_fp32=True, noise_mode='const', **self.G_kwargs)

        # Center crop.
        if self.crop:
            assert img.shape[2] == img.shape[3]
            c = img.shape[2] // 8
            img = img[:, :, c*3 : c*7, c*2 : c*6]

        # Downsample to 256x256.
        factor = self.G.img_resolution // 256
        if factor > 1:
            img = img.reshape([-1, img.shape[1], img.shape[2] // factor, factor, img.shape[3] // factor, factor]).mean([3, 5])

        # Scale dynamic range from [-1,1] to [0,255].
        img = (img + 1) * (255 / 2)
        if self.G.img_channels == 1:
            img = img.repeat([1, 3, 1, 1])

        # Evaluate differential LPIPS.
        lpips_t0, lpips_t1 = self.vgg16(img, resize_images=False, return_lpips=True).chunk(2)
        dist = (lpips_t0 - lpips_t1).square().sum(1) / self.epsilon ** 2
        return dist

#----------------------------------------------------------------------------

def compute_ppl(opts, num_samples, epsilon, space, sampling, crop, batch_size, jit=False):
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    vgg16_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    vgg16 = metric_utils.get_feature_detector(vgg16_url, num_gpus=opts.num_gpus, rank=opts.rank, verbose=opts.progress.verbose)

    # Setup sampler.
    sampler = PPLSampler(G=opts.G, G_kwargs=opts.G_kwargs, latent_encoder=opts.latent_encoder, epsilon=epsilon, space=space, sampling=sampling, crop=crop, vgg16=vgg16, dataset_kwargs=opts.dataset_kwargs)
    sampler.eval().requires_grad_(False).to(opts.device)
    if jit:
        sampler = torch.jit.trace(sampler, [None], check_trace=False)

    # Sampling loop.
    dist = []
    progress = opts.progress.sub(tag='ppl sampling', num_items=num_samples)
    for batch_start in range(0, num_samples, batch_size * opts.num_gpus):
        progress.update(batch_start)
        c = [dataset.get_label(np.random.randint(len(dataset))) for _i in range(batch_size)]
        c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)
        x = sampler(c)
        for src in range(opts.num_gpus):
            y = x.clone()
            if opts.num_gpus > 1:
                torch.distributed.broadcast(y, src=src)
            dist.append(y)
    progress.update(num_samples)

    # Compute PPL.
    if opts.rank != 0:
        return float('nan')
    dist = torch.cat(dist)[:num_samples].cpu().numpy()
    lo = np.percentile(dist, 1, interpolation='lower')
    hi = np.percentile(dist, 99, interpolation='higher')
    ppl = np.extract(np.logical_and(dist >= lo, dist <= hi), dist).mean()
    return float(ppl)

#----------------------------------------------------------------------------
