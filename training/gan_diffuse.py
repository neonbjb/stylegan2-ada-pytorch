import copy
import pickle
from collections import OrderedDict

import torch
import torch.nn as nn

import dnnlib
import legacy
from torch_utils import persistence, misc
import numpy as np

# Simplified GLEAN Encoder. Produces two distinct outputs:
# 1) Stack of convolutional inputs produced by a stack of Discriminator blocks.
# 2) Latent inputs w for each of the two convolutions inside of the synthesis network.
from training.networks import DiscriminatorBlock, DiscriminatorEpilogue, Generator, SynthesisLayer, FullyConnectedLayer, \
    ToRGBLayer


def opt_get(d, keys, default):
    for k in keys:
        if k not in d.keys():
            return default
        d = d[k]
    return d


@persistence.persistent_class
class DiffusionEncoder(nn.Module):
    def __init__(self,
                 c_dim,  # Conditioning label (C) dimensionality.
                 img_resolution,  # Input resolution.
                 img_channels,  # Number of input color channels.
                 architecture='orig',  # Architecture: 'orig', 'skip', 'resnet'. Recommend 'orig' for generative side of things.
                 channel_base=32768,  # Overall multiplier for the number of channels.
                 channel_max=512,  # Maximum number of channels in any layer.
                 num_fp16_res=0,  # Use FP16 for the N highest resolutions.
                 conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
                 block_kwargs={},  # Arguments for DiscriminatorBlock.
                 ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_output_resolutions = [2 ** i for i in range(self.img_resolution_log2, 1, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_output_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        first = True
        for output_res in self.block_output_resolutions:
            down_factor = 1 if first else 2
            input_res = output_res * down_factor
            in_channels = 0 if first else channels_dict[input_res]
            tmp_channels = channels_dict[input_res]
            out_channels = channels_dict[output_res]
            use_fp16 = (input_res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=input_res,
                                       first_layer_idx=cur_layer_idx, use_fp16=use_fp16, downsample=not first,
                                       **block_kwargs, **common_kwargs)
            setattr(self, f'conv_enc_{output_res}', block)
            cur_layer_idx += block.num_layers
            first = False

    def forward(self, img, **block_kwargs):
        x = None
        conv_outs = {}
        for output_res in self.block_output_resolutions:
            block = getattr(self, f'conv_enc_{output_res}')
            x, img = block(x, img, **block_kwargs)
            conv_outs[output_res] = x
        return conv_outs


@persistence.persistent_class
class DiffusionGenerator(nn.Module):
    def __init__(self,
                 z_dim,  # Input latent (Z) dimensionality.
                 w_dim,  # Intermediate latent (W) dimensionality.
                 img_resolution,  # Output resolution.
                 img_channels,  # Number of output color channels.
                 mapping_kwargs={},  # Arguments for MappingNetwork.
                 synthesis_kwargs={},  # Arguments for SynthesisNetwork.
                 enc_block_kwargs={},
                 ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channel_base = opt_get(synthesis_kwargs, ['channel_base'], 32768)
        channel_max = opt_get(synthesis_kwargs, ['channel_max'], 512)
        self.encoder = DiffusionEncoder(0, img_resolution, img_channels, channel_base=channel_base,
                                        channel_max=channel_max, block_kwargs=enc_block_kwargs)
        synthesis_kwargs['encoder_attachment_resolutions'] = [2 ** i for i in range(self.img_resolution_log2+1, 2, -1)]
        self.gen_bank = Generator(z_dim, w_dim, img_resolution, img_channels, mapping_kwargs, synthesis_kwargs)

    def do_encoder(self, prior, **synthesis_kwargs):
        block_args = opt_get(synthesis_kwargs, ['block_kwargs'], {})
        conv_outs = self.encoder(prior, **block_args)
        return conv_outs

    def do_latent_mapping_for_single(self, z, truncation_psi=1, truncation_cutoff=None,
                                     skip_w_avg_update=False):
        ws = self.gen_bank.mapping(z, skip_trunc_and_broadcast=True, skip_w_avg_update=skip_w_avg_update)
        ws = self.gen_bank.mapping.apply_truncation_and_broadcast(ws, truncation_psi=truncation_psi,
                                                                  truncation_cutoff=truncation_cutoff)
        return ws

    def do_latent_mapping_with_mixing(self, z, mixing_prob, truncation_psi=1, truncation_cutoff=None):
        ws = self.do_latent_mapping_for_single(z, truncation_psi, truncation_cutoff)
        if mixing_prob > 0:
            cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
            cutoff = torch.where(torch.rand([], device=ws.device) < mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
            ws[:, cutoff:] = self.do_latent_mapping_for_single(torch.randn_like(z),
                                                               truncation_psi=truncation_psi,
                                                               truncation_cutoff=truncation_cutoff,
                                                               skip_w_avg_update=True)[:, cutoff:]
        return ws

    # To assist with style mixing, this can be called in one of two ways:
    # 1) Let this function "do everything" for you:
    #    img = gen(z=<z>, prior=<prior>)
    # 2) Perform your own mapping and feed in the ws:
    #    enc_c = gen.do_encoder(prior=<prior>)
    #    ws = gen.do_latent_mapping_for_single(z=<z>)  # Or create your own ws.
    #    img = gen(ws=ws, enc_conv_outs=enc_c)
    def forward(self, z=None, c=None, prior=None, ws=None, enc_conv_outs=None, truncation_psi=1,
                truncation_cutoff=None, return_ws=False, **synthesis_kwargs):
        force_fp32 = opt_get(synthesis_kwargs, ['force_fp32'], False)
        block_args = opt_get(synthesis_kwargs, ['block_kwargs'], {'force_fp32': force_fp32})

        assert (z is not None and prior is not None) or (ws is not None and enc_conv_outs is not None)
        if prior is not None:
            conv_outs = self.do_encoder(prior, **synthesis_kwargs)
        else:
            conv_outs = enc_conv_outs
        if z is not None:
            ws = self.do_latent_mapping_for_single(z, truncation_psi, truncation_cutoff)

        # Split the latents.
        synth = self.gen_bank.synthesis
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, synth.num_ws, synth.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in synth.block_resolutions:
                block = getattr(synth, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        # Manually propagate the synthesis network.
        x = img = None
        for res, cur_ws in zip(synth.block_resolutions, block_ws):
            first = res == 4
            eres = res // 2
            if not first and eres in conv_outs.keys():
                x = torch.cat([x, conv_outs[eres]], dim=1)
            synblock = getattr(synth, f'b{res}')
            x, img = synblock(x, img, cur_ws, **block_args)
        if return_ws:
            return img, ws
        else:
            return img


if __name__ == '__main__':
    args = {
        'z_dim': 512,
        'w_dim': 512,
        'img_resolution': 64,
        'img_channels': 3,
        'mapping_kwargs': {'num_layers': 8},
        'synthesis_kwargs': {
            'channel_base': 16384,
            'channel_max': 512,
            'num_fp16_res': 4,
            'conv_clamp': 256
        }
    }
    gen = DiffusionGenerator(**args).to('cuda')
    z = torch.rand((1,512)).to('cuda')
    c = torch.zeros((1,0)).to('cuda')
    prior = torch.rand((1,3,48,64)).to('cuda')
    #gen(z, None, prior, truncation_psi=1, truncation_cutoff=None)
    # Also test the style mixing method
    enc_c = gen.do_encoder(prior=prior)
    ws = gen.do_latent_mapping_with_mixing(z=z,  mixing_prob=1)
    img = gen(ws=ws, enc_conv_outs=enc_c)
    #convert_stylegan2_pickle('results/imgset_256.pkl', 32)
