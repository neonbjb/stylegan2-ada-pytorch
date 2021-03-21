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
class GleanEncoder(nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 0,        # Use FP16 for the N highest resolutions.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        epilogue_kwargs     = {},       # Arguments for the "epilogue" - in this case, the latent encoder.
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
            out_channels = channels_dict[input_res // 2]
            use_fp16 = (input_res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=input_res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, downsample=not first, **block_kwargs, **common_kwargs)
            setattr(self, f'conv_enc_{output_res}', block)
            cur_layer_idx += block.num_layers
            first = False

        self.latent_encoder_final = DiscriminatorEpilogue(channels_dict[4], cmap_dim=channel_max, resolution=4,
                                                          **epilogue_kwargs, **common_kwargs)

    def forward(self, img, **block_kwargs):
        x = None
        conv_outs = {}
        for output_res in self.block_output_resolutions:
            block = getattr(self, f'conv_enc_{output_res}')
            x, img = block(x, img, **block_kwargs)
            conv_outs[output_res] = x
        lat = self.latent_encoder_final(x, img, None)
        return conv_outs, lat

@persistence.persistent_class
class GleanGenerator(nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        enc_input_resolution,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        synthesis_kwargs    = {},   # Arguments for SynthesisNetwork.
        enc_block_kwargs    = {},
        enc_epilogue_kwargs = {}
    ):
        super().__init__()
        channel_base = opt_get(synthesis_kwargs, ['channel_base'], 32768)
        channel_max = opt_get(synthesis_kwargs, ['channel_max'], 512)
        self.encoder = GleanEncoder(c_dim, enc_input_resolution, img_channels, channel_base=channel_base,
                                    channel_max=channel_max, block_kwargs=enc_block_kwargs,
                                    epilogue_kwargs=enc_epilogue_kwargs)
        self.gen_bank = Generator(z_dim, c_dim, w_dim, img_resolution, img_channels, mapping_kwargs, synthesis_kwargs)
        for p in self.gen_bank.parameters():
            p.requires_grad = True
        self.gen_bank.eval()

        # Generative bank attachments.
        conv_clamp = opt_get(synthesis_kwargs, ['block_kwargs', 'conv_clamp'], None)
        channels_last = opt_get(synthesis_kwargs, ['block_kwargs', 'channels_last'], None)
        layer_kwargs = opt_get(synthesis_kwargs, ['block_kwargs', 'layer_kwargs'], {})
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.enc_input_resolution_log2 = int(np.log2(enc_input_resolution))
        self.img_channels = img_channels
        self.enc_attachment_resolutions = [2 ** i for i in range(self.enc_input_resolution_log2, 1, -1)]
        self.dec_attachment_resolutions = [2 ** i for i in range(self.img_resolution_log2, self.enc_input_resolution_log2-1, -1)]
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}

        # Encoder attachments first
        self.enc_w_combiner = FullyConnectedLayer(w_dim*2, w_dim, bias_init=0)
        for res in self.enc_attachment_resolutions:
            first = res == 4
            out_channels = channels_dict[res]
            in_channels = out_channels * 2  # The input comes from both the previous gen bank output and the level-wise encoder output.
            layer = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=res, conv_clamp=conv_clamp,
                                   channels_last=channels_last, **layer_kwargs)
            setattr(self, f'enc_attachment_{res}', layer)

        # Decoder attachments second
        for res in self.dec_attachment_resolutions:
            in_channels = channels_dict[res]
            out_channels = channels_dict[res]
            layer = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=res, conv_clamp=conv_clamp,
                                   channels_last=channels_last, **layer_kwargs)
            setattr(self, f'dec_attachment_{res}', layer)
        self.dec_torgb = ToRGBLayer(channels_dict[max(self.dec_attachment_resolutions)], img_channels, w_dim=w_dim, conv_clamp=conv_clamp, channels_last=channels_last)


    def forward(self, z, c, lq, truncation_psi=1, truncation_cutoff=None, return_ws=False, **synthesis_kwargs):
        block_args = opt_get(synthesis_kwargs, ['block_kwargs'], {})
        force_fp32 = opt_get(synthesis_kwargs, ['force_fp32'], False)
        layer_args = opt_get(block_args, ['layer_kwargs'], {})

        # TODO: Optionally turn on grad.
        with torch.no_grad():
            ws = self.gen_bank.mapping(z, c, skip_trunc_and_broadcast=True)

        conv_outs, latent = self.encoder(lq, **block_args)
        wse = self.enc_w_combiner(torch.cat([ws, latent], dim=-1))
        # `wse` is used for the encoder blocks, while `ws` is used for the decoder blocks.
        ws = self.gen_bank.mapping.apply_truncation_and_broadcast(ws, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        wse = self.gen_bank.mapping.apply_truncation_and_broadcast(wse, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)

        # Manually propagate the synthesis network.
        synth = self.gen_bank.synthesis
        block_ws, block_wse = [], []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(wse, [None, synth.num_ws, synth.w_dim])
            ws = ws.to(torch.float32)
            wse = wse.to(torch.float32)
            w_idx = 0
            for res in synth.block_resolutions:
                block = getattr(synth, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                block_wse.append(wse.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        for res, cur_ws, cur_wse in zip(synth.block_resolutions, block_ws, block_wse):
            first = res == 4
            did_enc = False
            synblock = getattr(synth, f'b{res}')
            if first:
                x, img = synblock(x, img, cur_wse, **block_args)
            else:
                # First, convert to the correct memory format and dtype. The generator would normally have done this for us, but we're prepending it.
                dtype = torch.float16 if synblock.use_fp16 and not force_fp32 else torch.float32
                memory_format = torch.channels_last if synblock.channels_last and not force_fp32 else torch.contiguous_format
                x = x.to(dtype=dtype, memory_format=memory_format)

                # Encoder attachment
                eres = res // 2
                if eres in conv_outs.keys():
                    did_enc = True
                    x = torch.cat([x, conv_outs[eres].to(dtype=dtype, memory_format=memory_format)], dim=1)
                    #x = torch.cat([x, conv_outs[eres]], dim=1)
                    layer = getattr(self, f'enc_attachment_{eres}')
                    enc_w = cur_wse[:,0,:]  # Just use the first block's w for the encoder attachment.
                    x = layer(x, enc_w, **layer_args)

                # Generative bank
                lws = cur_wse if did_enc else cur_ws
                # TODO: Optionally turn on grad.
                with torch.no_grad():
                    x, img = synblock(x, img, lws, puppet=True, **block_args)

                # Decoder attachment
                if res in self.dec_attachment_resolutions:
                    layer = getattr(self, f'dec_attachment_{res}')
                    dec_w = lws[:,1,:]
                    x = layer(x, dec_w, **layer_args)

        with misc.suppress_tracer_warnings(): # this value will be treated as a constant
            fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)
        y = self.dec_torgb(x, lws[:,1,:], fused_modconv=fused_modconv)
        y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
        img = img.add_(y)
        if return_ws:
            return img, ws
        else:
            return img


def convert_stylegan2_pickle(pkl, enc_input_resolution):
    with dnnlib.util.open_url(pkl) as f:
        resume_data = legacy.load_network_pkl(f)
    G = dnnlib.util.construct_class_by_name(class_name='training.glean_networks.GleanGenerator', enc_input_resolution=enc_input_resolution, **resume_data['G'].init_kwargs)
    G_ema = copy.deepcopy(G).eval()
    for name, module in [('G_ema', G.gen_bank), ('G_ema', G_ema.gen_bank)]:   # Note: both G and G_ema start from the saved G_ema intentionally.
        module.load_state_dict(resume_data[name].state_dict(), strict=True)

    # Discriminator is a special case - we want to copy the parameters from the lower half of the network, but leave
    # the new init in the upper half.
    D = dnnlib.util.construct_class_by_name(class_name='training.networks.Discriminator', stop_training_at=enc_input_resolution, **resume_data['D'].init_kwargs)
    pretrained_sd = resume_data['D'].state_dict()
    scrubbed_sd = OrderedDict()
    for k,v in pretrained_sd.items():
        block_res = int(k.split('.')[0][1:])
        if block_res <= enc_input_resolution:  # This intentionally mismatches with the Discriminator "stop_training_at" implementation: We load the pretrained weights *at* the encoder dimension *and* train at that dimension.
            scrubbed_sd[k] = v
    D.load_state_dict(scrubbed_sd, strict=False)

    # Save snapshot.
    snapshot_data = dict(training_set_kwargs=resume_data['training_set_kwargs'])
    for name, module in [('G', G), ('D', D), ('G_ema', G_ema), ('augment_pipe', resume_data['augment_pipe'])]:
        if module is not None:
            module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
        snapshot_data[name] = module
        del module # conserve memory
    with open('converted-0.pkl', 'wb') as f:
        pickle.dump(snapshot_data, f)



if __name__ == '__main__':
    '''
    args = {
        'z_dim': 512,
        'c_dim': 0,
        'w_dim': 512,
        'img_resolution': 256,
        'img_channels': 3,
        'enc_input_resolution': 64,
        'mapping_kwargs': {'num_layers': 8},
        'synthesis_kwargs': {
            'channel_base': 32768,
            'channel_max': 512,
            'num_fp16_res': 4,
            'conv_clamp': 256
        }
    }
    gen = GleanGenerator(**args).to('cuda')
    z = torch.rand((1,512)).to('cuda')
    c = torch.zeros((1,0)).to('cuda')
    lq = torch.rand((1,3,64,64)).to('cuda')
    gen(z, c, lq, truncation_psi=1, truncation_cutoff=None)
    '''
    convert_stylegan2_pickle('results/imgset_256.pkl', 32)