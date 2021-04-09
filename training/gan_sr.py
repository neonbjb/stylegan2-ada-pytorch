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
class SrEncoder(nn.Module):
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
                 epilogue_kwargs={},  # Arguments for the "epilogue" - in this case, the latent encoder.
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

        self.latent_encoder_final = DiscriminatorEpilogue(channels_dict[4], cmap_dim=channel_max, resolution=4,
                                                          **epilogue_kwargs, **common_kwargs)

    def forward(self, img, no_latent=False, **block_kwargs):
        x = None
        conv_outs = {}
        for output_res in self.block_output_resolutions:
            block = getattr(self, f'conv_enc_{output_res}')
            x, img = block(x, img, **block_kwargs)
            conv_outs[output_res] = x
        if no_latent:
            lat = None
        else:
            lat = self.latent_encoder_final(x, img, None)
        return conv_outs, lat


@persistence.persistent_class
class SrGenerator(nn.Module):
    def __init__(self,
                 z_dim,  # Input latent (Z) dimensionality.
                 w_dim,  # Intermediate latent (W) dimensionality.
                 img_resolution,  # Output resolution.
                 img_channels,  # Number of output color channels.
                 enc_input_resolution,
                 freeze_latent_dict=True,  # Whether or not the latent dict should be trained.
                 freeze_mapping_network=True,
                 mapping_kwargs={},  # Arguments for MappingNetwork.
                 synthesis_kwargs={},  # Arguments for SynthesisNetwork.
                 enc_block_kwargs={},
                 enc_epilogue_kwargs={}
                 ):
        super().__init__()
        self.freeze_synthesis_network = freeze_latent_dict
        self.freeze_mapping_network = freeze_mapping_network
        channel_base = opt_get(synthesis_kwargs, ['channel_base'], 32768)
        channel_max = opt_get(synthesis_kwargs, ['channel_max'], 512)
        self.encoder = SrEncoder(0, enc_input_resolution, img_channels, channel_base=channel_base,
                                    channel_max=channel_max, block_kwargs=enc_block_kwargs,
                                    epilogue_kwargs=enc_epilogue_kwargs)
        self.gen_bank = Generator(z_dim, w_dim, w_dim, img_resolution, img_channels, mapping_kwargs, synthesis_kwargs)

        # StyleGAN Generator Attachments
        conv_clamp = opt_get(synthesis_kwargs, ['block_kwargs', 'conv_clamp'], None)
        channels_last = opt_get(synthesis_kwargs, ['block_kwargs', 'channels_last'], None)
        layer_kwargs = opt_get(synthesis_kwargs, ['block_kwargs', 'layer_kwargs'], {})
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.enc_input_resolution_log2 = int(np.log2(enc_input_resolution))
        self.img_channels = img_channels
        self.enc_attachment_resolutions = [2 ** i for i in range(self.enc_input_resolution_log2, 1, -1)]
        self.dec_attachment_resolutions = [2 ** i for i in
                                           range(self.img_resolution_log2, self.enc_input_resolution_log2 - 1, -1)]
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}

        # Encoder attachments
        self.enc_w_combiner = FullyConnectedLayer(w_dim * 2, w_dim, bias_init=0)
        for res in self.enc_attachment_resolutions:
            out_channels = channels_dict[res]
            in_channels = out_channels * 2  # The input comes from both the previous gen bank output and the level-wise encoder output.
            layer = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=res, conv_clamp=conv_clamp,
                                   channels_last=channels_last, **layer_kwargs)
            setattr(self, f'enc_attachment_{res}', layer)

    def do_encoder(self, lq, **synthesis_kwargs):
        block_args = opt_get(synthesis_kwargs, ['block_kwargs'], {})
        conv_outs, latent = self.encoder(lq, **block_args)
        return conv_outs, latent

    def do_encoder_no_latent(self, lq, **synthesis_kwargs):
        block_args = opt_get(synthesis_kwargs, ['block_kwargs'], {})
        conv_outs, latent = self.encoder(lq, no_latent=True, **block_args)
        return conv_outs, latent

    def do_latent_mapping_for_single(self, z, enc_latent, truncation_psi=1, truncation_cutoff=None,
                                     skip_w_avg_update=False):
        if self.freeze_mapping_network:
            for p in self.gen_bank.mapping.parameters():
                p.requires_grad = False
            self.gen_bank.mapping.eval()

        ws = self.gen_bank.mapping(z, enc_latent, skip_trunc_and_broadcast=True, skip_w_avg_update=skip_w_avg_update)
        ws = self.gen_bank.mapping.apply_truncation_and_broadcast(ws, truncation_psi=truncation_psi,
                                                                  truncation_cutoff=truncation_cutoff)
        return ws

    # Performs style mixing only within the SR region of the generator.
    def do_latent_mapping_with_mixing(self, z, enc_latent, mixing_prob, truncation_psi=1, truncation_cutoff=None):
        ws = self.do_latent_mapping_for_single(z, enc_latent, truncation_psi, truncation_cutoff)
        if mixing_prob > 0:
            enc_split_ind = (self.enc_input_resolution_log2-2)*2
            cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(enc_split_ind + 1, ws.shape[1])
            cutoff = torch.where(torch.rand([], device=ws.device) < mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
            ws[:, cutoff:] = self.do_latent_mapping_for_single(torch.randn_like(z), enc_latent=enc_latent,
                                                               truncation_psi=truncation_psi,
                                                               truncation_cutoff=truncation_cutoff,
                                                               skip_w_avg_update=True)[:, cutoff:]
        return ws

    # To assist with style mixing, this can be called in one of two ways:
    # 1) Let this function "do everything" for you:
    #    img = gen(z=<z>, lq=<lq>)
    # 2) Perform your own mapping and feed in the ws:
    #    enc_c, enc_l = gen.do_encoder(lq=<lq>)
    #    ws = gen.do_latent_mapping_for_single(z=<z>, enc_latent=enc_l)  # Or create your own ws.
    #    img = gen(ws=ws, enc_conv_outs=enc_c)
    def forward(self, z=None, c=None, lq=None, ws=None, enc_conv_outs=None, truncation_psi=1,
                truncation_cutoff=None, return_ws=False, **synthesis_kwargs):
        force_fp32 = opt_get(synthesis_kwargs, ['force_fp32'], False)
        block_args = opt_get(synthesis_kwargs, ['block_kwargs'], {'force_fp32': force_fp32})
        layer_args = opt_get(block_args, ['layer_kwargs'], {})

        assert (z is not None and lq is not None) or (ws is not None and enc_conv_outs is not None)
        if lq is not None:
            conv_outs, enc_latent = self.do_encoder(lq, **synthesis_kwargs)
        else:
            conv_outs = enc_conv_outs
        if z is not None:
            ws = self.do_latent_mapping_for_single(z, enc_latent, truncation_psi, truncation_cutoff)

        if self.freeze_synthesis_network:
            for p in self.gen_bank.synthesis.parameters():
                p.requires_grad = False
            self.gen_bank.synthesis.eval()

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
            synblock = getattr(synth, f'b{res}')
            if first:
                x, img = synblock(x, img, cur_ws, **block_args)
                # TODO: FIX ME - this needs to get the x output of the synblock to match the original lq image input.
                #x = torch.nn.functional.interpolate(x, scale_factor=4, mode='bilinear')
                #img = torch.nn.functional.interpolate(img, scale_factor=4, mode='bilinear')
            else:
                # First, convert to the correct memory format and dtype. The synthesis network would normally have done this for us, but we're prepending it.
                dtype = torch.float16 if synblock.use_fp16 and not force_fp32 else torch.float32
                memory_format = torch.channels_last if synblock.channels_last and not force_fp32 else torch.contiguous_format
                x = x.to(dtype=dtype, memory_format=memory_format)

                # Encoder attachment
                eres = res // 2
                if eres in conv_outs.keys():
                    x = torch.cat([x, conv_outs[eres].to(dtype=dtype, memory_format=memory_format)], dim=1)
                    # x = torch.cat([x, conv_outs[eres]], dim=1)
                    layer = getattr(self, f'enc_attachment_{eres}')
                    enc_w = cur_ws[:,0,:]  # Just use the first block's w for the encoder attachment.
                    x = layer(x, enc_w, **layer_args)

                # Call into the synthesis blocks.
                x, img = synblock(x, img, cur_ws, puppet=True, **block_args)
        if return_ws:
            return img, ws
        else:
            return img


def convert_stylegan2_pickle(pkl, enc_input_resolution):
    with dnnlib.util.open_url(pkl) as f:
        resume_data = legacy.load_network_pkl(f)
    G = dnnlib.util.construct_class_by_name(class_name='training.gan_sr.SrGenerator',
                                            enc_input_resolution=enc_input_resolution, **resume_data['G'].init_kwargs)
    G_ema = copy.deepcopy(G).eval()
    for name, module in [('G_ema', G.gen_bank),
                         ('G_ema', G_ema.gen_bank)]:  # Note: both G and G_ema start from the saved G_ema intentionally.
        module.load_state_dict(resume_data[name].state_dict(), strict=True)

    # Discriminator is a special case - we want to copy the parameters from the lower half of the network, but leave
    # the new init in the upper half.
    D = dnnlib.util.construct_class_by_name(class_name='training.networks.Discriminator',
                                            stop_training_at=enc_input_resolution, **resume_data['D'].init_kwargs)
    pretrained_sd = resume_data['D'].state_dict()
    scrubbed_sd = OrderedDict()
    for k, v in pretrained_sd.items():
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
        del module  # conserve memory
    with open('converted-0.pkl', 'wb') as f:
        pickle.dump(snapshot_data, f)


if __name__ == '__main__':
    args = {
        'z_dim': 512,
        'w_dim': 512,
        'img_resolution': 256,
        'img_channels': 3,
        'enc_input_resolution': 64,
        'mapping_kwargs': {'num_layers': 8},
        'synthesis_kwargs': {
            'channel_base': 16384,
            'channel_max': 512,
            'num_fp16_res': 4,
            'conv_clamp': 256
        }
    }
    gen = SrGenerator(**args).to('cuda')
    z = torch.rand((1,512)).to('cuda')
    c = torch.zeros((1,0)).to('cuda')
    lq = torch.rand((1,3,64,64)).to('cuda')
    #gen(z, None, lq, truncation_psi=1, truncation_cutoff=None)
    # Also test the style mixing method
    enc_c, enc_l = gen.do_encoder(lq=lq)
    ws = gen.do_latent_mapping_with_mixing(z=z, enc_latent=enc_l, mixing_prob=1)
    img = gen(ws=ws, enc_conv_outs=enc_c)
    #convert_stylegan2_pickle('results/imgset_256.pkl', 32)
