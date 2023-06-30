#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import math
import functools
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
from transformer import Transformer 
import BigGAN_layers as layers
from batchnorm import SynchronizedBatchNorm2d as SyncBatchNorm2d
from util import to_device, load_network
from networks import init_weights
from params import *


# In[2]:


def D_arch(ch=64, attention='64', input_nc=3, ksize='333333', dilation='111111'):
    arch = {}
    arch[256] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8, 8, 16]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 8, 16, 16]],
                 'downsample': [True] * 6 + [False],
                 'resolution': [128, 64, 32, 16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 8)}}
    arch[128] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8, 16]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 16, 16]],
                 'downsample': [True] * 5 + [False],
                 'resolution': [64, 32, 16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 8)}}
    arch[64] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8]],
                'out_channels': [item * ch for item in [1, 2, 4, 8, 16]],
                'downsample': [True] * 4 + [False],
                'resolution': [32, 16, 8, 4, 4],
                'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                              for i in range(2, 7)}}
    arch[63] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8]],
                'out_channels': [item * ch for item in [1, 2, 4, 8, 16]],
                'downsample': [True] * 4 + [False],
                'resolution': [32, 16, 8, 4, 4],
                'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                              for i in range(2, 7)}}
    arch[32] = {'in_channels': [input_nc] + [item * ch for item in [4, 4, 4]],
                'out_channels': [item * ch for item in [4, 4, 4, 4]],
                'downsample': [True, True, False, False],
                'resolution': [16, 16, 16, 16],
                'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                              for i in range(2, 6)}}
    arch[129] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8, 8, 16]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 8, 16, 16]],
                 'downsample': [True] * 6 + [False],
                 'resolution': [128, 64, 32, 16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 8)}}
    arch[33] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8, 16]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 16, 16]],
                 'downsample': [True] * 5 + [False],
                 'resolution': [64, 32, 16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 10)}}
    arch[31] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8, 16]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 16, 16]],
                 'downsample': [True] * 5 + [False],
                 'resolution': [64, 32, 16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 10)}}
    
    #relevant for set resolution
    arch[16] = {'in_channels': [input_nc] + [ch * item for item in [1, 8, 16]],
                 'out_channels': [item * ch for item in [1, 8, 16, 16]],
                 'downsample': [True] * 3 + [False],
                 'resolution': [16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 5)}}

    arch[17] = {'in_channels': [input_nc] + [ch * item for item in [1, 4]],
                 'out_channels': [item * ch for item in [1, 4, 8]],
                 'downsample': [True] * 3,
                 'resolution': [16, 8, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 5)}}


    arch[20] = {'in_channels': [input_nc] + [ch * item for item in [1, 8, 16]],
                 'out_channels': [item * ch for item in [1, 8, 16, 16]],
                 'downsample': [True] * 3 + [False],
                 'resolution': [16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 5)}}
    return arch


# In[3]:


'''# resolution is the resolution of the image and will decide the INPUT of the discriminator model
   # if res = 16 then [batch_size, channel_num, height, width] we have [batch_size, channel_num, 16, 16]
     assume channel 1 for grayscale.
'''
class Discriminator(nn.Module):

    def __init__(self, D_ch=64, D_wide=True, resolution=resolution,
                 D_kernel_size=3, D_attn='64', n_classes=VOCAB_SIZE,
                 num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False),
                 SN_eps=1e-8, output_dim=1, D_mixed_precision=False, D_fp16=False,
                 D_init='N02', skip_init=False, D_param='SN', gpu_ids=[0],bn_linear='SN', input_nc=1, one_hot=False, **kwargs):

        super(Discriminator, self).__init__()
        self.name = 'D'
        # gpu_ids
        self.gpu_ids = gpu_ids
        # one_hot representation
        self.one_hot = one_hot
        # Width multiplier
        self.ch = D_ch
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.D_wide = D_wide
        # Resolution, pretty important defines input image res
        self.resolution = resolution
        # Kernel size
        self.kernel_size = D_kernel_size
        # Attention?
        self.attention = D_attn
        # Number of classes
        self.n_classes = n_classes
        # Activation
        self.activation = D_activation
        # Initialization style
        self.init = D_init
        # Parameterization style
        self.D_param = D_param
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # Fp16?
        self.fp16 = D_fp16
        # Architecture
        self.arch = D_arch(self.ch, self.attention, input_nc)[resolution]
        '''
        # Which convs, batchnorms, and linear layers to use
        # SN-Spectral Normalisation applied to stabilise the GAN model since the discriminator
        learns faster than the generator the spectral norm i.e. the weight in the discriminator network which
        has the highest value has set limit and cannot exceed that certain value. So instead of normal layers
        we apply SN infused layers in the D.
        '''
        if self.D_param == 'SN':
            self.which_conv = functools.partial(layers.SNConv2d,
                                                kernel_size=3, padding=1,
                                                num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                eps=self.SN_eps)
            self.which_linear = functools.partial(layers.SNLinear,
                                                  num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                  eps=self.SN_eps)
            self.which_embedding = functools.partial(layers.SNEmbedding,
                                                     num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                     eps=self.SN_eps)
            if bn_linear=='SN':
                self.which_embedding = functools.partial(layers.SNLinear,
                                                         num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                         eps=self.SN_eps)
        else:
            self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = nn.Linear
            # We use a non-spectral-normed embedding here regardless;
            # For some reason applying SN to G's embedding seems to randomly cripple G
            self.which_embedding = nn.Embedding
        if one_hot:
            self.which_embedding = functools.partial(layers.SNLinear,
                                                         num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                         eps=self.SN_eps)
        # Prepare model
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[layers.DBlock(in_channels=self.arch['in_channels'][index],
                                           out_channels=self.arch['out_channels'][index],
                                           which_conv=self.which_conv,
                                           wide=self.D_wide,
                                           activation=self.activation,
                                           preactivation=(index > 0),
                                           downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]
            # If attention on this block, attach it to the end
            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index],
                                                     self.which_conv)]
        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        # Linear output layer. The output dimension is typically 1, but may be
        # larger if we're e.g. turning this into a VAE with an inference output
        self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)
        # Embedding for projection discrimination
        self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])

        # Initialize weights
        if not skip_init:
            self = init_weights(self, D_init)

    def forward(self, x, y=None, **kwargs):
        # Stick x into h for cleaner for loops without flow control
        h = x
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
        # Apply global sum pooling as in SN-GAN
        h = torch.sum(self.activation(h), [2, 3])
        # Get initial class-unconditional output
        out = self.linear(h)
        # Get projection of final featureset onto class vectors and add to evidence
        if y is not None:
            out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)
        return out

    def return_features(self, x, y=None):
        # Stick x into h for cleaner for loops without flow control
        h = x
        block_output = []
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
                block_output.append(h)
        # Apply global sum pooling as in SN-GAN
        # h = torch.sum(self.activation(h), [2, 3])
        return block_output


# In[1]:


class WDiscriminator(nn.Module):

    def __init__(self, D_ch=64, D_wide=True, resolution=resolution,
                 D_kernel_size=3, D_attn='64', n_classes=VOCAB_SIZE,
                 num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False),
                 SN_eps=1e-8, output_dim=339, D_mixed_precision=False, D_fp16=False,
                 D_init='N02', skip_init=False, D_param='SN', gpu_ids=[0],bn_linear='SN', input_nc=1, one_hot=False, **kwargs):
        super(WDiscriminator, self).__init__()
        self.name = 'D'
        # gpu_ids
        self.gpu_ids = gpu_ids
        # one_hot representation
        self.one_hot = one_hot
        # Width multiplier
        self.ch = D_ch
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.D_wide = D_wide
        # Resolution
        self.resolution = resolution
        # Kernel size
        self.kernel_size = D_kernel_size
        # Attention?
        self.attention = D_attn
        # Number of classes
        self.n_classes = n_classes
        # Activation
        self.activation = D_activation
        # Initialization style
        self.init = D_init
        # Parameterization style
        self.D_param = D_param
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # Fp16?
        self.fp16 = D_fp16
        # Architecture
        self.arch = D_arch(self.ch, self.attention, input_nc)[resolution]

        # Which convs, batchnorms, and linear layers to use
        # No option to turn off SN in D right now
        if self.D_param == 'SN':
            self.which_conv = functools.partial(layers.SNConv2d,
                                                kernel_size=3, padding=1,
                                                num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                eps=self.SN_eps)
            self.which_linear = functools.partial(layers.SNLinear,
                                                  num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                  eps=self.SN_eps)
            self.which_embedding = functools.partial(layers.SNEmbedding,
                                                     num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                     eps=self.SN_eps)
            if bn_linear=='SN':
                self.which_embedding = functools.partial(layers.SNLinear,
                                                         num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                         eps=self.SN_eps)
        else:
            self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = nn.Linear
            # We use a non-spectral-normed embedding here regardless;
            # For some reason applying SN to G's embedding seems to randomly cripple G
            self.which_embedding = nn.Embedding
        if one_hot:
            self.which_embedding = functools.partial(layers.SNLinear,
                                                         num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                         eps=self.SN_eps)
        # Prepare model
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[layers.DBlock(in_channels=self.arch['in_channels'][index],
                                           out_channels=self.arch['out_channels'][index],
                                           which_conv=self.which_conv,
                                           wide=self.D_wide,
                                           activation=self.activation,
                                           preactivation=(index > 0),
                                           downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]
            # If attention on this block, attach it to the end
            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index],
                                                     self.which_conv)]
        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        # Linear output layer. The output dimension is typically 1, but may be
        # larger if we're e.g. turning this into a VAE with an inference output
        self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)
        # Embedding for projection discrimination
        self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])
        self.cross_entropy = nn.CrossEntropyLoss()
        # Initialize weights
        if not skip_init:
            self = init_weights(self, D_init)

    def forward(self, x, y=None, **kwargs):
        # Stick x into h for cleaner for loops without flow control
        h = x
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
        # Apply global sum pooling as in SN-GAN
        h = torch.sum(self.activation(h), [2, 3])
        # Get initial class-unconditional output
        out = self.linear(h)
        # Get projection of final featureset onto class vectors and add to evidence
        #if y is not None:
        #out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)

        loss = self.cross_entropy(out, y.long())

        return loss

    def return_features(self, x, y=None):
        # Stick x into h for cleaner for loops without flow control
        h = x
        block_output = []
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
                block_output.append(h)
        # Apply global sum pooling as in SN-GAN
        # h = torch.sum(self.activation(h), [2, 3])
        return block_output


# In[ ]:


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, kernel_size=3, norm_layer=nn.Identity, sn=True,
                 num_D_SVs=1, num_D_SV_itrs=1, SN_eps=1e-8):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        self.sn = sn
        self.SN_eps = SN_eps
        if self.sn:
            self.which_conv = functools.partial(layers.SNConv2d,
                                                padding=1,
                                                num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                eps=self.SN_eps)

        kw = kernel_size
        padw = 1
        sequence = [self.which_conv(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.ReLU(inplace=False)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                self.which_conv(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=True),
                # norm_layer(ndf * nf_mult),
                nn.ReLU(inplace=False)
            ]

        nf_mult_prev = nf_mult
        # nf_mult = min(2 ** n_layers, 8)
        # sequence += [
        #     self.which_conv(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=True),
        #     # norm_layer(ndf * nf_mult),
        #     nn.ReLU(inplace=False)
        # ]

        sequence += [self.which_conv(nf_mult * ndf, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, x, x_lens, y_lens):
        """Standard forward."""
        h = self.model(x)
        h_lens = x_lens * h.size(-1) // (x.size(-1) + 1e-8)
        mask = _len2mask(h_lens.int(), h.size(-1), torch.float32).to(x.device).detach()
        mask = mask.view(mask.size(0), 1, 1, mask.size(1))
        h = torch.sum(h * mask, [2, 3])
        h = h / y_lens.unsqueeze(dim=-1)
        return h

