import torch
import pandas as pd
import torch.nn as nn
import torchvision.models as models
from transformer import *
from blocks import LinearBlock, Conv2dBlock, ResBlocks, ActFirstResBlock
from params import *
from OCR_network import *
from dataset import *

class FCNDecoder(nn.Module):
    def __init__(self, ups=3, n_res=2, dim=512, out_dim=1, res_norm='adain', activ='relu', pad_type='reflect'):
        super(FCNDecoder, self).__init__()

        self.model = []
        self.model += [ResBlocks(n_res, dim, res_norm,
                                 activ, pad_type=pad_type)]
        for i in range(ups):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2,
                                       norm='in',
                                       activation=activ,
                                       pad_type=pad_type)]
            dim //= 2
        self.model += [Conv2dBlock(dim, out_dim, 7, 1, 3,
                                   norm='none',
                                   activation='tanh',
                                   pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        y =  self.model(x)

        return y


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        INP_CHANNEL = NUM_EXAMPLES
        if IS_SEQ: INP_CHANNEL = 1


        encoder_layer = TransformerEncoderLayer(TN_HIDDEN_DIM, TN_NHEADS, TN_DIM_FEEDFORWARD,
                                                TN_DROPOUT, "relu", True)
        encoder_norm = nn.LayerNorm(TN_HIDDEN_DIM) if True else None
        self.encoder = TransformerEncoder(encoder_layer, TN_ENC_LAYERS, encoder_norm)

        decoder_layer = TransformerDecoderLayer(TN_HIDDEN_DIM, TN_NHEADS, TN_DIM_FEEDFORWARD,
                                                TN_DROPOUT, "relu", True)
        decoder_norm = nn.LayerNorm(TN_HIDDEN_DIM)
        self.decoder = TransformerDecoder(decoder_layer, TN_DEC_LAYERS, decoder_norm,
                                          return_intermediate=True)

        self.Feat_Encoder = nn.Sequential(*([nn.Conv2d(INP_CHANNEL, 64, kernel_size=7, stride=2, padding=3, bias=False)] +list(models.resnet18(pretrained=True).children())[1:-2]))

        self.query_embed = nn.Embedding(VOCAB_SIZE, TN_HIDDEN_DIM)


        self.linear_q = nn.Linear(TN_DIM_FEEDFORWARD, TN_DIM_FEEDFORWARD*8)

        self.DEC = FCNDecoder(res_norm = 'in')


        self._muE = nn.Linear(512,512)
        self._logvarE = nn.Linear(512,512)

        self._muD = nn.Linear(512,512)
        self._logvarD = nn.Linear(512,512)


        self.l1loss = nn.L1Loss()

        self.noise = torch.distributions.Normal(loc=torch.tensor([0.]), scale=torch.tensor([1.0]))
    def reparameterize(self, mu, logvar):

        mu = torch.unbind(mu , 1)
        logvar = torch.unbind(logvar , 1)

        outs = []

        for m,l in zip(mu, logvar):

            sigma = torch.exp(l)
            eps = torch.cuda.FloatTensor(l.size()[0],1).normal_(0,1)
            eps  = eps.expand(sigma.size())

            out = m + sigma*eps

            outs.append(out)


        return torch.stack(outs, 1)


    def Eval(self, ST, QRS):

        if IS_SEQ:
            B, N, R, C = ST.shape
            FEAT_ST = self.Feat_Encoder(ST.view(B*N, 1, R, C))
            FEAT_ST = FEAT_ST.view(B, 512, 1, -1)
        else:
            FEAT_ST = self.Feat_Encoder(ST)


        FEAT_ST_ENC = FEAT_ST.flatten(2).permute(2,0,1)

        memory = self.encoder(FEAT_ST_ENC)

        if IS_KLD:

            Ex = memory.permute(1,0,2)

            memory_mu = self._muE(Ex)
            memory_logvar = self._logvarE(Ex)

            memory = self.reparameterize(memory_mu, memory_logvar).permute(1,0,2)


        OUT_IMGS = []

        for i in range(QRS.shape[1]):

            QR = QRS[:, i, :]

            if ALL_CHARS:
                QR_EMB = self.query_embed.weight.repeat(batch_size,1,1).permute(1,0,2)
            else:
                QR_EMB = self.query_embed.weight[QR].permute(1,0,2)

            tgt = torch.zeros_like(QR_EMB)

            hs = self.decoder(tgt, memory, query_pos=QR_EMB)

            if IS_KLD:

                Dx = hs[0].permute(1,0,2)

                hs_mu = self._muD(Dx)
                hs_logvar = self._logvarD(Dx)

                hs = self.reparameterize(hs_mu, hs_logvar).permute(1,0,2).unsqueeze(0)


            h = hs.transpose(1, 2)[-1]#torch.cat([hs.transpose(1, 2)[-1], QR_EMB.permute(1,0,2)], -1)
            if ADD_NOISE: h = h + self.noise.sample(h.size()).squeeze(-1).to(DEVICE)

            h = self.linear_q(h)
            h = h.contiguous()

            if ALL_CHARS: h = torch.stack([h[i][QR[i]] for i in range(batch_size)], 0)

            h = h.view(h.size(0), h.shape[1]*2, 4, -1)
            h = h.permute(0, 3, 2, 1)

            h = self.DEC(h)


            OUT_IMGS.append(h.detach())



        return OUT_IMGS






    def forward(self, ST, QR, QRs = None, mode = 'train'):

        #Attention Visualization Init


        enc_attn_weights, dec_attn_weights = [], []

        self.hooks = [

            self.encoder.layers[-1].self_attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            ),
            self.decoder.layers[-1].multihead_attn.register_forward_hook(
                lambda self, input, output: dec_attn_weights.append(output[1])
            ),
        ]


        #Attention Visualization Init

        B, N, R, C = ST.shape
        FEAT_ST = self.Feat_Encoder(ST.view(B*N, 1, R, C))
        FEAT_ST = FEAT_ST.view(B, 512, 1, -1)



        FEAT_ST_ENC = FEAT_ST.flatten(2).permute(2,0,1)

        memory = self.encoder(FEAT_ST_ENC)

        QR_EMB = self.query_embed.weight[QR].permute(1,0,2)

        tgt = torch.zeros_like(QR_EMB)

        hs = self.decoder(tgt, memory, query_pos=QR_EMB)


        h = hs.transpose(1, 2)[-1]#torch.cat([hs.transpose(1, 2)[-1], QR_EMB.permute(1,0,2)], -1)

        if ADD_NOISE: h = h + self.noise.sample(h.size()).squeeze(-1).to(DEVICE)

        h = self.linear_q(h)
        h = h.contiguous()

        h = h.view(h.size(0), h.shape[1]*2, 4, -1)
        h = h.permute(0, 3, 2, 1)

        h = self.DEC(h)

        self.dec_attn_weights = dec_attn_weights[-1].detach()
        self.enc_attn_weights = enc_attn_weights[-1].detach()



        for hook in self.hooks:
            hook.remove()

        return h
















