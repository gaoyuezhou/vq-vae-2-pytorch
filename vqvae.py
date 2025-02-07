import torch
from torch import nn
from torch.nn import functional as F

import distributed as dist_fn


# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)
        # import pdb; pdb.set_trace()

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
    ):
        super().__init__()

        # self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        # # self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        # # self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        # # self.quantize_t = Quantize(embed_dim, n_embed)
        # # self.dec_t = Decoder(
        # #     embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        # # )
        # # self.quantize_conv_b = nn.Conv2d(channel, embed_dim, 4, stride=4, padding=4) # hack, just for dim to match
        # self.quantize_conv_b = Encoder(channel, channel, n_res_block, n_res_channel, stride=4)
        
        self.dino_enc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        # feeze dino encoder
        for param in self.dino_enc.parameters():
            param.requires_grad = False
        import torchvision.transforms as T
        self.dino_trans =  T.Compose([T.Resize(196)])

        self.quantize_b = Quantize(embed_dim, n_embed)
        # self.upsample_b = nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=4, padding=4) # the dims here are brute forced to match
        # # self.upsample_t = nn.ConvTranspose2d(
        # #     embed_dim, embed_dim, 4, stride=2, padding=1
        # # )
        self.upsample_b = Decoder(embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=4)
        self.dec = Decoder(
            embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )
        self.info = f"in_channel: {in_channel}, channel: {channel}, n_res_block: {n_res_block}, n_res_channel: {n_res_channel}, embed_dim: {embed_dim}, n_embed: {n_embed}, decay: {decay}"

    def forward(self, input):
        quant_b, diff, _ = self.encode(input)
        dec = self.decode(quant_b)

        return dec, diff

    def encode(self, input):
        # # import pdb; pdb.set_trace()
        # enc_b = self.enc_b(input) # (3, 256, 256) --> (128, 64, 64) desired: (3, 224, 224) --> (384, 16, 16)
        # # enc_t = self.enc_t(enc_b)

        # # quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        # # import pdb; pdb.set_trace()
        # # quant_t, diff_t, id_t = self.quantize_t(quant_t) # 32 * 32 vectors of dim 64. codebook contains vectors of dim 64.
        # # quant_t = quant_t.permute(0, 3, 1, 2)
        # # diff_t = diff_t.unsqueeze(0)

        # # dec_t = self.dec_t(quant_t)
        # # enc_b = torch.cat([dec_t, enc_b], 1)

        # quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1) # (batch, 14, 14, 384)
        # import pdb; pdb.set_trace()
        reshaped_input = self.dino_trans(input)
        # import pdb; pdb.set_trace()
        patch_tokens = self.dino_enc.forward_features(reshaped_input)['x_norm_patchtokens'] # (batch, 256, 384)
        patch_tokens = patch_tokens.reshape(patch_tokens.shape[0], 196//self.dino_enc.patch_size, 196//self.dino_enc.patch_size, -1) # (batch, 14, 14, 384) # TODO: double check dino patch feature order
        quant_b, diff_b, id_b = self.quantize_b(patch_tokens)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_b, diff_b, id_b

    # def decode(self, quant_t, quant_b):
    #     upsample_t = self.upsample_t(quant_t) # quant_t: (64, 32, 32) --> (64, 64, 64)
    #     quant = torch.cat([upsample_t, quant_b], 1)
    #     dec = self.dec(quant) # quant: (128, 64, 64)
    #     import pdb; pdb.set_trace()

    #     return dec
    def decode(self, quant_b):
        upsample_b = self.upsample_b(quant_b) 
        dec = self.dec(upsample_b) # quant: (128, 64, 64)
        # import pdb; pdb.set_trace()
        return dec

    # def decode_code(self, code_t, code_b):
    #     quant_t = self.quantize_t.embed_code(code_t)
    #     quant_t = quant_t.permute(0, 3, 1, 2)
    #     quant_b = self.quantize_b.embed_code(code_b)
    #     quant_b = quant_b.permute(0, 3, 1, 2)

    #     dec = self.decode(quant_t, quant_b)

    #     return dec
    def decode_code(self, code_b):
        # quant_t = self.quantize_t.embed_code(code_t)
        # quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_b)

        return dec
