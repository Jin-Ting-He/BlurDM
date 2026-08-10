import torch
import torch.nn as nn
import sys
sys.path.append('/home/jthe/FFTformer')
from thop import profile
from einops import rearrange


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.LeakyReLU(0.1, inplace=True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        # self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
    
class MLP(nn.Module):
    def __init__(self,
                 num_patches,
                 embed_dims,
                 patch_expansion,
                 channel_expansion,
                 **kwargs):

        super(MLP, self).__init__()

        patch_mix_dims = int(patch_expansion * num_patches)
        channel_mix_dims = int(channel_expansion * embed_dims)

        self.patch_mixer = nn.Sequential(
            nn.Linear(num_patches, patch_mix_dims),
            nn.GELU(),
            nn.Linear(patch_mix_dims, num_patches),
        )

        self.channel_mixer = nn.Sequential(
            nn.Linear(embed_dims, channel_mix_dims),
            nn.GELU(),
            nn.Linear(channel_mix_dims, embed_dims),
        )

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)

    def forward(self, x):
        x = x + self.patch_mixer(self.norm1(x).transpose(1,2)).transpose(1,2)
        x = x + self.channel_mixer(self.norm2(x))

        return x

class latent_encoder_gelu(nn.Module):

    def __init__(self, in_chans=6, embed_dim=64, block_num=6, stage=1, group=4, patch_expansion=0.5, channel_expansion=4):
        super(latent_encoder_gelu, self).__init__()

        assert in_chans == int(6//stage), "in chanel size is wrong"

        self.group = group

        self.pixel_unshuffle = nn.PixelUnshuffle(4)
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_chans*16, embed_dim, 3, 1, 1), 
                nn.GELU(),
                )

        self.blocks = nn.ModuleList()
        for i in range(block_num):
            block = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, 1, 1), 
                nn.GELU(),
                nn.Conv2d(embed_dim, embed_dim, 3, 1, 1))
            self.blocks.append(block)

        self.conv2 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d((group, group))
        self.mlp = MLP(num_patches=group*group, embed_dims=embed_dim, patch_expansion=patch_expansion, channel_expansion=channel_expansion)
        self.end = nn.Sequential(
                nn.Linear(embed_dim, embed_dim*4),
                nn.GELU(),)
        

    def forward(self, inp_img, gt=None):
        if gt is not None:
            x = torch.cat([gt, inp_img], dim=1)
        else:
            x = inp_img

        x = self.pixel_unshuffle(x)
        x = self.conv1(x)
        for block in self.blocks:
            x = block(x) + x
        x = self.pool(self.conv2(x))
        x = rearrange(x, 'b c h w-> b (h w) c')
        x = self.mlp(x)
        x = self.end(x)
        return x


class latent_encoder_lrelu(nn.Module):

    def __init__(self, in_chans=6, embed_dim=64, block_num=4, stage=1, group=4, patch_expansion=0.5, channel_expansion=4):
        super(latent_encoder_lrelu, self).__init__()

        assert in_chans == int(6//stage), "in chanel size is wrong"

        self.group = group

        self.pixel_unshuffle = nn.PixelUnshuffle(4)
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_chans*16, embed_dim, 3, 1, 1), 
                nn.LeakyReLU(0.1, True),
                )

        self.blocks = nn.ModuleList()
        for i in range(block_num):
            block = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, 1, 1), 
                nn.LeakyReLU(0.1, True),
                nn.Conv2d(embed_dim, embed_dim, 3, 1, 1))
            self.blocks.append(block)

        self.conv2 = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1, True),
                nn.Conv2d(embed_dim * 2, embed_dim * 2, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1, True),
                nn.Conv2d(embed_dim * 2, embed_dim * 4, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1, True),
        )
        self.pool = nn.AdaptiveAvgPool2d((group, group))
        self.mlp = MLP(num_patches=group*group, embed_dims=embed_dim*4, patch_expansion=patch_expansion, channel_expansion=channel_expansion)
        self.end = nn.Sequential(
                nn.Linear(embed_dim*4, embed_dim*4),
                nn.LeakyReLU(0.1, True),)
        

    def forward(self, inp_img, gt=None):
        if gt is not None:
            x = torch.cat([gt, inp_img], dim=1)
        else:
            x = inp_img

        x = self.pixel_unshuffle(x)
        x = self.conv1(x)
        for block in self.blocks:
            x = block(x) + x
        x = self.pool(self.conv2(x))
        x = rearrange(x, 'b c h w-> b (h w) c')
        x = self.mlp(x)
        x = self.end(x)
        return x

class LE_arch(nn.Module):
    def __init__(self,n_feats = 64, n_encoder_res = 6, bn = False):
        super(LE_arch, self).__init__()
        E1=[nn.Conv2d(96, n_feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True)]
        E2=[
            ResBlock(
                default_conv, n_feats, kernel_size=3, bn=bn
            ) for _ in range(n_encoder_res)
        ]
        E3=[
            nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        ]
        E=E1+E2+E3
        self.E = nn.Sequential(
            *E
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True)
        )
        
        self.pixel_unshuffle = nn.PixelUnshuffle(4)
    def forward(self, x,gt):
        gt0 = self.pixel_unshuffle(gt)
        x0 = self.pixel_unshuffle(x)
        x = torch.cat([x0, gt0], dim=1)
        fea = self.E(x).squeeze(-1).squeeze(-1)
        fea1 = self.mlp(fea)
        return fea1
    

if __name__ == '__main__':
    net = LE_arch(
    ).cuda()

    input = torch.randn((1, 3, 256, 256)).cuda()
    flops, params = profile(net, (input,input))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')

    out = net(input, input)
    print(out.shape)

    test = torch.randn((1, 16, 256))
    print(test.shape)