import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from thop import profile
    
class LayerNorm_Without_Shape(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm_Without_Shape, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return self.body(x)

class HIM(nn.Module):
    def __init__(self, dim, num_heads, bias, embed_dim, LayerNorm_type, qk_scale=None):
        super(HIM, self).__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.norm1 = LayerNorm_Without_Shape(dim, LayerNorm_type)
        self.norm2 = LayerNorm_Without_Shape(embed_dim*4, LayerNorm_type)

        self.q = nn.Linear(dim, dim, bias=bias)
        self.kv = nn.Linear(embed_dim*4, 2*dim, bias=bias)
        
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, prior):
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        _x = self.norm1(x)
        prior = self.norm2(prior)
        
        q = self.q(_x)
        kv = self.kv(prior)
        k,v = kv.chunk(2, dim=-1)   

        q = rearrange(q, 'b n (head c) -> b head n c', head=self.num_heads)
        k = rearrange(k, 'b n (head c) -> b head n c', head=self.num_heads)
        v = rearrange(v, 'b n (head c) -> b head n c', head=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head n c -> b n (head c)', head=self.num_heads)
        out = self.proj(out)

        # sum
        x = x + out
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W).contiguous()

        return x
    
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):

        super(DFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class FSAS(nn.Module):
    def __init__(self, dim, bias):
        super(FSAS, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')

        self.patch_size = 8

        self.kv = nn.Linear(256, 2 * dim, bias=bias)

    def forward(self, x, prior):
        B, C, H, W = x.shape

        kv = self.kv(prior)
        kv = kv.view(B, C * 2, 1, 1)
        k, v = kv.chunk(2, dim=1)  

        x = x * k + v

        hidden = self.to_hidden(x)

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())

        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)

        out = self.norm(out)

        output = v * out
        output = self.project_out(output)

        return output


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', att=False):
        super(TransformerBlock, self).__init__()

        self.att = att
        if self.att:
            self.norm1 = LayerNorm(dim, LayerNorm_type)
            self.attn = FSAS(dim, bias)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = DFFN(dim, ffn_expansion_factor, bias)

    def forward(self, inp):
        x = inp[0]
        prior = inp[1]
        # print("########")
        # print("Before", torch.mean(x), torch.var(x))
        if self.att:
            x = x + self.attn(self.norm1(x), prior)

        x = x + self.ffn(self.norm2(x))

        return [x, prior]


class Fuse(nn.Module):
    def __init__(self, n_feat):
        super(Fuse, self).__init__()
        self.n_feat = n_feat
        self.att_channel = TransformerBlock(dim=n_feat * 2)

        self.conv = nn.Conv2d(n_feat * 2, n_feat * 2, 1, 1, 0)
        self.conv2 = nn.Conv2d(n_feat * 2, n_feat * 2, 1, 1, 0)

    def forward(self, enc, dnc, prior):
        x = self.conv(torch.cat((enc, dnc), dim=1))
        x = self.att_channel([x, prior])[0]
        x = self.conv2(x)
        e, d = torch.split(x, [self.n_feat, self.n_feat], dim=1)
        output = e + d

        return output


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat * 2, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat // 2, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.body(x)


##########################################################################
##---------- FFTformer -----------------------
class fftformer_prior(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[6, 6, 12, 8],
                 num_refinement_blocks=4,
                 ffn_expansion_factor=3,
                 bias=False,
                 ):
        super(fftformer_prior, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in
            range(num_blocks[0])])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_blocks[2])])

        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_refinement_blocks)])

        self.fuse2 = Fuse(dim * 2)
        self.fuse1 = Fuse(dim)
        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, prior):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1([inp_enc_level1, prior])[0]

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2([inp_enc_level2, prior])[0]

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3([inp_enc_level3, prior])[0]

        out_dec_level3 = self.decoder_level3([out_enc_level3, prior])[0]

        inp_dec_level2 = self.up3_2(out_dec_level3)

        inp_dec_level2 = self.fuse2(inp_dec_level2, out_enc_level2, prior)

        out_dec_level2 = self.decoder_level2([inp_dec_level2, prior])[0]

        inp_dec_level1 = self.up2_1(out_dec_level2)

        inp_dec_level1 = self.fuse1(inp_dec_level1, out_enc_level1, prior)
        out_dec_level1 = self.decoder_level1([inp_dec_level1, prior])[0]

        out_dec_level1 = self.refinement([out_dec_level1, prior])[0]

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1

# if __name__ == '__main__':
#     img = torch.zeros((1, 3, 256, 256)).cuda()
#     prior = torch.randn((1, 256)).cuda()
#     model = fftformer_prior().eval().cuda()
#     with torch.no_grad():
#         flops, params = profile(model, (img, prior))
#         print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
#         print('Params = ' + str(params / 1000 ** 2) + 'M')

if __name__ == '__main__':
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True  # 為不同大小的輸入加速

    # 建立模型
    net = fftformer_prior().cuda()
    net.eval()

    # 建立 4K 測試影像輸入與 prior 向量
    input_tensor = torch.randn(1, 3, 2160, 3840).cuda()  # 4K 圖像大小
    prior = torch.randn(1, 256).cuda()  # 根據你的模型需求

    # 預熱 (warm-up)
    with torch.no_grad():
        for _ in range(5):
            _ = net(input_tensor, prior)

    # 監測記憶體與推理時間
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    with torch.no_grad():
        output = net(input_tensor, prior)
    end_time = time.time()
    inference_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

    # 計算 FLOPs 和參數量（注意：這裡是以 256x256 為估算基準）
    dummy_input = torch.randn(1, 3, 256, 256).cuda()
    dummy_prior = torch.randn(1, 256).cuda()
    flops, params = profile(net, (dummy_input, dummy_prior))

    # 輸出結果
    print(f"[4K Input] Inference Time: {inference_time:.3f} sec")
    print(f"[4K Input] Peak Memory: {peak_memory:.2f} MB")
    print(f"[256x256 Input] FLOPs = {flops / 1e9:.2f} GFLOPs")
    print(f"[256x256 Input] Params = {params / 1e6:.2f} M")