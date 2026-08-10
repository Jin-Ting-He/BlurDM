import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from thop import profile

from models.LatentEncoder import LE_arch
# from latent_encoder_arch import LE_arch

class ResMLP(nn.Module):
    def __init__(self,n_feats = 512):
        super(ResMLP, self).__init__()
        self.resmlp = nn.Sequential(
            nn.Linear(n_feats , n_feats ),
            nn.LeakyReLU(0.1, True),
        )
    def forward(self, x):
        res=self.resmlp(x)
        return res
    
class denoise(nn.Module):
    def __init__(self,n_feats = 64, n_denoise_res = 5,timesteps=5):
        super(denoise, self).__init__()
        # self.max_period=timesteps*10
        self.max_period=timesteps
        n_featsx4=4*n_feats
        resmlp = [
            nn.Linear(n_featsx4*2+1, n_featsx4),
            nn.LeakyReLU(0.1, True),
        ]
        for _ in range(n_denoise_res):
            resmlp.append(ResMLP(n_featsx4))
        self.resmlp=nn.Sequential(*resmlp)

    def forward(self,x, t,c):
        t=t.float()
        t =t/self.max_period
        t=t.view(-1,1)
        c = torch.cat([c,t,x],dim=1)
        fea = self.resmlp(c)

        return fea

class LatentExposureDiffusion(nn.Module):
    def __init__(self, total_timestamps=5, spvised_mid_out=False, grids = False, train_size=384):
        super().__init__()
        self.grids_flag = grids
        self.overlap_size = (32, 32)
        self.kernel_size = [train_size, train_size]
        self.lcr_model = denoise(timesteps = total_timestamps)
        self.noise_model = denoise(timesteps = total_timestamps)
        self.condition_encoder = LE_arch()

        self.total_timestamps = total_timestamps
        self.spvised_mid_out = spvised_mid_out

        # self.r = 0.2
        beta_start = 0.0
        beta_end = 0.02 # default 0.02
        alpha_start = 1.0
        alpha_end = 2.0
        betas = torch.linspace(beta_start, beta_end, self.total_timestamps, dtype=torch.float32)
        alphas = torch.linspace(alpha_start, alpha_end, self.total_timestamps+1, dtype=torch.float32)
        betas_bar_list = self.get_beta_bar(alphas, betas)
        time_stamps_list = torch.tensor([torch.tensor(i) for i in range(self.total_timestamps, 0, -1)])
        
        self.register_buffer("alphas", alphas)
        self.register_buffer("betas_bar", betas_bar_list)
        self.register_buffer("time_stamps_list", time_stamps_list)

    def get_beta_bar(self, alphas, betas):
        betas_bar_list = []
        for t in range(1, self.total_timestamps+1):
            sub_betas = betas[:t]  # 取前 t 个 beta 值
            weights = torch.tensor([(alphas[i-1] / alphas[t])**2 for i in range(1, t+1)], dtype=torch.float32)
            result = torch.sum(weights * sub_betas)
            betas_bar_list.append(result.clone().detach().sqrt())
        return torch.tensor(betas_bar_list)
    
    def q_sample_d(self, img):
        noise = torch.randn_like(img)
        return img + self.betas_bar[self.total_timestamps-1] * noise, noise
    
    # def model_loading(self):
    #     print("Load Pretrained LCRDDM !")

    #     ce_checkpoint_path = 'experiments/GoPro_fftformer_stage1_1e-5/models/net_le_100000.pth'
    #     ce_checkpoint = torch.load(ce_checkpoint_path)
    #     ce_checkpoint = ce_checkpoint['params']
    #     self.condition_encoder.load_state_dict(ce_checkpoint)

    #     lcr_path = "experiments/FFTformer_stage2v2/lcr_final.pth"
    #     noise_path = "experiments/FFTformer_stage2v2/noise_final.pth"

    #     lcr_checkpoint_path = lcr_path
    #     lcr_checkpoint = torch.load(lcr_checkpoint_path)
    #     self.lcr_model.load_state_dict(lcr_checkpoint)

    #     noise_checkpoint_path = noise_path
    #     noise_checkpoint = torch.load(noise_checkpoint_path)
    #     self.noise_model.load_state_dict(noise_checkpoint)

    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c, h, w)
        assert b == 1
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        overlap_size = self.overlap_size  # (64, 64)

        stride = (k1 - overlap_size[0], k2 - overlap_size[1])
        self.stride = stride
        num_row = (h - overlap_size[0] - 1) // stride[0] + 1
        num_col = (w - overlap_size[1] - 1) // stride[1] + 1
        self.nr = num_row
        self.nc = num_col

        # import math
        step_j = k2 if num_col == 1 else stride[1]  # math.ceil((w - stride[1]) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else stride[0]  # math.ceil((h - stride[0]) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        self.ek1, self.ek2 = None, None
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                # if not self.ek1:
                #     # print(step_i, i, k1, h)
                #     self.ek1 = i + k1 - h # - self.overlap_size[0]
                i = h - k1
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    # if not self.ek2:
                    #     self.ek2 = j + k2 - w # + self.overlap_size[1]
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts
    
    def forward(self, blur):
        if self.grids_flag:
            blur = self.grids(blur)
        pred_lcr_list = []
        pred_noise_list = []
        device = self.alphas.device
        b = blur.shape[0]
        T_z = self.condition_encoder(blur, blur)
        noise_img, noise = self.q_sample_d(T_z)
        for i in self.time_stamps_list:
            # t = i.unsqueeze(0)
            t = torch.full((b,), i,  device=device, dtype=torch.long)
            # print(t.shape, T_z.shape, noise_img.shape)
            pred_noise = self.noise_model(noise_img, t, T_z)
            pred_lcr = self.lcr_model(noise_img, t, T_z)

            if self.spvised_mid_out:
                pred_lcr_list.append(pred_lcr)
                pred_noise_list.append(pred_noise)

            if i == 1:
                noise_cof = self.betas_bar[i - 1]
            else: 
                beta_t_bar = self.betas_bar[i - 1]
                beta_t_minus1_bar = self.betas_bar[i - 2]
                noise_cof = (self.alphas[i]*beta_t_bar)/self.alphas[i-1] - beta_t_minus1_bar

            noise_img = ((self.alphas[i]*(noise_img) - pred_lcr)/self.alphas[i-1]) - noise_cof * pred_noise
            # if i.item()==5:
            #     break
        if self.spvised_mid_out:
            return noise_img, pred_lcr_list, pred_noise_list, noise
        else:
            return noise_img
        
class LatentExposureDiffusionCeFix(nn.Module):
    def __init__(self, total_timestamps, spvised_mid_out=False):
        super().__init__()

        self.lcr_model = latent_predictor()
        self.noise_model = latent_predictor()

        self.total_timestamps = total_timestamps
        self.spvised_mid_out = spvised_mid_out

        self.r = 0.2
        beta_start = 0.0 
        beta_end = 0.02
        alpha_start = 1.0
        alpha_end = 2.0
        betas = torch.linspace(beta_start, beta_end, self.total_timestamps, dtype=torch.float32)
        alphas = torch.linspace(alpha_start, alpha_end, self.total_timestamps+1, dtype=torch.float32)
        betas_bar_list = self.get_beta_bar(alphas, betas)
        time_stamps_list = torch.tensor([torch.tensor(i) for i in range(self.total_timestamps, 0, -1)])
        
        self.register_buffer("alphas", alphas)
        self.register_buffer("betas_bar", betas_bar_list)
        self.register_buffer("time_stamps_list", time_stamps_list)

    def get_beta_bar(self, alphas, betas):
        betas_bar_list = []
        for t in range(1, self.total_timestamps+1):
            sub_betas = betas[:t]  # 取前 t 个 beta 值
            weights = torch.tensor([(alphas[i-1] / alphas[t])**2 for i in range(1, t+1)], dtype=torch.float32)
            result = torch.sum(weights * sub_betas)
            betas_bar_list.append(result.clone().detach().sqrt())
        return torch.tensor(betas_bar_list)
    
    def q_sample_d(self, img):
        noise = torch.randn_like(img)
        return img + self.betas_bar[self.total_timestamps-1] * noise, noise
    
    def model_loading(self, lcr_ckpt_path ,noise_ckpt_path):

        lcr_checkpoint_path = lcr_ckpt_path
        lcr_checkpoint = torch.load(lcr_checkpoint_path)
        self.lcr_model.load_state_dict(lcr_checkpoint)

        noise_checkpoint_path = noise_ckpt_path
        noise_checkpoint = torch.load(noise_checkpoint_path)
        self.noise_model.load_state_dict(noise_checkpoint)
    
    def forward(self, T_z):
        pred_lcr_list = []
        pred_noise_list = []

        # T_z = self.condition_encoder(blur, blur)
        noise_img, noise = self.q_sample_d(T_z)
        for i in self.time_stamps_list:
            t = i.unsqueeze(0)
            pred_noise = self.noise_model(noise_img, T_z, t)
            pred_lcr = self.lcr_model(noise_img, T_z, t)

            if self.spvised_mid_out:
                pred_lcr_list.append(pred_lcr)
                pred_noise_list.append(pred_noise)

            if i == 1:
                noise_cof = self.betas_bar[i - 1]
            else: 
                beta_t_bar = self.betas_bar[i - 1]
                beta_t_minus1_bar = self.betas_bar[i - 2]
                noise_cof = (self.alphas[i]*beta_t_bar)/self.alphas[i-1] - beta_t_minus1_bar

            noise_img = ((self.alphas[i]*(noise_img) - pred_lcr)/self.alphas[i-1]) - noise_cof * pred_noise
        
        if self.spvised_mid_out:
            return noise_img, pred_lcr_list, pred_noise_list, noise
        else:
            return noise_img

if __name__ == '__main__':
    net = LatentExposureDiffusion(
    ).cuda()

    input = torch.randn((8, 3, 256, 256)).cuda()
    # flops, params = profile(net, (input,))
    # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    # print('Params = ' + str(params / 1000 ** 2) + 'M')

    out = net(input)
    print(out.shape)