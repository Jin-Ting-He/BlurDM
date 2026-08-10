import torch
import torch.nn as nn
from torchvision.utils import save_image
import os
import sys
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from dataloader import Test_Loader_DDP
from torch.utils.data import Dataset, DataLoader
from LoFormer.models.LoFormerBlurDM import LoFormerPrior
from LoFormer.models.LatentBlurDM import LatentExposureDiffusion
from utils.utils import calc_psnr, same_seed, count_parameters, tensor2cv, AverageMeter, judge_and_remove_module_dict
from torchvision.transforms import functional as F
from accelerate import Accelerator

@torch.no_grad()
def predict(model, model_dm, args, device):
    model.eval()
    model_dm.eval()
    if args.dataset == 'GoPro+HIDE':
        dataset_name = ['GoPro', 'HIDE']
    else:
        dataset_name = [args.dataset]

    for val_dataset_name in dataset_name:
        accelerator = Accelerator()
        dataset_path = os.path.join(args.data_path, val_dataset_name)

        dataset = Test_Loader_DDP(data_path=dataset_path,
                                crop_size=args.crop_size,
                                ZeroToOne=True)
        dataloader = DataLoader(
                        dataset,
                        batch_size=1,
                    )
        save_dir = os.path.join(args.dir_path, 'results', f'{val_dataset_name}')
        os.makedirs(save_dir, exist_ok=True)
        # dataset_len = len(dataset)
        # tq = tqdm.tqdm(range(dataset_len))
        # tq.set_description(f'Predict {val_dataset_name}')
        model, model_dm, dataloader = accelerator.prepare(model, model_dm, dataloader)

        for iter_idx, data in tqdm(enumerate(dataloader), disable=not accelerator.is_main_process):
            sample = data
            input = sample['blur'].to(device)
            gt = sample['sharp'].to(device)
            b, c, h, w = input.shape
            # factor=32
            # h_n = (factor - h % factor) % factor
            # w_n = (factor - w % factor) % factor
            # input = torch.nn.functional.pad(input, (0, w_n, 0, h_n), mode='reflect')
            # gt = torch.nn.functional.pad(gt, (0, w_n, 0, h_n), mode='reflect')
            z_pred = model_dm(input)
            output = model(input, z_pred)
            # output = output[:, :, :h, :w]
            # print(torch.max(output), torch.min(output), "##################")
            output = output.clamp(0, 1)

            # output = output + 0.5
            # output += 0.5 / 255
            image_name = sample['name'][0]
            save_img_path = os.path.join(save_dir, image_name)
            # pred = F.to_pil_image(output.squeeze(0).cpu(), 'RGB')
            # pred.save(save_img_path)
            save_image(output.squeeze(0).cpu(), save_img_path)



if __name__ == "__main__":
    # hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--data_path", default='./dataset/test', type=str)
    parser.add_argument("--dir_path", default='./results/LoFormer/GoPro', type=str)
    parser.add_argument("--model_path", default="/home/jthe/DeblurDM/weights/LoFormer/GoPro/epoch_2500_LoFormer.pth", type=str)
    parser.add_argument("--model_dm_path", default="/home/jthe/DeblurDM/weights/LoFormer/GoPro/epoch_2500_LoFormer.pth", type=str)
    parser.add_argument("--model", default='LoFormerPrior', type=str, choices=['LoFormerPrior'])
    parser.add_argument("--dataset", default='GoPro', type=str, choices=['GoPro+HIDE', 'GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R', 'RWBI'])
    parser.add_argument("--crop_size", default=None, type=int)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device :", device)
    load_model_state = torch.load(args.model_path)

    if not os.path.isdir(args.dir_path):
        os.makedirs(args.dir_path, exist_ok=True)

    # Model and optimizer
    net = LoFormerPrior(train_size=384)
    net_dm = LatentExposureDiffusion(total_timestamps=5, grids=True, train_size=384)
    
    # state_dict = torch.load(args.model_path)
    # state_dict = state_dict['params']
    # net.load_state_dict(state_dict,strict = True)

    # state_dict = torch.load(args.model_dm_path)    
    # state_dict = state_dict['params']
    # net_dm.load_state_dict(state_dict,strict = True)
    load_model_state = torch.load(args.model_path)
    load_model_dm_state = torch.load(args.model_dm_path)

    if 'model_state' in load_model_state.keys():
        load_model_state["model_state"] = judge_and_remove_module_dict(load_model_state["model_state"])
        net.load_state_dict(load_model_state["model_state"])
    elif 'model' in load_model_state.keys():
        load_model_state["model"] = judge_and_remove_module_dict(load_model_state["model"])
        net.load_state_dict(load_model_state["model"])
    else:
        load_model_state = judge_and_remove_module_dict(load_model_state)
        net.load_state_dict(load_model_state)

    load_model_dm_state["model_dm_state"] = judge_and_remove_module_dict(load_model_dm_state["model_dm_state"])
    net_dm.load_state_dict(load_model_dm_state["model_dm_state"])

    # net = nn.DataParallel(net)
    net.to(device)

    # net_dm = nn.DataParallel(net_dm)
    net_dm.to(device)

    print("device:", device)
    print(f'args: {args}')
    print(f'model: {net}')
    print(f'model parameters: {count_parameters(net)}')
    print(f'model dm parameters: {count_parameters(net_dm)}')

    same_seed(2023)
    predict(net, net_dm, args=args, device=device)





