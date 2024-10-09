import os
import torch
import numpy as np
import glob
from tqdm import tqdm
import matplotlib as mpl
from PIL import Image
mpl.use('Agg')
mpl.rcParams['figure.dpi'] = 200
import matplotlib.pyplot as plt
from ddpm.model import Diffusion_UNet
from ddpm.diffusion_sr3 import GaussianDiffusionSampler


def check_distributed():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = local_rank = world_size = -1
    is_distributed = world_size != -1
    return rank, local_rank, world_size, is_distributed


def model_eval(args, n_samples=8, model=None):
    with torch.no_grad():
        size = args.img_size
        dref_files = sorted(glob.glob("%s/dref/*.tif"%args.data_dir))
        ref_files = sorted(glob.glob("%s/ref/*.tif"%args.data_dir))
        dref_imgs, ref_imgs = [], []
        dref_rnd_choose = np.random.choice(len(dref_files), n_samples, replace=False)
        ref_rnd_choose = np.random.choice(len(ref_files), n_samples, replace=False)
        for i in tqdm(dref_rnd_choose, dynamic_ncols=True, desc='load dref images'):
            raw_dref = Image.open(dref_files[i]).resize((size, size))
            dref_imgs.append(np.array(raw_dref))
        for i in tqdm(ref_rnd_choose, dynamic_ncols=True, desc='load ref images'):
            raw_ref = Image.open(ref_files[i]).resize((size, size))
            ref_imgs.append(np.array(raw_ref))
        dref_imgs = torch.Tensor(np.array(dref_imgs)).unsqueeze(1)
        ref_imgs = torch.Tensor(np.array(ref_imgs)).unsqueeze(1)
        input_imgs = [dref_imgs[i]*ref_imgs[i] for i in range(n_samples)]
        input_imgs = torch.concatenate(input_imgs, dim=0).unsqueeze(1)
        pair_wise_maximum = input_imgs.view(n_samples, size**2).max(dim=1).values.view(-1, 1, 1, 1)
        input_imgs = input_imgs/pair_wise_maximum
        ref_imgs = ref_imgs/pair_wise_maximum

        if model is None:
            model = Diffusion_UNet().to(args.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(args.model_save_dir+'/'+args.checkpoint, map_location=args.device), strict=False)
            print("Model weight load down.")

        model.eval()
        sampler = GaussianDiffusionSampler(model, args.beta_1, args.beta_T, args.beta_sche, args.T).to(args.device)
        for i in range(n_samples):
            noisy_imgs = torch.randn(size=[1, 1, size, size], device=args.device)
            pred = sampler(input_imgs[i].view(1, 1, size, size).to(args.device), noisy_imgs).squeeze().cpu().numpy()
            fig = plt.figure()
            plt.subplot(231)
            plt.axis('off')
            plt.title('input img')
            plt.imshow(input_imgs[i, 0], cmap='gray', vmin=0, vmax=1)
            plt.subplot(234)
            plt.axis('off')
            plt.title('noise')
            plt.imshow(noisy_imgs.squeeze().cpu().numpy(), cmap='gray')
            plt.subplot(232)
            plt.axis('off')
            plt.title('ref pred')
            plt.imshow(pred, cmap='gray', vmin=0, vmax=1)
            plt.subplot(235)
            plt.axis('off')
            plt.title('ref gt')
            plt.imshow(ref_imgs[i, 0], cmap='gray', vmin=0, vmax=1)
            plt.subplot(233)
            plt.axis('off')
            plt.title('input-pred')
            plt.imshow(input_imgs[i, 0]/pred, cmap='gray')
            plt.subplot(236)
            plt.axis('off')
            plt.title('dref gt')
            plt.imshow(dref_imgs[i, 0], cmap='gray')
            fig.tight_layout()
            plt.savefig('figures/eval_visualize_%s.png'%str(i).zfill(3))


def model_eval_for_val(args, model=None, epoch=1):
    with torch.no_grad():
        size = args.img_size
        val_file = '20230505_Red2-b2-60s-m1/20230505_Red2-b2-60s-m1_0003.tif'
        raw_img = Image.open('./valid_data_n/data2/original/%s'%val_file).resize((size, size))
        raw_img = np.array(raw_img)
        dref_truth = Image.open('./valid_data_n/data2/gt_dref/%s'%val_file).resize((size, size))
        ref_truth = (raw_img/dref_truth)/raw_img.max()
        dref_truth = dref_truth/raw_img.max()
        print(ref_truth.max(), ref_truth.min())
        input_img = torch.Tensor(raw_img/raw_img.max())
        
        if model is None:
            model = Diffusion_UNet().to(args.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(args.model_save_dir+'/'+args.checkpoint, map_location=args.device), strict=False)
            print("Model weight load down.")
        model.eval()
        sampler = GaussianDiffusionSampler(model, args.beta_1, args.beta_T, args.beta_sche, args.T).to(args.device)
        noisyImage = torch.randn(size=[1, 1, size, size], device=args.device)
        pred = sampler(input_img.view(1, 1, size, size).to(args.device), noisyImage).squeeze().cpu().numpy()
        print(pred.max(), pred.min())
        obj_pred = input_img/pred
        fig = plt.figure()
        plt.subplot(231)
        plt.title('input img')
        plt.axis('off')
        plt.imshow(input_img, cmap='gray', vmin=0, vmax=1)
        plt.subplot(234)
        plt.title('noise')
        plt.axis('off')
        plt.imshow(noisyImage.squeeze().cpu().numpy(), cmap='gray')
        plt.subplot(232)
        plt.title('ref pred')
        plt.axis('off')
        plt.imshow(pred, cmap='gray', vmin=0, vmax=1)
        plt.subplot(235)
        plt.title('ref gt')
        plt.axis('off')
        plt.imshow(ref_truth, cmap='gray', vmin=0, vmax=1)
        plt.subplot(233)
        plt.title('input-pred')
        plt.axis('off')
        plt.imshow(obj_pred, cmap='gray')
        plt.subplot(236)
        plt.title('dref gt')
        plt.imshow(dref_truth, cmap='gray')
        plt.axis('off')
        fig.tight_layout()
        plt.savefig('figures/eval_visualize_ep%d.png'%(epoch))
    