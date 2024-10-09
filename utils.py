import os
import torch
import numpy as np
import glob
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


def model_eval(args, model=None):
    with torch.no_grad():
        size = args.img_size
        raw_dref = Image.open('./training_data_n/dref/20230320_MB-700-b2-60s2x2-m1_0004.tif').resize((size, size))
        raw_dref = np.array(raw_dref)
        ref_files = sorted(glob.glob("%s/ref/*.tif"%args.data_dir))
        ref_truth = Image.open(ref_files[38]).resize((size, size))
        ref_truth = np.array(ref_truth)
        input_img = raw_dref*ref_truth
        ref_truth = ref_truth/input_img.max()
        input_img = torch.Tensor(input_img/input_img.max())
        
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
        plt.imshow(raw_dref, cmap='gray')
        plt.axis('off')
        fig.tight_layout()
        plt.savefig('eval_visualize.png')


def model_eval_for_val(args, model=None, epoch=1):
    with torch.no_grad():
        size = args.img_size
        raw_img = Image.open('./valid_data_n/data2/original/20230505_Red2-b2-60s-m1/20230505_Red2-b2-60s-m1_0003.tif').resize((size, size))
        raw_img = np.array(raw_img)
        dref_truth = Image.open('./valid_data_n/data2/gt_dref/20230505_Red2-b2-60s-m1/20230505_Red2-b2-60s-m1_0003.tif').resize((size, size))
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
        plt.savefig('eval_visualize_ep%d.png'%(epoch))
    