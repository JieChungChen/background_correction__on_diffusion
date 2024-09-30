import os
import torch
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from ddpm.model import Diffusion_UNet
from ddpm.diffusion import GaussianDiffusionSampler


def check_distributed():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = local_rank = world_size = -1
    is_distributed = world_size != -1
    return rank, local_rank, world_size, is_distributed


def model_eval(model, dataset, args):
    with torch.no_grad():
        model.eval()
        sampler = GaussianDiffusionSampler(model, args.beta_1, args.beta_T, args.T).to(args.device)
        noisyImage = torch.randn(size=[1, 3, args.img_size, args.img_size], device=args.device)
        # saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        input_img = dataset[0][0].squeeze().numpy()
        ref_truth = dataset[0][1].squeeze().numpy()
        pred = sampler(dataset[0][0], noisyImage).squeeze().cpu().numpy()
        obj_pred = input_img-pred
        fig = plt.figure()
        plt.subplot(141)
        plt.title('input img')
        plt.imshow(input_img, cmap='gray', vmin=0, vmax=1)
        plt.subplot(142)
        plt.title('ref pred')
        plt.imshow(pred, cmap='gray', vmin=0, vmax=1)
        plt.subplot(143)
        plt.title('ground truth')
        plt.imshow(ref_truth, cmap='gray', vmin=0, vmax=1)
        plt.subplot(144)
        plt.title('input-pred')
        plt.imshow(obj_pred, cmap='gray', vmin=0, vmax=1)
        plt.savefig('eval_visualize.png')