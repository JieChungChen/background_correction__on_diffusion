import os
import torch
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['figure.dpi'] = 200
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


def model_eval(dataset, args):
    with torch.no_grad():
        model = Diffusion_UNet().to(args.device)
        model.load_state_dict(torch.load(args.model_save_dir+'/'+args.checkpoint, map_location=args.device), strict=False)
        print("Model weight load down.")
        model.eval()
        sampler = GaussianDiffusionSampler(model, args.beta_1, args.beta_T, args.T).to(args.device)
        noisyImage = torch.randn(size=[1, 1, args.img_size, args.img_size], device=args.device)
        # saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        input_img = dataset[200][0].squeeze().numpy()
        ref_truth = dataset[200][1].squeeze().numpy()
        pred = sampler(dataset[0][0].unsqueeze(0).to(args.device), noisyImage).squeeze().cpu().numpy()
        obj_pred = input_img-pred
        fig = plt.figure()
        plt.subplot(151)
        plt.title('input img')
        plt.axis('off')
        plt.imshow(input_img, cmap='gray')
        plt.subplot(152)
        plt.title('noise')
        plt.axis('off')
        plt.imshow(noisyImage.squeeze().cpu().numpy(), cmap='gray')
        plt.subplot(153)
        plt.title('ref pred')
        plt.axis('off')
        plt.imshow(pred, cmap='gray')
        plt.subplot(154)
        plt.title('ground truth')
        plt.axis('off')
        plt.imshow(ref_truth, cmap='gray')
        plt.subplot(155)
        plt.title('input-pred')
        plt.axis('off')
        plt.imshow(obj_pred, cmap='gray')
        plt.savefig('eval_visualize.png')

    